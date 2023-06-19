'''
Created on 13 avr. 2023

@author: verne
'''
# Importations
import matplotlib as mpl
from matplotlib import pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import newton
from networkx.algorithms.flow.capacityscaling import capacity_scaling
import math
from cmaes import CMA
import cma
import time
from bokeh.util.compiler import calc_cache_key
import os
from pathlib import Path
from pandas.tests.plotting.test_converter import dates

from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem
from pymoo.util.running_metric import RunningMetricAnimation

from sklearn.cluster import DBSCAN


from turbineEfficiency import TurbineEfficiency
from simulation import TurbineOperations
from simulation import Dam
from simulation import Revenue




def get_q_in_tethys(file, dates):
    
    data = pd.read_excel(file, index_col=0)
    data["Year"] = pd.DatetimeIndex(data.index).year
    data["Month"] = pd.DatetimeIndex(data.index).month
    data["Day"] = pd.DatetimeIndex(data.index).day
    start_index = np.where( (data['Year']==dates[0].year) & (data['Month']==dates[0].month) & (data['Day']==dates[0].day))[0][0]
    end_index = np.where((data['Year']==dates[0].year) & (data['Month']==dates[-1].month) & (data['Day']==dates[-1].day))[0][0]
    q_in = data.iloc[start_index:end_index+1]
    
    q_in_ = q_in.drop(columns=['Year','Month','Day','Targets'])
    #===========================================================================
    # q_in_mean = q_in_.mean(axis=1,skipna=True, numeric_only=False) # along the columns
    # q_in_min = q_in_.min(axis=1) 
    # q_in_max = q_in_.max(axis=1) 
    #===========================================================================
    
    weights = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
    
    final_q_in_= q_in_
    

    for j in range (len(weights)-1):
        current_q = q_in_.iloc[:,j] * weights[j]
        final_q_in_.iloc[:,j] = current_q
    
    
    final_q_in_ = final_q_in_.sum(axis=1)/sum(weights)
     
    return final_q_in_






def get_prices(dates):
    '''
    Gets the EPEX-SPOT prices depending on the period over which there is the optimization
    From data: data_SPOTPhelix 2005-2017
    '''
    
    data = pd.read_excel('data_SPOTPhelix.xlsx', index_col=0)
    data = data.set_index('Date')
    prices = data.iloc[:,0]
 
    data["Year"] = pd.DatetimeIndex(prices.index).year
    data["Month"] = pd.DatetimeIndex(prices.index).month
    data["Day"] = pd.DatetimeIndex(prices.index).day
     
    grouped_prices = data.groupby([pd.DatetimeIndex(prices.index).month,pd.DatetimeIndex(prices.index).day]).mean()
    grouped_prices.drop(columns = 'Year')
    
    #dates = pd.date_range('2017-04-20', '2017-05-05', freq=freq)
    dates = dates
    
    start_index = np.where((grouped_prices['Month']==dates[0].month) & (grouped_prices['Day']==dates[0].day))[0][0]
    end_index = np.where((grouped_prices['Month']==dates[-1].month) & (grouped_prices['Day']==dates[-1].day))[0][0]
    final_prices = grouped_prices.iloc[start_index:end_index+1]
    
    final_prices = final_prices.iloc[:,0]
    
    final_prices.index = dates
    final_prices = final_prices.resample(rule=freq).interpolate(method='linear').loc[dates[0]:dates[-1]]
    

    return final_prices














class Multi_Enguri(Problem):

    def __init__(self, q_in, price_in, ini_level, freq, dates):
        
        self.q_in = q_in
        self.price_in = price_in
        self.freq = freq
        self.dates = dates
        self.ini_level = ini_level
        
        self.q_target = self.get_q_target(dates)
        self.lower_bounds = np.zeros_like(self.q_target.values.ravel())
        self.upper_bounds = np.zeros_like(self.q_target.values.ravel()) + 1500
        self.n_var = len(self.q_target)


        self.pareto_points = None

        self.n_obj = 2
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         xl=self.lower_bounds,
                         xu=self.upper_bounds)



    #===========================================================================
    # def plot_pareto(self, result):
    #     '''
    #      
    #     '''
    #      
    #     self.pareto_front()
    #      
    #     if isinstance(self.pareto_points, type(None)):
    #         self.pareto_points = plt.plot(-result[:,0], -result[:,1], '.')[0]
    #         self.pareto_fig = plt.gcf()
    #         self.pareto_ax = plt.gca()
    #         plt.show(block=False)
    #     else:
    #         if self.pareto_fig!=plt.gcf():
    #             plt.close(plt.gcf())
    #          
    #         self.pareto_points.set_xdata(-result[:,0])
    #         self.pareto_points.set_ydata(-result[:,1])
    #         self.pareto_ax.set_xlim((min(-result[:,0]), max(-result[:,0])))
    #         self.pareto_ax.set_ylim((min(-result[:,1]), max(-result[:,1])))
    #         self.pareto_fig.canvas.draw()
    #         self.pareto_fig.canvas.flush_events()
    #===========================================================================


    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate each solution in the population
        result = np.empty((x.shape[0], self.n_obj)) * np.nan
        for i0 in range(x.shape[0]):
            x_ = x[i0, :]
            obj1, obj2 = self.objective_function(q_target = x_)
            result[i0, 0] = obj1
            result[i0, 1] = obj2
            
        #=======================================================================
        # self.plot_pareto(result)
        #=======================================================================
            
        out["F"] = result

    def objective_function(self, q_target, input_coherence_penalty=1, safety_penalty=1E6, spill_threshold=600, flood_penalty=1E4, penalty_factor_for_objective_2=100):
        """
        Objective function that should be maximized
        Inputs : revenue 
                 h_target = constraint: h must be equal to h_target at the end of the optimization
                 final_h = that should be equal to h_target
        
        """
                
        results = dam.runSimulation(self.q_in, q_target, self.price_in, self.ini_level, self.q_in.index[0], self.q_in.index[-1], runFrequency=self.freq, resultFrequency=self.freq, resample_series=False)
        revenue = results['Revenue [M EUR]'].sum()
        final_h = results['h [m a.s.l.]'].iloc[-1]
        
        real_outflow = results.loc[:,['Qspilled [m3/s]', 'Qturbined [m3/s]']].sum(axis=1)
        abs_deviation_from_target = np.abs(real_outflow-q_target)
        mean_abs_deviation_from_target = abs_deviation_from_target.mean()            
            
        comparison_h_hmax = max(0, (results['h [m a.s.l.]'] - results['h MWL [m a.s.l.]']).sum() )
        hmax_exceedance_occurrences = comparison_h_hmax                                         # count the number of times that the water levels exceeded the MWL
            
            
        mean_q_spill_above_threshold = max(0, (results['Qspilled [m3/s]'] - spill_threshold).sum())
        mean_q_spill_above_threshold = mean_q_spill_above_threshold
            
        penalties = -mean_abs_deviation_from_target*input_coherence_penalty -hmax_exceedance_occurrences*safety_penalty -mean_q_spill_above_threshold*flood_penalty
        
        objective1 = -(revenue + penalties) 
        objective2 = -(final_h + penalties*penalty_factor_for_objective_2) 


        return (objective1, objective2)



    @staticmethod
    def get_q_target(dates):
        '''
        Gets the q_target depending on the period over which there is the optimization
        From data: discharges turbined at Enguri (1995-2021)
        '''
        
        data = pd.read_excel('Discharges_turbined.xlsx', index_col=0)
        q_target = data.iloc[:,0]
     
        data["Year"] = pd.DatetimeIndex(q_target.index).year
        data["Month"] = pd.DatetimeIndex(q_target.index).month
        data["Day"] = pd.DatetimeIndex(q_target.index).day
         
        q_target = data.groupby([pd.DatetimeIndex(q_target.index).month,pd.DatetimeIndex(q_target.index).day]).mean()
        q_target.drop(columns = 'Year')
        
        #dates = pd.date_range('2017-04-20', '2017-05-05', freq=freq)
        dates = dates
        
        start_index = np.where((q_target['Month']==dates[0].month) & (q_target['Day']==dates[0].day))[0][0]
        end_index = np.where((q_target['Month']==dates[-1].month) & (q_target['Day']==dates[-1].day))[0][0]
        final_q_target = q_target.iloc[start_index:end_index+1]
        
        final_q_target = final_q_target.iloc[:,0]
        
        final_q_target.index = dates
        final_q_target = final_q_target.resample(rule=freq).interpolate(method='linear').loc[dates[0]:dates[-1]]
        
    
        return final_q_target




if __name__=='__main__':
     
    # Dam construction
    
    q_max_per_turbine = 98 # m3/s
    turbines = TurbineOperations('turbineEfficiency.csv', q_rated=q_max_per_turbine/1.15, number_of_groups=5, sep=';', index_col=0)
         
    dam = Dam(table_levels='Enguri reservoir levels.xls',
            table_dates='Enguri reservoir dates.xlsx',
            penstock_length=15000,
            penstock_diameter=9.5,
            ks=0.1,
            turbineOperations=turbines,
            revenues= None,
            downstream_level=200,
            tailrace=120
            )
    
    # q_in initialization
    data = pd.read_excel('Tethys data (Enguri dam, Q.Enguri water level).xlsx', index_col=0)
    data.index = pd.to_datetime(data.index, format='%y-%m-%d')
    q_in = data['Enguri dam, Q'].dropna().interpolate(method='linear')
    
    
    #===========================================================================
    # excel_prices = pd.read_excel('data_SPOTPhelix.xlsx', index_col='Date', usecols=['Date', 'Price [€/MWh]']).dropna()
    # excel_prices.to_pickle('prices.pkl', protocol=4)
    #===========================================================================
    excel_prices = pd.read_pickle('prices.pkl')
    
    mpl.use('QtAgg')
 
    # Initialization
    start_run = time.time()
    
    ini_level = 475
    
    freq = '1D'
    
    #===========================================================================
    # dates = pd.date_range('2017-04-20', '2017-05-05', freq=freq)
    # q_in_ = q_in.resample(rule=freq).interpolate(method='linear').loc[dates[0]:dates[-1]] 
    # price_in = excel_prices.resample(rule=freq).mean().interpolate(method='linear').loc[dates[0]:dates[-1]].iloc[:,0] 
    #===========================================================================
    
    dates = pd.date_range('2023-04-25', '2023-05-09', freq=freq)
    q_in_ = get_q_in_tethys('01.xlsx', dates)
    price_in = get_prices(dates)
    
    
        
    multi_enguri = Multi_Enguri(q_in=q_in_, price_in=price_in, freq=freq, dates=dates, ini_level=ini_level)
    algorithm = UNSGA3(get_reference_directions("das-dennis", 2, n_partitions=12),
                       pop_size=100,
                       eliminate_duplicates=True)
    termination = ('n_gen', 40)

    res = minimize(multi_enguri,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    X = res.X
    F = res.F


    end_run = time.time()
    duration = end_run-start_run
    print('duration: ' + str(duration))

    max_revenue = max(-F[:,0])
    idx_max_revenue = np.argmax(-F[:,0])
    max_water_level = max(-F[:,1])
    idx_max_water_level = np.argmax(-F[:,1])

    
    sol_max_revenue = dam.runSimulation(q_in_, X[idx_max_revenue], price_in, ini_level, q_in_.index[0], q_in_.index[-1], runFrequency='1D', resultFrequency='1D', resample_series=False)
    dam.plotSimulation()
    sol_max_revenue.to_excel('multi_rev.xlsx', index=False)
    
    sol_max_water_level = dam.runSimulation(q_in_, X[idx_max_water_level], price_in, ini_level, q_in_.index[0], q_in_.index[-1], runFrequency='1D', resultFrequency='1D', resample_series=False)
    dam.plotSimulation()
    sol_max_water_level.to_excel('multi_wl.xlsx', index=False)
   
   
    sol_middle = dam.runSimulation(q_in_, X[3], price_in, ini_level, q_in_.index[0], q_in_.index[-1], runFrequency='1D', resultFrequency='1D', resample_series=False)
    dam.plotSimulation()
    sol_middle.to_excel('multi_middle.xlsx', index=False)

    # Plot solutions
    fig1, ax1 = plt.subplots()
    ax1.plot(-F[:,0], -F[:,1], '.')
    ax1.set_xlabel('Revenue [mln.EUR]')
    ax1.set_ylabel('Water level [m.a.s.l.]')
    plt.show(block=True)

    # Hypervolume
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population
    
    for algo in res.history:
    
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
    
        # retrieve the optimum from the algorithm
        opt = algo.opt
    
        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
    
        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])



    from pymoo.indicators.hv import Hypervolume
    
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    
    metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)
    
    hv = [metric.do(_F) for _F in hist_F]
    
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()

    
    #
    plt.figure()
    for f0 in hist_F[:-1]:
        plt.plot(-f0[:,0], -f0[:, 1], 'k.')
    plt.plot(-hist_F[-1][:,0], -hist_F[-1][:, 1], 'ro')
    plt.show()
    
    
    # Running Metric
    running = RunningMetricAnimation(delta_gen=10,
                        n_plots=4,
                        key_press=False,
                        do_show=True)

    for algorithm in res.history:
        running.update(algorithm)
    
    
    
    
    # Pareto front approximation

    results_for_pareto = np.vstack([-F[:,0], -F[:,1]]).T

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    clusters = dbscan.fit_predict(results_for_pareto)

    # Select the best solution from each cluster
    pareto_front = []
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        best_idx = np.argmin(results_for_pareto[idx, 0] + results_for_pareto[idx, 1])
        pareto_front.append(results_for_pareto[idx[best_idx]])

        pareto_front = np.array(pareto_front)

    # Plot the results
    fig, ax = plt.subplots()
    ax.scatter(-F[:,0], -F[:,1], c=clusters, cmap='viridis')
    ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', s=100, marker='o')
    ax.set_xlabel('Revenue [mln.EUR]')
    ax.set_ylabel('Water level [m.a.s.l.]')
    ax.set_title('Pareto Front Estimate')
    plt.show()
    
    
    









    a=1
    pass
    pass
