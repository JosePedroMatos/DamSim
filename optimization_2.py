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
from turbineEfficiency import TurbineEfficiency
from simulation import TurbineOperations
from simulation import Dam
from simulation import Revenue
from bokeh.util.compiler import calc_cache_key
import os
from pathlib import Path
from pandas.tests.plotting.test_converter import dates


#from simulation import *

# Functions



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



def opti(dam, ini_level, q_in, q_target, price_in, h_target, freq='1D'):
    """
    Optimizes the energy production
    """
    
    #===========================================================================
    # #Change the current working directory to the simulation directory
    # simulation_dir = r'C:\Users\verne\Universidade de Lisboa\IST-IST151906-PDM LOUISE VERNET - General\Python\PDM\simulation'
    # os.chdir(simulation_dir)
    #===========================================================================
    
    
    # first try: safety_penalty = 1E9, spill_threshold = 300, flood_penalty = 1E6
    # second try: safety_penalty = 1E6, spill_threshold = 300, flood_penalty = 1E4
    
    
    def objective_function(q_target, q_in, price_in, freq, deviation_penalty=1E3, deviation_tolerance=0.25, within_tolerance_factor=0.01, input_coherence_penalty=1, safety_penalty = 1E6, spill_threshold = 600, flood_penalty = 1E4):
        """
        Objective function that should be maximized
        Inputs : revenue 
                 h_target = constraint: h must be equal to h_target at the end of the optimization
                 final_h = that should be equal to h_target
        
        """
        
        results = dam.runSimulation(q_in, q_target, price_in, ini_level, q_in.index[0], q_in.index[-1], runFrequency=freq, resultFrequency=freq, resample_series=False)
        revenue = results['Revenue [M EUR]'].sum()
        
        
        real_outflow = results.loc[:,['Qspilled [m3/s]', 'Qturbined [m3/s]']].sum(axis=1)
        abs_deviation_from_target = np.abs(real_outflow-q_target)
        mean_abs_deviation_from_target = abs_deviation_from_target.mean()
        
        final_h = results['h [m a.s.l.]'].iloc[-1]
        deviation = final_h-h_target
        if np.abs(deviation-deviation_tolerance*0.9)<deviation_tolerance:
            deviation_penalty *= within_tolerance_factor
        
        
        comparison_h_hmax = max(0, (results['h [m a.s.l.]'] - results['h MWL [m a.s.l.]']).sum() )
        hmax_exceedance_occurrences = comparison_h_hmax                                         # count the number of times that the water levels exceeded the MWL
        
        
        mean_q_spill_above_threshold = max(0, (results['Qspilled [m3/s]'] - spill_threshold).sum())
        mean_q_spill_above_threshold = mean_q_spill_above_threshold
        
        
        #mean_q_spill_above_threshold = max(0, (results['Qspilled [m3/s]'] - spill_threshold).mean())
        #=======================================================================
        # if mean_q_spill_above_threshold <0:     # if >0 : too much water downstream, we add a penalty
        #     mean_q_spill_above_threshold = 0    # if <0 : no too much spilled (no peak), so no penalty added
        #=======================================================================
        
        
        #=======================================================================
        # dam.plotSimulation()
        # proposed = real_outflow.copy()*np.nan
        # proposed.loc[:] = q_target
        # proposed.plot(ax=plt.gca(), style=':', color='g', linewidth=3, label='Qout proposed')
        # results['Price [EUR/MWh]'].plot(ax=plt.gca(), style=':', color='k', linewidth=3)
        # plt.title('The revenue was %03.3f M EUR (deviation of %02.3f m).' % (revenue, deviation))
        #=======================================================================
        
        return revenue - ((deviation**2)*deviation_penalty) - mean_abs_deviation_from_target*input_coherence_penalty - hmax_exceedance_occurrences*safety_penalty - mean_q_spill_above_threshold*flood_penalty # maximizing in CMA-ES


    # Second version
     
    start_run = time.time()  
     

    # Set up the file path for the output files
    
    #output_path = Path(os.getcwd()) / 'results' # opti 3
    output_path = Path('C:/Users/verne/Documents') / 'results' 
    output_path.mkdir(parents=True, exist_ok=True)
     
     
     
    #output_path = os.path.join("C:", os.sep, "Users", "verne", "Universidade de Lisboa", "IST-IST151906-PDM LOUISE VERNET - General", "Python", "PDM", "simulation")
     
    #output_path = r"C:\Users\verne\Universidade de Lisboa\IST-IST151906-PDM LOUISE VERNET - General\Python\PDM\simulation\outcmaes"    #opti 1+2
    
    #output_path = r"C:\Users\verne\Universidade de Lisboa\IST-IST151906-PDM LOUISE VERNET - General\Python\PDM\simulation\resultsaxlen.dat"
    
  
     
    # Use the output_path variable in the CMA-ES options
    opts = cma.CMAOptions()
    opts.set("verb_disp", 10)
    opts.set("tolfun", 1e-4)
    opts.set("maxiter", 500)   # 3000
    

    opts.set("verb_filenameprefix", str(output_path /  'outcmaes'))    #does not work
    
    

    opts['bounds']=[np.zeros_like(q_target.values.ravel()) + 0, np.zeros_like(q_target.values.ravel()) + 1500]
    opts['popsize']=30
     
    to_optimize = lambda x: -objective_function(x, freq=freq, q_in=q_in, price_in=price_in)
    es = cma.CMAEvolutionStrategy(q_target.values.ravel(), 200, opts)
 
 
    es.optimize(to_optimize)
    es.result_pretty()
    
    es.logger.add() #
    
    
    solution = es.result.xbest
    best_results = dam.runSimulation(q_in, solution, price_in, ini_level, q_in.index[0], q_in.index[-1], runFrequency=freq, resultFrequency=freq, resample_series=False)
     
    end_run = time.time()
    duration_run = end_run - start_run
    print('duration: ' +  str(duration_run))
     
    dam.plotSimulation()
    
    #===========================================================================
    # plt.grid()
    # ax=best_results['Price [EUR/MWh]'].plot(ax=plt.gca(), style=':', color='k', linewidth=3, legend = 'Price [EUR/MWh]')
    # ax.legend(loc='upper center', bbox_to_anchor=(-0.1, 1))
    #   
    # plt.gcf().axes[0].grid(linestyle = ':')
    #===========================================================================
    
    #plt.show()
    es.logger.plot()    #
    plt.savefig("my_plot.png")      #  provides a plot with some information

    best_results.to_excel('results_opti.xlsx', index=False)
    

    return dam.results




# Main ----------------------------------------------------------------------------------------------------------------------------

     
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
    
    mpl.use('TkAgg')
 
    # Optimization
    
    freq = '1D'
    dates = pd.date_range('2017-04-20', '2017-05-05', freq=freq)
    q_in_ = q_in.resample(rule=freq).interpolate(method='linear').loc[dates[0]:dates[-1]]
    price_in = excel_prices.resample(rule=freq).mean().interpolate(method='linear').loc[dates[0]:dates[-1]].iloc[:,0]
     
    #q_target = q_in_.copy()*0 + 200
     
    q_target = get_q_target(dates)
    
    #opti_1 = opti(dam, 470, q_in_, q_target=q_target, price_in=price_in, h_target=475, freq=freq)
    
    #opti_2 = opti(dam, 500, q_in_*5, q_target=q_target, price_in=price_in, h_target=475,  freq=freq)
    
    opti_3 = opti(dam, 500, q_in_*2, q_target=q_target, price_in=price_in, h_target=475, freq=freq)
    
    #opti_4 = opti(dam, 500, q_in_*6, q_target=q_target, price_in=price_in, h_target=475,  freq=freq)
    
    #opti_5 = opti(dam, 500, q_in_*12, q_target=q_target, price_in=price_in, h_target=475,  freq=freq)
    
    

  


   


    #--------------------With Tethys---------------------------------------------------------------
    
    


    
    
    #===========================================================================
    # freq = '1D'
    # dates = pd.date_range('2023-04-25', '2023-05-09', freq=freq)
    #   
    # q_in_ = get_q_in_tethys('01.xlsx', dates)
    #    
    # price_in = get_prices(dates)
    #   
    # q_target = get_q_target(dates)
    #===========================================================================
    
 
 
    #opti_1 = opti(dam, 470, q_in_, q_target=q_target, price_in=price_in, h_target=475, freq=freq)
    
    #opti_2 = opti(dam, 500, q_in_*2, q_target=q_target, price_in=price_in, h_target=475, freq=freq)

    #opti_3 = opti(dam, 470, q_in_, q_target=q_target, price_in= -price_in, h_target=475,  freq=freq)



     




    pass
    pass
    a=1







