'''
Created on 7 mars 2023

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
import time
#from .turbineEfficiency import TurbineEfficiency

from turbineEfficiency import TurbineEfficiency #@UnresolvedImport <- this can be used to suppress the error
from bokeh.util.compiler import calc_cache_key

class TurbineOperations(object):
    '''
    Helper class that handles turbine operations
    '''

    def __init__(self, efficiency_csv, q_rated, number_of_groups, *args, **kwargs):
        turbineEfficiency = pd.read_csv(efficiency_csv, *args, **kwargs)
        turbineEfficiency.iloc[:, 0] = turbineEfficiency.iloc[:, 0].str.replace('%','').astype(float)/100
     
        self.q_rated = q_rated
        self.number_of_groups = number_of_groups
        self.efficiency_table = turbineEfficiency
        self.q_max_per_group = turbineEfficiency.index[-1] * q_rated
        
        self.turbineEfficiencyInterpolator = {}
        for g0 in range(1, self.number_of_groups+1):
            turb = TurbineEfficiency(turbineEfficiency)                 # creation of the object "turbine"
            efficiencyCurve = turb.aggregate(number=g0)
        

            self.turbineEfficiencyInterpolator[g0] = interpolate.interp1d(efficiencyCurve.index.values, efficiencyCurve.values.ravel(),
                                                                          fill_value=(efficiencyCurve.iloc[0], efficiencyCurve.iloc[-1]), bounds_error=False)
            
    def getEfficiency(self, Q, number_of_groups):
        return self.turbineEfficiencyInterpolator[number_of_groups](Q/self.q_rated)

    def getMaxDischarge(self, number_of_groups):
        return self.q_max_per_group * number_of_groups
    
    
class Revenue(object):
         
    def __init__(self, energy):
           
        self.energy = energy
        
    def compute_revenue(self):
        """
        Compute the total revenue
        Inputs : E : produced energy
                selling_price (single price or list of prices or function)
        Output : Revenue [€]
        """
        selling_price= self.get_prices( runFrequency='3H')
        return(self.energy*selling_price)
      
      
      
    def get_prices(self, runFrequency='3H'):   
        """
        Gets the prices for the Simulation period
        """
        #=======================================================================
        # excel_prices = pd.read_excel('data_SPOTPhelix.xlsx').dropna()
        # excel_prices = excel_prices.set_index(["Date"])            # dates as index
        # prices=excel_prices.iloc[:,1]
        # prices = prices.loc[(prices.index>=self.energy.first_valid_index()) & (prices.index<=self.energy.last_valid_index())].resample(runFrequency).mean().interpolate(method='linear')
        # 
        #=======================================================================
        
        #prices=prices.resample(resultFrequency).sum()
        
        
        excel_prices = pd.read_excel('data_SPOTPhelix.xlsx').dropna()
        excel_prices = excel_prices.set_index(["Date"])            # dates as index
    
        excel_prices.index = pd.to_datetime(excel_prices.index, format='%y-%m-%d')
        dates = pd.date_range(start=self.energy.first_valid_index(), end=self.energy.last_valid_index(), freq=runFrequency)
        shifts = dates - dates.map(lambda x: dt.datetime(x.year, 1, 1))
        
        
        prices = excel_prices.iloc[:,1]
        prices = prices.interpolate(method='linear').dropna()
        
        prices = prices[~prices.index.duplicated()]
        prices = prices.reindex(prices.index[0] + shifts)
        prices.index = dates
        prices.interpolate(method='ffill', inplace=True) # method='linear'
        
        
        
        return(prices)
      
      
    def plotRevenues(self):
              
        # Data
        x= self.energy.index
        revenue = self.compute_revenue()
             
         
        # Figure and axis creation
              
        fig, ax1 = plt.subplots()
              
        # y1
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Energy produced [MWh]')
        ax1.plot(x, self.energy, color='red', label='Energy produced [MWh]')
          
              
        # y2
              
        ax2 = ax1.twinx()
        ax2.set_ylabel('Revenue [€]')
        ax2.plot(x, revenue, color='green', label='Revenue [€]')
   
      
        
        ax1.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.1))
        ax2.legend(loc='upper center', bbox_to_anchor=(-0.1, 1))
         
        plt.grid()
                  
        
     
    
    
    
    
    
    
class Dam(object):
    
    '''
    classdocs
    '''

    TO_SUM = ['Revenue [M EUR]', 'Energy [MWh]']

    # Functions ----------------------------------------------------------------------------------------------------
    
    def __init__(self, table_levels, table_dates, turbineOperations, revenues, #mOL, fsl, MWL,
                 penstock_length, penstock_diameter, ks, downstream_level, tailrace, verbose=False):
        '''
        Constructor
        '''
        
        self.verbose = verbose
        
        # Revenues
        self.revenues = revenues
        
        # Turbines
        self.turbineOperations = turbineOperations
        
        # For head losses
        self.penstock_length = penstock_length            # length of the penstock (head losses computation)
        self.penstock_diameter = penstock_diameter        # diameter of the penstock (head losses computation)
        self.ks = ks                                    # rugosity [m] (head losses computation)
        self.downstream_level = downstream_level          # downstream level : powerplant
        self.tailrace = tailrace
        
        #=======================================================================
        # # Others
        # self.mOL = mOL                                      # minimum operational level (could be function of the date too)
        # self.fsl = fsl                                      # full supply level (could be function of the date too)
        # self.MWL = MWL                                      # maximum water level (could be function of the date too)
        #=======================================================================
        
        self.interpolator_max_spilling = self._initialize_level_interpolator(table_levels, key_x='water level', key_y='Spillage capacity [m3/s]')    # initialization for the maximum spilling
        self.interpolator_storage_volume = self._initialize_level_interpolator(table_levels, key_x='water level', key_y='reservoir capacity mln. m3')              # initialization for the storage volume
        self.interpolator_water_level = self._initialize_level_interpolator(table_levels, key_x='reservoir capacity mln. m3', key_y='water level')                 # initialization for the water level
        self.interpolator_seepage = self._initialize_level_interpolator(table_levels, key_x='water level', key_y='Seepage [m3/s]') 
        
        self.interpolator_max_fill_rate = self._initialize_level_interpolator(table_levels, key_x='water level', key_y='Max. fill rate [m/day]') 
        self.interpolator_max_empty_rate = self._initialize_level_interpolator(table_levels, key_x='water level', key_y='Max. empty rate [m/day]') 
        
        self.interpolator_full_supply_level = self._initialize_date_interpolator(table_dates, key_x='Date', key_y='Full supply level [m.a.s.l.]') 
        self.interpolator_minimum_level = self._initialize_date_interpolator(table_dates, key_x='Date', key_y='Minimum level [m.a.s.l.]') 
        self.interpolator_maximum_level = self._initialize_date_interpolator(table_dates, key_x='Date', key_y='Maximum level [m.a.s.l.]') 
        self.interpolator_eflow = self._initialize_date_interpolator(table_dates, key_x='Date', key_y='Environmental flow [m3/s]') 
        self.interpolator_target_level = self._initialize_date_interpolator(table_dates, key_x='Date', key_y='Target level [m.a.s.l.]') 
        
        self.water_use_order = ['Qeflow [m3/s]', 'Qturbined [m3/s]', 'Qspilled [m3/s]']
                          
    def get_spillway_capacity(self, h):
        '''
        Returns the spillway capacity based on the current water level in the reservoir
        Input: h : the water level(s) for which the spillway capacity is wanted [m.a.s.l.] as a float or an array
        Output: the spillway capacity, depending on the water level [m3/s]
        '''
        
        Qmax = self.interpolator_max_spilling(h)
        
        return Qmax
        
    def get_storage_volume(self, h):
        
        """
        Returns the volume stored in the reservoir at each time step [mio.m3]
        Input: h : Water level - array [m.a.sl.]
        Output: Returns an array of volumes [mio.m3] --> needs to be converted to m3 when used later on
        """
        
        stored_volume = self.interpolator_storage_volume(h)
        return stored_volume
    
    def get_seepage(self, h):
        return self.interpolator_seepage(h)
    
    def get_max_empty_rate(self, h):
        return self.interpolator_max_empty_rate(h)
    
    def get_max_fill_rate(self, h):
        return self.interpolator_max_fill_rate(h)
    
    def get_water_level(self, vol):
        
        """
        Returns the water level in the reservoir at each time step [m.a.s.l.]
        Input: vol : the volume stored in the reservoir [mio.m3]
        Output: Returns an array of water levels [m.a.s.l.] 
        """

        h = self.interpolator_water_level(vol)
        return h
    
    def get_minimum_level(self, date):
        return self._interpolate_date(self.interpolator_minimum_level, date)
    
    def get_maximum_level(self, date):
        return self._interpolate_date(self.interpolator_maximum_level, date)
    
    def get_eflow(self, date):
        return self._interpolate_date(self.interpolator_eflow, date)
    
    def get_target_level(self, date):
        return self._interpolate_date(self.interpolator_target_level, date)
    
    def get_full_supply_level(self, date):
        return self._interpolate_date(self.interpolator_full_supply_level, date)
    
    def get_efficiency(self, Q_turb, nGroups):
        
        """
        Returns the efficiency [-] corresponding to Q_turb
        Input: vol : the turbined discharge Q_turb  [m3/s]
        Output: Returns the efficiency [-]
        """
        
        return self.turbineOperations.getEfficiency(Q_turb, nGroups)
     
    # Checked
    
    def runSimulation(self, q_in, q_target, price_in, ini_level, start_date, end_date, number_of_groups=None, runFrequency='3H', resultFrequency='1D', resample_series=True):
        
        ###### q_spilled can also be optimized...
        
        '''
        Runs the simulation
        Inputs: q_in the initial entering flow [m3/s]     --> known
                q_target = instruction : q_spill = q_target - q_turb_max
                ini_level the initial water level [m.a.s.l.]
                start_date = starting date of the simulation (year,month,day)
                end_date = end date of the simulation (year,month,day)
                runFrequency = frequency at which the simulation is run
                resultFrequency = frequency at which the results are saved
        Output : Results of the simulation : water level [m.a.s.l.], energy produced, etc.    
                
        For the simulation (runFrequency) : https://pandas.pydata.org/docs/user_guide/timeseries.html
        
        To save the results : pandas resample
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
        '''
  
        if isinstance(number_of_groups, type(None)):
            number_of_groups = self.turbineOperations.number_of_groups
  
        dates = pd.date_range(start=start_date, end=end_date, freq=runFrequency)
        dt_seconds = (dates[1]-dates[0]).total_seconds() # in seconds
        dt = dt_seconds / 3600 # in hours
       
        # missing a check for the completeness of the inflow series over the chosen period between start and end (not to worry about now; should be done elsewhere)
        
        if resample_series:
            q_target = q_target.resample(runFrequency).mean().interpolate(method='linear')
            price_in = price_in.resample(runFrequency).mean().interpolate(method='linear')
            q_in_calc = q_in.loc[(q_in.index>=start_date) & (q_in.index<=end_date)].resample(runFrequency).mean().interpolate(method='linear')  # Select only
            q_target.reindex(q_in_calc.index)
        else:
            q_in_calc = q_in.copy()
            
        tmp = np.empty_like(q_in_calc)*np.NaN               # to create an empty array (tmp stands for "template")
        calc = pd.DataFrame({'Qin [m3/s]': q_in_calc,       # dataframe construction (to store the results)
                             'Qtarget [m3/s]': q_target,
                             'V [mio.m3]': tmp,
                             'h [m a.s.l.]': tmp,
                             'h target [m a.s.l.]': self.get_target_level(q_in_calc.index),
                             'h FSL [m a.s.l.]': self.get_full_supply_level(q_in_calc.index),
                             'h mOL [m a.s.l.]': self.get_minimum_level(q_in_calc.index),
                             'h MWL [m a.s.l.]': self.get_maximum_level(q_in_calc.index),
                             'Qout [m3/s]': tmp,
                             'Qeflow [m3/s]': self.get_eflow(q_in_calc.index),
                             'Qseepage [m3/s]': tmp,
                             'Qspilled [m3/s]': tmp,
                             'Qturbined [m3/s]': tmp,
                             'Qturbined_max [m3/s]': tmp,
                             'Qspilled_max [m3/s]': tmp,
                             'Qin-Qout [m3/s]': tmp,
                             'Max. fill rate [m/day]': tmp,
                             'Max. empty rate [m/day]': tmp,
                             'Max. fill rate [m3/s]': tmp,
                             'Max. empty rate [m3/s]': tmp,
                             'Net head [m]': tmp,
                             'Power [MW]': tmp,
                             'Energy [MWh]': tmp,
                             'Price [EUR/MWh]': price_in,
                             'Revenue [M EUR]': tmp,
                             }, index=q_in_calc.index,
        )
        
        calc.loc[start_date, 'V [mio.m3]'] = self.get_storage_volume(ini_level) # initialization
        
        
        n2i = {n: i for i, n in enumerate(calc.columns)}    # dictionary : index to column name
        
        
        marker = 0                                          # to see the calculations evolution
        size = calc.shape[0]                                # to get the number of rows
        for i0, d0 in enumerate(calc.index):                # i0 = index , d0= date
            '''
            Where the magic happens
            '''
            
            
            V = calc.iloc[i0, n2i['V [mio.m3]']]
            h = self.get_water_level(V)
            
            min_vol = self.get_storage_volume(calc.iloc[i0, n2i['h mOL [m a.s.l.]']])       # h min and h max depend on the date
            max_vol = self.get_storage_volume(calc.iloc[i0, n2i['h MWL [m a.s.l.]']])
            
            calc.iloc[i0, n2i['h [m a.s.l.]']] = h
            calc.iloc[i0, n2i['Max. fill rate [m/day]']] = self.get_max_fill_rate(h)
            calc.iloc[i0, n2i['Max. empty rate [m/day]']] = self.get_max_empty_rate(h)
            
            
            
            calc.iloc[0:i0,n2i['Max. fill rate [m/day]']]  = calc.iloc[0:i0,n2i['Max. fill rate [m/day]']].rolling(window=1).mean()
            calc.iloc[0:i0, n2i['Max. empty rate [m/day]']] = calc.iloc[0:i0,n2i['Max. empty rate [m/day]']].rolling(window=1).mean()
            
            #===================================================================
            # print("max fill rate")
            # print(calc['Max. fill rate [m/day]'])
            # print("max empty rate")
            # print(calc['Max. empty rate [m/day]'])
            #===================================================================
            
            
            
#===============================================================================
#             # Define the window size
#             window_size = 8 # 8*3 h = 24h
# 
#             # Loop through each data point in the original data
#             for i in range(window_size-1, len(calc['Max. fill rate [m/day]'])):
#                 # Calculate the moving average using the current data point and the previous window_size-1 data points
#                 current_avg_fill = sum(calc.iloc[i-window_size+1:i+1, n2i['Max. fill rate [m/day]']]) / window_size
#                 current_avg_empty = sum(calc.iloc[i-window_size+1:i+1, n2i['Max. empty rate [m/day]']]) / window_size
#                 # Append the moving average to the list
#                 calc.iloc[i0, n2i['Max. fill rate [m/day]']] = current_avg_fill
#                 calc.iloc[i0, n2i['Max. empty rate [m/day]']] = current_avg_empty
#===============================================================================
            
            
            calc.iloc[i0, n2i['Qseepage [m3/s]']] = self.get_seepage(h)
            calc.iloc[i0, n2i['Qspilled_max [m3/s]']] = self.get_spillway_capacity(h)
            
            # Initial values of Qturbined and Qspilled that will be updated below
            q_turbined_max = self.turbineOperations.getMaxDischarge(number_of_groups)
            calc.iloc[i0, n2i['Qturbined_max [m3/s]']] = q_turbined_max
            calc.iloc[i0, n2i['Qturbined [m3/s]']] = min((q_turbined_max, calc.iloc[i0, n2i['Qtarget [m3/s]']]))
            calc.iloc[i0, n2i['Qspilled [m3/s]']] = calc.iloc[i0, n2i['Qtarget [m3/s]']] - calc.iloc[i0, n2i['Qturbined [m3/s]']]
            
            # Verification of max emptying rate
            V_after_seepage_and_inflows = V + (calc.iloc[i0, n2i['Qin [m3/s]']]-calc.iloc[i0, n2i['Qseepage [m3/s]']]) * dt_seconds /1E6 # M m3            remaining_water = max((0, V_after_seepage-min_vol)) # what is left in the reservoir
            V_remaining_at_maximum_empty_rate = self.get_storage_volume(h - calc.iloc[i0, n2i['Max. empty rate [m/day]']]*dt/24) # M m3
            max_empty_discharge = (V_after_seepage_and_inflows-V_remaining_at_maximum_empty_rate) * 1E6 / dt_seconds # m3/s
            available_release = (V_after_seepage_and_inflows - min_vol) / dt_seconds *1E6
            available_release = max((0, min((available_release, max_empty_discharge))))
            
            calc.iloc[i0, n2i['Max. empty rate [m3/s]']] = available_release
            
            q_out = 0
            for u1 in self.water_use_order:
                q1 = min((calc.iloc[i0, n2i[u1]], available_release))
                calc.iloc[i0, n2i[u1]] = q1
                available_release -= q1
                q_out += q1
            
            # Verification of max filling rate
            V_after_outflow_operations = V_after_seepage_and_inflows - (q_out*dt_seconds/1E6)
            V_reached_at_maximum_filling_rate = self.get_storage_volume(h + calc.iloc[i0, n2i['Max. fill rate [m/day]']]*dt/24) # M m3
            V_max_admissible = min((V_reached_at_maximum_filling_rate, max_vol))
            
            calc.iloc[i0, n2i['Max. fill rate [m3/s]']] = (V_max_admissible-V)/ dt_seconds *1E6
            
            
            
            required_additional_discharge = max((0, (V_after_outflow_operations-V_max_admissible) * 1E6 / dt_seconds)) # m3/s
            additional_turbined_discharge = min((required_additional_discharge,q_turbined_max-calc.iloc[i0, n2i['Qturbined [m3/s]']]))
            calc.iloc[i0, n2i['Qturbined [m3/s]']] += additional_turbined_discharge # Qturbined is updated to the required additional discharge
            calc.iloc[i0, n2i['Qspilled [m3/s]']] += required_additional_discharge - additional_turbined_discharge # Qspilled is updated to the required additional discharge
            

            
            # Q spilled is limited to the capacity of the spillway
            calc.iloc[i0, n2i['Qspilled [m3/s]']] = min((calc.iloc[i0, n2i['Qspilled [m3/s]']], calc.iloc[i0, n2i['Qspilled_max [m3/s]']]))
            
            
            
            # In case prices are <0
            
            if calc.iloc[i0, n2i['Price [EUR/MWh]']] < 0 :
                         
                remaining_spilling_capacity = calc.iloc[i0, n2i['Qspilled_max [m3/s]']] - calc.iloc[i0, n2i['Qspilled [m3/s]']]
   
                if calc.iloc[i0, n2i['Qturbined [m3/s]']] <= remaining_spilling_capacity :
                    
                    calc.iloc[i0, n2i['Qspilled [m3/s]']] = calc.iloc[i0, n2i['Qspilled [m3/s]']] + calc.iloc[i0, n2i['Qturbined [m3/s]']]
                    calc.iloc[i0, n2i['Qturbined [m3/s]']] = 0
                    
                else :     
                    calc.iloc[i0, n2i['Qturbined [m3/s]']] = calc.iloc[i0, n2i['Qturbined [m3/s]']] - remaining_spilling_capacity
                    calc.iloc[i0, n2i['Qspilled [m3/s]']] = calc.iloc[i0, n2i['Qspilled_max [m3/s]']]
            
            
            
            # Calculation of Qout
            calc.iloc[i0, n2i['Qout [m3/s]']] = calc.iloc[i0, [n2i['Qseepage [m3/s]'], n2i['Qeflow [m3/s]'], n2i['Qturbined [m3/s]'], n2i['Qspilled [m3/s]']]].sum()
            
            # Net head, power and energy calculation
            if calc.iloc[i0, n2i['Qturbined [m3/s]']] <= 0 :
                calc.iloc[i0, n2i['Net head [m]']] = h
                calc.iloc[i0, n2i['Power [MW]']] = 0
        
            else:
                calc.iloc[i0, n2i['Net head [m]']], _, calc.iloc[i0, n2i['Power [MW]']] = self.hydropower(calc.iloc[i0, :], number_of_groups)
            
            calc.iloc[i0, n2i['Qout [m3/s]']] = calc.iloc[i0, n2i['Qeflow [m3/s]']] + calc.iloc[i0, n2i['Qseepage [m3/s]']] + calc.iloc[i0, n2i['Qturbined [m3/s]']] + calc.iloc[i0, n2i['Qspilled [m3/s]']]
            calc.iloc[i0, n2i['Energy [MWh]']] = calc.iloc[i0, n2i['Power [MW]']] * dt
            
            calc.iloc[i0, n2i['Qin-Qout [m3/s]']] = (calc.iloc[i0, n2i['Qin [m3/s]']] - calc.iloc[i0, n2i['Qout [m3/s]']]) 
            
            if i0<size-1:
                calc.iloc[i0+1, n2i['V [mio.m3]']] = calc.iloc[i0, n2i['V [mio.m3]']] + (calc.iloc[i0, n2i['Qin [m3/s]']] - calc.iloc[i0, n2i['Qout [m3/s]']]) * dt_seconds / 1E6 #*3600 --> back to [hr] and /10^6 because Vol in [mio.m3]
               
            if self.verbose and i0/size*100 > marker:
                print('%02u%%' % marker)
                marker += 10 
        
        calc.loc[:, 'Revenue [M EUR]'] = calc.loc[:, 'Energy [MWh]'] * calc.loc[:, 'Price [EUR/MWh]'] / 1E6
        
   

        
        
        if resample_series:
            self.results = calc.resample(resultFrequency).mean()
            self.results.loc[:, self.TO_SUM] = self.results.loc[:, self.TO_SUM].resample(resultFrequency).sum()
        else:
            self.results = calc.copy()
        
        return self.results
            
    def hydropower(self, simulation, number_of_groups):
        
        #=======================================================================
        # self.downstream_level
        # self.penstock_length
        # self.penstock_diameter
        # best_efficiency_for_flow, number_of_active_groups = self.somefunction(turbined_flow)
        #=======================================================================
                
        # Q > number of groups        
        # number of groups > Q per hydraulic circuit
        # Q per hydraulic circuit + characteristics of the circuit > distributed (Colebrook-White) and localized headlosses (bends, changes of diameter, other obstacles) > net head
        
        # function that calculates the efficiency of the groups > based on the hill chart information
        
        
        
        ####
        efficiency = self.turbineOperations.getEfficiency(simulation.loc['Qturbined [m3/s]'], number_of_groups)

        
        V=simulation.loc['Qturbined [m3/s]']/(math.pi*pow(self.penstock_diameter,2)/4 )         # velocity
        
        
        ###### TO CORRECT with the true tailwater level (that must be declared with dam or turbineOperations
        net_head = simulation.loc['h [m a.s.l.]'] + pow(V,2)/2/9.81 - self.head_losses(V) - self.tailrace       # H1=H0-head_losses
        
        
        power = net_head * simulation.loc['Qturbined [m3/s]'] * efficiency * 9810 / 1000000 # MW
        
        return (net_head,number_of_groups, power)
    
    def head_losses(self, V ):
       
        """
        Computes the total head losses
        Inputs : Velocity V [m/s]
        Outputs : linear head losses, localized head losses [m]
        """
        # Linear head losses
     
        
        def colebrook(f0, eps, Re):
            f1 = pow(math.sqrt(f0),3)*(-2*math.log((eps/3.71)+(2.51/(Re*math.sqrt(f0)))))
            return np.abs(f0-f1)
        
        nu=1.32*pow(10,-6)                        # kinematic viscosity at 10°C [m2/s]
        eps = self.ks/self.penstock_diameter           # Epsilon = relative rugosity [-]
        Re = V*self.penstock_diameter/nu
      
  
        if Re<2500 :
            f=64/Re
        else:  
            f = newton(colebrook, x0=1, args=(eps, Re), disp=False)

        h_l = f*self.penstock_length*V*V/self.penstock_diameter/2/9.81       # linear losses
        
        # Localized head losses
        
        # Intake : coeff : 0.5 (Assumption)
         
        coeff=0.5
        h_s = coeff*V*V/2/9.81 
        
        h_tot= h_l+h_s
        return(h_tot)
        
    def plotSimulation(self):
        '''
        Function creating plots
        '''


        # First plot : Main results
        # Data
        x= self.results.index
        
        y1_1=self.results['h [m a.s.l.]']
        y1_2=self.results['h FSL [m a.s.l.]']
        
        y2_1=self.results['Qturbined [m3/s]']
        y2_2=self.results['Qspilled [m3/s]']
        y2_3=self.results['Qin [m3/s]']

        # Figure and axis creation
        
        fig1, ax1 = plt.subplots()
        
        # y1
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Water level [m.a.s.l.]')
        ax1.plot(x, y1_1, color='red', label='h [m.a.s.l.]')
        ax1.plot(x, y1_2, color='red', linestyle='dashed', label='f.s.l [m a.s.l.]')
        
        # y2
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Discharge [m3/s]')
        ax2.plot(x, y2_1, color='green', label='Qturbined [m3/s]')
        ax2.plot(x, y2_2, color='blue', label='Qspilled [m3/s]')
        ax2.plot(x, y2_3, color='cyan', label='Qin [m3/s]')
        ax2.plot(x, self.results['Price [EUR/MWh]'], color='k', linestyle = ':', linewidth =3, label = 'Price [EUR/MWh]')
        
        #self.results['Price [EUR/MWh]'].plot(ax=plt.gca(), style=':', color='k', linewidth=3, legend = 'Price [EUR/MWh]')
  
        
        ax1.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.1))
        plt.gcf().axes[0].grid(linestyle = ':')
        ax2.legend(loc='upper center', bbox_to_anchor=(-0.1, 1))
        plt.grid()
       
        
        #=======================================================================
        # ax1.legend(loc='upper left', bbox_to_anchor=(-0.15, 1), ncol=1, mode='expand')
        # ax2.legend(loc='upper left', bbox_to_anchor=(-0.15, 0.6), ncol=1, mode='expand')
        #=======================================================================

        
        #=======================================================================
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # fig.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=len(labels))
        #=======================================================================
        
        
        # Plot #2 : Checking the limits

        fig2, ax3 = plt.subplots()
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Discharges [m3/s]')
        ax3.plot(x,np.abs(self.results['Max. fill rate [m3/s]']), color='cyan', linestyle='dashed',label = 'Max. fill rate [m3/s]')     
        ax3.plot(x,self.results['Max. empty rate [m3/s]'], color='red', linestyle='dashed', label = 'Max. empty rate [m3/s]')      
                             
    
        ax3.plot(x,self.results['Qspilled [m3/s]'], color = 'blue', label = 'Qspilled [m3/s]')
        ax3.plot(x,self.results['Qspilled_max [m3/s]'], color='blue', linestyle='dashed', label = 'Qspilled_max [m3/s]')
        ax3.plot(x,self.results['Qturbined [m3/s]'], color='green',label = 'Qturbined [m3/s]')
        ax3.plot(x,self.results['Qturbined_max [m3/s]'], color='green', linestyle='dashed',label = 'Qturbined_max [m3/s]')
        
        ax3.plot(x,self.results['Qout [m3/s]'], color='orange',label = 'Total outflow [m3/s]')
        #ax3.plot(x, self.results['Qin [m3/s]'], color='cyan', label='Inflow [m3/s]')
        ax3.plot(x,self.results['Qin-Qout [m3/s]'], color='magenta',label = 'Qin-Qout [m3/s]')
                          
        ax3.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.1), fontsize='x-small')
        plt.grid()
        
        plt.show()
        


          
    def _interpolate_date(self, interpolator, date):
        '''
        
        '''
        
        return interpolator(date.dayofyear)
        
    def _initialize_date_interpolator(self, table, key_x, key_y):
        """
        This function prepares interpolator based on dates (day of year).
      
        Input: an Excel file with rows corresponding to dates (one year), column names of the x and y variables that are used
        """
        res_excel = pd.read_excel(table, usecols=[key_x, key_y])
        try:
            res_excel = pd.read_excel(table, usecols=[key_x, key_y])
        except Exception as ex:
            res_excel = pd.read_excel(table, usecols=[key_x, key_y], encoding='unicode_escape')
        res_excel.dropna(how='any', inplace=True)
        
        res_excel = res_excel.loc[:, [key_x, key_y]]
        
        #training_x = pd.to_datetime(res_excel.iloc[:, 0]).dt.day_of_year
        training_x = pd.to_datetime(res_excel.iloc[:, 0]).dt.dayofyear
        training_y = res_excel.iloc[:, 1].astype(float)     
        
        # Repeat the start at the end
        training_x = pd.concat((training_x, pd.Series([365])), axis=0)
        training_y = pd.concat((training_y, training_y.iloc[[0]]), axis=0)
        
        return interpolate.interp1d(training_x, training_y, fill_value=(training_y.iloc[0], training_y.iloc[-1]), bounds_error=False)
              
    def _initialize_level_interpolator(self, table, key_x, key_y):      
        """
        This function prepares interpolator based on levels.
      
        Input: an Excel file with rows corresponding to levels, column names of the x and y variables that are used
        """
        try:
            res_excel = pd.read_excel(table, usecols=[key_x, key_y])
        except Exception as ex:
            res_excel = pd.read_excel(table,usecols=[key_x, key_y],encoding='unicode_escape')
        res_excel.dropna(how='any', inplace=True)
        
        res_excel = res_excel.loc[:, [key_x, key_y]]
        
        res_excel= res_excel.astype(float)  
        
        #=======================================================================
        # if key_x == 'reservoir capacity mln. m3':
        #     min_index = res_excel[res_excel[key_y]==430].index.values
        #     max_index = res_excel[res_excel[key_y]==510].index.values
        #     res_excel = res_excel.iloc[int(min_index):int(max_index)]
        #=======================================================================
  
        training_x = res_excel.iloc[:, 0]
        training_y = res_excel.iloc[:, 1]
        
        #return interpolate.interp1d(training_x.values.ravel(), training_y.values.ravel(), fill_value=(training_y.min(), training_y.max()), bounds_error=False)
        
        #=======================================================================
        # 
        # if key_x == 'reservoir capacity mln. m3':
        #     min_index = res_excel.get_loc(430)
        #     max_index = res_excel.get_loc(510)
        #     res_excel = res_excel.iloc[min_index:max_index,:]
        #     
        #     return interpolate.interp1d(training_x.values.ravel(), training_y.values.ravel(), fill_value=(430,510), bounds_error=False)
        # else:  
        #  
        #=======================================================================
        
        if key_x == 'Qspilled_max [m3/s]':
            return interpolate.interp1d(training_x.values.ravel(), training_y.values.ravel(), fill_value=(training_y.min(), training_y.max()), bounds_error=False)
        else:
            return interpolate.interp1d(training_x.values.ravel(), training_y.values.ravel(), fill_value='extrapolate')
        
        #return interpolate.interp1d(training_x.values.ravel(), training_y.values.ravel(), fill_value=(training_y.min(), training_y.max()), bounds_error=False)
        
       
    
       
        
if __name__=='__main__':
    
    
    #===========================================================================
    # #Example
    # optimizer = CMA(mean=np.zeros(2), sigma=1, bounds=np.array([[0,2],[0,2]]), n_max_resampling=2)        # population sampling : multivariate Gaussian distribution, parameters: mean + variance
    #                                                     # sigma = 1.3 
    # def quadratic(x1, x2):
    #     """
    #     Test for the CMA-ES
    #     """
    #     return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2 
    #   
    # for generation in range(50):
    #     solutions = []
    #     for _ in range(optimizer.population_size):
    #         x = optimizer.ask()                     # array with x1,x2 : what is changed to obtain the best value
    #         print('x :' + str(x))
    #         value = quadratic(x[0], x[1])     # result 
    #         print('value :' + str(value))
    #         solutions.append((x, value))
    #         print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
    #     optimizer.tell(solutions)
    # 
    # #-----------------------------------------------------------------------------------------------------------------------------------------------
    #===========================================================================

    # Dam construction
    q_max_per_turbine = 98 # m3/s
    turbines = TurbineOperations('turbineEfficiency.csv', q_rated=q_max_per_turbine/1.15, number_of_groups=5, sep=';', index_col=0)
    
    dam_1 = Dam(table_levels='Enguri reservoir levels.xls',
                table_dates="Enguri reservoir dates.xlsx",
                penstock_length=15000,
                penstock_diameter=9.5,
                ks=0.1,
                #===============================================================
                # mOL=430,
                # fsl=510,
                # MWL=511.5,
                #===============================================================
                turbineOperations=turbines,
                revenues= None,
                downstream_level=200,
                tailrace=120
                )

         
    # Import inflow data
    data = pd.read_excel('Tethys data (Enguri dam, Q.Enguri water level).xlsx', index_col=0)
    data.index = pd.to_datetime(data.index, format='%y-%m-%d')
    q_in = data['Enguri dam, Q'].interpolate(method='linear').dropna()
    q_out = q_in.copy()*0 + 200
     
         
    excel_prices = pd.read_excel('data_SPOTPhelix.xlsx', index_col='Date', usecols=['Date', 'Price [€/MWh]']).dropna()
    excel_prices.to_pickle('prices.pkl', protocol=4)
    excel_prices = pd.read_pickle('prices.pkl')     
         
         
    # Simulation
    #===========================================================================
    # freq = '1D'
    # dates = pd.date_range('2015-01-01', '2016-01-01', freq=freq)
    # q_in_ = q_in.resample(rule=freq).interpolate(method='linear').loc[dates[0]:dates[-1]]
    # price_in = excel_prices.resample(rule=freq).mean().interpolate(method='linear').loc[dates[0]:dates[-1]].iloc[:,0]
    # dam_1.runSimulation(q_in*2, q_out, price_in, ini_level=470, start_date=dt.datetime(2015, 1, 1), end_date=dt.datetime(2016, 1, 1))
    #===========================================================================
    
    #===========================================================================
    # dam_1.results.plot()
    # plt.show()
    #===========================================================================
    
    
    freq = '1D'
    dates = pd.date_range('2016-07-15', '2016-07-31', freq=freq)
    q_in_ = q_in.resample(rule=freq).interpolate(method='linear').loc[dates[0]:dates[-1]]
    price_in = excel_prices.resample(rule=freq).mean().interpolate(method='linear').loc[dates[0]:dates[-1]].iloc[:,0]
    dam_1.runSimulation(q_in, q_out, -price_in*100, ini_level=470, start_date=dt.datetime(2016, 7, 15), end_date=dt.datetime(2016, 7, 31))
    
    
    dam_1.plotSimulation()
    
    print(dam_1.results)
    

    #dam_1.revenues.plotRevenues()

    plt.show()
    
#===============================================================================
#     # Optimization
#     
#     start_time = time.time()
#     
#     selling_price= dam_1.get_prices(start_date=dt.datetime(2015, 1, 1), end_date=dt.datetime(2017, 1, 1), runFrequency='3H', resultFrequency='1D')                         # 27 ct/kWh : /100:ct-->euros & /1000 kWh-->MWh
#     size=selling_price.size
#     Q_turb_assumption = 30      # m3/s
#            
#     optimizer = CMA(mean=np.ones(size)*Q_turb_assumption, sigma=1.3)        # population sampling : multivariate Gaussian distribution, parameters: mean + variance
#     
#     def objective_function(x):
#         dam_1.runSimulation(q_in,x, start_date=dt.datetime(2015, 1, 1), end_date=dt.datetime(2017, 1, 1))
#         revenue=dam_1.compute_revenue(dam_1.results.loc[:,'Energy [MWh]'], selling_price)
#         return(-revenue)
#     
#     
#     max_generation = 3
#     for generation in range(max_generation): 
#         print("iteration n° : " + str(generation))                       
#         solutions = []
#         for _ in range(optimizer.population_size):
#             
#                    
#             #dam_1.runSimulation( q_in, start_date=dt.datetime(2015, 1, 1), end_date=dt.datetime(2017, 1, 1))
#             #produced_E= dam_1.results.loc[:,'Energy [MWh]']
#               
#             x = optimizer.ask()                                          # array with x1,x2 : what is changed to obtain the best value
#             #print('x :' + str(x))
#             value = objective_function(x)   # result : minimize - revenue to obtain max(revenue)
#             #print('value :' + str(value))
#             solutions.append((x, value))
#             
#             print(f"#{generation} {value} (x)")
#             
#             #print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
#            
#         #optimizer.tell(solutions)            # error : does not work ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
#     
#     
#     
#     end_time = time.time()
#     
#     optimization_duration= end_time-start_time
#     print("Duration of the optimization : ")
#     print(optimization_duration)
# 
#     Q_turb=x
#     revenue= - value
#     total_revenue=revenue.sum()
#===============================================================================

    pass
    pass
    a = 1
    b = 2




