'''
Created on 20/03/2023

@author: zepedro
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class TurbineEfficiency(object):
    '''
    classdocs
    '''


    def __init__(self, efficiency, precision=0.5):
        '''
        Constructor
        efficiency is a pandas dataframe with an index (Q/Qrated) and one column (efficiency as a decimal)
        '''

        self.precision = precision
        self.efficiency = self._setPrecision(efficiency)

    def _setPrecision(self, groupEfficiency):
        '''
        Resamples the groupEfficiency according to the precision (if required)
        '''
        
        groupEfficiency.dropna(inplace=True)
        if groupEfficiency.index.to_series().diff().max()>self.precision or groupEfficiency.index.min()!=0:
            groupEfficiency.loc[groupEfficiency.iloc[:,0]==0, :] = None # Remove 0's for abrupt start of the curve (no operation for Q below...)
            index = np.arange(0, groupEfficiency.index.max()+self.precision, self.precision)
            groupEfficiency = groupEfficiency.reindex(index)
            groupEfficiency.interpolate(method='linear', inplace=True)
            groupEfficiency.fillna(0, inplace=True)
        
        return groupEfficiency
    
    def aggregate(self, groupEfficiencies=[], number=2):
        '''
        Calculates the aggregated turbine efficiency for several groups
        '''
        
        # Inputs are handled here
        while len(groupEfficiencies)<number:
            groupEfficiencies.append(self.efficiency)
        
        # Resample efficiencies (if required)
        groupEfficiencies = [self._setPrecision(g) for g in groupEfficiencies]
        
        
        if len(groupEfficiencies)==1:
            return groupEfficiencies[0]
        else:
            # Where the work gets done by aggregating each pair of groups incrementally (the function calls itself repeatedly)
            # f(a, f(b, f(c, f(d, ...))))
            efficiency = self.aggregate(groupEfficiencies[1:], number=number-1)
            discharges = groupEfficiencies[0].index.values + np.expand_dims(efficiency.index.values, 1)
            tmp = groupEfficiencies[0].iloc[:, 0].values*groupEfficiencies[0].index.values + np.expand_dims(efficiency.iloc[:, 0].values*efficiency.index.values, 1)
            efficiencies = tmp/discharges
             
            efficiency = pd.DataFrame(efficiencies.ravel(), index=discharges.ravel(), columns=groupEfficiencies[0].columns)
            efficiency.index.name = groupEfficiencies[0].index.name
            efficiency = efficiency.groupby(axis=0, level=0).max()
            return efficiency.fillna(0)
        
if __name__=='__main__':
    #===========================================================================
    # import matplotlib
    # matplotlib.use('TkAgg') # I need these two for my conda env, but you probably don't
    #===========================================================================
    
    # load data
    turbineEfficiency = pd.read_csv('turbineEfficiency.csv', sep=';', index_col=0)
    turbineEfficiency.iloc[:, 0] = turbineEfficiency.iloc[:, 0].str.replace('%','').astype(float)/100
    
    turb = TurbineEfficiency(turbineEfficiency)
    efficiencyCurve = turb.aggregate(number=5)
    efficiencyCurve.plot()
    
    a=1