# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:46:11 2015

@author: 535873
"""

# 1. show most similar players at the same age for a given player and age, on variables included in WAR
    # total WAR vs. WAR components
    # cosine similarity (which disregards magnitude) vs. Euclidean norm
# 2. predict future WAR for a given player based on age and historical WAR variable performance
    # using random forest regression
    # using k-nearest neighbors
    # use same training and test sets across both algorithms

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy import spatial
import random
import bokeh

def MakePastComparables(playerList,measure,alg):
    RC_past_comparables = {}
    for playerID in playerList:
        playerAgeDict = {}
        for ages in player_age_mapping[playerID]:
            pos = player_pos_mapping[playerID][0]
            tempDF = past_stats[pos][ages]
            if playerID in tempDF.index:
                if measure == 'RC':
                    tempDF = tempDF[['RC']]
                    meanRC = tempDF['RC'].mean()
                    stdRC = tempDF['RC'].std()
                    #tempDF['RC_norm'] = tempDF['RC']
                    #tempDF['RC_norm'] = (tempDF.loc[:,'RC']-meanRC)/stdRC
                    tempDF['RC_norm'] = tempDF.apply(lambda x:(x['RC']-meanRC)/stdRC, axis=1)
                    compRC = tempDF['RC_norm'][playerID]
                    tempDF['dist'] = tempDF.apply(lambda x:(x['RC_norm']-compRC)**2, axis=1)
                if measure == 'component':
                    tempDF = tempDF[['H','X2B','X3B','HR','BB','SO','IBB']]
                    if alg == 'cossim':
                        tempDF['dist'] = tempDF.apply(lambda x:(fn_ComputeCosSim(x,tempDF.ix[playerID])), axis=1)
                    if alg == 'euclidean':
                        tempDF['dist'] = tempDF.apply(lambda x:(scipy.spatial.distance.euclidean(x,tempDF.ix[playerID])), axis=1)
                # remove row corresponding to playerID, sort by nearest distance, and keep top 20 only
                tempDF = tempDF[tempDF.index!=playerID].sort_values(by=['dist']).ix[0:19]
                # create result dict
                result = {}
                # add RC_dist_top10 key with its dataframe as its value to result dict        
                result['Comparables_Distances']= tempDF['dist']
                # add RC performance stats for each of the 10 closest fits
                Performance = {}
                CompAges = pd.Series()
                for ind in tempDF.index:
                    Performance[ind] = fn_getCumRCByAge(ind)
                    CompAges = CompAges.append(Performance[ind]['Age'])
                result['Comparables_Performance'] = Performance
                CompAges = pd.Series(CompAges.unique())
                RC_Trajectories = pd.DataFrame(index=CompAges)
                ct = 0
                for ind in tempDF.index:
                    playerDF = Performance[ind].copy()
                    playerDF.index = playerDF['Age']
                    playerDF = playerDF['RC'] 
                    playerDF = playerDF.to_frame()
                    playerDF.columns=[ind]
                    if ct==0:
                        RC_Trajectories = playerDF
                    else:
                        RC_Trajectories = RC_Trajectories.merge(playerDF,left_index=True,right_index=True,how='outer')
                    ct += 1
                RC_Trajectories = RC_Trajectories.fillna(0)
                result['Comparables_Age_vs_RC'] = RC_Trajectories
                # make result dict the value for playerAgeDict[ages]
                playerAgeDict[ages]=result
        # make playerAgeDict the value for RC_past_comparables[playerID]
        RC_past_comparables[playerID] = playerAgeDict
    return RC_past_comparables

def fn_getCumRCByAge(player):
    # a function that returns a dataframe with [playerID | age | RC | TotRC]
    pos = player_pos_mapping[player][0]
    ages = player_age_mapping[player]
    ages.sort()
    RC = pd.Series()
    TotRC = pd.Series()
    diff = 0
    ind = 0
    b = 0
    for age in ages:
        tempDF = past_stats[pos][age]
        if player in tempDF.index:
            b = tempDF.ix[player]['RC']
            TotRC.set_value(ind,b)
            if ind == 0:
                a = 0
            else:
                a = TotRC[ind-1]
            diff = b-a
            RC.set_value(ind,diff)
            ind += 1
    Age = pd.Series(ages)
    PctTotRC = RC/TotRC.max()
    CumPctTotRC = TotRC/TotRC.max()
    InvCumPctTotRC = 1-PctTotRC
    # result = pd.DataFrame([Age,RC,TotRC,PctTotRC],columns=['Age','RC','TotRC','PctTotRC'])
    result = pd.concat([Age,RC,TotRC,PctTotRC,CumPctTotRC,InvCumPctTotRC],axis=1)
    result.columns = ['Age','RC','TotRC','PctTotRC','CumPctTotRC','InvCumPctTotRC']
    return result

def fn_ComputeRunsCreated(stats): # to be used with 'apply' and 'axis=1' 
    if (stats.AB+stats.BB+stats.IBB) > 0:
        TB = stats.BB + stats.IBB + stats.H + 2*stats.X2B + 3*stats.X3B + 4*stats.HR
        RC = ((stats.H+stats.BB+stats.IBB)*TB)/(stats.AB+stats.BB+stats.IBB)
    else:
        RC = 0
    return RC
    
def fn_ComputeCosSim(vec1,vec2):
    result = 1 - spatial.distance.cosine(vec1,vec2)
    return result

### FOR STEP 1
## FOR INPUTS
# read in hitting table (with age and position already in)
hittingDF = pd.read_csv('C:/Users/Tim/Documents/BAH Data Sci Course (Fall 2015)/Capstone/hittingTable.csv')
# weed out pitchers and deadball-era hitters
hittingDF = hittingDF.ix[hittingDF['POS']!='P']
hittingDF = hittingDF.ix[hittingDF['yearID']>=1920]
playerIDs = hittingDF['playerID'].unique()
positions = hittingDF['POS'].unique()
player_pos_mapping = {}
player_age_mapping = {}
for playerID in playerIDs:
    subTable = hittingDF.ix[hittingDF['playerID']==playerID]
    player_pos_mapping[playerID] = subTable['POS'].unique()
    player_age_mapping[playerID] = subTable['AGE'].unique()

# create a dictionary of groupby dataframes for historical performance
    # key on age and position
    # apply max age filter and position filter to master batting dataframe
    # turn filtered dataframe to group by object, grouping on playerID, summing WAR variables
    # add in RC value as a column
    # add in normalized versions of all the performance variables
past_stats = {}
for pos in positions:
    posDF = hittingDF.ix[hittingDF['POS']==pos]
    posAgeDict = {}
    for age in posDF['AGE'].unique():
        tempDF = posDF.ix[posDF['AGE']<=age]
        tempDF = tempDF[['playerID','AB','R','H','X2B','X3B','HR','RBI','BB','SO','IBB']]
        if tempDF.shape[0]>0:
            groupedDF = tempDF.groupby(['playerID']).sum()
            # remove any rows with AB==0
            groupedDF = groupedDF.fillna(0)
            groupedDF['RC'] = groupedDF.apply(fn_ComputeRunsCreated, axis=1)
            groupedDF = groupedDF.ix[groupedDF['AB']>=400]
            posAgeDict[age] = groupedDF
    past_stats[pos]=posAgeDict
# trout = MakeRCPastComparables(['troutmi01'])
# create a similar dictionary of dataframes for future performance, following the same process, but with the complementary age filter
future_stats = {}
for pos in positions:
    posDF = hittingDF.ix[hittingDF['POS']==pos]
    posAgeDict = {}
    for age in posDF['AGE'].unique():
        tempDF = posDF.ix[posDF['AGE']>age]
        tempDF = tempDF[['playerID','AB','R','H','X2B','X3B','HR','RBI','BB','SO','IBB']]
        if tempDF.shape[0]>0:
            groupedDF = tempDF.groupby(['playerID']).sum()
            # remove any rows with AB==0
            groupedDF = groupedDF.fillna(0)
            groupedDF['RC'] = groupedDF.apply(fn_ComputeRunsCreated, axis=1)
            groupedDF = groupedDF.ix[groupedDF['AB']>=400]
            posAgeDict[age] = groupedDF
    future_stats[pos]=posAgeDict
## FOR RESULTS
# create a dictionary of dataframes, but now keyed by playerID and age
# create a random set of players to use as a test bed. they must have at least 10 seasons, and we'll pick an age to predict in between the 25% and 50% mark in their career
# we will also exclude pitchers from the sample, since we're focusing on batting statistics
player_age_mapping_select = {}
for k in list(player_age_mapping.keys()):
    if len(player_age_mapping[k]) >= 10:
        if player_pos_mapping[k][0] != 'P':
            player_age_mapping_select[k] = player_age_mapping[k]
player_sample = random.sample(list(player_age_mapping_select.keys()),200)
ages = list()
for p in player_sample:
    a = player_age_mapping_select[p]
    a.sort()
    numAges = len(a)
    lb = int(numAges/4)
    ub = int(numAges/2)
    age = a[random.randint(lb,ub)]
    ages.append(age)
player_age_sample = pd.DataFrame(player_sample)
ages = pd.DataFrame(ages)
player_age_sample = player_age_sample.merge(ages,left_index=True,right_index=True)
player_age_sample.columns = ['player','age']
    # a dataframe with the euclidean and cossim distance scores to both RC and the RC components
    # for every other playerID in that age and position group
sample_comparables_RC_Euc = MakePastComparables(player_age_sample.loc[:,'player'],'RC','euclidean')
sample_comparables_component_Euc = MakePastComparables(player_age_sample.loc[:,'player'],'component','euclidean')
sample_comparables_component_cossim = MakePastComparables(player_age_sample.loc[:,'player'],'component','cossim')
# create another dictionary of dataframes keyed on playerID and age (actually, one dictionary for each of the distance combinations from the previous step)
# or simply find the average errors for each of the comparable sets for each player in the sample
player_age_sample = player_age_sample.set_index('player')
player_age_sample['pos'] = ""
player_age_sample['avgError_RC_Euc'] = 0
player_age_sample['avgError_component_Euc'] = 0
player_age_sample['avgError_component_cossim'] = 0
for playerID in player_age_sample.index:
    refAge = player_age_sample.loc[playerID][0]
    refPos = player_pos_mapping[playerID][0]
    if (playerID in future_stats[refPos][refAge].index and refAge in sample_comparables_RC_Euc[playerID].keys() and refAge in sample_comparables_component_Euc[playerID].keys() and refAge in sample_comparables_component_cossim[playerID].keys()):
        refFutureRC = future_stats[refPos][refAge].loc[playerID,'RC']
        #refTotalDist_RC_Euc = sample_comparables_RC_Euc[playerID][refAge]['Comparables_Distances'].sum()
        #refMaxDist_RC_Euc = sample_comparables_RC_Euc[playerID][refAge]['Comparables_Distances'].max()
        #refTotalDist_component_Euc = sample_comparables_component_Euc[playerID][refAge]['Comparables_Distances'].sum()
        #refMaxDist_component_Euc = sample_comparables_component_Euc[playerID][refAge]['Comparables_Distances'].max()
        #refTotalDist_component_cossim = sample_comparables_component_cossim[playerID][refAge]['Comparables_Distances'].sum()
        #refMaxDist_component_cossim = sample_comparables_component_cossim[playerID][refAge]['Comparables_Distances'].max()
        refInvDist_RC_Euc = 1/sample_comparables_RC_Euc[playerID][refAge]['Comparables_Distances']
        refInvScaledDist_RC_Euc = refInvDist_RC_Euc/refInvDist_RC_Euc.sum()
        refInvDist_component_Euc = 1/sample_comparables_component_Euc[playerID][refAge]['Comparables_Distances']
        refInvScaledDist_component_Euc = refInvDist_component_Euc/refInvDist_component_Euc.sum()
        refInvDist_component_cossim = 1/sample_comparables_component_cossim[playerID][refAge]['Comparables_Distances']
        refInvScaledDist_component_cossim = refInvDist_component_cossim/refInvDist_component_cossim.sum()
        errorVec_RC_Euc = []
        errorVec_component_Euc = []
        errorVec_component_cossim = []
        for compPlayer in sample_comparables_RC_Euc[playerID][refAge]['Comparables_Distances'].index:
            if compPlayer in future_stats[refPos][refAge].index:
                compFutureRC = future_stats[refPos][refAge].loc[compPlayer,'RC']
                errorRC = abs(compFutureRC-refFutureRC)/refFutureRC
                # weigh each error score by the scaled inverse distance of compPlayer, and scale by the total distance
                adjustedErrorRC = errorRC * refInvScaledDist_RC_Euc[compPlayer]
                errorVec_RC_Euc.append(adjustedErrorRC)
        for compPlayer in sample_comparables_component_Euc[playerID][refAge]['Comparables_Distances'].index:
            if compPlayer in future_stats[refPos][refAge].index:        
                compFutureRC = future_stats[refPos][refAge].loc[compPlayer,'RC']
                errorRC = abs(compFutureRC-refFutureRC)/refFutureRC
                # weigh each error score by the distance of compPlayer, and scale by the total distance
                adjustedErrorRC = errorRC * refInvScaledDist_component_Euc[compPlayer]
                errorVec_component_Euc.append(errorRC)
        for compPlayer in sample_comparables_component_cossim[playerID][refAge]['Comparables_Distances'].index:
            if compPlayer in future_stats[refPos][refAge].index:
                compFutureRC = future_stats[refPos][refAge].loc[compPlayer,'RC']
                errorRC = abs(compFutureRC-refFutureRC)/refFutureRC
                # weigh each error score by the distance of compPlayer, and scale by the total distance
                adjustedErrorRC = errorRC * refInvScaledDist_component_cossim[compPlayer]
                errorVec_component_cossim.append(errorRC)
        player_age_sample.loc[playerID,'avgError_RC_Euc'] = np.asarray(errorVec_RC_Euc).sum()
        player_age_sample.loc[playerID,'avgError_component_Euc'] = np.asarray(errorVec_component_Euc).sum()
        player_age_sample.loc[playerID,'avgError_component_cossim'] = np.asarray(errorVec_component_cossim).sum()
# summarize eror rates by age and plot them
player_age_sample_ageGB = player_age_sample.groupby(['age']).mean()
#from bokeh.plotting import figure, output_file, show
#p = figure(plot_width=400, plot_height=300)
#p.line(player_age_sample['avgError_RC_Euc'],player_age_sample['avgError_component_Euc'],player_age_sample['avgError_component_cossim'])
#show(p)
#output_file('similarity_error_summaries')
plt.plot(player_age_sample_ageGB.index,player_age_sample_ageGB['avgError_RC_Euc'],label='RC comparison, Euclidean distance')
plt.plot(player_age_sample_ageGB['avgError_component_Euc'],label='component comparison, Euclidean distance')
plt.plot(player_age_sample_ageGB['avgError_component_cossim'],label='component comparison, cosine distance')
plt.legend(prop={'size':8})
plt.xlabel('age of comparison') 
plt.ylabel('weighted average error among 20 closest matches')
plt.show
# lastly, for a few players, make a scatterplot of distance vs. future performance from the last dictionary above
    # then overlay the actual performance to see how the estimates worked out
player_age_sample.to_csv('C:\\Users\\Tim\\Documents\\BAH Data Sci Course (Fall 2015)\\Capstone\\PlayerSampleDistanceMetricComparison_20160109.csv')

