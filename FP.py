'''
331 FINAL PROJECT
'''

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#dict w/ arthropod taxons & common names mapped to abbreviations
abbs_taxon = {'COLE':['Coleoptera','Beetles'],'HYMA':['Hymenoptera Apocrita','Wasps, bees, ants'],'ARAN':['Araneae','Spiders'],'UNKI':['Unknown','Unknown insect'],
                  'ORTH':['Orthoptera','Grasshoppers, crickets'],'LEPI':['Lepidoptera','Butterflies, moths'],'SCOR':['Scorpiones','Scorpions'],'HYMB':['Hymenoptera Symphyta','Sawflies'],
                  'CRUS':['Crustacea','Crustaceans'],'THYS':['Thysanoptera','Thrips'],'DIPT':['Diptera','Flies'],'HETE':['Heteroptera','True bugs'],'SOLI':['Solifugae','Sun spiders'],
                  'AUCH':['Actinotrichida','Ticks, mites'],'MANT':['Mantodea','Mantises'],'NEUR':['Neuroptera','Lacewings'],'PSUE':['Pseudoscorpiones','Pseudoscorpions'],
                  'BLAT':['Blattodea','Cockroaches, termites'],'CHIL':['Chilopoda','Centipedes, millipedes'],'DERM':['Dermestidae','Larder beetle']}
#dict w/ reptile species codes mapped to abbreviations
abbs_spp = {'ASTI':['Aspidoscelis tigris','Tiger whiptail'],'UTST':['Uta stansburiana','Common side-blotches lizard'],'SCMA':['Sceloporus magister','Desert spiny lizard'],
                'COVA':['Coleonyx variegates','Western banded gecko'],'CADR':['Callisaurus draconoides','Zebra-tailed lizard'],'CRSC':[],'RHLE':[],'LEHU':[],'MAFL':[],
                'UNKS':['Unknown','Unknown reptile'],'PICA':[],'PHSO':['Phrynosoma solare','Regal horned lizard']}


#graph arthropod pred/prey and reptile population frequencies, fit a curve to the data (scatter plot w/ regression line?) maybe apply k means or something else here to show something about the data
'''
Reptiles: one point for count of total rows for each month of the year (or for each year?)
Arthropods: one point that sums the count column for each month of the year (or for each year?)
'''
#create/train model using global reptile information dataset (naive bayes?)
'''
Use info from global reptile dataset to create model & confusion matrix, y=scientific name and X=weight, measurements, region, habitat, elevation, etc.
'''
#apply model to AZ reptile data (confusion matrix? show decision boundaries? which feature/s are best for predicting reptile species given biological data?)
'''
Apply given reptile data from AZ to see how well it classifies it 
(using mean/max M/F for each species), make suggestion about data to be collected that would work best with the model
'''


def PhoenixAir_arthro_df():
    #read everything except notes col to df
    df = pd.read_csv('691_arthropods.csv',index_col=0, usecols=[0,1,2,3,4,6,7])
    #replace str index w/ datetime objects
    df.index = pd.to_datetime(df.index)
    return df

def CentralPhoenix_arthro_df():
    df = pd.read_csv('Invertabratedryweightdata_2.csv')
    return df

def PhoenixAir_reptile_df():
    df = pd.read_csv('691_captures.csv', index_col=0, usecols=[0,1,3,4,5,6,9,10,11,13,17])
    df.index = pd.to_datetime(df.index)
    return df

def format_rept_df(df):
    #remove all entries that aren't lizards
    df = df[df['Taxa'] == 'Lizard'] 
    df = df[df['SppCode'] != 'PHSO']
    #remove hatchlings
    df = df[df['Hatchling'].isin(['N','FALSE'])]
    return df

def arth_rept_pop_plot(adf, rdf):
    '''
    Plot frequency of arthropods & reptiles over time, x axis years and y axis count

    Reptiles: 1 entry for each lizard
    Arthropods: 
    '''
    plt.figure(figsize=(12,6))
    plt.title(r"Arthropod & Reptile Populations", fontsize=24)
    plt.xlabel(r"Day of the Year", fontsize=20)
    plt.ylabel(r"Count", fontsize=20)
    #plot arth pred count (total # of count for each month)
    plt.plot(rdf.index, adf[adf['Predator'] == 'Y']['count'], marker='o', linestyle='', markersize=3, label='Arthropod (Predators)')
    plt.show()
    #plot arth prey count (total # of count for each month)
    
    #plot rept count (total of rept samples for each month)


#scatter plot of reptile & arthropod frequency over time w/ reg line
#curve fitting? sine curve fit to arthro pred/prey & reptile population frequency

#confusion matrix predict classify reptile species given features (SVL, VTL, weight, trap type)


'''
x = rdf.loc[:,'SVL']
#x = rdf.loc[:,'VTL']
y = rdf.loc[:, 'weight']
#plt.ylim((0, 500))
plt.scatter(x, y)
plt.show()

rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL']
rdf.loc[rdf['SppCode'] == 'UTST', 'SVL']
rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL']
rdf.loc[rdf['SppCode'] == 'COVA', 'SVL']
rdf.loc[rdf['SppCode'] == 'CADR', 'SVL']

##
plt.scatter((rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL']), rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], c='red')
plt.scatter((rdf.loc[rdf['SppCode'] == 'UTST', 'SVL']/rdf.loc[rdf['SppCode'] == 'UTST', 'VTL']), rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], c='blue')
plt.scatter((rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL']), rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], c='green')
plt.scatter((rdf.loc[rdf['SppCode'] == 'COVA', 'SVL']/rdf.loc[rdf['SppCode'] == 'COVA', 'VTL']), rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], c='pink')
plt.scatter((rdf.loc[rdf['SppCode'] == 'CADR', 'SVL']/rdf.loc[rdf['SppCode'] == 'CADR', 'VTL']), rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], c='black')
plt.xlabel('SVL/VTL (mm)')
plt.ylabel('Body Weight (g)')
plt.show()
##
plt.plot((rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL']), marker='o', linestyle='', markersize=3, c='red')
plt.plot((rdf.loc[rdf['SppCode'] == 'UTST', 'SVL']/rdf.loc[rdf['SppCode'] == 'UTST', 'VTL']), marker='o', linestyle='', markersize=3, c='blue')
plt.plot((rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL']), marker='o', linestyle='', markersize=3, c='green')
plt.plot((rdf.loc[rdf['SppCode'] == 'COVA', 'SVL']/rdf.loc[rdf['SppCode'] == 'COVA', 'VTL']), marker='o', linestyle='', markersize=3, c='pink')
plt.plot((rdf.loc[rdf['SppCode'] == 'CADR', 'SVL']/rdf.loc[rdf['SppCode'] == 'CADR', 'VTL']), marker='o', linestyle='', markersize=3, c='black')
plt.xlabel('count')
plt.ylabel('SVL/VTL (mm)')
plt.show()
##
plt.scatter((rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL']), rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], c='red')
plt.scatter((rdf.loc[rdf['SppCode'] == 'UTST', 'SVL']/rdf.loc[rdf['SppCode'] == 'UTST', 'VTL']), rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], c='blue')
plt.scatter((rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL']), rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], c='green')
plt.scatter((rdf.loc[rdf['SppCode'] == 'COVA', 'SVL']/rdf.loc[rdf['SppCode'] == 'COVA', 'VTL']), rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], c='pink')
plt.scatter((rdf.loc[rdf['SppCode'] == 'CADR', 'SVL']/rdf.loc[rdf['SppCode'] == 'CADR', 'VTL']), rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], c='black')
plt.xlabel('SVL/VTL (mm)')
plt.ylabel('Body Weight (g)')
plt.show()
##
plt.scatter((rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'weight']), rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], c='red')
plt.scatter((rdf.loc[rdf['SppCode'] == 'UTST', 'SVL']/rdf.loc[rdf['SppCode'] == 'UTST', 'weight']), rdf.loc[rdf['SppCode'] == 'UTST', 'VTL']/rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], c='blue')
plt.scatter((rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'weight']), rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], c='green')
plt.scatter((rdf.loc[rdf['SppCode'] == 'COVA', 'SVL']/rdf.loc[rdf['SppCode'] == 'COVA', 'weight']), rdf.loc[rdf['SppCode'] == 'COVA', 'VTL']/rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], c='pink')
plt.scatter((rdf.loc[rdf['SppCode'] == 'CADR', 'SVL']/rdf.loc[rdf['SppCode'] == 'CADR', 'weight']), rdf.loc[rdf['SppCode'] == 'CADR', 'VTL']/rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], c='black')
plt.xlabel('SVL/Body Weight (mm/g)')
plt.ylabel('VTL/Body Weight (mm/g)')
plt.show()
##

ax.scatter(rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL'], rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL'], rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], c='red')
ax.scatter(rdf.loc[rdf['SppCode'] == 'UTST', 'SVL'], rdf.loc[rdf['SppCode'] == 'UTST', 'VTL'], rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], c='blue')
ax.scatter(rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL'], rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL'], rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], c='green')
ax.scatter(rdf.loc[rdf['SppCode'] == 'COVA', 'SVL'], rdf.loc[rdf['SppCode'] == 'COVA', 'VTl'], rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], c='pink')
ax.scatter(rdf.loc[rdf['SppCode'] == 'CADR', 'SVL'], rdf.loc[rdf['SppCode'] == 'CADR', 'VTL'], rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], c='black')


x = rdf.loc[:,'SVL']
y = rdf.loc[:,'VTL']
z = rdf.loc[:, 'weight']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL'], rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL'], rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], c='red')
ax.scatter(rdf.loc[rdf['SppCode'] == 'UTST', 'SVL'], rdf.loc[rdf['SppCode'] == 'UTST', 'VTL'], rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], c='blue')
ax.scatter(rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL'], rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL'], rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], c='green')
ax.scatter(rdf.loc[rdf['SppCode'] == 'COVA', 'SVL'], rdf.loc[rdf['SppCode'] == 'COVA', 'VTl'], rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], c='pink')
ax.scatter(rdf.loc[rdf['SppCode'] == 'CADR', 'SVL'], rdf.loc[rdf['SppCode'] == 'CADR', 'VTL'], rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], c='black')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter((rdf.loc[rdf['SppCode'] == 'ASTI', 'SVL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'weight']), rdf.loc[rdf['SppCode'] == 'ASTI', 'VTL']/rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], rdf.loc[rdf['SppCode'] == 'ASTI', 'weight'], c='red')
ax.scatter((rdf.loc[rdf['SppCode'] == 'UTST', 'SVL']/rdf.loc[rdf['SppCode'] == 'UTST', 'weight']), rdf.loc[rdf['SppCode'] == 'UTST', 'VTL']/rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], rdf.loc[rdf['SppCode'] == 'UTST', 'weight'], c='blue')
ax.scatter((rdf.loc[rdf['SppCode'] == 'SCMA', 'SVL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'weight']), rdf.loc[rdf['SppCode'] == 'SCMA', 'VTL']/rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], rdf.loc[rdf['SppCode'] == 'SCMA', 'weight'], c='green')
ax.scatter((rdf.loc[rdf['SppCode'] == 'COVA', 'SVL']/rdf.loc[rdf['SppCode'] == 'COVA', 'weight']), rdf.loc[rdf['SppCode'] == 'COVA', 'VTL']/rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], rdf.loc[rdf['SppCode'] == 'COVA', 'weight'], c='pink')
ax.scatter((rdf.loc[rdf['SppCode'] == 'CADR', 'SVL']/rdf.loc[rdf['SppCode'] == 'CADR', 'weight']), rdf.loc[rdf['SppCode'] == 'CADR', 'VTL']/rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], rdf.loc[rdf['SppCode'] == 'CADR', 'weight'], c='black')
plt.show()
'''


#create scatter plot of arthropod species frequency over time, with x axis datetime and y axis frequency
#one line for each species?
#regression line describing total frequency of all species combined

#plot of SVL by VTL with species groups