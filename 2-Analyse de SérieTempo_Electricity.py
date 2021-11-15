# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 21:14:57 2021

@author: ceecy
"""

# Import des librairies 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import detrend
from pylab import rcParams


# Chemin
path=r"C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S1_Algorithmique&Python/data/"

# Importation des csv
df_conso=pd.read_csv(path+"bilan-electrique-heure.csv",sep=";")

# Pour que ce soit dans le bon ordre
df_conso = df_conso.sort_values(by=['Horodate'])

# On supprime la variable à valeur manquante
df_conso.drop(columns = ["Pseudo rayonnement"], inplace = True)
# On met la variable 'horodate' en datetime format.
df_conso["Horodate"] = pd.to_datetime(df_conso["Horodate"])


# Importation et concaténation des prix 
dossiers = { 'prices' : [2016,2022]}

dfs = []
k = 'prices'
for i in range(dossiers[k][0],dossiers[k][1]):
    try:
        dfs.append(pd.read_excel(path+'{}\\{}.xlsx'.format(k,i)))
    except:
        pass
ganisland = pd.concat(dfs)
df_price = ganisland.reset_index()

df_price.drop(columns = ["index"], inplace = True)


# On récupère la période qui nous intéresse 
df_price = df_price[(df_price["Date"] >= "2016-10-23") & (df_price["Date"] <= "2021-10-22" )]


#Pour supprimer les lignes en trop avec un nan dans la colonne prix
index_with_nan = df_price.index[df_price["Prix"].isnull()]
df_price.drop(index_with_nan, 0, inplace=True)

#Suppression des colonnes 'Hours', 'Date' dans  pour qu'il ne reste que les prix
df_price = df_price.drop(['Hours', 'Date'], 1) 


#Remettre l'index à 0, pour les deux fichiers 
df_conso = df_conso.reset_index(drop=False, inplace=False)
df_price = df_price.reset_index(drop=False, inplace=False)


#On concatène les deux fichiers :
df_final = df_conso.merge(df_price, how='inner', left_index=True, right_index=True)

#On supprime les index créés par défaut en trop :
df_final.drop(columns = ["index_x","index_y"], inplace = True)

# On créé une nouvelle variable Year :
df_final.insert(1, 'Year',df_final["Horodate"].astype(str).str[:4] )
# On créé une nouvelle variable Hours :
df_final.insert(2, 'Hours',df_final["Horodate"].astype(str).str[11:13] )


#Export du df_final :
df_final.to_csv(path+"df_final.csv",index = False)


#Import du df_final :
path=r"C:/Users/ceecy/OneDrive - Université Paris-Dauphine/Economie/M2/S1_Algorithmique&Python/data/"
df_final = pd.read_csv(path+"df_final.csv",sep=",")


# ## Modélisation de la série temporelle 

# On essaye de modéliser la série en dégageant la `tendance`, la `saisonnalité` et le `bruit`


#Modélisation globale par type de consommation : 
    #Consommation des particuliers :
        
#Définition de la variable qu'on va décomposer.
var_tokeep = ["Horodate","Consommation résidentielle profilée (W)"] #Pour définir la var d'intérêt
Conso_par = df_final[var_tokeep] #On met la var d'intérêt et la timeseries dans un dataframe
Conso_par.set_index("Horodate", inplace = True) #Pour mettre la timeseries en index

#1 - FONCTIONNE PAS SUR L'ORDI A CECE
rcParams['figure.figsize'] = 11, 9 

decomposition = sm.tsa.seasonal_decompose(Conso_par, model='additive') # Période de 1 an donc 365 x 24 car données au pas horaire
fig = decomposition.plot()
plt.show()

### TypeError: Index(...) must be called with a collection of some kind, 'seasonal' was passed

#2
notrend = detrend(Conso_par["Consommation résidentielle profilée (W)"])
Conso_par["notrend"] = notrend
Conso_par["trend"] = Conso_par['Consommation résidentielle profilée (W)'] - notrend
Conso_par.tail()

Conso_par.plot(y=["Consommation résidentielle profilée (W)", "notrend", "trend"], figsize=(14,4));

    #Consommation des professionnels :
  
var_tokeep = ["Horodate","Consommation professionnelle profilée (W)"]
Conso_pro = df_final[var_tokeep]
Conso_pro.set_index("Horodate", inplace = True)

#1
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(Conso_pro, model='additive', period = 8760) # Période de 1 an donc 365 x 24 car données au pas horaire
fig = decomposition.plot()
plt.show()

#2
notrend = detrend(Conso_pro["Consommation professionnelle profilée (W)"])
Conso_pro["notrend"] = notrend
Conso_pro["trend"] = Conso_pro['Consommation professionnelle profilée (W)'] - notrend
Conso_pro.tail()

Conso_pro.plot(y=["Consommation professionnelle profilée (W)", "notrend", "trend"], figsize=(14,4));


#Création des dataframes par année
# Année random
df_2018 = df_final[df_final["Year"] == 2018]
# Année que l'on vise à prédire pour tester la qualité de la prédiction
df_2019 = df_final[df_final["Year"] == 2019]
# Année du confinement : on veut observer les différences avec les autres années.
df_2020 = df_final[df_final["Year"] == 2020]

#Modélisation annuelle par type de consommation : 
    #Consommation des particuliers :
        
var_tokeep = ["Horodate","Consommation résidentielle profilée (W)"]
Par_an = df_2020[var_tokeep]
Par_an.set_index("Horodate", inplace = True)

#1
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(Par_an, model='additive', period = 720) # Période de 1 mois donc 30 x 24 car données au pas horaire
fig = decomposition.plot()
plt.show()

#2
notrend = detrend(Par_an["Consommation résidentielle profilée (W)"])
Par_an["notrend"] = notrend
Par_an["trend"] = Par_an['Consommation résidentielle profilée (W)'] - notrend
Par_an.tail()

Par_an.plot(y=["Consommation résidentielle profilée (W)", "notrend", "trend"], figsize=(14,4));

    #Consommation des professionnels :
  
var_tokeep = ["Horodate","Consommation professionnelle profilée (W)"]
Pro_an = df_2020[var_tokeep]
Pro_an.set_index("Horodate", inplace = True)

#1
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(Pro_an, model='additive', period = 720) # Période de 1 mois donc 30 x 24 car données au pas horaire
fig = decomposition.plot()
plt.show()

#2
notrend = detrend(Pro_an["Consommation professionnelle profilée (W)"])
Pro_an["notrend"] = notrend
Pro_an["trend"] = Pro_an['Consommation professionnelle profilée (W)'] - notrend
Pro_an.tail()

Pro_an.plot(y=["Consommation professionnelle profilée (W)", "notrend", "trend"], figsize=(14,4));

#Création des dataframes par jour
# Jour d'été random en semaine pour 2018 et 2019
df_day18_S = df_2018[df_2018["Horodate"].astype(str).str.startswith("2018-06-12")] 
df_day19_S = df_2019[df_2019["Horodate"].astype(str).str.startswith("2019-06-12")]

# Jour d'hiver random en semaine pour 2018 et 2019
df_day18_w = df_2018[df_2018["Horodate"].astype(str).str.startswith("2018-12-12")] 
df_day19_w = df_2019[df_2019["Horodate"].astype(str).str.startswith("2019-12-12")]

#Est ce qu'on fait le week-end ? 

# Jour random en semaine pendant les 1er et deuxième confinement :
df_day20_1 = df_2020[df_2020["Horodate"].astype(str).str.startswith("2020-04-15")]
df_day20_2 = df_2020[df_2020["Horodate"].astype(str).str.startswith("2020-04-19")]

#Modélisation journée type par type de consommation : 
    #Consommation des particuliers :
        
var_tokeep = ["Horodate","Consommation résidentielle profilée (W)"]
Par_day = df_day18_w[var_tokeep]
Par_day.set_index("Horodate", inplace = True)

#1
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(Par_day, model='additive', period = 1) # Période de 1 heure donc 1 car données au pas horaire
fig = decomposition.plot()
plt.show()

#2
notrend = detrend(Par_day["Consommation résidentielle profilée (W)"])
Par_day["notrend"] = notrend
Par_day["trend"] = Par_day['Consommation résidentielle profilée (W)'] - notrend
Par_day.tail()

Par_day.plot(y=["Consommation résidentielle profilée (W)", "notrend", "trend"], figsize=(14,4));

    #Consommation des professionnels :
  
var_tokeep = ["Horodate","Consommation professionnelle profilée (W)"]
Pro_day = df_day18_w[var_tokeep]
Pro_day.set_index("Horodate", inplace = True)

#1
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(Pro_day, model='additive', period = 1) # Période de 1 heure donc 1 car données au pas horaire
fig = decomposition.plot()
plt.show()

#2
notrend = detrend(Pro_day["Consommation professionnelle profilée (W)"])
Pro_day["notrend"] = notrend
Pro_day["trend"] = Pro_day['Consommation professionnelle profilée (W)'] - notrend
Pro_day.tail()

Pro_day.plot(y=["Consommation professionnelle profilée (W)", "notrend", "trend"], figsize=(14,4));

# Si ça ne va pas même chose avec 'multiplicative' dans le cadre d'usage du modèle 1. 
    
# Il faut retirer la trend et la saisonnalité & trouver le pattern des résidus.