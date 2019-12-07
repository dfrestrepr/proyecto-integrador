# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:38:34 2019

@author: nicol
"""


import funciones
import pandas as pd
import numpy as np


################# Correr parte inicial para crear algunas variables

from sklearn.preprocessing import StandardScaler
datos = pd.read_csv('data/data_gapminder_proc2.csv')
### Variables a usar 
datos = datos[datos.columns[:]]  
#### Preprocesamiento de datos
datos = funciones.data_preprocessing(datos, alpha_outlier_detection =0.96, 
                                     columns_not_numeric = {'country','Date'},
                                     column_id = 'country', 
                                     shrinkage = False)
### Estandarizo todos los datos
datos_e = datos.copy()
scaler_es = StandardScaler()
datos_e[datos_e.columns[2:]] = scaler_es.fit_transform(datos_e[datos_e.columns[2:]])



#### De aqui necesitaba scaler_es. ademas uso
year_i = 1980
periodos_incluir = 30
k = 3





###################### Lectura resultados spark


#### Leer csv con resultados de spark
datos_e = pd.read_csv('datosspark/datos_e.csv')
datos_pca = pd.read_csv('datosspark/datos_pca.csv')
X_data_df = pd.read_csv('datosspark/X_data_df.csv')
grad_per = pd.read_csv('datosspark/grad_per.csv')
etiquetas_glo = pd.read_csv('datosspark/etiquetas_glo.csv')
centroids_ite = pd.read_csv('datosspark/centroids_ite.csv')
imp_periods_var = pd.read_csv('datosspark/imp_periods_var.csv')




#### Corregir X_data_df
X_data_df = X_data_df.drop(columns='index')



#### Corregir datos_e
datos_e = datos_e.drop(columns='index')

#### Corregir datos_pca
datos_pca = datos_pca.drop(columns='index')


#### Corregir grad_per
new_grad_per = []
for i in range(len(grad_per)):
    grau = list(grad_per.loc[i][1:])
    formo = []
    for b in grau:
        bu = eval(b)
        formo.append(bu)
    new_grad_per.append(formo)
grad_per = new_grad_per.copy()



#### Corregir etiquetas_glo
new_eti_glo = []
for i in range(len(etiquetas_glo)):
    grab = np.array(etiquetas_glo.loc[i][1:])
    new_eti_glo.append(grab)
etiquetas_glo = new_eti_glo.copy()



#### Corregir centroids_ite
new_cent = []
coun = 0
### Creo np array
este_paso = np.zeros((3,9))

for i in range(len(centroids_ite)):

    ### Voy agregando
    grab = list(centroids_ite.loc[i][1:])
    
    for u in grab:        
        
        ### Asigno
        este_paso[int(coun/9)][coun%9] = u
        
        coun = coun + 1
        
        
        
        ### Reiniciar
        if coun == 27:
            new_cent.append(este_paso)
            este_paso = np.zeros((3,9))
            coun=0

            
centroids_ite = new_cent.copy()




#### Corregir imp_periods_var
new_i_v = []
for i in range(len(imp_periods_var)):
    grau = list(imp_periods_var.loc[i][1:])
    formo = []
    for b in grau:
        buo = []
        for ki in b.split(' '):
            ki1 = ki.replace('[','')
            ki2 = ki1.replace(']','')
            if len(ki2)>=1:

                buo.append(float(ki2.strip()))
        bu = np.array(buo)
        formo.append(bu)
    new_i_v.append(formo)
    
    
    
imp_periods_var = new_i_v.copy()




###############################################################################
########################### NUEVA VISUALIZACION  ##############################
###############################################################################

funciones.gapminder_plot_bokeh(datos_e, datos_pca, year_i, X_data_df, grad_per,
                         etiquetas_glo, periodos_incluir, k, imp_periods_var,
                         centroids_ite, scaler_es,
                         title = 'Gapminder data',
                         xlabel='Componente principal 1',
                         ylabel='Componente principal 2')

#    
#from bokeh.plotting import output_file, save
#output_file("test.html")
#save(layout)
#
