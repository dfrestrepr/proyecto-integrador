# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:38:34 2019

@author: nicol
"""
#%%
import os
import funciones
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%
file_path = os.path.dirname(os.path.abspath(__file__))
#%% 
### Configuracion del logger
import logging
from logging.handlers import RotatingFileHandler

file_name = 'Segmentacion_dinamica'
logger = logging.getLogger()
dir_log = os.path.join(file_path, f'logs/{file_name}.log')

if not os.path.exists('logs'):
    os.mkdir('logs')

handler = RotatingFileHandler(dir_log, maxBytes=2000000, backupCount=10)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
                    handlers = [handler])
#%%
logger.info('*'*80)
logger.info('Inicia ejecucion del programa')
#### Version propia de kmeans para proyecto integrador

### Leemos los datos 
datos = pd.read_csv('outputs/data_gapminder_proc2.csv')

### Variables a usar 
datos = datos[datos.columns[:]]  

### Nombres variables
nomb_vars = datos.columns[2:]

#### Preprocesamiento de datos
datos = funciones.data_preprocessing(datos, alpha_outlier_detection =0.96, 
                                     columns_not_numeric = {'country','Date'},
                                     column_id = 'country', 
                                     shrinkage = False)

### Estandarizo todos los datos
datos_e = datos.copy()
scaler_es = StandardScaler()
datos_e[datos_e.columns[2:]] = scaler_es.fit_transform(datos_e[datos_e.columns[2:]])
datos_e = datos_e.fillna(0)

#### Aplicarle PCA a todos los datos
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_e[datos_e.columns[2:]])


### Fijar seed aleatoria
np.random.seed(1)

##### Inicio tomando los del primer year
year_i = min(datos_e['Date'])   ### Year inicial a considerar
filtro = datos_e['Date']==year_i
X_data_df = datos_e[filtro].reset_index(drop=True)
X_data = np.array(X_data_df[X_data_df.columns[2:]])


### Numero de periodos que incluire en el estudio, sin incluir el inicial
periodos_incluir = max(datos_e['Date']) - min(datos_e['Date'])

### Los que usare para el PCA seran
X_data_pca = np.array(datos_pca[filtro])


#### Lista donde ire guardando las listas de grados de pertenencia 
grad_per = []

### Lista donde ire guardando las etiquetas asignadas del cluster
etiquetas_glo = []

### Lista donde ire guardando las variables mas importantes por cluster y por periodo
imp_periods_var = []

### Lista donde ire guardando los centroides de cada iteracion
centroids_ite = []


## Numero de observaciones en cada periodo
numdata = len(X_data)
  

### Define cantidad de clusters, numero maximo de iteraciones, y la distancia
### que se utilizara en el metodo de kmeans
k = 3
numiter = 5
p_dista = 2   ### 0 para mahalanobis


#### Inicializar los centroides
centroids = funciones.init_centroids(X_data,k)
centroids_pca = pca.transform(centroids)

### Para la fase 2 de las importancias
centroids_p = centroids.copy()

### Aplicar kmeans
grados_pertenencia,etiquetas,centroids = funciones.kmeans(X_data,
                                                          numiter,
                                                          centroids,
                                                          p_dista = p_dista)


### Guardo grados de pertenencia
grad_per.append(grados_pertenencia.copy())

### Guardo etiquetas
etiquetas_glo.append(etiquetas.copy())

### Guardo centroids
centroids_ite.append(centroids.copy())

### Variable global donde ire guardando la importancia de las variables en cada 
### iteracion
imp_iters = []


### Obtener la importancia de las variables de cada cluster (de mayor a menor)
importancias_cluster = []

### Para cada cluster
for clu in range(k):
    ### Dejo solo las observaciones de cada cluster
    datax_i = pd.DataFrame(X_data)
    datay_i = etiquetas.copy()
    #### Solo clasifico binario si si pertence o no a cada cluster
    distintos_cluster = np.where(datay_i!=clu)
    ### Lo que no pertenece al cluster, lo pongo en -1
    datay_i[distintos_cluster] = -1
    datay_i = pd.DataFrame(datay_i)
    
    
    ### Calcular relevancias
    relevancias, _ = funciones.variables_relevantes_arbol(datax_i, datay_i, 0)
    
    importancias_cluster.append(relevancias)

##### Calculo los promedios de importancia de cada variable
imp_clus_prom =  np.mean(importancias_cluster, axis=0)

### Guardo las importancias de esta iteracion
imp_iters.append(imp_clus_prom)


### Guardo importancias generales (para el plot)
imp_periods_var.append(importancias_cluster)


###############################################################################
################## Ahora, empiezo a iterar para t >=2 #########################
###############################################################################



for periodos in range(periodos_incluir):
    
    ### Guardo la X_data anterior
    X_data_viej = X_data.copy()
    centroids_viej = centroids.copy()
    

    ### Los datos para este year ya serian
    X_data_df = datos_e[datos_e['Date']==year_i+1+periodos].reset_index(drop=True)
    X_data = np.array(X_data_df[X_data_df.columns[2:]])

    #### Obtener los 2 componentes principales de los datos para plotear estos
    X_data_pca = np.array(datos_pca[datos_e['Date']==year_i+1+periodos])
    
    ### Calulo los cambios en las variables
    cambios_variables = X_data_viej - X_data    
    
    ###########################################################################
    ######################### Ponderacion dinamica ############################
    ###########################################################################
    
    ### Pondero X_data
    X_data_ori = X_data.copy()  ### X data original (sin ponderacion)
    
    
    ### Obtengo la importancia promedio de cada variable (promedio de todas las
    ### iteraciones)
    importancia_prom = np.mean(imp_iters, axis=0)
    
    ### Rankeo las variables de menor a mayor importancia
    rank_variables = np.argsort(importancia_prom)
    rankpeso_variables = np.zeros(len(rank_variables))
    
    cont = 0
    for i in rank_variables:
        rankpeso_variables[i] = (cont+1)/len(rank_variables)
        cont= cont+1

    #### Usar rankings o usar los promedios para el peso
    peso_variables = importancia_prom.copy()*100  ### Escalarlos con 100 para reducir errores numericos

    
    ### Escalo entonces la X para cambiar los pesos (segun las importancias)
    X_data_pond = X_data.copy()
    for peso in range(len(peso_variables)):
        X_data_pond[:,peso] = X_data_pond[:,peso] * peso_variables[peso]

    
    ###########################################################################
    ######################## K means para los plots ###########################
    ###########################################################################
    
    ### Etiquetas actuales de cada elemento para cada cluster
    etiquetas_prev = etiquetas.copy()
    
    ###########################################################################
    #################### Clusters con k means ponderado #######################
    ###########################################################################
    
    grados_pertenencia,etiquetas,centroids = funciones.kmeans(X_data_pond,
                                                              numiter,
                                                              centroids,
                                                              p_dista = p_dista,
                                                              etiquetas = etiquetas)


    ### Guardo grados de pertenencia
    grad_per.append(grados_pertenencia.copy())
    
    ### Guardo etiquetas
    etiquetas_glo.append(etiquetas.copy())
    
    ### Guardo centroids (con los valores originales sin ponderar)
    centroids_ite.append(centroids.copy()* (1/peso_variables))



    ###### Esta importancia la necesito para los labels
    ### Obtener la importancia de las variables de cada cluster (de mayor a menor)
    importancias_cluster = []
    ### Para cada cluster
    for clu in range(k):
        ### Dejo solo las observaciones de cada cluster
        datax_i = pd.DataFrame(X_data_pond)
        datay_i = etiquetas.copy()
        
        #### Solo clasifico binario si si pertence o no a cada cluster
        distintos_cluster = np.where(datay_i!=clu)
        
        ### Lo que no pertenece al cluster, lo pongo en -1
        datay_i[distintos_cluster] = -1
        datay_i = pd.DataFrame(datay_i)
        
        ### Calcular relevancias
        relevancias, _ = funciones.variables_relevantes_arbol(datax_i, datay_i, 0)
        
        importancias_cluster.append(relevancias)

    ### Guardo importancias generales (para el plot)
    imp_periods_var.append(importancias_cluster)




    ###########################################################################
    ################ K means para la seleccion de variables ###################
    ###########################################################################
    
    ###### Para la proxima iteracion, los pesos
    grados_pertenencia_p,etiquetas_p,centroids_p = funciones.kmeans(X_data_ori,
                                                                    numiter,
                                                                    centroids_p,
                                                                    p_dista = p_dista,
                                                                    etiquetas = etiquetas)
    
    ### Obtener la importancia de las variables de cada cluster (de mayor a menor)
    importancias_cluster = []
    ### Para cada cluster
    for clu in range(k):
        ### Dejo solo las observaciones de cada cluster
        datax_i = pd.DataFrame(X_data_ori)
        datay_i = etiquetas_p.copy()
        
        #### Solo clasifico binario si si pertence o no a cada cluster
        distintos_cluster = np.where(datay_i!=clu)
        
        ### Lo que no pertenece al cluster, lo pongo en -1
        datay_i[distintos_cluster] = -1
        datay_i = pd.DataFrame(datay_i)
        
        
        ### Calcular relevancias
        relevancias, _ = funciones.variables_relevantes_arbol(datax_i, datay_i, 0)
        
        importancias_cluster.append(relevancias)
    
    ### Calculo los promedios de importancia de cada variable
    imp_clus_prom =  np.mean(importancias_cluster, axis=0)
    
    ### Guardo las importancias de esta iteracion
    imp_iters.append(imp_clus_prom)


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
