# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:20:05 2019

@author: Pablo Saldarriaga
"""
### Paquetes necesarios para el funcionamiento de las funciones en este script
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import distance
from sklearn.covariance import LedoitWolf
from sklearn.tree import ExtraTreeClassifier
#%%
logger = logging.getLogger(__name__)
#%%

### La funcion toma como entrada el conjunto de datos en el cual se desea
### realizar la deteccion de outliers multivariante. El metodo utiliza la
### distancia de Mahalanobis y la matriz de covarianza habitual. Se define
### alpha como el percentil en el cual se realizara el corte de la matriz de 
### distancias para determinar que registros son considerados outliers. 
def outlier_detection_mahal(X,alpha = 0.95):
    logger.info('Comienza la deteccion de outliers')
    if len(X)==0:
        logger.info('No hay datos para realizar deteccion de outliers')
        return None
    try:
        X = np.array(X)
        X_arr = np.array(X)
        X_mean = X_arr.mean(axis = 0)
        cov = np.array(pd.DataFrame(X).cov())
        inv_cov = np.linalg.inv(cov)
        
        dist = []
        i=0
        while i<len(X):
            distances = distance.mahalanobis(X_arr[i], X_mean, inv_cov)
            dist.append(distances)
            i +=1
        
        dist = np.array(dist)
        cutoff = np.quantile(dist,alpha)
        outliers = (dist>cutoff).astype(int)
    except Exception as e:
        logger.info(f'Error en la deteccion de outliers:{e}')
    
    logger.info('Deteccion de outliers exitoso')
    return outliers


### Esta funcion toma como entrada un dataframe con el conjunto de datos, la
### etiqueta de 1s y 0s, y realiza un stepwise en la regresion logistica para
### conservar las variables mas relevantes para explicar la etiqueta. La
### funcion devuelve una lista con las variables relevantes, y el modelo final
### de la regresion logistica
def stepwise_logistic(X,Y,alpha =0.1):
    
    try:
        n = len(X.columns)
        for i in range(n):
            model = sm.Logit(Y, X).fit()
            pvalues = pd.DataFrame(model.pvalues,columns = ['pvalues']).reset_index()
            if pvalues['pvalues'].max()<alpha:
                break
            to_drop = pvalues[pvalues['pvalues']==pvalues['pvalues'].max()]['index'].values[0]
            X = X.drop(columns = to_drop)
    
        model = sm.Logit(Y, X).fit()
        variables_relevantes = list(X.columns) 
    except Exception as e:
        logger.info(f'Error en el stepwise logistic:{e}')
        model = None
        variables_relevantes = []
        
    return variables_relevantes,model


### Se construye la funcion de variables relevantes utilizando metodos de
### arboles. Esta funcion entrena un arbol para el conjunto de datos y
### retorna las variables mas relevantes para explicar la variable Y. Hay
### que tener en cuenta que esta funcion toma como entrada DataFrames, donde
### para la variable X, las columnas tienen los nombres de las variables.
def variables_relevantes_arbol(X,Y,alpha = None):
    
    if len(X)==0:
        logger.info("No se ingreso informacion de variables")
        return []
    
    features = list(X.columns)
    
    if alpha == None:
        alpha = 1.0/len(features)
        logger.info(f'Se calcula el valor minimo de aceptacion de importancia: {alpha}')
    
    try:    
        model = ExtraTreeClassifier()
        model.fit(X,Y)
        
        importance = model.feature_importances_
        
        relevant_features = []
        for i in range(len(features)):
            if importance[i]>alpha:
                relevant_features.append(features[i])
                
    except Exception as e:
        logger.info(f'Error con el metodo de arboles, no se determinaron variables relevantes: {e}')
        relevant_features = []
        
    return relevant_features

### Esta funcion calcula la matriz de covarianza de Ledoit and Wolf, retorna
### la matriz de covarianza despues de aplicar el metodo de Shrinkage, ademas
### de retornar la media estimada. Esta funcion toma como parametros de entrada
### el conjunto de datos
def LedoitWolf_covMatrix(X):
    
    cov = LedoitWolf().fit(X)
    cov_matrix = cov.covariance_
    mean_vector = cov.location_
    
    return cov_matrix, mean_vector


### Funcion de distancia p
def distancia(a,b, p):
    ## if p = 0, que use la de mahalanobis (aun no esta programada)
    
    ### por ahora, la cuadratica
    return np.linalg.norm(a-b, p)


### Inicializa los centroides para el metodo de kmeans, inicializa de forma
### aleatoria tantos centros como clusters
def init_centroids(X_data,k):
    
    centroids = []
    numdata = len(X_data)
    for i in range(k):
        centroids.append(X_data[np.random.randint(0, numdata)])
        
    return centroids

### Metodo de kmeans
def kmeans(X_data,k,numiter,centroids,p_dista = 2,etiquetas = []):

    numdata = len(X_data)    
    if len(etiquetas)==0:
         ### Etiquetas actuales de cada elemento para cada cluster
        etiquetas = np.ones(numdata)*-1   ### Inicialmente, ningun elemento esta asignado
    
    ### Grados de pertenencia a cada cluster
    grados_pertenencia = []
        
    
    ### Ahora empiezo las iteraciones
    for it in range(numiter):
        
        ### En cada iteracion, itero para todos los elementos
        for element in range(numdata):
            
            ### Evaluo las distancias a cada centroides
            distc = []
            for c in centroids:
                distc.append(distancia(X_data[element], c, p_dista))
            
            ### Encuentro el centroide al que tiene menor distancia
            nearest_centroid = np.argmin(distc)
            
            ### Asigno el elemento a este cluster
            etiquetas[element] = nearest_centroid
            
            ### Grados de pertenencia a cada cluster
            grados_pertenencia.append(str(list(np.around(1/(distc/sum(distc))/sum(1/(distc/sum(distc))),3))))
            
            ### Recalculo el centroide 
            centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas==nearest_centroid)], axis=0)   
        
        centroids = np.array(centroids)
    return grados_pertenencia,etiquetas,centroids









