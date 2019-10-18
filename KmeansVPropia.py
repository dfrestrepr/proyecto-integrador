# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:38:34 2019

@author: nicol
"""

#### Version propia de kmeans para proyecto integrador


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Leemos los datos 
datos = pd.read_csv('outputs/data_gapminder_proc.csv')
datos = datos[datos.columns[[0,1,4,6]]]  ### Por ahora solo usare 2 variables

### Estandarizo todos los datos
from sklearn.preprocessing import StandardScaler
datos_e = datos.copy()
datos_e[datos_e.columns[2:]] = StandardScaler().fit_transform(datos_e[datos_e.columns[2:]])
datos_e = datos_e.fillna(0)

### Fijar seed
np.random.seed(1)






#### Datos dummy (3 distribuciones normal aletaorias distintas)
#numda = 200## Num dats por cada distrib
#
#mu1, sigma1 = 0, 0.3 # mean and standard deviation
#x1 = np.random.normal(mu1, sigma1, numda)
#y1 = np.random.normal(mu1, sigma1, numda)
#mu2, sigma2 = 1, 0.3 # mean and standard deviation
#x2 = np.random.normal(mu2, sigma2, numda)
#y2 = np.random.normal(mu2, sigma2, numda)
#mu3, sigma3 = 2, 0.3 # mean and standard deviation
#x3 = np.random.normal(mu3, sigma3, numda)
#y3 = np.random.normal(mu3, sigma3, numda)
#### Junto los datos en una sola matriz de datos
#x=np.append(x1,[x2,x3])
#y=np.append(y1,[y2,y3])
#
#
#
#X_data = np.vstack((x, y)).T
#
#
#### Ploteo inicial de los valores
#plt.figure()
#plt.plot(x,y,'o')


##### Inicio tomando los del primer year
year_i = 2012
X_data_df = datos_e[datos_e['Date']==year_i].reset_index(drop=True)
X_data = np.array(X_data_df[X_data_df.columns[2:]])



## Numero de datos
numdata = len(X_data)



### Funcion de distancia
def distancia(a,b):
    ### por ahora, la cuadratica
    return np.linalg.norm(a-b, 2)
    

#### Ahora si, empezar con el metodo de kmeans



### el numero de clusters sera
k=3


### Inicialmente, asignar 3 puntos de forma arbitraria como centroides
centroids = X_data[[np.random.randint(0, numdata), np.random.randint(0, numdata), np.random.randint(0, numdata)]]



### Numero de iteraciones del k means
numiter = 5

### Etiquetas actuales de cada elemento para cada cluster
etiquetas = np.ones(numdata)*-1   ### Inicialmente, ningun elemento esta asignado

### Grados de pertenencia a cada cluster
grados_pertenencia = []
    

### Ahora empiezo las iteraciones
for it in range(numiter):
    
    ### En cada iteracion, itero para todos los elementos
    for el in range(numdata):
        
        ### Evaluo las distancias a cada centroides
        distc = []
        for c in centroids:
            distc.append(distancia(X_data[el], c))
        
        ### Encuentro el centroide al que tiene menor distancia
        nearest_centroid = np.argmin(distc)
        
        ### Asigno el elemento a este cluster
        etiquetas[el] = nearest_centroid
        
        ### Grados de pertenencia a cada cluster
        grados_pertenencia.append(str(list(np.around(1/(distc/sum(distc))/sum(1/(distc/sum(distc))),3))))
        
        ### Recalculo el centroide 
        centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas==nearest_centroid)], axis=0)
            
            
        
#        
#### Finalmente, pinto los resultados finales de los clusters
#plt.figure()
#plt.plot(X_data[:,0][np.where(etiquetas==0)],X_data[:,1][np.where(etiquetas==0)], 'ro')
#plt.plot(X_data[:,0][np.where(etiquetas==1)],X_data[:,1][np.where(etiquetas==1)], 'go')
#plt.plot(X_data[:,0][np.where(etiquetas==2)],X_data[:,1][np.where(etiquetas==2)], 'bo')
#plt.plot(centroids[:,0],centroids[:,1], 'yo')
#


#PLotear con hover tool
from math import sin
from random import random
from bokeh.io import output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, BoxZoomTool, WheelZoomTool, ResetTool, PanTool
from bokeh.palettes import plasma
from bokeh.plotting import figure
from bokeh.transform import transform

list_x = X_data[:,0]
list_y = X_data[:,1]
list_pais = X_data_df['country'].values
grados_pertenencia = np.array(grados_pertenencia)
desc = [str(i) for i in list_y]
source = ColumnDataSource(data=dict(x=list_x, y=list_y))


hover = HoverTool(tooltips=[
    ("pais","@pais"),
    ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
    ("cluster_id", "@cluster_id"),
    ("Pertenencia_clusters", "@grados_p"),

])
mapper = LinearColorMapper(palette=plasma(256), low=min(list_y), high=max(list_y))

p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), ResetTool(), BoxZoomTool(), WheelZoomTool()], title=str(year_i))

### PLoteo cada conjunto
source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==0)], 'y':list_y[np.where(etiquetas==0)], 'pais':list_pais[np.where(etiquetas==0)],'grados_p':grados_pertenencia[np.where(etiquetas==0)],'cluster_id':etiquetas[np.where(etiquetas==0)]})
p.circle('x','y', size=12, 
         fill_color='blue', source=source)
source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==1)], 'y':list_y[np.where(etiquetas==1)], 'pais':list_pais[np.where(etiquetas==1)],'grados_p':grados_pertenencia[np.where(etiquetas==1)],'cluster_id':etiquetas[np.where(etiquetas==1)]})
p.circle('x','y', size=12, 
         fill_color='yellow', source=source)
source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==2)], 'y':list_y[np.where(etiquetas==2)], 'pais':list_pais[np.where(etiquetas==2)],'grados_p':grados_pertenencia[np.where(etiquetas==2)],'cluster_id':etiquetas[np.where(etiquetas==2)]})
p.circle('x','y', size=12, 
         fill_color='red', source=source)

### Ploteo centroides
p.square(centroids[:,0], centroids[:,1], size=15, 
         fill_color='black')

### Labels
p.xaxis.axis_label = datos.columns[-2]
p.yaxis.axis_label = datos.columns[-1]

output_file('outputs/ClustersGenerados/cluster_inicial_'+str(year_i)+'.html')
save(p)





#################### Ahora, empiezo a iterar para t >=2

for periodos in range(4):
    
    ### Guardo la X_data anterior
    X_data_viej = X_data.copy()
    centroids_viej = centroids.copy()
    
#    #### Perturbo levemente todos los valroes de la matriz X
#    for i in range(X_data.shape[0]):
#        for j in range(X_data.shape[1]):
#            X_data[i][j] = X_data[i][j] + np.random.normal(0, 0.01, 1)[0]
#    
    
    ### Los datos para este year ya serian
    X_data_df = datos_e[datos_e['Date']==year_i+1+periodos].reset_index(drop=True)
    X_data = np.array(X_data_df[X_data_df.columns[2:]])
    
    ### Calulo los cambios en las variables
    cambios_variables = X_data_viej - X_data    
    

    
    ### Etiquetas actuales de cada elemento para cada cluster
    etiquetas_prev = etiquetas.copy()
#    etiquetas = np.ones(numdata)*-1   ### Inicialmente, ningun elemento esta asignado
    
    ### Grados de pertenencia a cada cluster
    grados_pertenencia = []
    
    ### Ahora empiezo las iteraciones
    for it in range(numiter):
        
        ### En cada iteracion, itero para todos los elementos
        for el in range(numdata):
            
            ### Evaluo las distancias a cada centroides
            distc = []
            for c in centroids:
                distc.append(distancia(X_data[el], c))
            
            ### Encuentro el centroide al que tiene menor distancia
            nearest_centroid = np.argmin(distc)
            
            ### Asigno el elemento a este cluster
            etiquetas[el] = nearest_centroid
            
            ### Grados de pertenencia a cada cluster
            grados_pertenencia.append(str(list(np.around(1/(distc/sum(distc))/sum(1/(distc/sum(distc))),3))))
            
            ### Recalculo el centroide 
            centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas==nearest_centroid)], axis=0)
                
                
    print(centroids)

    #PLotear con hover tool
    list_x = X_data[:,0]
    list_y = X_data[:,1]
    list_xv = cambios_variables[:,0]
    list_yv = cambios_variables[:,1]
    list_pais = X_data_df['country'].values
    grados_pertenencia = np.array(grados_pertenencia)

    desc = [str(i) for i in list_y]
    
    source = ColumnDataSource(data={'x':list_x, 'y':list_y, 'xv':list_xv, 'yv': list_yv})
    
    
    hover = HoverTool(tooltips=[
                ("pais","@pais"),
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ("(Cambio_x,Cambio_y)", "(@xv, @yv)"),    
    ("cluster_id", "@cluster_id"),
    ("Pertenencia_clusters", "@grados_p"),
    ])
    mapper = LinearColorMapper(palette=plasma(256), low=min(list_y), high=max(list_y))
    
    p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), ResetTool(), BoxZoomTool(), WheelZoomTool()], title=str(year_i+1+periodos))
    
    ### PLoteo cada conjunto
    source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==0)], 'y':list_y[np.where(etiquetas==0)], 'xv':list_xv[np.where(etiquetas==0)], 'yv': list_yv[np.where(etiquetas==0)],'pais':list_pais[np.where(etiquetas==0)],'grados_p':grados_pertenencia[np.where(etiquetas==0)],'cluster_id':etiquetas[np.where(etiquetas==0)]})
    p.circle('x','y', size=12, 
             fill_color='blue', source=source)
    source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==1)], 'y':list_y[np.where(etiquetas==1)], 'xv':list_xv[np.where(etiquetas==1)], 'yv': list_yv[np.where(etiquetas==1)],'pais':list_pais[np.where(etiquetas==1)],'grados_p':grados_pertenencia[np.where(etiquetas==1)],'cluster_id':etiquetas[np.where(etiquetas==1)]})
    p.circle('x','y', size=12, 
             fill_color='yellow', source=source)
    source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==2)], 'y':list_y[np.where(etiquetas==2)], 'xv':list_xv[np.where(etiquetas==2)], 'yv': list_yv[np.where(etiquetas==2)],'pais':list_pais[np.where(etiquetas==2)],'grados_p':grados_pertenencia[np.where(etiquetas==2)],'cluster_id':etiquetas[np.where(etiquetas==2)]})
    p.circle('x','y', size=12, 
             fill_color='red', source=source)
    
    ### Veo cuales cambiaron de cluster
    etiquetas_cambios = np.where(etiquetas_prev-etiquetas != 0)
    etiqs = etiquetas[etiquetas_cambios]

    ### PLoteo los elementos de cada conjunto que cambiaron de cluster
    X_data_cambios = X_data[etiquetas_cambios]
    list_x = X_data_cambios[:,0]
    list_y = X_data_cambios[:,1]
    list_xv = cambios_variables[:,0]
    list_yv = cambios_variables[:,1]
    list_pais = list_pais[etiquetas_cambios]
    grados_pertenencia = grados_pertenencia[etiquetas_cambios]

    source = ColumnDataSource(data={'x':list_x[np.where(etiqs==0)], 'y':list_y[np.where(etiqs==0)], 'xv':list_xv[np.where(etiqs==0)], 'yv': list_yv[np.where(etiqs==0)],'pais':list_pais[np.where(etiqs==0)],'grados_p':grados_pertenencia[np.where(etiqs==0)],'cluster_id':etiqs[np.where(etiqs==0)]})
    p.square('x','y', size=6, 
             fill_color='white', source=source)
    source = ColumnDataSource(data={'x':list_x[np.where(etiqs==1)], 'y':list_y[np.where(etiqs==1)], 'xv':list_xv[np.where(etiqs==1)], 'yv': list_yv[np.where(etiqs==1)],'pais':list_pais[np.where(etiqs==1)],'grados_p':grados_pertenencia[np.where(etiqs==1)],'cluster_id':etiqs[np.where(etiqs==1)]})
    p.square('x','y', size=6, 
             fill_color='white', source=source)
    source = ColumnDataSource(data={'x':list_x[np.where(etiqs==2)], 'y':list_y[np.where(etiqs==2)], 'xv':list_xv[np.where(etiqs==2)], 'yv': list_yv[np.where(etiqs==2)],'pais':list_pais[np.where(etiqs==2)],'grados_p':grados_pertenencia[np.where(etiqs==2)],'cluster_id':etiqs[np.where(etiqs==2)]})
    p.square('x','y', size=6, 
             fill_color='white', source=source)

    ### Ploteo centroides
    cambios_centroids = centroids_viej - centroids
    list_x = centroids[:,0]
    list_y = centroids[:,1]
    list_xv = cambios_centroids[:,0]
    list_yv = cambios_centroids[:,1]    
    source = ColumnDataSource(data={'x':list_x, 'y':list_y, 'xv':list_xv, 'yv': list_yv})
    p.square('x','y', size=15, 
             fill_color='black',source=source)
    
    ### Labels
    p.xaxis.axis_label = datos.columns[-2]
    p.yaxis.axis_label = datos.columns[-1]
    
    output_file('outputs/ClustersGenerados/cluster'+str(year_i+1+periodos)+'.html')
    save(p)
    
#    time.sleep(8)
               