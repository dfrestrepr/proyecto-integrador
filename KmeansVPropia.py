# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:38:34 2019

@author: nicol
"""

#### Version propia de kmeans para proyecto integrador
import numpy as np
import pandas as pd


### Leemos los datos 
datos = pd.read_csv('outputs/data_gapminder_proc.csv')

### Por ahora, dejamos solo nombre pais y year, y dos variables explicativas
datos = datos[datos.columns[[0,1,4,6]]]  

### Estandarizo todos los datos
from sklearn.preprocessing import StandardScaler
datos_e = datos.copy()
datos_e[datos_e.columns[2:]] = StandardScaler().fit_transform(datos_e[datos_e.columns[2:]])
datos_e = datos_e.fillna(0)

### Fijar seed aleatoria
np.random.seed(1)








##### Inicio tomando los del primer year
year_i = 2012   ### Year inicial a considerar
X_data_df = datos_e[datos_e['Date']==year_i].reset_index(drop=True)
X_data = np.array(X_data_df[X_data_df.columns[2:]])



## Numero de observaciones en cada periodo
numdata = len(X_data)



### Funcion de distancia p
def distancia(a,b, p):
    ## if p = 0, que use la de mahalanobis (aun no esta programada)
    
    ### por ahora, la cuadratica
    return np.linalg.norm(a-b, p)
    

### La p para la distancia que usare
p_dista = 2


############# El numero de clusters sera
k=3


### Inicialmente, asignar 3 puntos de forma aleatoria como centroides
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
            distc.append(distancia(X_data[el], c, p_dista))
        
        ### Encuentro el centroide al que tiene menor distancia
        nearest_centroid = np.argmin(distc)
        
        ### Asigno el elemento a este cluster
        etiquetas[el] = nearest_centroid
        
        ### Grados de pertenencia a cada cluster
        grados_pertenencia.append(str(list(np.around(1/(distc/sum(distc))/sum(1/(distc/sum(distc))),3))))
        
        ### Recalculo el centroide 
        centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas==nearest_centroid)], axis=0)
            
            
        



#PLotear con hover tool
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, PanTool
from bokeh.plotting import figure


### Colores a usar para cada cluster
colores = ['blue', 'yellow', 'red', 'green', 'orange', 'purple']


### Consolido en listas las x, las y y las demas variables que vere para cada punto
list_x = X_data[:,0]
list_y = X_data[:,1]
list_pais = X_data_df['country'].values
grados_pertenencia = np.array(grados_pertenencia)

### Creo la herramienta de hover tool
hover = HoverTool(tooltips=[
    ("pais","@pais"),
    ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
    ("cluster_id", "@cluster_id"),
    ("Pertenencia_clusters", "@grados_p"),

])

### Creo la figura
p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), ResetTool(), BoxZoomTool(), WheelZoomTool()], title=str(year_i))

### PLoteo cada cluster
for i in range(k):
    source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==i)], 'y':list_y[np.where(etiquetas==i)], 'pais':list_pais[np.where(etiquetas==i)],'grados_p':grados_pertenencia[np.where(etiquetas==i)],'cluster_id':etiquetas[np.where(etiquetas==i)]})
    p.circle('x','y', size=12, 
             fill_color=colores[i], source=source)


### Ploteo centroides
p.square(centroids[:,0], centroids[:,1], size=15, 
         fill_color='black')

### Labels
p.xaxis.axis_label = datos.columns[-2]
p.yaxis.axis_label = datos.columns[-1]

### Guardo el resultado
output_file('outputs/ClustersGenerados/cluster_inicial_'+str(year_i)+'.html')
save(p)





#################### Ahora, empiezo a iterar para t >=2

### Numero de periodos que incluire en el estudio, sin incluir el inicial
periodos_incluir = 4

for periodos in range(periodos_incluir):
    
    ### Guardo la X_data anterior
    X_data_viej = X_data.copy()
    centroids_viej = centroids.copy()
    

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
                distc.append(distancia(X_data[el], c, p_dista))
            
            ### Encuentro el centroide al que tiene menor distancia
            nearest_centroid = np.argmin(distc)
            
            ### Asigno el elemento a este cluster
            etiquetas[el] = nearest_centroid
            
            ### Grados de pertenencia a cada cluster
            grados_pertenencia.append(str(list(np.around(1/(distc/sum(distc))/sum(1/(distc/sum(distc))),3))))
            
            ### Recalculo el centroide 
            centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas==nearest_centroid)], axis=0)
                
                

    ########### Plotting
    
    ### Consolido en listas las x, las y y las demas variables que vere para cada punto
    list_x = X_data[:,0]
    list_y = X_data[:,1]
    list_xv = cambios_variables[:,0]
    list_yv = cambios_variables[:,1]
    list_pais = X_data_df['country'].values
    grados_pertenencia = np.array(grados_pertenencia)

    ### Hover tool para los datos
    hover = HoverTool(tooltips=[
                ("pais","@pais"),
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ("(Cambio_x,Cambio_y)", "(@xv, @yv)"),    
    ("cluster_id", "@cluster_id"),
    ("Pertenencia_clusters", "@grados_p"),
    ])

    
    ### Crear la figura
    p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), ResetTool(), BoxZoomTool(), WheelZoomTool()], title=str(year_i+1+periodos))
    
    ### PLoteo cada conjunto
    for i in range(k):
        source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==i)], 'y':list_y[np.where(etiquetas==i)], 'xv':list_xv[np.where(etiquetas==i)], 'yv': list_yv[np.where(etiquetas==i)],'pais':list_pais[np.where(etiquetas==i)],'grados_p':grados_pertenencia[np.where(etiquetas==i)],'cluster_id':etiquetas[np.where(etiquetas==i)]})
        p.circle('x','y', size=12, 
                 fill_color=colores[i], source=source)

    
    ### Veo cuales cambiaron de cluster
    etiquetas_cambios = np.where(etiquetas_prev-etiquetas != 0)
    etiqs = etiquetas[etiquetas_cambios]

    ### PLoteo los elementos de cada conjunto que cambiaron de cluster
    
    ### Listaas para los elementos que cambiaron de cluster
    X_data_cambios = X_data[etiquetas_cambios]
    list_x = X_data_cambios[:,0]
    list_y = X_data_cambios[:,1]
    list_xv = cambios_variables[:,0]
    list_yv = cambios_variables[:,1]
    list_pais = list_pais[etiquetas_cambios]
    grados_pertenencia = grados_pertenencia[etiquetas_cambios]

    ## Plotear elementos que cambiaron de cluster
    for i in range(k):
        source = ColumnDataSource(data={'x':list_x[np.where(etiqs==i)], 'y':list_y[np.where(etiqs==i)], 'xv':list_xv[np.where(etiqs==i)], 'yv': list_yv[np.where(etiqs==i)],'pais':list_pais[np.where(etiqs==i)],'grados_p':grados_pertenencia[np.where(etiqs==i)],'cluster_id':etiqs[np.where(etiqs==i)]})
        p.square('x','y', size=6, 
                 fill_color='white', source=source)



    ### Ploteo centroides
    
    ### Listas para centroiodes
    cambios_centroids = centroids_viej - centroids
    list_x = centroids[:,0]
    list_y = centroids[:,1]
    list_xv = cambios_centroids[:,0]
    list_yv = cambios_centroids[:,1]

    # Plotar centroids    
    source = ColumnDataSource(data={'x':list_x, 'y':list_y, 'xv':list_xv, 'yv': list_yv})
    p.square('x','y', size=15, 
             fill_color='black',source=source)
    
    ### Labels de la grafica
    p.xaxis.axis_label = datos.columns[-2]
    p.yaxis.axis_label = datos.columns[-1]
    
    
    ### Guardar resultados
    output_file('outputs/ClustersGenerados/cluster'+str(year_i+1+periodos)+'.html')
    save(p)

               