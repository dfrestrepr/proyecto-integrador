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
from bokeh.plotting import figure
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, PanTool
#%%
file_path = os.path.dirname(os.path.abspath(__file__))
#%% 
### Configuracion del logger
import logging
from logging.handlers import RotatingFileHandler

file_name = 'Segmentacion_dinamica_'
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
datos = pd.read_csv('outputs/data_gapminder_proc.csv')

### Por ahora, dejamos solo nombre pais y year, y dos variables explicativas
datos = datos[datos.columns[[0,1,3,4,5,6]]]  


datos = funciones.data_preprocessing(datos, alpha_outlier_detection =0.98, 
                                     columns_not_numeric = {'country','Date'},
                                     column_id = 'country')

### Estandarizo todos los datos
datos_e = datos.copy()
datos_e[datos_e.columns[2:]] = StandardScaler().fit_transform(datos_e[datos_e.columns[2:]])
datos_e = datos_e.fillna(0)

#### Aplicarle PCA a todos los datos
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_e[datos_e.columns[2:]])


### Fijar seed aleatoria
np.random.seed(1)

##### Inicio tomando los del primer year
year_i = 2012   ### Year inicial a considerar
filtro = datos_e['Date']==year_i
X_data_df = datos_e[filtro].reset_index(drop=True)
X_data = np.array(X_data_df[X_data_df.columns[2:]])


### Los que usare para el PCA seran
X_data_pca = np.array(datos_pca[filtro])


#### Lista donde ire guardando las listas de grados de pertenencia 
grad_per = []

### Lista donde ire guardando las etiquetas asignadas del cluster
etiquetas_glo = []



## Numero de observaciones en cada periodo
numdata = len(X_data)
  

### Define cantidad de clusters, numero maximo de iteraciones, y la distancia
### que se utilizara en el metodo de kmeans
k = 4
numiter = 5
p_dista = 2


#### Inicializar los centroides
centroids = funciones.init_centroids(X_data,k)
centroids_pca = pca.transform(centroids)

### Para la fase 2 de las importancias
centroids_p = centroids.copy()

### Aplicar kmeans
grados_pertenencia,etiquetas,centroids = funciones.kmeans(X_data,numiter,
                                                          centroids,p_dista = p_dista)


### Guardo grados de pertenencia
grad_per.append(grados_pertenencia)

### Guardo etiquetas
etiquetas_glo.append(etiquetas)


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


### Consolido en listas las x, las y y las demas variables que vere para cada punto
list_x = X_data_pca[:,0]
list_y = X_data_pca[:,1]
list_pais = X_data_df['country'].values
grados_pertenencia = np.array(grados_pertenencia)

funciones.plot_clusters_bokeh(list_x, list_y, list_pais, k, etiquetas, 
                              grados_pertenencia,title = str(year_i),to_save = True)

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

    #### Obtener los 2 componentes principales de los datos para plotear estos
    X_data_pca = np.array(datos_pca[datos_e['Date']==year_i+1+periodos])
    
    ### Calulo los cambios en las variables
    cambios_variables = X_data_viej - X_data    
    

    ################## Ponderacion dinamica
    ######## Pondero X_data
    X_data_ori = X_data.copy()  ### X data original (sin ponderacion)
    
    
    ### Obtengo la importancia promedio de cada variable (promedio de todas las iteraciones)
    importancia_prom = np.mean(imp_iters, axis=0)
    
    #### Rankeo las variables de menor a mayor importancia
    rank_variables = np.argsort(importancia_prom)
    rankpeso_variables = np.zeros(len(rank_variables))
    cont = 0
    for i in rank_variables:
        rankpeso_variables[i] = (cont+1)/len(rank_variables)
        cont= cont+1

    #### Usar rankings o usar los promedios para el peso
    peso_variables = importancia_prom.copy()
#    peso_variables = rankpeso_variables.copy()

    
    ### Escalo entonces la X para cambiar los pesos (segun las importancias)
    X_data_pond = X_data.copy()
    for peso in range(len(peso_variables)):
        X_data_pond[:,peso] = X_data_pond[:,peso] * peso_variables[peso]


    ############ K means para los plots
    ### Etiquetas actuales de cada elemento para cada cluster
    etiquetas_prev = etiquetas.copy()
    
    ######### Clusters con k means ponderado
    grados_pertenencia,etiquetas,centroids = funciones.kmeans(X_data_pond,numiter,
                                                              centroids,
                                                              p_dista = p_dista,
                                                              etiquetas = etiquetas)


    ### Guardo grados de pertenencia
    grad_per.append(grados_pertenencia)
    
    ### Guardo etiquetas
    etiquetas_glo.append(etiquetas)


    ################# K means para la seleccion de variables
    ###### Para la proxima iteracion, los pesos
    grados_pertenencia_p,etiquetas_p,centroids_p = funciones.kmeans(X_data_ori,numiter,
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
    
    ##### Calculo los promedios de importancia de cada variable
    imp_clus_prom =  np.mean(importancias_cluster, axis=0)
    
    ### Guardo las importancias de esta iteracion
    imp_iters.append(imp_clus_prom)



    ###########################################################################
    ################################ Plotting #################################
    ########################################################################### 
    
    ### Consolido en listas las x, las y y las demas variables que vere para cada punto
    list_x = X_data_pca[:,0]
    list_y = X_data_pca[:,1]
    list_xv = cambios_variables[:,0]
    list_yv = cambios_variables[:,1]
    list_pais = X_data_df['country'].values
    grados_pertenencia = np.array(grados_pertenencia)


    title = str(year_i+1+periodos)           
    
    
    funciones.plot_cluster_bokeh_cambios(X_data_pca, list_x, list_y, list_xv, list_yv, k, 
                               cambios_variables, list_pais, grados_pertenencia, 
                               etiquetas, etiquetas_prev, centroids, 
                               centroids_viej, centroids_pca, title = title,
                               to_save = True)   
    

########################### NUEVA VISUALIZACION  ##############################


from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (ColumnDataSource, HoverTool, SingleIntervalTicker,
                          Slider, Button, Label, CategoricalColorMapper)
from bokeh.palettes import Spectral6
from bokeh.plotting import figure

    
### Lista years
years_plot = []
for o in range(periodos_incluir+1):
    years_plot.append(year_i + o)
    
######## Dataframes necesarias
pca1 = pd.DataFrame(columns = years_plot)
pca2 = pd.DataFrame(columns = years_plot)



##### PCA de cada year
for year in years_plot:
    filtro = datos_e['Date']==year
    
    ### Los que usare para el PCA seran
    X_data_pca_y = np.array(datos_pca[filtro])
    
    pca1[year] =  X_data_pca_y[:,0]
    pca2[year] =  X_data_pca_y[:,1]

### Nombres de los individuos
pca1.index = X_data_df.country
pca2.index = X_data_df.country



### Grados de pertenencia
grados_pert = pd.DataFrame(columns = years_plot)


##### Grados de pertenencia de cada year
coun = 0
for year in years_plot:    
    grados_pert[year] =  np.max(grad_per[coun], axis=1)
    coun = coun+1
grados_pert.index = X_data_df.country

 
##### Cluster al que pertenece cada dato en cada periodo de tiempo
etiqs_plot = []
couu = 0
for year in years_plot:
    eti = pd.DataFrame(columns = years_plot)
    eti['region'] = list(etiquetas_glo[couu])
    eti.index = X_data_df.country
    
    etiqs_plot.append(eti)
    couu = couu+1


##### Regions_list son los id de los cluster
regions_list = []
for cu in range(k):
    regions_list.append(str(cu))


### fertility df seria componente principal 1
### life expectancy df seria componente principal 2
### population_df_size es el maximo grado de pertenencia
### regions_df es el cluster id al que se asigno cada uno
### years es la lista de years a modelar
### regions list seria el "nombre " de cada cluster (top variables mas importantes)



df = pd.concat({'Componente_1': pca1,
                'Componente_2': pca2,
                'Grado_Pertenencia': grados_pert},
               axis=1)
    
    

### Construir data
data = {}

#regions_df.rename({'Group':'region'}, axis='columns', inplace=True)
counta = 0
for year in years_plot:
    df_year = df.iloc[:,df.columns.get_level_values(1)==year]
    df_year.columns = df_year.columns.droplevel(1)
    data[year] = df_year.join(etiqs_plot[counta].region).reset_index().to_dict('series')
    counta = counta+1




source = ColumnDataSource(data=data[years_plot[0]])

plot = figure(x_range=(1, 9), y_range=(20, 100), title='Gapminder Data', plot_height=450, plot_width = 900)
plot.xaxis.ticker = SingleIntervalTicker(interval=1)
plot.xaxis.axis_label = "Componente principal 1"
plot.yaxis.ticker = SingleIntervalTicker(interval=20)
plot.yaxis.axis_label = "Componente principal 2"

label = Label(x=1.1, y=18, text=str(years_plot[0]), text_font_size='70pt', text_color='#eeeeee')
plot.add_layout(label)

color_mapper = CategoricalColorMapper(palette=Spectral6, factors=regions_list)
plot.circle(
    x='Componente_1',
    y='Componente_2',
    size='Grado_Pertenencia',
    source=source,
    fill_color={'field': 'region', 'transform': color_mapper},
    fill_alpha=0.8,
    line_color='#7c7e71',
    line_width=0.5,
    line_alpha=0.5,
#    legend_group='region',
)
plot.add_tools(HoverTool(tooltips="@Country", show_arrow=False, point_policy='follow_mouse'))


def animate_update():
    year = slider.value + 1
    if year > years_plot[-1]:
        year = years_plot[0]
    slider.value = year


def slider_update(attrname, old, new):
    year = slider.value
    label.text = str(year)
    source.data = data[year]

slider = Slider(start=years_plot[0], end=years_plot[-1], value=years_plot[0], step=1, title="Year")
slider.on_change('value', slider_update)

callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [plot],
    [slider, button],
])

    

    
curdoc().add_root(layout)
curdoc().title = "Gapminder"

    
#    
#    
#from bokeh.plotting import output_file, save
#output_file("test.html")
#save(layout)





