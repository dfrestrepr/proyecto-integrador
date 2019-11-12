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
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.palettes import Spectral6
from bokeh.io import output_file, save, curdoc
from bokeh.models import (ColumnDataSource, HoverTool, BoxZoomTool, 
                          WheelZoomTool, ResetTool, PanTool,SingleIntervalTicker,
                          Slider, Button, Label, CategoricalColorMapper)
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
        logger.info('Error en la deteccion de outliers:{0}'.format(e))
    
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
        logger.info('Error en el stepwise logistic:{0}'.format(e))
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
        logger.info('Se calcula el valor minimo de aceptacion de importancia: {0}'.format(alpha))
    
    try:    
        model = ExtraTreeClassifier()
        model.fit(X,Y)
        
        importance = model.feature_importances_
        
        relevant_features = []
        for i in range(len(features)):
            if importance[i]>alpha:
                relevant_features.append(features[i])
                
    except Exception as e:
        logger.info('Error con el metodo de arboles, no se determinaron variables relevantes: {0}'.format(e))
        relevant_features = []
        
    return importance, relevant_features

### Esta funcion toma como entrada un dataframe, el nombre de las variables no
### numericas, el nombre de la columna identificadora de los registros (ID del
### cliente, pais, etc) y realiza una deteccion de outliers utilizando la 
### distancia de Mahalanobis, adicional a esto, realiza la eliminacion de
### variables redundantes en el conjunto de datos mirando la relacion lineal
### de una variable contra todas las demas
def data_preprocessing(datos, alpha_outlier_detection =0.95, columns_not_numeric = {}, 
                       column_id = '', remove_vars = True, r_sqr_threshold = 0.9):
    
    ### Realiza la identificaciohn de outliers usando la distancia de 
    ### Mahalanobis
    logger.info('Realiza deteccion de outlier en el preprocesamiento de los datos')
    outliers = outlier_detection_mahal(datos.drop(columns = columns_not_numeric),alpha = alpha_outlier_detection)
    
    ### En caso que no exista una columna identificadora para eliminar IDs
    ### outliers, se eliminan todos los registros que se encontraron como
    ### outlier    
    if column_id == '':
        logger.info('Realiza la eliminacion de outliers por registros encontrados')
        datos = datos[outliers==0].reset_index(drop = True)    
        
    else:
        logger.info('Realiza la eliminacion de outliers por IDs detectados')              
        ids_outliers = datos[outliers==1][column_id].unique()
        datos = datos[~datos[column_id].isin(ids_outliers)].reset_index(drop = True) 
    
    ### Esta seccion realiza eliminacion iterativa de variables de forma tal
    ### que el R^2 de todas las variables sea menor a un umbral dado
    if remove_vars:
        logger.info('Se realiza el analisis del R^2 de todas las variables para eliminar variables redundantes')
        to_drop = []
        X_aux = datos.drop(columns = columns_not_numeric).copy()
        while True:
            var_names = list(X_aux.columns)
            
            inver = np.linalg.inv(X_aux.cov())
    
            ### Este es el R2 de cada variable con respecto a todas las demas, 
            ### por eso seria un R2 por cada variable
            r_squared = 1 - 1/(np.diagonal(X_aux.cov())*np.diagonal(inver))
    
            if any(r_squared > r_sqr_threshold):
                
                var_m = np.argmax(r_squared)
                X_aux = X_aux.drop(columns = var_names[var_m])
                
                to_drop.append(var_names[var_m])
                logger.info('Se elimina la variable:var_names[var_m] con el umbral: {0}'.format(r_sqr_threshold))
                
            else:
                break
    
    datos = datos.drop(columns = to_drop)
    logger.info('Fin del preprocesamiento de los datos eliminando Outliers y variables redundantes')
    return datos


### Esta funcion calcula la matriz de covarianza de Ledoit and Wolf, retorna
### la matriz de covarianza despues de aplicar el metodo de Shrinkage, ademas
### de retornar la media estimada. Esta funcion toma como parametros de entrada
### el conjunto de datos
def LedoitWolf_covMatrix(X):
    logger.info('Se realiza el calculo de la matriz de covarianza con Shrinkage')
    cov = LedoitWolf().fit(X)
    cov_matrix = cov.covariance_
    mean_vector = cov.location_
    
    return cov_matrix, mean_vector


### Funcion de distancia p, tenga en cuenta que si p es cero, entonces se
### asume que la distancia deseada es la de Mahalanobis
def distancia(a,b, p, cov = np.array([])):
    
    if p == 0:
        inv_cov = np.linalg.inv(cov)
        dist = distance.mahalanobis(a, b, inv_cov)
    else:
        dist = np.linalg.norm(a-b, p) 
    

    return dist


### Inicializa los centroides para el metodo de kmeans, inicializa de forma
### aleatoria tantos centros como clusters
def init_centroids(X_data,k):
    logger.info('Inicializacion de {0} centroides para el metodo de kmeans'.format(k))
    centroids = []
    numdata = len(X_data)
    for i in range(k):
        centroids.append(X_data[np.random.randint(0, numdata)])

    return centroids

### Metodo de kmeans
def kmeans(X_data,numiter,centroids,p_dista = 2,etiquetas = [], shrinkage = True):
    logger.info('Inicializa el metodo de kmeans')
    numdata = len(X_data)    
    if len(etiquetas)==0:
        logger.info('Se crean etiquetas ya que no fueron pasadas incialmente')
         ### Etiquetas actuales de cada elemento para cada cluster
        etiquetas = np.ones(numdata)*-1   ### Inicialmente, ningun elemento esta asignado
    
    ### Grados de pertenencia a cada cluster
    grados_pertenencia = []
    
    ### Se realiza el calculo de la matriz de varianzas y covarianzas en caso
    ### de utilizar la distancia de Mahalanobis, ademas si se desea, se utiliza
    ### una version bien condicionada de la matriz utilizando el metodo de
    ### Shrinkage
    if shrinkage:      
        covariance_matrix, _ = LedoitWolf_covMatrix(X_data)
    else:      
        covariance_matrix = np.cov(X_data,rowvar=False)
        
    ### Ahora empiezo las iteraciones
    for it in range(numiter):
        logger.info('Iteracion {0} de {1} para el metodo de kmeans'.format(it, numiter))
        ### En cada iteracion, itero para todos los elementos
        for element in range(numdata):
            
            np.seterr(all='raise')
            ### Evaluo las distancias a cada centroides
            ### le sumo 0.00001 a cada distancia para evitar division sobre cero
            distc = []
            for c in centroids:
                distc.append(distancia(X_data[element], c, p_dista,covariance_matrix)+0.00001)
            

            ### Encuentro el centroide al que tiene menor distancia
            nearest_centroid = np.argmin(distc)
            
            ### Asigno el elemento a este cluster
            etiquetas[element] = nearest_centroid
            
            
            ### Recalculo el centroide 
            centroids[nearest_centroid] = np.mean(X_data[np.where(etiquetas==nearest_centroid)], axis=0)   
        
        centroids = np.array(centroids)


    ### Guardar grados de pertenencia a cada cluster
    for element in range(numdata):
        ### Evaluo las distancias a cada centroides
        ### le sumo 0.00001 a cada distancia para evitar division sobre cero
        distc = []
        for c in centroids:
            distc.append(distancia(X_data[element], c, p_dista,covariance_matrix)+0.00001)        
        grados_pert = list(np.around(1/(distc/sum(distc))/sum(1/(distc/sum(distc))),4))
        grados_pertenencia.append(grados_pert)

    logger.info('Fin del algoritmo kmeans')
    return grados_pertenencia,etiquetas,centroids


def plot_clusters_bokeh(list_x, list_y, list_pais, k, etiquetas, grados_pertenencia, 
                        title = 'Title', to_save = True):

    ### Plotear con hover tool
    
    ### Colores a usar para cada cluster
    colores = ['blue', 'yellow', 'red', 'green', 'orange', 'purple']

    
    ### Creo la herramienta de hover tool
    hover = HoverTool(tooltips=[
        ("pais","@pais"),
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ("cluster_id", "@cluster_id"),
        ("Pertenencia_clusters", "@grados_p"),
        ])

    ### Creo la figura
    p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), 
                                                       ResetTool(), BoxZoomTool(), 
                                                       WheelZoomTool()], title=title)
    
    ### PLoteo cada cluster
    for i in range(k):
        source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==i)], 'y':list_y[np.where(etiquetas==i)], 'pais':list_pais[np.where(etiquetas==i)],'grados_p':grados_pertenencia[np.where(etiquetas==i)],'cluster_id':etiquetas[np.where(etiquetas==i)]})
        p.circle('x','y', size=12, 
                 fill_color=colores[i], source=source)

    ### Ploteo centroides
    #p.square(centroids_pca[:,0], centroids_pca[:,1], size=15,    fill_color='black')
    
    ### Labels (componentes principales)
    p.xaxis.axis_label = 'Componente principal 1'
    p.yaxis.axis_label = 'Componente principal 2'
    #p.xaxis.axis_label = datos.columns[-2]
    #p.yaxis.axis_label = datos.columns[-1]
    
    if to_save:
    ### Guardo el resultado
        output_file('outputs/ClustersGenerados/cluster_inicial_'+title+'.html')
        save(p)
    
    return None

def plot_cluster_bokeh_cambios(X_data_pca, list_x, list_y, list_xv, list_yv, k, 
                               cambios_variables, list_pais, grados_pertenencia, 
                               etiquetas, etiquetas_prev, centroids, 
                               centroids_viej, centroids_pca, title = 'Title',
                               to_save = True):
    
    ### Hover tool para los datos
    hover = HoverTool(tooltips=[
                                ("pais","@pais"),
                                ("index", "$index"),
                                ("(x,y)", "(@x, @y)"),
                                ("(Cambio_x,Cambio_y)", "(@xv, @yv)"),    
                                ("cluster_id", "@cluster_id"),
                                ("Pertenencia_clusters", "@grados_p"),
                                ])
    
    ### Colores a usar para cada cluster
    colores = ['blue', 'yellow', 'red', 'green', 'orange', 'purple']
    
    ### Crear la figura
    p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), 
                                                       ResetTool(), 
                                                       BoxZoomTool(), 
                                                       WheelZoomTool()],
                                                       title=title)
    
    ### PLoteo cada conjunto
    for i in range(k):
        source = ColumnDataSource(data={'x':list_x[np.where(etiquetas==i)], 
                                        'y':list_y[np.where(etiquetas==i)], 
                                        'xv':list_xv[np.where(etiquetas==i)], 
                                        'yv': list_yv[np.where(etiquetas==i)],
                                        'pais':list_pais[np.where(etiquetas==i)],
                                        'grados_p':grados_pertenencia[np.where(etiquetas==i)],
                                        'cluster_id':etiquetas[np.where(etiquetas==i)]})
        p.circle('x','y', size=12, 
                 fill_color=colores[i], source=source)
    
    
    ### Veo cuales cambiaron de cluster
    etiquetas_cambios = np.where(etiquetas_prev-etiquetas != 0)
    etiqs = etiquetas[etiquetas_cambios]
    
    ### PLoteo los elementos de cada conjunto que cambiaron de cluster
    
    ### Listas para los elementos que cambiaron de cluster
    X_data_cambios = X_data_pca[etiquetas_cambios]
    list_x = X_data_cambios[:,0]
    list_y = X_data_cambios[:,1]
    list_xv = cambios_variables[:,0]
    list_yv = cambios_variables[:,1]
    list_pais = list_pais[etiquetas_cambios]
    grados_pertenencia = grados_pertenencia[etiquetas_cambios]
    
    ## Plotear elementos que cambiaron de cluster
    for i in range(k):
        source = ColumnDataSource(data={'x':list_x[np.where(etiqs==i)], 
                                        'y':list_y[np.where(etiqs==i)], 
                                        'xv':list_xv[np.where(etiqs==i)], 
                                        'yv': list_yv[np.where(etiqs==i)],
                                        'pais':list_pais[np.where(etiqs==i)],
                                        'grados_p':grados_pertenencia[np.where(etiqs==i)],
                                        'cluster_id':etiqs[np.where(etiqs==i)]})
        p.square('x','y', size=6, 
                 fill_color='white', source=source)
    
    
    
    ### Ploteo centroides
    
    ### Listas para centroiodes
    cambios_centroids = centroids_viej - centroids
    list_x = centroids_pca[:,0]
    list_y = centroids_pca[:,1]
    list_xv = cambios_centroids[:,0]
    list_yv = cambios_centroids[:,1]
    
    # Plotar centroids    
    source = ColumnDataSource(data={'x':list_x, 'y':list_y, 'xv':list_xv, 'yv': list_yv})
    #    p.square('x','y', size=15,             fill_color='black',source=source)
    
    ### Labels de la grafica (componentes principales)
    p.xaxis.axis_label = 'Componente principal 1'
    p.yaxis.axis_label = 'Componente principal 2'
    #p.xaxis.axis_label = datos.columns[-2]
    #p.yaxis.axis_label = datos.columns[-1]
    
    
    if to_save:
        ### Guardar resultados
        output_file('outputs/ClustersGenerados/cluster'+title+'.html')
        save(p)
    
    return None










### Para ploteo de datos con estilo de gapminder
def gapminder_plot_bokeh(datos_e, datos_pca, year_i, X_data_df, grad_per,
                         etiquetas_glo, periodos_incluir, k, imp_periods_var,
                         centroids_ite, scaler_es,
                         title = 'Titulo',
                         xlabel='Eje x',
                         ylabel='Eje y'):
    
    

    
    
    
    ### Lista years
    years_plot = []
    for o in range(periodos_incluir+1):
        years_plot.append(year_i + o)
        
    ### Dataframes necesarias
    pca1 = pd.DataFrame(columns = years_plot)
    pca2 = pd.DataFrame(columns = years_plot)
    
    ### PCA de cada year
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
        grados_pert[year] =  np.max(grad_per[coun], axis=1)*40  ### Aumento escala para que se vean bien
        coun = coun+1
    grados_pert.index = X_data_df.country
    
     
    ##### Cluster al que pertenece cada dato en cada periodo de tiempo
    etiqs_plot = []
    couu = 0
    for year in years_plot:
        eti = pd.DataFrame()
        eti['region'] = [str(i)[0] for i in list(etiquetas_glo[couu])] ### Solo 1 caracter
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
    
    
    
    
    ### Consolidar data
    df = pd.concat({'Componente_1': pca1,
                    'Componente_2': pca2,
                    'Grado_Pertenencia': grados_pert},
                   axis=1)
        
        
    
    ### Construir data
    data = {}
    
    counta = 0
    for year in years_plot:
        df_year = df.iloc[:,df.columns.get_level_values(1)==year]
        df_year.columns = df_year.columns.droplevel(1)
        data[year] = df_year.join(etiqs_plot[counta].region).reset_index().to_dict('series')
        counta = counta+1
    
    
    source = ColumnDataSource(data=data[years_plot[0]])




    ############### Para las labels ########################
    
    #### Numero de variables a plotear
    num_v_plot = 4
    
    #### Nombres variables
    nomb_v = datos_e.columns[2:]
    
    #### Desestandarizar centroides
    centroids_ito = scaler_es.inverse_transform(centroids_ite)
    
    #### Consolidar strings de las legends de cada iteracion
    strings_legends = []
    c=0
    for y in years_plot:
        esta_iter = []
        estas_imp = imp_periods_var[c]
        cc = 0
        for clu in estas_imp:
            ### Variables mas importantes
            orden_v = np.argsort(clu)[::-1][:num_v_plot]
            
            ### Construir string
            stri = ''
            
            for i in orden_v:
                stri = stri+nomb_v[i][:12]+': ' + str(np.around(centroids_ito[c][cc][i],2))+', '
            stri=stri[:-2]
            esta_iter.append(stri)
            cc=cc+1
        strings_legends.append(esta_iter)
        c=c+1



    #### PLoteos    
    global plot
    plot = figure(title=title, y_range=(-3, 5), plot_height=520, plot_width = 900)
    plot.xaxis.ticker = SingleIntervalTicker(interval=1)
    plot.xaxis.axis_label = xlabel
    plot.yaxis.ticker = SingleIntervalTicker(interval=20)
    plot.yaxis.axis_label = ylabel
    
    label = Label(x=1.1, y=18, text=str(years_plot[0]), text_font_size='70pt', text_color='#eeeeee')
    plot.add_layout(label)
    
    color_mapper = CategoricalColorMapper(palette=Spectral6, factors=regions_list)
    global r
    
    r = plot.circle(
        x='Componente_1',
        y='Componente_2',
        size='Grado_Pertenencia',
        source=source,
        fill_color={'field': 'region', 'transform': color_mapper},
        fill_alpha=0.8,
        line_color='#7c7e71',
        line_width=0.5,
        line_alpha=0.5,
#        legend_group='region',
    )
    
    from bokeh.models import Legend, LegendItem
    
    global legend   
    
    items_son=[]
    co = 0
    for a in strings_legends[0]:
        color_ =  list(etiquetas_glo[0]).index(co)
        items_son.append(LegendItem(label=a, renderers=[r], index=color_))
        co=co+1
        
    legend = Legend(items=items_son)
    plot.add_layout(legend)    
    

    plot.add_tools(HoverTool(tooltips="@country", show_arrow=False, point_policy='follow_mouse'))    


    def animate_update():
        year = slider.value + 1
        if year > years_plot[-1]:
            year = years_plot[0]
        slider.value = year
    
    
    def slider_update(attrname, old, new):
        year = slider.value
        label.text = str(year)
        source.data = data[year]
        pos = years_plot.index(year)
        global legend
        global r
        global plot

    
        items_son=[]
        bo = 0
        for a in strings_legends[pos]:
            color_ =  list(etiquetas_glo[pos]).index(bo)
            items_son.append(LegendItem(label=a, renderers=[r], index=color_))
            bo=bo+1
        legend.items = items_son
        plot.add_layout(legend)   
        

        
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
    
    layout_plot = layout([
        [plot],
        [slider, button],
    ])
    
    
    curdoc().add_root(layout_plot)
    curdoc().title = "Gapminder"

    return None