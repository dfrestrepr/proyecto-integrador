import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sb
import funciones as fn

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

PATH = '/home/david/Descargas/SEG_DYN.csv'

df = pd.read_csv(PATH, sep="|", parse_dates=[2,27])
df = df.replace('\\N', np.nan )
#project.profiler_documentation(df, 'SEG_DYN')

data=df[['CUST_CODE','TKT_VALUE','TKT_QTY','ORDER_DATE']].copy()
data.columns = ['cliente', 'precio', 'cantidad', 'fecha']
data.info()
data.describe()
data.hist()
#plt.show()

total = data['cantidad'] * data['precio']
data['precio_total'] = total
data['ano'] = data.fecha.dt.year


# conocer máxima fecha
data['fecha'].max()

NUMERO_RECENTLY =1
PRESENT = data['fecha'].max()+ timedelta(days=NUMERO_RECENTLY)

# calculo del variables del RFM
rfm= data.groupby(['cliente', 'ano']).agg({'fecha': lambda date: (PRESENT - date.max()).days,
                                        'cantidad': lambda num: len(num),
                                        'precio_total': lambda price: price.sum()})

rfm = rfm.reset_index(level='ano')
rfm = rfm.reset_index(level='cliente')

rfm = rfm.rename(columns={'precio_total': 'monetary',
                            'fecha': 'recency',
                            'cantidad': 'frequency'})

rfm.describe()
rfm.hist()

df_list= list()
for item in rfm.ano.unique():
    filtro = rfm[rfm['ano'] == item].copy()
    filtro.recency = filtro.recency.apply(lambda x: (filtro.recency.max()-x)/(filtro.recency.max()-filtro.recency.min()))
    filtro.monetary = filtro.monetary.apply(lambda x: (x-filtro.monetary.min())/(filtro.monetary.max()-filtro.monetary.min()))
    filtro.frequency = filtro.frequency.apply(lambda x: (x-filtro.frequency.min())/(filtro.frequency.max()-filtro.frequency.min()))
    df_list.append(filtro)

rfm = pd.concat(df_list)

#plt.show()

outlier= fn.outlier_detection_mahal(rfm.drop(['ano', 'cliente'], axis=1), 0.8)
index = np.where(outlier==1)[0]
rfm = rfm.drop(rfm.index[index])
############ CALCULO DEL RFM: esto se puede mejor para no usar quintiles si no otra medida de distancia
rfm['r_quartile'] = pd.qcut(rfm['recency'], 5, [1,2,3,4,5]).astype(int)
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 5, [1,2,3,4,5]).astype(int)
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 5, [1,2,3,4,5]).astype(int)
rfm['rfm'] = rfm.r_quartile + rfm.f_quartile + rfm.m_quartile
rfm['rfm'] = rfm['rfm'].astype(int)
rfm.rfm = rfm.rfm.apply(lambda x: (x-rfm.rfm.min())/(rfm.rfm.max()-rfm.rfm.min()))


rfm = rfm[['cliente','ano','recency','frequency','monetary']]
rfm.hist()
#plt.show()
#sb.pairplot(rfm.dropna(), hue="rfm",size=4,vars=['recency','frequency','monetary'],kind='scatter')
#plt.show()

datos2017 = rfm[rfm.ano==2017]
datos2018 = rfm[rfm.ano==2018]
datos2019 = rfm[rfm.ano==2019]
datosm = pd.merge(datos2017, datos2018, how='inner', on='cliente')
datosm = pd.merge(datosm, datos2019, how='inner', on='cliente')
rfm = rfm[rfm['cliente'].isin(datosm['cliente'])]

datos_e = rfm
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, PanTool
#%%
#### Aplicarle PCA a todos los datos
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_e[datos_e.columns[2:]])

### Fijar seed aleatoria
np.random.seed(1)

##### Inicio tomando los del primer year
year_i = 2017  ### Year inicial a considerar
filtro = datos_e['ano'] == year_i
X_data_df = datos_e[filtro].reset_index(drop=True)
X_data = np.array(X_data_df[X_data_df.columns[2:]])

### Los que usare para el PCA seran
X_data_pca = np.array(datos_pca[filtro])

## Numero de observaciones en cada periodo
numdata = len(X_data)

### Define cantidad de clusters, numero maximo de iteraciones, y la distancia
### que se utilizara en el metodo de kmeans
k = 4
numiter = 10
p_dista = 2

#### Inicializar los centroides
centroids = fn.init_centroids(X_data, k)
centroids_pca = pca.transform(centroids)

### Para la fase 2 de las importancias
centroids_p = centroids.copy()

### Aplicar kmeans
grados_pertenencia, etiquetas, centroids = fn.kmeans(X_data, k, numiter,
                                                            centroids, p_dista=p_dista)

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
    distintos_cluster = np.where(datay_i != clu)
    ### Lo que no pertenece al cluster, lo pongo en -1
    datay_i[distintos_cluster] = -1
    datay_i = pd.DataFrame(datay_i)

    ### Calcular relevancias
    relevancias = fn.variables_relevantes_arbol(datax_i, datay_i, 0)

    importancias_cluster.append(relevancias)

##### Calculo los promedios de importancia de cada variable
imp_clus_prom = np.mean(importancias_cluster, axis=0)

### Guardo las importancias de esta iteracion
imp_iters.append(imp_clus_prom)

# PLotear con hover tool

### Colores a usar para cada cluster
colores = ['blue', 'yellow', 'red', 'green', 'orange', 'purple']

### Consolido en listas las x, las y y las demas variables que vere para cada punto
list_x = X_data_pca[:, 0]
list_y = X_data_pca[:, 1]
list_pais = X_data_df.index.values ################################### corregir deben ser los id de los clientes
grados_pertenencia = np.array(grados_pertenencia)

### Creo la herramienta de hover tool
hover = HoverTool(tooltips=[
    ("cliente", "@cliente"),
    ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
    ("cluster_id", "@cluster_id"),
    ("Pertenencia_clusters", "@grados_p"),

])

### Creo la figura
p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(), ResetTool(), BoxZoomTool(), WheelZoomTool()],
           title=str(year_i))

### PLoteo cada cluster
for i in range(k):
    source = ColumnDataSource(data={'x': list_x[np.where(etiquetas == i)], 'y': list_y[np.where(etiquetas == i)],
                                    'pais': list_pais[np.where(etiquetas == i)],
                                    'grados_p': grados_pertenencia[np.where(etiquetas == i)],
                                    'cluster_id': etiquetas[np.where(etiquetas == i)]})
    p.circle('x', 'y', size=12,
             fill_color=colores[i], source=source)

### Ploteo centroides
# p.square(centroids_pca[:,0], centroids_pca[:,1], size=15,    fill_color='black')

### Labels (componentes principales)
p.xaxis.axis_label = 'Componente principal 1'
p.yaxis.axis_label = 'Componente principal 2'
# p.xaxis.axis_label = datos.columns[-2]
# p.yaxis.axis_label = datos.columns[-1]


### Guardo el resultado
output_file('/home/david/proyecto-integrador/outputs/rfm_' + str(year_i) + '.html')
save(p)

#################### Ahora, empiezo a iterar para t >=2

### Numero de periodos que incluire en el estudio, sin incluir el inicial
periodos_incluir = 2

for periodos in range(periodos_incluir):

    ### Guardo la X_data anterior
    X_data_viej = X_data.copy()
    centroids_viej = centroids.copy()

    ### Los datos para este year ya serian
    X_data_df = datos_e[datos_e['ano'] == year_i + 1 + periodos].reset_index(drop=True)
    X_data = np.array(X_data_df[X_data_df.columns[2:]])

    #### Obtener los 2 componentes principales de los datos para plotear estos
    X_data_pca = np.array(datos_pca[datos_e['ano'] == year_i + 1 + periodos])

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
        rankpeso_variables[i] = (cont + 1) / len(rank_variables)
        cont = cont + 1

    #### Usar rankings o usar los promedios para el peso
    #peso_variables = importancia_prom.copy()
    peso_variables = rankpeso_variables.copy()

    ### Escalo entonces la X para cambiar los pesos (segun las importancias)
    X_data_pond = X_data.copy()
    for peso in range(len(peso_variables)):
        X_data_pond[:, peso] = X_data_pond[:, peso] * peso_variables[peso]

    ############ K means para los plots
    ### Etiquetas actuales de cada elemento para cada cluster
    etiquetas_prev = etiquetas.copy()

    ######### Clusters con k means ponderado
    grados_pertenencia, etiquetas, centroids = fn.kmeans(X_data_pond, k, numiter,
                                                                centroids,
                                                                p_dista=p_dista,
                                                                etiquetas=etiquetas)

    ################# K means para la seleccion de variables
    ###### Para la proxima iteracion, los pesos
    grados_pertenencia_p, etiquetas_p, centroids_p = fn.kmeans(X_data_ori, k, numiter,
                                                                      centroids_p,
                                                                      p_dista=p_dista,
                                                                      etiquetas=etiquetas)
    ### Obtener la importancia de las variables de cada cluster (de mayor a menor)
    importancias_cluster = []
    ### Para cada cluster
    for clu in range(k):
        ### Dejo solo las observaciones de cada cluster
        datax_i = pd.DataFrame(X_data_ori)
        datay_i = etiquetas_p.copy()
        #### Solo clasifico binario si si pertence o no a cada cluster
        distintos_cluster = np.where(datay_i != clu)
        ### Lo que no pertenece al cluster, lo pongo en -1
        datay_i[distintos_cluster] = -1
        datay_i = pd.DataFrame(datay_i)

        ### Calcular relevancias
        relevancias = fn.variables_relevantes_arbol(datax_i, datay_i, 0)

        importancias_cluster.append(relevancias)

    ##### Calculo los promedios de importancia de cada variable
    imp_clus_prom = np.mean(importancias_cluster, axis=0)

    ### Guardo las importancias de esta iteracion
    imp_iters.append(imp_clus_prom)

    ###########################################################################
    ################################ Plotting #################################
    ###########################################################################

    ### Consolido en listas las x, las y y las demas variables que vere para cada punto
    list_x = X_data_pca[:, 0]
    list_y = X_data_pca[:, 1]
    list_xv = cambios_variables[:, 0]
    list_yv = cambios_variables[:, 1]
    list_pais = X_data_df['cliente'].values
    grados_pertenencia = np.array(grados_pertenencia)

    ### Hover tool para los datos
    hover = HoverTool(tooltips=[
        ("cliente", "@cliente"),
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ("(Cambio_x,Cambio_y)", "(@xv, @yv)"),
        ("cluster_id", "@cluster_id"),
        ("Pertenencia_clusters", "@grados_p"),
    ])

    ### Crear la figura
    p = figure(plot_width=700, plot_height=500, tools=[hover, PanTool(),
                                                       ResetTool(),
                                                       BoxZoomTool(),
                                                       WheelZoomTool()],
               title=str(year_i + 1 + periodos))

    ### PLoteo cada conjunto
    for i in range(k):
        source = ColumnDataSource(data={'x': list_x[np.where(etiquetas == i)],
                                        'y': list_y[np.where(etiquetas == i)],
                                        'xv': list_xv[np.where(etiquetas == i)],
                                        'yv': list_yv[np.where(etiquetas == i)],
                                        'cliente': list_pais[np.where(etiquetas == i)],
                                        'grados_p': grados_pertenencia[np.where(etiquetas == i)],
                                        'cluster_id': etiquetas[np.where(etiquetas == i)]})
        p.circle('x', 'y', size=12,
                 fill_color=colores[i], source=source)

    ### Veo cuales cambiaron de cluster
    etiquetas_cambios = np.where(etiquetas_prev - etiquetas != 0)
    etiqs = etiquetas[etiquetas_cambios]

    ### PLoteo los elementos de cada conjunto que cambiaron de cluster

    ### Listas para los elementos que cambiaron de cluster
    X_data_cambios = X_data_pca[etiquetas_cambios]
    list_x = X_data_cambios[:, 0]
    list_y = X_data_cambios[:, 1]
    list_xv = cambios_variables[:, 0]
    list_yv = cambios_variables[:, 1]
    list_pais = list_pais[etiquetas_cambios]
    grados_pertenencia = grados_pertenencia[etiquetas_cambios]

    ## Plotear elementos que cambiaron de cluster
    for i in range(k):
        source = ColumnDataSource(data={'x': list_x[np.where(etiqs == i)], 'y': list_y[np.where(etiqs == i)],
                                        'xv': list_xv[np.where(etiqs == i)], 'yv': list_yv[np.where(etiqs == i)],
                                        'cliente': list_pais[np.where(etiqs == i)],
                                        'grados_p': grados_pertenencia[np.where(etiqs == i)],
                                        'cluster_id': etiqs[np.where(etiqs == i)]})
        p.square('x', 'y', size=6,
                 fill_color='white', source=source)

    ### Ploteo centroides

    ### Listas para centroiodes
    cambios_centroids = centroids_viej - centroids
    list_x = centroids_pca[:, 0]
    list_y = centroids_pca[:, 1]
    list_xv = cambios_centroids[:, 0]
    list_yv = cambios_centroids[:, 1]

    # Plotar centroids
    source = ColumnDataSource(data={'x': list_x, 'y': list_y, 'xv': list_xv, 'yv': list_yv})
    #    p.square('x','y', size=15,             fill_color='black',source=source)

    ### Labels de la grafica (componentes principales)
    p.xaxis.axis_label = 'Componente principal 1'
    p.yaxis.axis_label = 'Componente principal 2'
    # p.xaxis.axis_label = datos.columns[-2]
    # p.yaxis.axis_label = datos.columns[-1]

    ### Guardar resultados
    output_file('/home/david/proyecto-integrador/outputs/rfm_' + str(year_i + 1 + periodos) + '.html')
    save(p)

