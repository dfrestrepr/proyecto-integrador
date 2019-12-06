import pandas as pd
import numpy as np
import funcionesv2 as fn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

### Lectura de datos reales (este archivo lo tenemos de forma local)
PATH = '/home/david/Descargas/SEG_DYN.csv'
variables = ['precio_total', 'cantidad', 'frecuencia','CEMENT_QTY', 'C_FLYASH_QTY', 'CEMENTITIOUS_QTY',
         'COARSE_QTY', 'SAND_QTY', 'ADMIXTURE_QTY']
MAP = {
    'CUST_CODE': 'cliente',
    'TKT_VALUE': 'precio_total',
    'TKT_QTY': 'cantidad',
    'ORDER_DATE': 'fecha'
}

datos = pd.read_csv(PATH, sep="|", parse_dates=[2,27])
datos = datos.replace('\\N', np.nan )

data=datos[['CUST_CODE','TKT_VALUE','TKT_QTY','ORDER_DATE', 'CEMENT_QTY', 'C_FLYASH_QTY', 'CEMENTITIOUS_QTY',
         'COARSE_QTY', 'SAND_QTY', 'ADMIXTURE_QTY']].copy()

data = data.rename(columns=MAP)
data['ano'] = data.fecha.dt.year
data['trimestre'] = data.fecha.dt.quarter


## filtro los clientes del primer periodo
filtro = data[(data.ano == 2017) & (data.trimestre == 1)]
filtro = filtro.cliente.unique()
data = data[data.cliente.isin(filtro)]
data['frecuencia'] = 1

"""
pd.set_option('display.max_rows', 50)
data[variables].describe()
data[variables].hist()
plt.show()
sb.pairplot(data, size=4,vars=variables,kind='scatter')
plt.show()
"""
# calcula de la fecha final para cada trimestre
data['final_trimestre'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in  data.fecha]
data = data.sort_values(['cliente', 'fecha'])

# acumulados por cliente
data_sum = pd.DataFrame()
for each in data.cliente.unique():
    filtro = data[data.cliente == each].copy()
    filtro = filtro[variables].cumsum()
    data_sum = data_sum.append(filtro)

data.update(data_sum)


# procesamiento para que cada cliente este en todos los periodos, y obtener el valor de cada cliente en cada trimestre
df_list = pd.DataFrame(columns=data.columns
                       )
for each in data.cliente.unique():
    filtro = data[data.cliente == each].copy()
    for value in data.ano.unique():
        filtro2 = filtro[filtro.ano == value]
        if filtro2.empty:
            df = df_list[(df_list.cliente == each) & (df_list.ano == value - 1)].tail(1)
            df.ano = value
            for i in data.trimestre.unique():
                df2 = df
                df2.trimestre = i
                df_list = df_list.append(df2)
        else:
            for item in data.trimestre.unique():
                filtro3 = filtro2[filtro2.trimestre == item]
                if filtro3.empty:
                    if item == 1:
                        df = df_list[(df_list.cliente == each) & (df_list.ano == value - 1)].tail(1)
                        df.ano = value
                        df.trimestre = item
                        df_list = df_list.append(df)
                    else:
                        df = df_list[(df_list.cliente == each) & (df_list.ano == value) & (df_list.trimestre == item -1)]
                        df.trimestre = item
                        df_list = df_list.append(df.tail(1))
                else:
                    df_list = df_list.append(filtro3.tail(1))

# procesamiento para obtener el recency con base el cual es la difrencia entre la ultima fecha de pedido y
# la fecha de fin de cada trismestre
df_fecha = data[['ano', 'trimestre', 'final_trimestre']].drop_duplicates()
df = pd.merge(df_list, df_fecha, how='left', on=['ano', 'trimestre'])
df['recency'] = df['final_trimestre_y'] - df['fecha']
df['recency']=df['recency']/np.timedelta64(1,'D')
df['trimestre'] = df['trimestre']/10
df['periodo'] = df.ano + df.trimestre

df = df.drop(['trimestre', 'final_trimestre_x', 'final_trimestre_y', 'fecha','ano'], axis=1)



"""
#### Preprocesamiento de datos

df2 = fn.data_preprocessing(df, alpha_outlier_detection =0.70,
                                     columns_not_numeric = {'cliente','periodo'},
                                     column_id = 'cliente')
variables = ['C_FLYASH_QTY', 'CEMENTITIOUS_QTY', 'ADMIXTURE_QTY', 'recency']

"""
# deteccion outlayer
outlier= fn.outlier_detection_mahal(df.drop(['periodo', 'cliente'], axis=1), 0.80)
index = np.where(outlier==1)[0]
df_out = df.iloc[index].cliente.unique()
df = df[~df.cliente.isin(df_out)]

#df_final.shape
# normalizacion de variables
scaler_es = StandardScaler()
def normalizacion(df):
    df_list= list()
    for item in df.periodo.unique():
        filtro = df[df['periodo'] == item].copy()
        filtro.recency = filtro.recency.apply(
            lambda x: (filtro.recency.max()-x)/(filtro.recency.max()-filtro.recency.min())
        )
        filtro[variables] = scaler_es.fit_transform(filtro[variables])
        df_list.append(filtro)
    estandar = pd.concat(df_list)
    return estandar

datos_e = normalizacion(df)

# Aplicarle PCA a todos los datos
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_e[variables])

# Fijar seed aleatoria
np.random.seed(1)

# Inicio tomando los del primer year
year_i = min(datos_e['periodo'])   ### Year inicial a considerar
filtro = datos_e['periodo']==year_i
X_data_df = datos_e[filtro].reset_index(drop=True)
X_data = np.array(X_data_df[variables])

# Numero de periodos que incluire en el estudio, sin incluir el inicial
periodos_totales = datos_e.periodo.unique()
periodos_incluir = periodos_totales[1:]

# Los que usare para el PCA seran
X_data_pca = np.array(datos_pca[filtro])

# Lista donde ire guardando las listas de grados de pertenencia
grad_per = []

# Lista donde ire guardando las etiquetas asignadas del cluster
etiquetas_glo = []

# Lista donde ire guardando las variables mas importantes por cluster y por periodo
imp_periods_var = []

# Lista donde ire guardando los centroides de cada iteracion
centroids_ite = []

# Numero de observaciones en cada periodo
numdata = len(X_data)

### Define cantidad de clusters, numero maximo de iteraciones, y la distancia
### que se utilizara en el metodo de kmeans
k = 4
numiter = 5
p_dista = 2   ### 0 para mahalanobis


#### Inicializar los centroides
centroids = fn.init_centroids(X_data,k)
centroids_pca = pca.transform(centroids)

### Para la fase 2 de las importancias
centroids_p = centroids.copy()

### Aplicar kmeans
grados_pertenencia,etiquetas,centroids = fn.kmeans(X_data,
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
    distintos_cluster = np.where(datay_i != clu)
    ### Lo que no pertenece al cluster, lo pongo en -1
    datay_i[distintos_cluster] = -1
    datay_i = pd.DataFrame(datay_i)

    ### Calcular relevancias
    relevancias, _ = fn.variables_relevantes_arbol(datax_i, datay_i, 0)

    importancias_cluster.append(relevancias)

##### Calculo los promedios de importancia de cada variable
imp_clus_prom = np.mean(importancias_cluster, axis=0)

### Guardo las importancias de esta iteracion
imp_iters.append(imp_clus_prom)

### Guardo importancias generales (para el plot)
imp_periods_var.append(importancias_cluster)

for periodos in periodos_incluir:

    ### Guardo la X_data anterior
    X_data_viej = X_data.copy()
    centroids_viej = centroids.copy()

    ### Los datos para este year ya serian
    X_data_df = datos_e[datos_e['periodo'] == periodos].reset_index(drop=True)
    X_data = np.array(X_data_df[variables])

    #### Obtener los 2 componentes principales de los datos para plotear estos
    X_data_pca = np.array(datos_pca[datos_e['periodo'] == periodos])

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
        rankpeso_variables[i] = (cont + 1) / len(rank_variables)
        cont = cont + 1

    #### Usar rankings o usar los promedios para el peso
    peso_variables = importancia_prom.copy() * 100  ### Escalarlos con 100 para reducir errores numericos

    ### Escalo entonces la X para cambiar los pesos (segun las importancias)
    X_data_pond = X_data.copy()
    for peso in range(len(peso_variables)):
        X_data_pond[:, peso] = X_data_pond[:, peso] * peso_variables[peso]

    ###########################################################################
    ######################## K means para los plots ###########################
    ###########################################################################

    ### Etiquetas actuales de cada elemento para cada cluster
    etiquetas_prev = etiquetas.copy()

    ###########################################################################
    #################### Clusters con k means ponderado #######################
    ###########################################################################

    grados_pertenencia, etiquetas, centroids = fn.kmeans(X_data_pond,
                                                                numiter,
                                                                centroids,
                                                                p_dista=p_dista,
                                                                etiquetas=etiquetas)

    ### Guardo grados de pertenencia
    grad_per.append(grados_pertenencia.copy())

    ### Guardo etiquetas
    etiquetas_glo.append(etiquetas.copy())

    ### Guardo centroids (con los valores originales sin ponderar)
    centroids_ite.append(centroids.copy() * (1 / peso_variables))

    ###### Esta importancia la necesito para los labels
    ### Obtener la importancia de las variables de cada cluster (de mayor a menor)
    importancias_cluster = []
    ### Para cada cluster
    for clu in range(k):
        ### Dejo solo las observaciones de cada cluster
        datax_i = pd.DataFrame(X_data_pond)
        datay_i = etiquetas.copy()

        #### Solo clasifico binario si si pertence o no a cada cluster
        distintos_cluster = np.where(datay_i != clu)

        ### Lo que no pertenece al cluster, lo pongo en -1
        datay_i[distintos_cluster] = -1
        datay_i = pd.DataFrame(datay_i)

        ### Calcular relevancias
        relevancias, _ = fn.variables_relevantes_arbol(datax_i, datay_i, 0)

        importancias_cluster.append(relevancias)

    ### Guardo importancias generales (para el plot)
    imp_periods_var.append(importancias_cluster)

    ###########################################################################
    ################ K means para la seleccion de variables ###################
    ###########################################################################

    ###### Para la proxima iteracion, los pesos
    grados_pertenencia_p, etiquetas_p, centroids_p = fn.kmeans(X_data_ori,
                                                                      numiter,
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
        relevancias, _ = fn.variables_relevantes_arbol(datax_i, datay_i, 0)

        importancias_cluster.append(relevancias)

    ### Calculo los promedios de importancia de cada variable
    imp_clus_prom = np.mean(importancias_cluster, axis=0)

    ### Guardo las importancias de esta iteracion
    imp_iters.append(imp_clus_prom)

###############################################################################
########################### NUEVA VISUALIZACION  ##############################
###############################################################################

#scaler_es = StandardScaler()

fn.gapminder_plot_bokeh(datos_e, datos_pca, year_i, X_data_df, grad_per,
                               etiquetas_glo, periodos_totales, k, imp_periods_var,
                               centroids_ite, scaler_es,
                               title='Gapminder data',
                               xlabel='Componente principal 1',
                               ylabel='Componente principal 2')

#
# from bokeh.plotting import output_file, save
# output_file("test.html")
# save(layout)
#



