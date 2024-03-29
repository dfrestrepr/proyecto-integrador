# Segmentación Dinámica

Este proyecto consiste en la implementación de un método que permita hacer segmentación (clustering) de un conjunto de individuos para los cuales se tiene un componente temporal, de tal forma que se tenga un efecto de dinamismo en la segmentación a través del tiempo.



Los siguientes son los pasos para ejecutar los scripts y las visualizaciones del programa:


Para que funcionen correctamente los comandos de los visualizadores, 
tener en cuenta que bokeh es una librería de python y desde la terminal 
donde se ejecute este comando se debe tener acceso a python (puede ser 
en un ambiente de python, o usarlo directamente si se tiene instalado 
python en el PATH del computador). 

***********************************************************************
Crear ambiente de Anaconda para el proyecto a partir de archivo yml:

En el anaconda prompt, navegar hasta la carpeta donde se encuentra el
archivo "SegmentacionDinamica.yml" y ejecutar el comando:

conda env create -f SegmentacionDinamica.yml

***********************************************************************

- Para ejecutar la segmentación dinámica con datos de Gapminder usando 
nuestro k-means de Python, sería necesario ejecutar en una linea de comandos 
ubicada actualmente en la carpeta "proyecto-integrador" el comando 
"bokeh serve main.py --show" el cual se encuentra en el archivo 'Ejecutar_gapminder.bat'. 

Para realizar esa ejecución, desde el anaconda prompt, activamos el ambiente 
de segmentación dinámica con el comando:

conda activate segmentacion-dinamica

Posteriormente desde el anaconda prompt con el ambiente activo, se navega 
hasta la carpeta donde se encuentra el archivo Ejecutar_gapminder.bat, luego
directamente se ejecuta el archivo con el comando:

Ejecutar_gapminder.bat

Esto cargará la visualización de la solución del método
(se debe abrir un navegador con la visualización).


- Para ejecutar la segmentación dinámica con datos de Gapminder usando
la implementación de pyspark.ml de kmeans, sería necesario correr 
el notebook llamado "Ejemplo_pyspark_final.ipynb", ya que este notebook 
genera los .csv con los distintos reusltados de salida que usará el visualizador 
(este requiere usar pyspark, nosotros pudimos usarlo tanto con nuestras instalaciones 
locales de Pyspark como con Pyspark en un cluster de Azure databricks). 

Luego de generar estos resultados y de que queden guardados los .csv generados 
en la carpeta "datosspark", se puede usar el comando 
"bokeh serve visualizar_spark.py --show" (el cual se encuentra en el archivo 
'Visualizar_pyspark_gapminder.bat') para generar la visualización. 

Para realizar esta ejecución, repetimos los pasos utilizados para correr el
archivo Ejecutar_gapminder.bat, solo cambiando el nombre del archivo por Visualizar_pyspark_gapminder.bat

- Para la segmentación dinámica con datos reales usando nuestro k-means de Python, 
no hemos subido aquí los datos necesarios para ejecutarla debido a temas de 
confidencialidad. Sin embargo, sí quedan en este github los distintos códigos que se 
usaron para la ejecución y visualización usando nosotros los datos (los datos estaban 
en un archivo llamado "SEG_DYN.csv" que nosotros teníamos de forma local). Para usar 
el programa, nosotros lo que hacíamos era ejecutar en una linea de comandos ubicada 
actualmente en la carpeta "proyecto-integrador" el comando "bokeh serve transaccional.py --show" 
el cual se encuentra en el archivo 'Ejecutar_datos_reales.bat'. 