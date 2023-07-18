# Predictor de Retrasos de Vuelos

## Introducción

Este proyecto consistió en realizar un análisis de un estudio realizado por un ingeniero (Juan) el cual consistía en crear un modelo de _machine learning_ que pueda predecir si un vuelo se retrasará o no. Una vez hecho el análisis de aquel estudio yo escogí uno de los modelos que propuso Juan, éste fue  **<span style="color:green">XGBoost</span>**. La razón de mi decisión fue netamente al funcionamiento de aquel modelo ya que tiene muchas consideraciones internas importantes como la ponderación a ciertos _features_ los cuales se consideran más importantes, el _tuning_ interno que se realizan a los parámetros aprendibles, entre otros que se encuentran más detallados en mi análisis hecho en el _Jupyter Notebook_ cuyo nombre es `Challenge Machine Learning Engineer.ipnyb`.

En la selección del modelo tuve que hacer varias pruebas con las bases de datos para probar y encontrar un mejor resultado. A modo resumen no obtuve resultados tan diferentes con los procesamientos realizados, Repito en el Notebook se aprecia con más detalles los análisis, resultados y conclusiones.

## Contenido del github

En este repositorio nos encontramos con las bases de datos:
  - `dataset_SCL.csv` que es la base de datos original.
  - `final_feat.csv` es la base de datos que tiene variables seleccionadas por el ingeniero.
  - `synthetic_features.csv` es la base de datos con variables creadas por el ingeniero.

también se encuentra los script:
  - `predictor_retrasos.py` que es el código base de la predicción
  - `main.py` el cual es la API REST que se debe ejecutar con flask (más adelante se encuentra la forma de ejecutarlo)

Pesos pre-entrenados:
  - `xgb_trained_model.json` 

Jupyter Notebook:
  - `Challenge Machine Learning Engineer.ipynb` notebook donde se encuentra mi análisis acerca de la selección del modelo y sugerencias para mejorarlo.
  - `to-expose.ipynb` notebook de Juan donde hace su análisis.




## Ejecución

Con respecto a la ejecución del modelo escogí usar `flask` para la API REST. Con el modelo se podrá entrenar y/o hacer la predicción. Lamentablemente, por motivos de tiempo no pudeo hacerlo tan automatizado como yo quisiera, entonces para hacer que el modelo entrene o prediga hay que realizar un cambio en el archivo `main.py` dentro de la carpeta **server**. Este cambio consiste en modificar la variable `Train` por `False ` si se quiere predecir o por `True` si se quiere entrenar, esta variable se encuentra dentro del diccionario `PARAMS` y listo, luego en la terminal se puede iniciar de la forma `python main.py` o `FLASK_APP='main.py' flask run`.
Entonces, aparecerá un link del puerto en donde se mostrarán los resultados, hay que abrir ese enlace.

Ahora, hay que mencionar un par de cosas más con respecto al código. Cuando se entrena hay este código va a guardar los pesos cuyo nombre de archivo será `xgb_trained_model.json` este mismo se necesita para lograr predecir el modelo, de todas formas ya está subido estos pesos si es que solo se desea predecir. Además, para el entrenamiento el modelo va a recibir un archivo cuyo nombre es `final_feat.csv` el cual contiene la información seleccionada en primera instancia por el ingeniero (estas variables son: `OPERA`, `MES`, `TIPOVUELO`,`SIGLADES`,`DIANOM`,`atraso_15`), luego se hace una transformación a _one hot encoding_ y se entrena. Si se desea predecir, el modelo cargará los pesos y hará la predicción de la data ingresada, este input debe ser un un archivo csv con los mismo features que fueron ingresados al entrenamiento. En la variable `PARAMS_FILENAME`, del script _main.py_, se debe ingresar la ruta donde se encuentra la base de datos para que sea cargada.

### Output: 

Cuando se ejecuta el entrenamiento con flask se abrirá el navegador y se apreciará un diccionario con la tabla de reportes de clasificación más la exactitud de validación. Cuando se ejecuta la predicción se verá una tabla con el ID del vuelo, un par de características de él y la predicción en la columna `atraso_15`.

___
Por último, como bien mencioné más arriba, por temas de tiempo no pudeo hacer más automatizado el código también me faltó hacer la prueba de estrés y hacer más pruebas para encontrar la mejor configuración de hiperparámetros que conllevaría a una mejor clasificación.