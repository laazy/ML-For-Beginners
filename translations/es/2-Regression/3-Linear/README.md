# Construir un modelo de regresión usando Scikit-learn: regresión de cuatro maneras

## Nota para principiantes

La regresión lineal se usa cuando queremos predecir un **valor numérico** (por ejemplo, precio de una casa, temperatura o ventas).
Funciona encontrando una línea recta que represente mejor la relación entre las características de entrada y la salida.

En esta lección, nos enfocamos en entender el concepto antes de explorar técnicas de regresión más avanzadas.
![Linear vs polynomial regression infographic](../../../../translated_images/es/linear-polynomial.5523c7cb6576ccab.webp)
> Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introducción

Hasta ahora has explorado qué es la regresión con datos de ejemplo obtenidos del conjunto de datos de precios de calabazas que usaremos a lo largo de esta lección. También lo has visualizado usando Matplotlib.

Ahora estás listo para profundizar en regresión para ML. Mientras que la visualización te permite entender los datos, el verdadero poder del Aprendizaje Automático proviene de _entrenar modelos_. Los modelos se entrenan con datos históricos para capturar automáticamente las dependencias de los datos y permiten predecir resultados para datos nuevos, que el modelo no ha visto antes.

En esta lección, aprenderás más sobre dos tipos de regresión: _regresión lineal básica_ y _regresión polinómica_, junto con algo de la matemática que subyace a estas técnicas. Estos modelos nos permitirán predecir los precios de las calabazas dependiendo de diferentes datos de entrada.

[![ML para principiantes - Entendiendo la regresión lineal](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML para principiantes - Entendiendo la regresión lineal")

> 🎥 Haz clic en la imagen arriba para un video corto con una visión general de la regresión lineal.

> A lo largo de este currículo, asumimos un conocimiento mínimo de matemáticas y buscamos hacerlo accesible para estudiantes de otras áreas, así que presta atención a notas, 🧮 llamadas, diagramas y otras herramientas de aprendizaje que ayudan en la comprensión.

### Prerrequisitos

Ya deberías estar familiarizado con la estructura de los datos de calabazas que estamos examinando. Puedes encontrarlos precargados y preprocesados en el archivo _notebook.ipynb_ de esta lección. En el archivo, el precio de la calabaza está mostrado por bushel en un nuevo dataframe. Asegúrate de poder ejecutar estos notebooks en kernels de Visual Studio Code.

### Preparación

Como recordatorio, cargas estos datos para poder hacer preguntas sobre ellos.

- ¿Cuándo es el mejor momento para comprar calabazas?
- ¿Qué precio puedo esperar para un paquete de calabazas pequeñas?
- ¿Debería comprarlas en cestas de medio bushel o en cajas de 1 1/9 bushel?
Sigamos indagando en estos datos.

En la lección anterior, creaste un dataframe de Pandas y lo llenaste con parte del conjunto de datos original, estandarizando el precio por bushel. Sin embargo, al hacer eso, solo pudiste obtener alrededor de 400 puntos de datos y solo para los meses de otoño.

Mira los datos que precargamos en el notebook que acompaña esta lección. Los datos están precargados y se ha graficado un diagrama de dispersión inicial para mostrar los datos del mes. Quizá podamos obtener un poco más de detalle sobre la naturaleza de los datos limpiándolos más.

## Una línea de regresión lineal

Como aprendiste en la Lección 1, el objetivo de un ejercicio de regresión lineal es poder trazar una línea para:

- **Mostrar relaciones entre variables**. Mostrar la relación entre variables.
- **Hacer predicciones**. Hacer predicciones precisas sobre dónde caerá un nuevo punto de datos en relación con esa línea.

Es típico de la **regresión por mínimos cuadrados** dibujar este tipo de línea. El término "mínimos cuadrados" se refiere al proceso de minimizar el error total en nuestro modelo. Para cada punto de datos, medimos la distancia vertical (llamada residual) entre el punto real y nuestra línea de regresión.

Elevamos al cuadrado estas distancias por dos razones principales:

1. **Magnitud sobre dirección:** Queremos tratar un error de -5 igual que un error de +5. Al elevar al cuadrado todas las valores se vuelven positivos.

2. **Penalización de valores atípicos:** Elevar al cuadrado da más peso a errores grandes, forzando a la línea a mantenerse más cerca de puntos que están lejos.

Luego sumamos todos estos valores al cuadrado. Nuestro objetivo es encontrar la línea específica donde esta suma final sea la menor (el valor más pequeño posible), de ahí el nombre "mínimos cuadrados".

> **🧮 Muéstrame las matemáticas**
> 
> Esta línea, llamada _línea de mejor ajuste_, puede expresarse con [una ecuación](https://en.wikipedia.org/wiki/Simple_linear_regression):
> 
> ```
> Y = a + bX
> ```
>
> `X` es la 'variable explicativa'. `Y` es la 'variable dependiente'. La pendiente de la línea es `b` y `a` es la intersección en y, que se refiere al valor de `Y` cuando `X = 0`.
>
>![calcular la pendiente](../../../../translated_images/es/slope.f3c9d5910ddbfcf9.webp)
>
> Primero, calcula la pendiente `b`. Infografía por [Jen Looper](https://twitter.com/jenlooper)
>
> En otras palabras, y refiriéndonos a la pregunta original de nuestros datos de calabazas: "predecir el precio de una calabaza por bushel según el mes", `X` se referiría al precio y `Y` al mes de venta.
>
>![completar la ecuación](../../../../translated_images/es/calculation.a209813050a1ddb1.webp)
>
> Calcula el valor de Y. Si estás pagando alrededor de $4, ¡debe ser abril! Infografía por [Jen Looper](https://twitter.com/jenlooper)
>
> Las matemáticas que calculan la línea deben mostrar la pendiente de la línea, que también depende de la intersección, o dónde se sitúa `Y` cuando `X = 0`.
>
> Puedes observar el método de cálculo para estos valores en el sitio web [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). También visita [este calculador de mínimos cuadrados](https://www.mathsisfun.com/data/least-squares-calculator.html) para ver cómo los valores de los números impactan la línea.

## Correlación

Un término más que hay que entender es el **Coeficiente de Correlación** entre dos variables X y Y dadas. Usando un diagrama de dispersión, puedes visualizar rápidamente este coeficiente. Un gráfico con puntos de datos alineados en una línea ordenada tiene alta correlación, pero un gráfico con puntos dispersos por todo el plano entre X y Y tiene baja correlación.

Un buen modelo de regresión lineal será aquel que tenga un alto coeficiente de correlación (más cercano a 1 que a 0) usando el método de regresión por mínimos cuadrados con una línea de regresión.

✅ Ejecuta el notebook que acompaña esta lección y mira el diagrama de dispersión entre Mes y Precio. ¿Parece que hay una alta o baja correlación entre Mes y Precio para las ventas de calabazas, según tu interpretación visual del diagrama? ¿Cambia si usas una medida más detallada en vez de `Mes`, por ejemplo, *día del año* (es decir, número de días desde el inicio del año)?

En el código a continuación, asumiremos que hemos limpiado los datos y obtenido un dataframe llamado `new_pumpkins`, similar al siguiente:

ID | Mes | DíaDelAño | Variedad | Ciudad | Paquete | Precio Bajo | Precio Alto | Precio
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | TIPO PARA PASTEL | BALTIMORE | Cartones de 1 1/9 bushel | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | TIPO PARA PASTEL | BALTIMORE | Cartones de 1 1/9 bushel | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | TIPO PARA PASTEL | BALTIMORE | Cartones de 1 1/9 bushel | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | TIPO PARA PASTEL | BALTIMORE | Cartones de 1 1/9 bushel | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | TIPO PARA PASTEL | BALTIMORE | Cartones de 1 1/9 bushel | 15.0 | 15.0 | 13.636364

> El código para limpiar los datos está disponible en [`notebook.ipynb`](notebook.ipynb). Hemos realizado los mismos pasos de limpieza que en la lección anterior, y hemos calculado la columna `DayOfYear` usando la siguiente expresión:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Ahora que tienes una comprensión de las matemáticas detrás de la regresión lineal, vamos a crear un modelo de Regresión para ver si podemos predecir qué paquete de calabazas tendrá los mejores precios. Alguien que compre calabazas para un parche de calabazas en una fiesta podría querer esta información para optimizar sus compras de paquetes de calabazas para el parche.

## Buscando correlación

[![ML para principiantes - Buscando correlación: La clave para la regresión lineal](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML para principiantes - Buscando correlación: La clave para la regresión lineal")

> 🎥 Haz clic en la imagen arriba para un video corto con una visión general de la correlación.

Probablemente en la lección anterior viste que el precio promedio por diferentes meses se ve así:

<img alt="Precio promedio por mes" src="../../../../translated_images/es/barchart.a833ea9194346d76.webp" width="50%"/>

Esto sugiere que debería haber alguna correlación, y podemos intentar entrenar un modelo de regresión lineal para predecir la relación entre `Mes` y `Precio`, o entre `DíaDelAño` y `Precio`. Aquí está el gráfico de dispersión que muestra esta última relación:

<img alt="Diagrama de dispersión de Precio vs Día del Año" src="../../../../translated_images/es/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vamos a ver si hay correlación usando la función `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Parece que la correlación es bastante baja, -0.15 por `Mes` y -0.17 por el `DíaDelAño`, pero podría haber otra relación importante. Parece que hay diferentes grupos de precios correspondientes a diferentes variedades de calabazas. Para confirmar esta hipótesis, grafiquemos cada categoría de calabaza usando colores diferentes. Pasando un parámetro `ax` a la función de gráfica `scatter`, podemos trazar todos los puntos en el mismo gráfico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Diagrama de dispersión de Precio vs Día del Año" src="../../../../translated_images/es/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Nuestra investigación sugiere que la variedad tiene más efecto en el precio general que la fecha de venta real. Podemos ver esto con un gráfico de barras:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Gráfico de barras de precio vs variedad" src="../../../../translated_images/es/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Vamos a enfocarnos por el momento solo en una variedad de calabaza, la 'tipo pastel', y ver qué efecto tiene la fecha sobre el precio:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Diagrama de dispersión de Precio vs Día del Año" src="../../../../translated_images/es/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Si ahora calculamos la correlación entre `Precio` y `DíaDelAño` usando la función `corr`, obtendremos algo como `-0.27` – lo que significa que entrenar un modelo predictivo tiene sentido.

> Antes de entrenar un modelo de regresión lineal, es importante asegurarse de que nuestros datos estén limpios. La regresión lineal no funciona bien con valores faltantes, por lo que tiene sentido eliminar todas las celdas vacías:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Otra opción sería llenar esos valores vacíos con los valores promedio de la columna correspondiente.

## Regresión lineal simple

[![ML para principiantes - Regresión lineal y polinómica usando Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML para principiantes - Regresión lineal y polinómica usando Scikit-learn")

> 🎥 Haz clic en la imagen arriba para un video corto con una visión general de la regresión lineal y polinómica.

Para entrenar nuestro modelo de Regresión Lineal usaremos la biblioteca **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Comenzamos separando los valores de entrada (características) y la salida esperada (etiqueta) en arreglos numpy separados:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Ten en cuenta que tuvimos que realizar un `reshape` en los datos de entrada para que el paquete de Regresión Lineal lo entienda correctamente. Regresión Lineal espera un arreglo 2D como entrada, donde cada fila del arreglo corresponde a un vector de características de entrada. En nuestro caso, dado que tenemos solo una entrada, necesitamos un arreglo con forma N&times;1, donde N es el tamaño del conjunto de datos.

Luego, necesitamos dividir los datos en conjuntos de entrenamiento y prueba, para poder validar nuestro modelo después del entrenamiento:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finalmente, entrenar el modelo actual de Regresión Lineal toma solo dos líneas de código. Definimos el objeto `LinearRegression` y lo ajustamos a nuestros datos usando el método `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

El objeto `LinearRegression` después de ajustar (`fit`) contiene todos los coeficientes de la regresión, a los cuales se puede acceder mediante la propiedad `.coef_`. En nuestro caso, hay solo un coeficiente, que debería estar alrededor de `-0.017`. Esto significa que los precios parecen bajar un poco con el tiempo, pero no mucho, alrededor de 2 centavos por día. También podemos acceder al punto de intersección de la regresión con el eje Y usando `lin_reg.intercept_` — que estará alrededor de `21` en nuestro caso, indicando el precio al comienzo del año.

Para ver qué tan preciso es nuestro modelo, podemos predecir precios en un conjunto de datos de prueba, y luego medir qué tan cercanas están nuestras predicciones a los valores esperados. Esto se puede hacer usando la métrica del error cuadrático medio (MSE), que es la media de todas las diferencias cuadradas entre los valores esperados y predichos.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Nuestro error parece estar alrededor de 2 puntos, lo que es ~17%. No muy bueno. Otro indicador de la calidad del modelo es el **coeficiente de determinación**, que se puede obtener así:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

Si el valor es 0, significa que el modelo no toma en cuenta los datos de entrada y actúa como el *peor predictor lineal*, que es simplemente el valor medio del resultado. El valor 1 significa que podemos predecir perfectamente todas las salidas esperadas. En nuestro caso, el coeficiente es alrededor de 0.06, que es bastante bajo.

También podemos graficar los datos de prueba junto con la línea de regresión para ver mejor cómo funciona la regresión en nuestro caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/es/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regresión Polinómica

Otro tipo de Regresión Lineal es la Regresión Polinómica. Aunque a veces hay una relación lineal entre variables — mientras más grande el volumen de la calabaza, mayor el precio — a veces estas relaciones no pueden representarse como un plano o línea recta.

✅ Aquí hay [más ejemplos](https://online.stat.psu.edu/stat501/lesson/9/9.8) de datos que podrían utilizar Regresión Polinómica

Observa nuevamente la relación entre Fecha y Precio. ¿Parece que este diagrama de dispersión deba analizarse necesariamente con una línea recta? ¿No pueden fluctuar los precios? En este caso, puedes probar la regresión polinómica.

✅ Los polinomios son expresiones matemáticas que pueden consistir en una o más variables y coeficientes

La regresión polinómica crea una curva para ajustarse mejor a datos no lineales. En nuestro caso, si incluimos una variable cuadrática `DayOfYear` en los datos de entrada, deberíamos poder ajustar nuestros datos con una curva parabólica, que tendrá un mínimo en un cierto punto dentro del año.

Scikit-learn incluye una útil [API de pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) para combinar diferentes pasos del procesamiento de datos juntos. Un **pipeline** es una cadena de **estimadores**. En nuestro caso, crearemos un pipeline que primero agrega características polinómicas al modelo y luego entrena la regresión:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usar `PolynomialFeatures(2)` significa que incluiremos todos los polinomios de segundo grado a partir de los datos de entrada. En nuestro caso solo significará `DayOfYear`<sup>2</sup>, pero dado dos variables de entrada X y Y, esto agregará X<sup>2</sup>, XY y Y<sup>2</sup>. También podemos usar polinomios de grado superior si queremos.

Los pipelines pueden usarse de la misma manera que el objeto original `LinearRegression`, es decir, podemos ajustar (`fit`) el pipeline y luego usar `predict` para obtener los resultados de la predicción. Aquí está el gráfico que muestra los datos de prueba y la curva de aproximación:

<img alt="Polynomial regression" src="../../../../translated_images/es/poly-results.ee587348f0f1f60b.webp" width="50%" />

Usando Regresión Polinómica, podemos obtener un MSE ligeramente menor y un coeficiente de determinación más alto, pero no de forma significativa. ¡Necesitamos tener en cuenta otras características!

> Puedes ver que los precios mínimos de las calabazas se observan alrededor de Halloween. ¿Cómo puedes explicar esto?

🎃 ¡Felicidades, acabas de crear un modelo que puede ayudar a predecir el precio de las calabazas para pastel! Probablemente puedas repetir el mismo procedimiento para todos los tipos de calabaza, pero eso sería tedioso. ¡Ahora aprendamos cómo tener en cuenta la variedad de calabaza en nuestro modelo!

## Características Categóricas

En un mundo ideal, queremos poder predecir precios para diferentes variedades de calabaza usando el mismo modelo. Sin embargo, la columna `Variety` es algo diferente de columnas como `Month`, porque contiene valores no numéricos. Estas columnas se llaman **categóricas**.

[![ML para principiantes - Predicciones con características categóricas usando Regresión Lineal](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML para principiantes - Predicciones con características categóricas usando Regresión Lineal")

> 🎥 Haz clic en la imagen arriba para un breve video sobre el uso de características categóricas.

Aquí puedes ver cómo depende el precio promedio de la variedad:

<img alt="Average price by variety" src="../../../../translated_images/es/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Para tener en cuenta la variedad, primero necesitamos convertirla a forma numérica, o **codificarla**. Hay varias formas de hacerlo:

* La simple **codificación numérica** construirá una tabla de diferentes variedades y luego reemplazará el nombre de la variedad por un índice en esa tabla. Esto no es lo mejor para regresión lineal, porque la regresión lineal toma el valor numérico real del índice y lo agrega al resultado, multiplicado por algún coeficiente. En nuestro caso, la relación entre el número de índice y el precio es claramente no lineal, incluso si nos aseguramos de que los índices estén ordenados de alguna forma específica.
* La **codificación one-hot** reemplazará la columna `Variety` por 4 columnas diferentes, una para cada variedad. Cada columna contendrá `1` si la fila correspondiente es de esa variedad, y `0` de lo contrario. Esto significa que habrá cuatro coeficientes en la regresión lineal, uno para cada variedad de calabaza, responsables del "precio inicial" (o más bien "precio adicional") para esa variedad en particular.

El código a continuación muestra cómo codificar one-hot una variedad:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Para entrenar la regresión lineal usando la variedad codificada one-hot como entrada, solo necesitamos inicializar datos `X` y `y` correctamente:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

El resto del código es igual al que usamos antes para entrenar la Regresión Lineal. Si lo pruebas, verás que el error cuadrático medio es aproximadamente el mismo, pero obtenemos un coeficiente de determinación mucho más alto (~77%). Para obtener predicciones aún más precisas, podemos tener en cuenta más características categóricas, así como características numéricas, como `Month` o `DayOfYear`. Para obtener un gran arreglo de características, podemos usar `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Aquí también tomamos en cuenta `City` y tipo de `Package`, lo que nos da MSE de 2.84 (10%), y determinación de 0.94!

## Integrando todo junto

Para crear el mejor modelo, podemos usar datos combinados (categóricos codificados one-hot + numéricos) del ejemplo anterior junto con Regresión Polinómica. Aquí tienes el código completo para tu comodidad:

```python
# configurar datos de entrenamiento
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# realizar la división de entrenamiento-prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configurar y entrenar la tubería
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predecir resultados para datos de prueba
pred = pipeline.predict(X_test)

# calcular MSE y determinación
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Esto debería darnos el mejor coeficiente de determinación de casi 97%, y MSE=2.23 (~8% de error de predicción).

| Modelo | MSE | Determinación |
|-------|-----|---------------|
| `DayOfYear` Lineal | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polinómico | 2.73 (17.0%) | 0.08 |
| `Variety` Lineal | 5.24 (19.7%) | 0.77 |
| Todas las características Lineal | 2.84 (10.5%) | 0.94 |
| Todas las características Polinómico | 2.23 (8.25%) | 0.97 |

🏆 ¡Bien hecho! Creaste cuatro modelos de Regresión en una lección, y mejoraste la calidad del modelo a 97%. En la sección final sobre Regresión, aprenderás sobre Regresión Logística para determinar categorías.

---
## 🚀Desafío

Prueba varias variables diferentes en este cuaderno para ver cómo la correlación corresponde a la precisión del modelo.

## [Cuestionario posterior a la clase](https://ff-quizzes.netlify.app/en/ml/)

## Repaso y Autoestudio

En esta lección aprendimos sobre Regresión Lineal. Existen otros tipos importantes de Regresión. Lee sobre las técnicas Stepwise, Ridge, Lasso y Elasticnet. Un buen curso para profundizar es el [curso de Aprendizaje Estadístico de Stanford](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Tarea

[Construir un Modelo](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda la traducción profesional realizada por humanos. No nos responsabilizamos por malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->