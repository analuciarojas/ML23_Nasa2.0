# Proyecto 1: Identificacion de digitos

## Introducción
En este proyecto, entraremos en el mundo de la visión por computadora y el aprendizaje automático para identificar dígitos escritos a mano a partir de imágenes. Utilizaremos el conjunto de datos Digits, que contiene 1797 imágenes de 8x8 píxeles, para entrenar y validar nuestros modelos. Nuestro objetivo es predecir con la mayor precisión posible a qué número corresponde cada imagen, aplicando diferentes técnicas y métodos aprendidos en clase.

## Modelos de clasificación
### K-means (agrupamiento)
### Regresión Logística (clasificación)

## Visualización en baja dimensionalidad

A continuación, hemos aplicado tanto TSNE como PCA para reducir la dimensionalidad de nuestros datos de entrenamiento a 2 dimensiones. Observemos ambas visualizaciones y determinemos cuál nos brinda una representación más clara y distintiva de los grupos en nuestros datos.

```python
#Reducimos la dimensionalidad de los datos de validacion data_val
# a 2 dimensiones usando TSNE y/o PCA
reduced_data = TSNE(n_components=2).fit_transform(data_train) #esta mejor este porque los datos se ven mejor separados
#reduced_data = PCA(n_components=2).fit_transform(data_train)

labels = np.unique(target_train)
fig, ax_tsne = plt.subplots(1, 1, figsize=(4,4))
fig.suptitle("Puntos reducidos a dos dimensiones")
for c in labels:
    indices = np.where(target_train == c)
    plot_data = reduced_data[indices]
    ax_tsne.scatter(plot_data[:, 0], plot_data[:, 1], label=f"Grupo {c}")
plt.show()
```

![download](https://github.com/analuciarojas/ML23_Nasa2.0/assets/101476793/209ef41f-2f3c-4aa2-baa9-4997e07d170e)

## Entrenamiento

Se tomó la decisión de entrenar en baja dimensionalidad con ambos K-means y Regresión Logística. Se normalizaron los datos, y no hubo ningún método extra de preprocesamiento

```python
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

model = {
    "regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42),
    "kmeans": KMeans(n_clusters=10, random_state=42),
}

# Entontramos los valores de normalización USANDO LOS DATOS DE ENTRENAMIENTO
scaler = StandardScaler()
scaler.fit(data_train)

def train(X, label, model_type:str):
    # Normalizamos los datos de entrenamiento
    data = scaler.transform(X)

    # TODO: Entrena el modelo y regresa el modelo entrenado en los datos de entrenamiento
    # model puede ser tanto la instancia de la clase que quieras usar, como un string indicando
    if model_type == "kmeans":
        model = KMeans(n_clusters=10, random_state=42)
    elif model_type == "regression":
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    else:
        raise ValueError(f"Modelo no reconocido: {model_type}")

    model.fit(data, label)
    return model

def inference(trained_model, X_val, model_type):
    # En inferencia, podemos recibir un solo dato entonces X_val.shape seria (D, )
    # Las clases de sklearn siempre esperan todo en la forma de  N, D
    if X_val.ndim == 1:
        X_val = X_val.reshape(1, -1)

    # TODO: Normaliza los datos de validación
    # El mismos preprocesamiento de datos se aplica a
    # tanto inferencia como entrenamiento
    data = scaler.transform(X_val)

    # TODO: Utiliza el modelo para predecir valores para los datos de validación
    # Regresa las predicciones de tu modelo para X_val
    # En este caso, modelo tiene que ser una instancia de una clase para la cual quieres hacer predicción
    if model_type == "kmeans":  # predecir los clústeres
        preds = trained_model.predict(data)
    elif model_type == "regression": #predecir las clases
        preds = trained_model.predict(data)
    return preds

trained_models = {
    "kmeans": None,
    "regression": None,
}
for model_type in trained_models.keys():
    modelo = train(data_train, target_train, model_type=model_type)
    trained_models[model_type] = modelo
```
## Evaluación y Análisis de predicciones

Se define una función que realizará la reducción de datos con el método TSNE a dos dimensiones, para que después se puedan visualizar los datos gráficamente.

```python
def vis_low_dim(data_val, preds, model_type):
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    fig.suptitle(f"Puntos clasificados {model_type} (2 dimensiones)")

    # Buscamos la cantidad de grupos que hay en los datos de validación
    groups = np.unique(preds)
    n_groups = len(groups)
    # Graficamos los datos, con un color diferente para cada clase/grupo
    print(f"Datos {data_val.shape}, predicciones {preds.shape}, clases/grupos {n_groups}")

    # TODO: Reduce los datos de VALIDACIÓN data_val a dos dimensiones para poder visualizarlos
    if model_type == "kmeans":
        reduced_data = TSNE(n_components=2).fit_transform(data_val)
    elif model_type == "regression":
        reduced_data = TSNE(n_components=2).fit_transform(data_val)

    for g in groups:
        # Filtrar los datos correspondientes a la clase actual
        data_group = reduced_data[preds == g]
        # Graficar los puntos de la clase actual
        ax.scatter(data_group[:, 0], data_group[:, 1], label=f"Grupo {g}")
    fig.show()
    fig.legend()
```

Realizar la inferencia

```python
for model_type, modelo in trained_models.items():
    preds = inference(modelo, data_val,model_type)
    vis_low_dim(data_val, preds, model_type)
```
Grupos: 10, se realizaron 450 predicciones
![download](https://github.com/analuciarojas/ML23_Nasa2.0/assets/101476793/27f28c78-3119-46ae-a8f2-f8fe1290c4a9)
![download](https://github.com/analuciarojas/ML23_Nasa2.0/assets/101476793/4502d8ef-6f79-43df-9a4b-cf8d97558e16)


Cada grupo en la imagen representa un digito del 0 al 9, se puede observar que ambos modelos si tuvieron aprendizaje y ambos hicieron un relativamente buen trabajo pues se pueden observar los grupos claramente aunque hay imperfecciones, como se puede notar en los puntos que no se acercaron a ningún grupo.

## Visualización de imágenes

A continuación se presentan los resultados de las predicciones, se utilizan los modelos entrenados y se pone a prueba la precisión de la predicción.
Realizar función para la visualización:

```python
def vis_preds(trained_model, data_val, target_val, model_name,model_type):
    # Llamamos a inferencia de su modelo
    # Este método regresará una cantidad definida de clases
    # Que haya encontrado para los datos de validación
    preds = inference(trained_model, data_val,model_type)
    group_pred = np.unique(preds)
    n_groups = len(group_pred)

    # Graficar
    n_cols = 5
    fig, axes = plt.subplots(n_groups//n_cols, n_cols, figsize=(10,6))
    axes = axes.flatten()
    side = 8
    for group, ax in zip(group_pred, axes):
        #======================= Start  solution=====================
        # TODO: Filtra data_val para quedarte solamente con aquellos elementos
        # donde la predicción de tu modelo sea igual a group
        # Haz lo mismo para las etiquetas
        group_data = data_val[preds == group]
        group_labels = target_val[preds == group]

        # TODO: Selecciona una imagen de los datos en data_val donde pred == group
        # y selecciona la etiqueta real para dicha imagén para mostrarlos juntos
        # Investiga: np.random.randint, np.random.choice etc.
        random_index = np.random.choice(len(group_data)) #seleccionar un índice aleatorio
        gt = group_labels[random_index] # utiliza el mismo índice aleatorio para seleccionar la etiqueta real correspondiente de group_labels. gt (ground truth) contiene la etiqueta real asociada a la imagen seleccionada.
        img_vector = group_data[random_index] #índice para seleccionar el vector de características correspondiente de group_data. El resultado es un vector que representa una imagen.
        
        # TODO: Calcula la predicción del modelo para la imagen aleatoria
        # usando el modelo entrenado "trained_model"
        pred = inference(trained_model,img_vector,model_type)

        # TODO: La predicción del modelo usa la imagen en forma de vector (1xD)
        # pero para visualizarla tenemos que cambia de forma a una imagen de 8x8 pixeles
        # Cambia la forma de la imagen usando np.reshape a (8, 8)
        img = np.reshape(img_vector, (side, side))
        
        # TODO: Visualiza la imagen de 8x8 usando ax.matshow Similar al inicio del ejercicio
        # Revisa la documentación de ser necesario
        ax.matshow(img, cmap='gray')

        #======================= end  solution=====================
        ax.set_title(f"Pred:{pred}, GT: {gt}")
        ax.axis('off')
    fig.suptitle(f"Muestras por grupo({model_name})")
    plt.tight_layout()
    plt.show()

for name, trained_model in trained_models.items():
    vis_preds(trained_model, data_val, target_val, name,model_type)
```

### K-means (agrupamiento)
![download](https://github.com/analuciarojas/ML23_Nasa2.0/assets/101476793/91215e2f-6ec9-4b0a-82c2-8d34668b9632)

### Regresión Logística (clasificación)
![download](https://github.com/analuciarojas/ML23_Nasa2.0/assets/101476793/4e19093a-b67d-4975-96ad-fefcfb57448a)

Al observar ambas imagenes, se puede ver claramente que el modelo de Regresión puede identificar todos los números. Mientras que K-means falló en casi todos menos el 5, además confunde el 4 con el 9.


## Comparación de rendimiento

```python
from sklearn import metrics

# Crear un diccionario para almacenar las métricas de cada modelo
model_metrics = {}

# TODO: Para todos los modelos que entrenaste, calcula un valor que indique la calidad de las predicciones en los datos de validación
# utiliza: data_val y target_val
for name, trained_model in trained_models.items():
    # Realizar predicciones en los datos de validación
    preds = inference(trained_model, data_val, model_type)
    
    # Calcular el accuracy (exactitud) como métrica
    accuracy = metrics.accuracy_score(target_val, preds)
    
    # Calcular el F1-score como métrica
    f1_score = metrics.f1_score(target_val, preds, average='macro')
    
    # Calcular la precisión y la exhaustividad (precision y recall) como métricas
    precision = metrics.precision_score(target_val, preds, average='macro')
    recall = metrics.recall_score(target_val, preds, average='macro')
    
    # Almacenar las métricas en el diccionario
    model_metrics[name] = {
        'Accuracy': accuracy,
        'F1 Score': f1_score,
        'Precision': precision,
        'Recall': recall
    }

# Imprimir un resumen de métricas de todos los modelos
print("Resumen de Métricas:")
for name, metrics in model_metrics.items():
    print(f"Modelo {name}:")
    print(f"Accuracy: {metrics['Accuracy']}")
    print(f"F1 Score: {metrics['F1 Score']}")
    print(f"Precision: {metrics['Precision']}")
    print(f"Recall: {metrics['Recall']}")
    print()
```

### K-means
Accuracy: 0.05555555555555555
F1 Score: 0.06756756756756757
Precision: 0.08333333333333334
Recall: 0.05681818181818182

### Regresión Logística
Accuracy: 0.9666666666666667
F1 Score: 0.9667957492952493
Precision: 0.9665632527868884
Recall: 0.9678913306011511

## Conclusión
### Alta o baja dimensionalidad
Al analizar los resultados cualitativos, observamos que la reducción de dimensionalidad a 2D a través de t-SNE permitió una mejor separación de los datos en comparación con PCA. Los puntos reducidos en t-SNE estaban más claramente agrupados en distintas clases, lo que sugiere que la alta dimensionalidad puede afectar negativamente la capacidad de los modelos para separar los datos.

### Supervisado o no supervisado
Nuestro análisis mostró que el aprendizaje supervisado, en particular el modelo de Regresión Logística, superó al enfoque no supervisado (K-Means) en términos de rendimiento. Los modelos supervisados alcanzaron una mayor precisión y F1-score en la identificación de números en las imágenes. Esto sugiere que en problemas de reconocimiento de números, donde tenemos etiquetas de clase disponibles, el aprendizaje supervisado puede ser más efectivo para aprovechar la información de las etiquetas y mejorar la precisión de las predicciones.

### Preprocesamiento
En el proyecto, aplicamos la normalización de datos como método de preprocesamiento.
No exploramos otros métodos de preprocesamiento en esta iteración del proyecto, pero se podría considerar la eliminación de características irrelevantes o la reducción de ruido en futuros proyectos para evaluar su impacto en el rendimiento de los modelos.

### Normalización de datos
La normalización de imágenes resultó ser beneficiosa en nuestro proyecto. Ayudó a mejorar la precisión de los modelos al garantizar que las características tuvieran escalas comparables, lo que facilitó el proceso de entrenamiento y predicción.
