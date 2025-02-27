#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install --user scikit-learn --upgrade')
import sklearn
print(sklearn.__version__)


# In[2]:


import numpy as np
import pandas as pd
import math

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from IPython.display import display


# In[3]:


df = pd.read_csv('/datasets/insurance_us.csv')


# In[4]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[5]:


df.sample(10)


# In[6]:


df.info()


# In[7]:


# puede que queramos cambiar el tipo de edad (de float a int) aunque esto no es crucial

# escribe tu conversión aquí si lo deseas:
df['age'] = df['age'].astype(int)


# In[8]:


# comprueba que la conversión se haya realizado con éxito
df.info()


# In[9]:


# ahora echa un vistazo a las estadísticas descriptivas de los datos.# ¿Se ve todo bien?
print(df.describe())


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

print("Valores únicos en 'gender':", df['gender'].unique())

# Punto 2: Visualizar la distribución de `income`
plt.figure(figsize=(10, 6))
sns.histplot(df['income'], kde=True)
plt.title('Distribución de Income')
plt.xlabel('Income')
plt.ylabel('Frecuencia')
plt.show()

# Punto 3: Contar valores en `family_members` para revisar si hay clientes con 0 familiares
print("Conteo de valores en 'family_members':")
print(df['family_members'].value_counts())

# Punto 4: Visualizar la distribución de `insurance_benefits` para observar el desbalance
plt.figure(figsize=(10, 6))
sns.countplot(x='insurance_benefits', data=df)
plt.title('Distribución de Insurance Benefits')
plt.xlabel('Insurance Benefits')
plt.ylabel('Frecuencia')
plt.show()


# In[11]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# In[12]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[13]:


def get_knn(df, n, k, metric):
    
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar    
    :param n: número de objetos para los que se buscan los vecinos más cercanos    
    :param k: número de vecinos más cercanos a devolver
    :param métrica: nombre de la métrica de distancia    """

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(df[feature_names].values)
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names].values], return_distance=True)
    
    # Creación de un DataFrame con los resultados
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# In[14]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[15]:


df_scaled.sample(5)


# In[16]:


# Ejemplo de uso: encontrar los 5 vecinos más cercanos al cliente en el índice 0 usando distancia Euclidiana en el conjunto escalado
n = 0  # Índice del cliente para el cual queremos encontrar vecinos
k = 5  # Número de vecinos más cercanos
metric = "euclidean"  # Métrica de distancia

# Obtener vecinos
result = get_knn(df_scaled, n, k, metric)
print(result)


# In[17]:


n = 0  # Índice del cliente para el cual queremos encontrar vecinos
k = 5  # Número de vecinos más cercanos
metric = "manhattan"  # Métrica de distancia

# Obtener vecinos usando distancia Manhattan
result = get_knn(df_scaled, n, k, metric)
print(result)


# In[18]:


# сalcula el objetivo
df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)


# In[19]:


# comprueba el desequilibrio de clases con value_counts()

# <tu código aquí>
print("Distribución de la variable objetivo (insurance_benefits_received):")
print(df['insurance_benefits_received'].value_counts())


# In[20]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# si tienes algún problema con la siguiente línea, reinicia el kernel y ejecuta el cuaderno de nuevo    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    cm = confusion_matrix(y_true, y_pred)
    print('Matriz de confusión')
    print(cm)


# In[21]:


# generar la salida de un modelo aleatorio

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[22]:


for P in [df['insurance_benefits_received'].mean(), 0.5, 1]:
    print(f'Probabilidad de predicción aleatoria: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, size=len(df['insurance_benefits_received']))
    
    # Evaluar el modelo aleatorio
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    print()


# In[23]:


# Definición de los datos y el objetivo
X = df[feature_names]
y = df['insurance_benefits_received']

# Divición de los datos en conjuntos de entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalamiento de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[24]:


# Evaluar kNN en datos sin escalar
print("Resultados con datos sin escalar:")
for k in range(1, 11):
    print(f"Modelo kNN con k={k}")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    eval_classifier(y_test, y_pred_knn)
    print()


# In[25]:


# Evaluar kNN en datos escalados
print("Resultados con datos escalados:")
for k in range(1, 11):
    print(f"Modelo kNN con k={k}")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn_scaled = knn.predict(X_test_scaled)
    eval_classifier(y_test, y_pred_knn_scaled)
    print()


# In[26]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y) # <tu código aquí>

    def predict(self, X):
        
        # añadir las unidades
        X2 = np.append(np.ones((len(X), 1)), X, axis=1) # <tu código aquí>
        y_pred = X2.dot(self.weights) # <tu código aquí>
        
        return y_pred


# In[27]:


def eval_regressor(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f}")
    
    r2 = r2_score(y_true, y_pred)
    print(f"R2: {r2:.2f}") 


# In[28]:


# Definir X e y
X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Crear el modelo de regresión lineal y entrenarlo
lr = MyLinearRegression()
lr.fit(X_train, y_train)
print("Pesos del modelo:", lr.weights)

# Predecir y evaluar en el conjunto de prueba
y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)

# Escalado de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo en datos escalados y evaluar
lr.fit(X_train_scaled, y_train)
y_test_pred_scaled = lr.predict(X_test_scaled)
print("\nEvaluación en datos escalados:")
eval_regressor(y_test, y_test_pred_scaled)


# In[29]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[30]:


X = df_pn.to_numpy()


# Generar una matriz aleatoria $P$.

# In[31]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# Comprobar que la matriz P sea invertible

# In[32]:


if np.linalg.det(P) == 0:
    print("La matriz P no es invertible. Genera una nueva matriz.")
else:
    print("La matriz P es invertible.")

    # Ocultar los datos 
X_ofuscado = X @ P


# In[33]:


# Calculo de la matriz inversa de P
P_inv = np.linalg.inv(P)

# Recuperación los datos originales de X_ofuscado
X_recuperado = X_ofuscado @ P_inv

# Mostrar los tres casos: Datos originales, transformados y recuperados
print("Datos originales:\n", X[:3])
print("Datos transformados (X'):\n", X_ofuscado[:3])
print("Datos recuperados:\n", X_recuperado[:3])


# In[34]:


np.random.seed(42)
P = np.random.rand(X.shape[1], X.shape[1])

# Comprobación de si P es invertible
if np.linalg.det(P) == 0:
    print("La matriz P no es invertible. Genera una nueva matriz.")
else:
    print("La matriz P es invertible.")


# In[35]:


X_ofuscado = X @ P


# In[36]:


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred))
r2_original = r2_score(y_test, y_pred)

# Aplicación de regresión lineal a los datos ofuscados
model.fit(X_train @ P, y_train) 
y_pred_ofuscado = model.predict(X_test @ P)
rmse_ofuscado = np.sqrt(mean_squared_error(y_test, y_pred_ofuscado))
r2_ofuscado = r2_score(y_test, y_pred_ofuscado)

# Resultados
print(f"Resultados con datos originales: RMSE = {rmse_original:.2f}, R2 = {r2_original:.2f}")
print(f"Resultados con datos ofuscados: RMSE = {rmse_ofuscado:.2f}, R2 = {r2_ofuscado:.2f}")

