import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el archivo CSV
file_path = 'taller_regresion_casas/SaratogaHouses.csv'  # Cambia la ruta del archivo
df = pd.read_csv(file_path)

# Convertir variables categóricas a numéricas usando one-hot encoding
df_encoded = pd.get_dummies(df, columns=['heating', 'fuel', 'sewer', 'waterfront', 'newConstruction', 'centralAir'], drop_first=True)

# Definir las variables independientes (X) y la variable dependiente (y)
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) y la raíz cuadrática media (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")

# Gráfico de valores reales
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Reales')
plt.title('Valores Reales del Precio de las Viviendas')
plt.xlabel('Índice')
plt.ylabel('Precio de la Vivienda')
plt.legend()
plt.show()

# Gráfico de valores predichos con la línea de regresión
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_pred, color='red', label='Valores Predichos')
plt.plot(range(len(y_test)), y_pred, color='green', label='Línea de Regresión')
plt.title('Valores Predichos y Línea de Regresión')
plt.xlabel('Índice')
plt.ylabel('Precio de la Vivienda')
plt.legend()
plt.show()



# Gráfico de puntos de valores reales y la línea de regresión
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Reales')
plt.plot(range(len(y_test)), y_pred, color='green', label='Línea de Regresión')
plt.title('Valores Reales del Precio de las Viviendas y Línea de Regresión')
plt.xlabel('Índice')
plt.ylabel('Precio de la Vivienda')
plt.legend()
plt.grid(True)
plt.show()



# Gráfico de dispersión: Valores Reales vs Valores Predichos
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='green', label='Valores Reales')
plt.scatter(range(len(y_test)), y_pred, color='blue', label='Valores Predichos')
plt.title('Valores Reales y Valores Predichos')
plt.xlabel('Índice')
plt.ylabel('Precio de la Vivienda')
plt.legend()
plt.grid(True)
plt.show()


# Crear una lista de características para graficar
features = ['livingArea', 'bedrooms', 'bathrooms', 'fireplaces', 'landValue', 'age']

# Graficar cada característica contra el precio
for feature in features:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[feature], y_test, color='green', label='Valores Reales')
    plt.scatter(X_test[feature], y_pred, color='blue', label='Valores Predichos')
    plt.title(f'Valores Reales y Predichos en función de {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Precio de la Vivienda')
    plt.legend()
    plt.grid(True)
    plt.show()
