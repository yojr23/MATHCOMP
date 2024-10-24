import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Funciones para los distintos modelos de regresión
def modelo_lineal(x, y):
    x = np.array(x).reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print(f"\nRegresión Lineal:")
    print(f"Coeficiente de determinación (R^2): {r_sq}")
    print(f"Pendiente: {model.coef_[0]}")
    print(f"Intersección: {model.intercept_}")
    
    plt.scatter(x, y, color="blue")
    plt.plot(x, model.predict(x), color="red")
    plt.title('Regresión Lineal')
    plt.show()

def modelo_exponencial(x, y):
    def exponencial(x, a, b):
        return a * np.exp(b * x)
    
    popt, pcov = curve_fit(exponencial, x, y)
    print(f"\nRegresión Exponencial:")
    print(f"Parámetros: a = {popt[0]}, b = {popt[1]}")
    
    plt.scatter(x, y, color="blue")
    plt.plot(x, exponencial(np.array(x), *popt), color="red")
    plt.title('Regresión Exponencial')
    plt.show()

def modelo_logaritmico(x, y):
    def logaritmico(x, a, b):
        return a + b * np.log(x)
    
    popt, pcov = curve_fit(logaritmico, x, y)
    print(f"\nRegresión Logarítmica:")
    print(f"Parámetros: a = {popt[0]}, b = {popt[1]}")
    
    plt.scatter(x, y, color="blue")
    plt.plot(x, logaritmico(np.array(x), *popt), color="red")
    plt.title('Regresión Logarítmica')
    plt.show()

def modelo_potencia(x, y):
    def potencia(x, a, b):
        return a * x**b
    
    popt, pcov = curve_fit(potencia, x, y)
    print(f"\nRegresión de Potencias:")
    print(f"Parámetros: a = {popt[0]}, b = {popt[1]}")
    
    plt.scatter(x, y, color="blue")
    plt.plot(x, potencia(np.array(x), *popt), color="red")
    plt.title('Regresión de Potencias')
    plt.show()

# Función principal
def cargar_datos():
    x = input("Ingrese los valores del eje x separados por comas: ")
    y = input("Ingrese los valores del eje y separados por comas: ")
    
    # Convertir las entradas a listas de flotantes
    x = list(map(float, x.split(',')))
    y = list(map(float, y.split(',')))
    
    if len(x) != len(y):
        print("Error: Las longitudes de los datos de x e y no coinciden.")
        return None, None
    
    return np.array(x), np.array(y)

def mostrar_menu():
    print("\nSeleccione el tipo de regresión que desea aplicar:")
    print("1. Regresión Lineal")
    print("2. Regresión Exponencial")
    print("3. Regresión Logarítmica")
    print("4. Regresión de Potencias")
    print("5. Salir")

def ejecutar_regresion():
    x, y = cargar_datos()
    
    if x is None or y is None:
        return
    
    while True:
        mostrar_menu()
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            modelo_lineal(x, y)
        elif opcion == "2":
            modelo_exponencial(x, y)
        elif opcion == "3":
            modelo_logaritmico(x, y)
        elif opcion == "4":
            modelo_potencia(x, y)
        elif opcion == "5":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

# Ejecutar el programa
ejecutar_regresion()
