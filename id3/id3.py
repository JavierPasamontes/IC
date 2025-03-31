import math
import pandas as pd

# Leer los nombres de las columnas desde AtributosJuego.txt
atributos = []
with open("AtributosJuego.txt") as file:
    atributos = file.read().strip().split(',')  # Nos aseguramos de eliminar espacios

# Cargar los datos de Juego.txt con los nombres de las columnas de AtributosJuego.txt
data = pd.read_csv("Juego.txt", header=None, names=atributos)

# Limpiar espacios adicionales en las columnas y las filas
data.columns = data.columns.str.strip()  # Eliminar espacios de los nombres de las columnas
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # Eliminar espacios de los valores

# Verificar las primeras filas del DataFrame
print(data)

# Función para calcular la entropía de un conjunto de datos
def calcular_entropia(data):
    valores = data.value_counts(normalize=True)
    entropia = -sum(valores * valores.apply(lambda x: math.log2(x) if x > 0 else 0))
    return entropia

# Función para calcular la ganancia de información
def ganancia_informacion(data, atributo):
    # Entropía original del conjunto de datos
    entropia_original = calcular_entropia(data['Jugar'])
    
    # Dividir los datos por los valores del atributo
    valores_atributo = data[atributo].unique()
    entropia_condicional = 0
    for valor in valores_atributo:
        subset = data[data[atributo] == valor]
        probabilidad = len(subset) / len(data)
        entropia_condicional += probabilidad * calcular_entropia(subset['Jugar'])
    
    return entropia_original - entropia_condicional

# Función recursiva para construir el árbol de decisión
def id3(data, atributos):
    # Si todos los ejemplos tienen la misma clase, devolver esa clase
    if len(data['Jugar'].unique()) == 1:
        return data['Jugar'].iloc[0]
    
    # Si no hay más atributos para dividir, devolver la clase más frecuente
    if len(atributos) == 0:
        return data['Jugar'].mode()[0]
    
    # Calcular la ganancia de información para cada atributo
    ganancias = {atributo: ganancia_informacion(data, atributo) for atributo in atributos}
    
    # Elegir el atributo con la mayor ganancia de información
    mejor_atributo = max(ganancias, key=ganancias.get)
    
    # Crear un nodo para el árbol con el mejor atributo
    arbol = {mejor_atributo: {}}
    
    # Eliminar el mejor atributo de la lista de atributos
    atributos_restantes = [atributo for atributo in atributos if atributo != mejor_atributo]
    
    # Recursivamente dividir los datos para cada valor del mejor atributo
    for valor in data[mejor_atributo].unique():
        subset = data[data[mejor_atributo] == valor]
        arbol[mejor_atributo][valor] = id3(subset, atributos_restantes)
    
    return arbol

# Ejecutar ID3
arbol_decision = id3(data, atributos[:-1])  # Excluimos el atributo 'Jugar'
print(arbol_decision)
