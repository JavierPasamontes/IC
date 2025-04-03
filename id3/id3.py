import tkinter as tk
from tkinter import ttk
import pandas as pd
import math

# Leer los nombres de las columnas desde AtributosJuego.txt
atributos = []
with open("AtributosJuego.txt") as file:
    atributos = file.read().strip().split(',')

# Cargar los datos de Juego.txt con los nombres de las columnas de AtributosJuego.txt
data = pd.read_csv("Juego.txt", header=None, names=atributos)
data.columns = data.columns.str.strip()
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def calcular_entropia(data):
    valores = data.value_counts(normalize=True)
    return -sum(valores * valores.apply(lambda x: math.log2(x) if x > 0 else 0))

def ganancia_informacion(data, atributo):
    entropia_original = calcular_entropia(data['Jugar'])
    valores_atributo = data[atributo].unique()
    entropia_condicional = sum(
        (len(data[data[atributo] == valor]) / len(data)) * calcular_entropia(data[data[atributo] == valor]['Jugar'])
        for valor in valores_atributo
    )
    return entropia_original - entropia_condicional

def id3(data, atributos):
    if len(data['Jugar'].unique()) == 1:
        return data['Jugar'].iloc[0]
    if not atributos:
        return data['Jugar'].mode()[0]
    mejor_atributo = max(atributos, key=lambda attr: ganancia_informacion(data, attr))
    arbol = {mejor_atributo: {}}
    for valor in data[mejor_atributo].unique():
        subset = data[data[mejor_atributo] == valor]
        arbol[mejor_atributo][valor] = id3(subset, [a for a in atributos if a != mejor_atributo])
    return arbol

def predecir(arbol, valores):
    if not isinstance(arbol, dict):
        return arbol
    atributo = next(iter(arbol))
    valor = valores.get(atributo, None)
    return predecir(arbol[atributo].get(valor, 'Desconocido'), valores)

arbol_decision = id3(data, atributos[:-1])
print(arbol_decision)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Clasificador ID3")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

tk.Label(frame, text="Seleccione los atributos", font=("Arial", 14), bg="#f0f0f0").grid(row=0, column=0, columnspan=2, pady=10)

valores_usuario = {}

for i, atributo in enumerate(atributos[:-1]):
    tk.Label(frame, text=atributo + ":", font=("Arial", 12), bg="#f0f0f0").grid(row=i + 1, column=0, padx=10, pady=5, sticky="w")
    valores_usuario[atributo] = tk.StringVar()
    valores_usuario[atributo].set(data[atributo].unique()[0])
    menu = ttk.Combobox(frame, textvariable=valores_usuario[atributo], values=list(data[atributo].unique()), state="readonly")
    menu.grid(row=i + 1, column=1, padx=10, pady=5)

def obtener_resultado():
    seleccion = {atributo: valores_usuario[atributo].get() for atributo in atributos[:-1]}
    resultado = predecir(arbol_decision, seleccion)
    resultado_label.config(text=f"Resultado: {resultado}")

tk.Button(root, text="Predecir", command=obtener_resultado, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=20)

resultado_label = tk.Label(root, text="Resultado:", font=("Arial", 14), bg="#f0f0f0")
resultado_label.pack()

root.mainloop()
