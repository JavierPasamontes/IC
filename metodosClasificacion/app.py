import numpy as np
import pandas as pd
from scipy.stats import norm
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys
import os

class BayesClassifier:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.stds = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        for c in self.classes:
            self.priors[c] = np.mean(y == c)
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.stds[c] = np.std(X_c, axis=0)
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = 0
                
                for i in range(len(x)):
                    likelihood += np.log(norm.pdf(x[i], loc=self.means[c][i], scale=self.stds[c][i]))
                
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            pred = self.classes[np.argmax(posteriors)]
            predictions.append(pred)
        
        return predictions

class LloydClassifier:
    def __init__(self, tol=1e-10, max_iter=10, learning_rate=0.1):
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.centroids = None
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = {}
        
        # Inicialización de centroides con valores específicos del enunciado
        if len(self.classes) == 2:
            self.centroids[self.classes[0]] = np.array([4.6, 3.0, 4.0, 0.0])
            self.centroids[self.classes[1]] = np.array([6.8, 3.4, 4.6, 0.7])
        else:
            # Si hay más clases, inicializar con la media de cada clase
            for c in self.classes:
                self.centroids[c] = np.mean(X[y == c], axis=0)
        
        # Algoritmo de Lloyd
        for _ in range(self.max_iter):
            old_centroids = {k: v.copy() for k, v in self.centroids.items()}
            
            # Asignar cada punto al centroide más cercano
            distances = np.array([[np.linalg.norm(x - self.centroids[c]) for c in self.classes] for x in X])
            closest = np.argmin(distances, axis=1)
            
            # Actualizar centroides
            for i, c in enumerate(self.classes):
                points = X[closest == i]
                if len(points) > 0:
                    self.centroids[c] += self.learning_rate * (np.mean(points, axis=0) - self.centroids[c])
            
            # Verificar convergencia
            converged = True
            for c in self.classes:
                if np.linalg.norm(old_centroids[c] - self.centroids[c]) > self.tol:
                    converged = False
                    break
            if converged:
                break
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            distances = [np.linalg.norm(x - self.centroids[c]) for c in self.classes]
            pred = self.classes[np.argmin(distances)]
            predictions.append(pred)
        
        return predictions

class KMeansClassifier:
    def __init__(self, tol=0.01, max_iter=100, b=2):
        self.tol = tol
        self.max_iter = max_iter
        self.b = b  # Peso exponencial
        self.centroids = None
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = {}
        
        # Inicialización de centroides con valores específicos del enunciado
        if len(self.classes) == 2:
            self.centroids[self.classes[0]] = np.array([4.6, 3.0, 4.0, 0.0])
            self.centroids[self.classes[1]] = np.array([6.8, 3.4, 4.6, 0.7])
        else:
            # Si hay más clases, inicializar aleatoriamente
            for i, c in enumerate(self.classes):
                self.centroids[c] = X[np.random.choice(len(X))]
        
        # Algoritmo K-medias
        for _ in range(self.max_iter):
            old_centroids = {k: v.copy() for k, v in self.centroids.items()}
            
            # Calcular pertenencia borrosa (fuzzy membership)
            membership = np.zeros((len(X), len(self.classes)))
            
            for i, x in enumerate(X):
                for j, c in enumerate(self.classes):
                    dist = np.linalg.norm(x - self.centroids[c])
                    if dist == 0:
                        membership[i, j] = 1
                    else:
                        sum_terms = 0
                        for k in range(len(self.classes)):
                            sum_terms += (dist / np.linalg.norm(x - self.centroids[self.classes[k]])) ** (2/(self.b-1))
                        membership[i, j] = 1 / sum_terms
            
            # Actualizar centroides
            for j, c in enumerate(self.classes):
                numerator = np.sum([(membership[i, j]**self.b) * x for i, x in enumerate(X)], axis=0)
                denominator = np.sum([membership[i, j]**self.b for i in range(len(X))])
                self.centroids[c] = numerator / denominator
            
            # Verificar convergencia
            converged = True
            for c in self.classes:
                if np.linalg.norm(old_centroids[c] - self.centroids[c]) > self.tol:
                    converged = False
                    break
            if converged:
                break
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calcular pertenencia borrosa para el punto nuevo
            membership = np.zeros(len(self.classes))
            
            for j, c in enumerate(self.classes):
                dist = np.linalg.norm(x - self.centroids[c])
                if dist == 0:
                    membership[j] = 1
                else:
                    sum_terms = 0
                    for k in range(len(self.classes)):
                        sum_terms += (dist / np.linalg.norm(x - self.centroids[self.classes[k]])) ** (2/(self.b-1))
                    membership[j] = 1 / sum_terms
            
            pred = self.classes[np.argmax(membership)]
            predictions.append(pred)
        
        return predictions

def load_data(filename):
    try:
        data = []
        labels = []
        
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    features = list(map(float, parts[:-1]))
                    label = parts[-1]
                    data.append(features)
                    labels.append(label)
        
        return np.array(data), np.array(labels)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo: {str(e)}")
        return None, None

def load_test_example(filename):
    try:
        with open(filename, 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            features = list(map(float, parts[:-1]))
            true_label = parts[-1]
            return np.array([features]), true_label
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el ejemplo: {str(e)}")
        return None, None

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Iris - Bayes, Lloyd y K-medias")
        self.root.geometry("1000x700")
        
        # Cargar icono (compatible con PyInstaller)
        try:
            icon_path = resource_path("icon.ico")
            self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Variables
        self.X_train, self.y_train = None, None
        self.bayes_clf = BayesClassifier()
        self.lloyd_clf = LloydClassifier(tol=1e-10, max_iter=10, learning_rate=0.1)
        self.kmeans_clf = KMeansClassifier(tol=0.01, max_iter=100, b=2)
        
        # Crear interfaz
        self.create_widgets()
        
        # Cargar datos por defecto si el archivo existe
        default_data = resource_path("Iris2Clases.txt")
        if os.path.exists(default_data):
            self.load_training_data(default_data)
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Botón para cargar datos de entrenamiento
        ttk.Button(config_frame, text="Cargar Datos de Entrenamiento", 
                  command=self.load_training_dialog).pack(side=tk.LEFT, padx=5)
        
        # Frame de entrada
        input_frame = ttk.LabelFrame(main_frame, text="Entrada", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Campos para características
        ttk.Label(input_frame, text="Longitud del sépalo (cm):").grid(row=0, column=0, sticky=tk.W)
        self.sepal_length = ttk.Entry(input_frame)
        self.sepal_length.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Ancho del sépalo (cm):").grid(row=1, column=0, sticky=tk.W)
        self.sepal_width = ttk.Entry(input_frame)
        self.sepal_width.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Longitud del pétalo (cm):").grid(row=2, column=0, sticky=tk.W)
        self.petal_length = ttk.Entry(input_frame)
        self.petal_length.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Ancho del pétalo (cm):").grid(row=3, column=0, sticky=tk.W)
        self.petal_width = ttk.Entry(input_frame)
        self.petal_width.grid(row=3, column=1, padx=5, pady=2)
        
        # Botones
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=4, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Clasificar", command=self.classify).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cargar TestIris01", 
                  command=lambda: self.load_test_case(resource_path("TestIris01.txt"))).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cargar TestIris02", 
                  command=lambda: self.load_test_case(resource_path("TestIris02.txt"))).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cargar TestIris03", 
                  command=lambda: self.load_test_case(resource_path("TestIris03.txt"))).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Mostrar Gráfico", command=self.show_plot).pack(side=tk.LEFT, padx=5)
        
        # Frame de resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Resultados de los clasificadores
        ttk.Label(result_frame, text="Bayes:").grid(row=0, column=0, sticky=tk.W)
        self.bayes_result = ttk.Label(result_frame, text="", font=('Arial', 10, 'bold'))
        self.bayes_result.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(result_frame, text="Lloyd:").grid(row=1, column=0, sticky=tk.W)
        self.lloyd_result = ttk.Label(result_frame, text="", font=('Arial', 10, 'bold'))
        self.lloyd_result.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(result_frame, text="K-medias:").grid(row=2, column=0, sticky=tk.W)
        self.kmeans_result = ttk.Label(result_frame, text="", font=('Arial', 10, 'bold'))
        self.kmeans_result.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Texto para detalles
        self.details = tk.Text(result_frame, height=10, width=80, state=tk.DISABLED)
        self.details.grid(row=3, columnspan=2, pady=10)
        
        # Configurar scrollbar
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.details.yview)
        scrollbar.grid(row=3, column=2, sticky=tk.NS)
        self.details['yscrollcommand'] = scrollbar.set
    
    def load_training_dialog(self):
        filename = filedialog.askopenfilename(title="Seleccionar archivo de entrenamiento",
                                            filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if filename:
            self.load_training_data(filename)
    
    def load_training_data(self, filename):
        self.X_train, self.y_train = load_data(filename)
        if self.X_train is not None and self.y_train is not None:
            self.bayes_clf.fit(self.X_train, self.y_train)
            self.lloyd_clf.fit(self.X_train, self.y_train)
            self.kmeans_clf.fit(self.X_train, self.y_train)
            
            self.details.config(state=tk.NORMAL)
            self.details.delete(1.0, tk.END)
            self.details.insert(tk.END, f"Datos de entrenamiento cargados desde: {filename}\n")
            self.details.insert(tk.END, f"Número de muestras: {len(self.X_train)}\n")
            self.details.insert(tk.END, f"Clases encontradas: {', '.join(np.unique(self.y_train))}\n")
            self.details.config(state=tk.DISABLED)
            
            messagebox.showinfo("Éxito", "Datos de entrenamiento cargados y modelos entrenados correctamente")
    
    def load_test_case(self, filename):
        X_test, true_label = load_test_example(filename)
        if X_test is not None and true_label is not None:
            self.sepal_length.delete(0, tk.END)
            self.sepal_length.insert(0, str(X_test[0][0]))
            self.sepal_width.delete(0, tk.END)
            self.sepal_width.insert(0, str(X_test[0][1]))
            self.petal_length.delete(0, tk.END)
            self.petal_length.insert(0, str(X_test[0][2]))
            self.petal_width.delete(0, tk.END)
            self.petal_width.insert(0, str(X_test[0][3]))
            
            self.details.config(state=tk.NORMAL)
            self.details.delete(1.0, tk.END)
            self.details.insert(tk.END, f"Ejemplo cargado de {filename}\n")
            self.details.insert(tk.END, f"Clase verdadera: {true_label}\n")
            self.details.config(state=tk.DISABLED)
    
    def classify(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Primero cargue datos de entrenamiento")
            return
        
        try:
            # Obtener valores de entrada
            sl = float(self.sepal_length.get())
            sw = float(self.sepal_width.get())
            pl = float(self.petal_length.get())
            pw = float(self.petal_width.get())
            
            X_test = np.array([[sl, sw, pl, pw]])
            
            # Realizar predicciones
            bayes_pred = self.bayes_clf.predict(X_test)[0]
            lloyd_pred = self.lloyd_clf.predict(X_test)[0]
            kmeans_pred = self.kmeans_clf.predict(X_test)[0]
            
            # Mostrar resultados
            self.bayes_result.config(text=bayes_pred)
            self.lloyd_result.config(text=lloyd_pred)
            self.kmeans_result.config(text=kmeans_pred)
            
            # Mostrar detalles
            self.details.config(state=tk.NORMAL)
            self.details.delete(1.0, tk.END)
            self.details.insert(tk.END, f"Características: [{sl:.1f}, {sw:.1f}, {pl:.1f}, {pw:.1f}]\n\n")
            self.details.insert(tk.END, f"Bayes predice: {bayes_pred}\n")
            self.details.insert(tk.END, f"Lloyd predice: {lloyd_pred}\n")
            self.details.insert(tk.END, f"K-medias predice: {kmeans_pred}\n\n")
            
            # Mostrar centroides
            self.details.insert(tk.END, "Centroides finales:\n")
            for c in self.lloyd_clf.classes:
                self.details.insert(tk.END, f"Lloyd {c}: {np.round(self.lloyd_clf.centroids[c], 2)}\n")
                self.details.insert(tk.END, f"K-medias {c}: {np.round(self.kmeans_clf.centroids[c], 2)}\n")
            
            self.details.config(state=tk.DISABLED)
            
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos")
    
    def show_plot(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Primero cargue datos de entrenamiento")
            return
        
        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Longitud vs Ancho del pétalo
        for c in np.unique(self.y_train):
            X_c = self.X_train[self.y_train == c]
            ax1.scatter(X_c[:, 2], X_c[:, 3], label=c)
        
        # Graficar centroides de Lloyd
        for c in self.lloyd_clf.centroids:
            ax1.scatter(self.lloyd_clf.centroids[c][2], self.lloyd_clf.centroids[c][3], 
                       marker='x', s=200, linewidths=3, color='black', label=f'Lloyd {c}')
        
        ax1.set_xlabel('Longitud del pétalo (cm)')
        ax1.set_ylabel('Ancho del pétalo (cm)')
        ax1.set_title('Lloyd: Distribución y centroides')
        ax1.legend()
        
        # Gráfico 2: Longitud vs Ancho del sépalo
        for c in np.unique(self.y_train):
            X_c = self.X_train[self.y_train == c]
            ax2.scatter(X_c[:, 0], X_c[:, 1], label=c)
        
        # Graficar centroides de K-medias
        for c in self.kmeans_clf.centroids:
            ax2.scatter(self.kmeans_clf.centroids[c][0], self.kmeans_clf.centroids[c][1], 
                       marker='D', s=100, linewidths=2, color='red', label=f'K-medias {c}')
        
        ax2.set_xlabel('Longitud del sépalo (cm)')
        ax2.set_ylabel('Ancho del sépalo (cm)')
        ax2.set_title('K-medias: Distribución y centroides')
        ax2.legend()
        
        plt.tight_layout()
        
        # Mostrar en ventana
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Visualización de Clasificadores")
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()