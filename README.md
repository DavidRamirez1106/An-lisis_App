# Aplicación de Métodos Numéricos

Aplicación web desarrollada en Flask para resolver problemas de métodos numéricos en tres capítulos principales.

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución

1. Ejecuta la aplicación:
```bash
python app.py
```

2. Abre tu navegador y ve a:
```
http://localhost:5000
```

## Estructura del Proyecto

```
AppAnálisis/
├── app.py                 # Aplicación principal de Flask
├── methods/               # Módulos con los métodos numéricos
│   ├── __init__.py
│   ├── chapter1.py        # Capítulo 1: Búsqueda de raíces
│   ├── chapter2.py        # Capítulo 2: Sistemas lineales iterativos
│   └── chapter3.py        # Capítulo 3: Interpolación
├── templates/             # Plantillas HTML
│   ├── base.html
│   ├── index.html         # Menú principal
│   ├── chapter1.html       # Interfaz del Capítulo 1
│   ├── chapter2.html       # Interfaz del Capítulo 2
│   └── chapter3.html       # Interfaz del Capítulo 3
├── static/                # Archivos estáticos
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

---

## 📚 Capítulos

### Capítulo 1: Búsqueda de Raíces

**Archivo:** `methods/chapter1.py`

Métodos implementados:
- **Bisección**: Requiere intervalo [a, b]
- **Regla Falsa**: Requiere intervalo [a, b]
- **Punto Fijo**: Requiere valor inicial x₀ y función g(x)
- **Newton-Raphson**: Requiere valor inicial x₀ y derivada f'(x)
- **Secante**: Requiere dos valores iniciales x₀ y x₁
- **Raíces Múltiples (Newton)**: Requiere valor inicial x₀, primera y segunda derivada

**Ejemplos de funciones:**
- `x**2 - 4`
- `sin(x) - x`
- `exp(x) - 2*x`
- `x**3 - 2*x - 5`

**Características:**
- Cálculo automático de derivadas usando SymPy
- Gráficas de la función y raíz encontrada
- Tabla de iteraciones detallada
- Informe comparativo opcional entre todos los métodos
- Validación de parámetros y manejo de errores mejorado

**Clase principal:** `Chapter1Methods`

---

### Capítulo 2: Sistemas Lineales Iterativos

**Archivo:** `methods/chapter2.py`

Métodos implementados:
- **Jacobi**: Método iterativo básico
- **Gauss-Seidel**: Método iterativo mejorado
- **SOR (Successive Over-Relaxation)**: Método con parámetro de relajación ω

**Formato de entrada:**
- Matriz A: Una fila por línea, valores separados por espacios
- Vector b: Valores separados por espacios
- Vector inicial x₀ (opcional): Valores separados por espacios

**Ejemplo de matriz 3x3:**
```
4  -1   0
-1   4  -1
 0  -1   4
```

**Ejemplo de vector b:**
```
3 2 3
```

**Características:**
- Cálculo del radio espectral para análisis de convergencia
- Verificación de convergencia basada en el radio espectral
- Tabla de iteraciones con error y residual
- Informe comparativo opcional entre todos los métodos
- Soporte para diferentes tipos de error (relativo, absoluto, condición)

**Clase principal:** `Chapter2Methods`

---

### Capítulo 3: Interpolación

**Archivo:** `methods/chapter3.py`

Métodos implementados:
- **Vandermonde**: Interpolación polinomial usando matriz de Vandermonde
- **Newton Interpolante**: Interpolación usando diferencias divididas
- **Lagrange**: Interpolación usando polinomios de Lagrange
- **Spline Lineal**: Interpolación por partes con funciones lineales
- **Spline Cúbico**: Interpolación por partes con funciones cúbicas (natural)

**Formato de entrada:**
- Valores de x: Separados por espacios (máximo 8 valores)
- Valores de y: Separados por espacios (mismo número que x)
- Punto de evaluación (opcional): Valor único para evaluar el polinomio

**Ejemplo:**
- x: `0 1 2 3`
- y: `1 4 9 16`
- Evaluar en: `1.5`

**Características:**
- Polinomio interpolado mostrado en formato texto y LaTeX
- Gráfica de la interpolación con puntos de datos
- Cálculo de errores absolutos y relativos en puntos de datos
- Tabla de diferencias divididas (método de Newton)
- Informe comparativo opcional entre todos los métodos

**Clase principal:** `Chapter3Methods`

---

## 🔧 Tecnologías Utilizadas

- **Flask**: Framework web
- **SymPy**: Manipulación simbólica y cálculo de derivadas
- **NumPy**: Operaciones numéricas y álgebra lineal
- **SciPy**: Métodos avanzados (interpolación, álgebra lineal)
- **Matplotlib**: Generación de gráficas
- **HTML/CSS/JavaScript**: Interfaz de usuario

---

## 📝 Notas Importantes

- La aplicación usa SymPy para manipulación simbólica y cálculo automático de derivadas
- Los gráficos se generan usando Matplotlib y se codifican en base64 para mostrar en el navegador
- Los informes comparativos se generan automáticamente si se selecciona la opción correspondiente
- La aplicación identifica el mejor método según los criterios de convergencia y errores
- Todos los valores booleanos se convierten a tipos nativos de Python para compatibilidad JSON
- Se incluye validación robusta de parámetros y manejo de errores en todos los métodos

---

## 📖 Ejemplos de Uso

### Capítulo 1: Búsqueda de Raíces

**Ejemplo - Método de Bisección:**
- Función: `x**2 - 4`
- a: `1`
- b: `3`
- Tolerancia: `1e-6`
- Tipo de error: `relative`

**Ejemplo - Método de Newton-Raphson:**
- Función: `x**3 - 2*x - 5`
- x₀: `2`
- Derivada: (se calcula automáticamente)
- Tolerancia: `1e-6`

### Capítulo 2: Sistemas Lineales

**Ejemplo - Método de Jacobi:**
- Matriz A:
  ```
  4  -1  0
  -1  4  -1
  0  -1  4
  ```
- Vector b: `3 2 3`
- Tolerancia: `1e-6`
- Tipo de error: `relative`

### Capítulo 3: Interpolación

**Ejemplo - Método de Lagrange:**
- Valores x: `0 1 2 3`
- Valores y: `1 4 9 16`
- Evaluar en: `1.5`

---

## 🚀 Contribuciones

Este proyecto está organizado por capítulos para facilitar el mantenimiento y la comprensión del código. Cada capítulo contiene métodos numéricos relacionados y está completamente documentado.

---

## 📄 Licencia

Este proyecto es de uso educativo y académico.
