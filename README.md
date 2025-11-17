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

## Capítulos

### Capítulo 1: Búsqueda de Raíces

Métodos implementados:
- **Bisección**: Requiere intervalo [a, b]
- **Regla Falsa**: Requiere intervalo [a, b]
- **Punto Fijo**: Requiere valor inicial x₀ y función g(x)
- **Newton-Raphson**: Requiere valor inicial x₀ y derivada f'(x)
- **Secante**: Requiere dos valores iniciales x₀ y x₁
- **Raíces Múltiples (Newton)**: Requiere valor inicial x₀, primera y segunda derivada

### Capítulo 2: Sistemas Lineales Iterativos

Métodos implementados:
- **Jacobi**: Método iterativo básico
- **Gauss-Seidel**: Método iterativo mejorado
- **SOR (Successive Over-Relaxation)**: Método con parámetro de relajación ω

### Capítulo 3: Interpolación

Métodos implementados:
- **Vandermonde**: Interpolación polinomial usando matriz de Vandermonde
- **Newton Interpolante**: Interpolación usando diferencias divididas
- **Lagrange**: Interpolación usando polinomios de Lagrange
- **Spline Lineal**: Interpolación por partes con funciones lineales
- **Spline Cúbico**: Interpolación por partes con funciones cúbicas (natural)
