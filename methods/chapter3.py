import numpy as np
import sympy as sp
from sympy import symbols
import matplotlib.pyplot as plt
import io
import base64
from scipy.interpolate import interp1d, CubicSpline

class Chapter3Methods:
    """Métodos de interpolación"""
    
    def __init__(self):
        self.x = symbols('x')
    
    def vandermonde(self, x_data, y_data):
        """Interpolación usando matriz de Vandermonde"""
        n = len(x_data)
        V = np.vander(x_data, increasing=True)
        coeffs = np.linalg.solve(V, y_data)
        
        # Crear polinomio de SymPy
        poly = sum(coeffs[i] * self.x**i for i in range(n))
        
        return {
            'coefficients': coeffs.tolist(),
            'polynomial': str(sp.expand(poly)),
            'polynomial_latex': sp.latex(sp.expand(poly))
        }
    
    def newton_interpolating(self, x_data, y_data):
        """Interpolación usando diferencias divididas de Newton"""
        n = len(x_data)
        
        # Calcular diferencias divididas
        f = np.zeros((n, n))
        f[:, 0] = y_data
        
        for j in range(1, n):
            for i in range(n - j):
                f[i, j] = (f[i+1, j-1] - f[i, j-1]) / (x_data[i+j] - x_data[i])
        
        # Construir polinomio: P(x) = f[0,0] + f[0,1]*(x-x0) + f[0,2]*(x-x0)*(x-x1) + ...
        poly_expr = f[0, 0]
        
        for j in range(1, n):
            term = f[0, j]
            for i in range(j):
                term = term * (self.x - x_data[i])
            poly_expr = poly_expr + term
        
        poly_expr = sp.simplify(poly_expr)
        
        return {
            'divided_differences': f.tolist(),
            'polynomial': str(poly_expr),
            'polynomial_latex': sp.latex(poly_expr)
        }
    
    def lagrange(self, x_data, y_data):
        """Interpolación usando polinomios de Lagrange"""
        n = len(x_data)
        poly = 0
        
        for i in range(n):
            Li = 1
            for j in range(n):
                if i != j:
                    Li = Li * (self.x - x_data[j]) / (x_data[i] - x_data[j])
            poly = poly + y_data[i] * Li
        
        poly = sp.simplify(poly)
        
        return {
            'polynomial': str(poly),
            'polynomial_latex': sp.latex(poly)
        }
    
    def linear_spline(self, x_data, y_data):
        """Interpolación usando splines lineales"""
        spline = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Crear expresión por partes
        pieces = []
        for i in range(len(x_data) - 1):
            m = (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
            b = y_data[i] - m * x_data[i]
            pieces.append({
                'interval': (x_data[i], x_data[i+1]),
                'coefficient': m,
                'intercept': b,
                'equation': f'{m}*x + {b}'
            })
        
        return {
            'pieces': pieces,
            'polynomial': 'Spline lineal por partes'
        }
    
    def cubic_spline(self, x_data, y_data):
        """Interpolación usando splines cúbicos"""
        cs = CubicSpline(x_data, y_data, bc_type='natural')
        
        # Obtener coeficientes
        pieces = []
        for i in range(len(x_data) - 1):
            coeffs = cs.c[:, i]
            a, b, c, d = coeffs[3], coeffs[2], coeffs[1], coeffs[0]
            x0 = x_data[i]
            # Polinomio: a*(x-x0)^3 + b*(x-x0)^2 + c*(x-x0) + d
            pieces.append({
                'interval': (x_data[i], x_data[i+1]),
                'coefficients': [float(a), float(b), float(c), float(d)],
                'equation': f'{a:.6f}*(x-{x0})^3 + {b:.6f}*(x-{x0})^2 + {c:.6f}*(x-{x0}) + {d:.6f}'
            })
        
        return {
            'pieces': pieces,
            'polynomial': 'Spline cúbico por partes'
        }
    
    def plot_interpolation(self, method_name, result, x_data, y_data, x_eval=None):
        """Genera gráfico de la interpolación"""
        x_plot = np.linspace(min(x_data), max(x_data), 1000)
        
        if method_name == 'vandermonde':
            coeffs = result['coefficients']
            y_plot = np.polyval(coeffs[::-1], x_plot)
        
        elif method_name == 'newton_interpolating':
            poly_expr = sp.sympify(result['polynomial'])
            poly_func = sp.lambdify(self.x, poly_expr, 'numpy')
            y_plot = poly_func(x_plot)
        
        elif method_name == 'lagrange':
            poly_expr = sp.sympify(result['polynomial'])
            poly_func = sp.lambdify(self.x, poly_expr, 'numpy')
            y_plot = poly_func(x_plot)
        
        elif method_name == 'linear_spline':
            spline = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_plot = spline(x_plot)
        
        elif method_name == 'cubic_spline':
            cs = CubicSpline(x_data, y_data, bc_type='natural')
            y_plot = cs(x_plot)
        
        else:
            raise ValueError(f"Método desconocido: {method_name}")
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_data, y_data, 'ro', markersize=10, label='Puntos de datos')
        plt.plot(x_plot, y_plot, 'b-', label=f'Interpolación ({method_name})')
        
        if x_eval is not None:
            if method_name == 'linear_spline':
                spline = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
                y_eval = spline(x_eval)
            elif method_name == 'cubic_spline':
                cs = CubicSpline(x_data, y_data, bc_type='natural')
                y_eval = cs(x_eval)
            else:
                poly_expr = sp.sympify(result['polynomial'])
                poly_func = sp.lambdify(self.x, poly_expr, 'numpy')
                y_eval = poly_func(x_eval)
            plt.plot(x_eval, y_eval, 'go', markersize=8, label=f'Evaluación en x={x_eval}')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Interpolación usando {method_name}')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def calculate_error(self, method_name, result, x_data, y_data):
        """Calcula el error de interpolación en los puntos de datos"""
        errors = []
        
        for i, x in enumerate(x_data):
            if method_name == 'vandermonde':
                coeffs = result['coefficients']
                y_pred = np.polyval(coeffs[::-1], x)
            elif method_name == 'newton_interpolating':
                poly_expr = sp.sympify(result['polynomial'])
                poly_func = sp.lambdify(self.x, poly_expr, 'numpy')
                y_pred = poly_func(x)
            elif method_name == 'lagrange':
                poly_expr = sp.sympify(result['polynomial'])
                poly_func = sp.lambdify(self.x, poly_expr, 'numpy')
                y_pred = poly_func(x)
            elif method_name == 'linear_spline':
                spline = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
                y_pred = spline(x)
            elif method_name == 'cubic_spline':
                cs = CubicSpline(x_data, y_data, bc_type='natural')
                y_pred = cs(x)
            else:
                y_pred = y_data[i]
            
            abs_error = abs(y_pred - y_data[i])
            rel_error = abs_error / abs(y_data[i]) if y_data[i] != 0 else abs_error
            
            errors.append({
                'x': x,
                'y_true': y_data[i],
                'y_pred': float(y_pred),
                'absolute_error': float(abs_error),
                'relative_error': float(rel_error)
            })
        
        return errors
    
    def parse_data(self, x_str, y_str):
        """Parsea los datos de entrada"""
        x_data = np.array([float(x.strip()) for x in x_str.strip().split()])
        y_data = np.array([float(y.strip()) for y in y_str.strip().split()])
        
        if len(x_data) != len(y_data):
            raise ValueError("Los arreglos x e y deben tener la misma longitud")
        
        if len(x_data) > 8:
            raise ValueError("Máximo 8 puntos de datos permitidos")
        
        return x_data, y_data
    
    def execute_method(self, method_name, params):
        """Ejecuta un método específico"""
        x_str = params.get('x_data')
        y_str = params.get('y_data')
        
        x_data, y_data = self.parse_data(x_str, y_str)
        
        if method_name == 'vandermonde':
            result = self.vandermonde(x_data, y_data)
        
        elif method_name == 'newton_interpolating':
            result = self.newton_interpolating(x_data, y_data)
        
        elif method_name == 'lagrange':
            result = self.lagrange(x_data, y_data)
        
        elif method_name == 'linear_spline':
            result = self.linear_spline(x_data, y_data)
        
        elif method_name == 'cubic_spline':
            result = self.cubic_spline(x_data, y_data)
        
        else:
            raise ValueError(f"Método desconocido: {method_name}")
        
        # Calcular errores
        errors = self.calculate_error(method_name, result, x_data, y_data)
        result['errors'] = errors
        
        # Generar gráfico
        x_eval = params.get('x_eval')
        if x_eval:
            x_eval = float(x_eval)
        result['plot'] = self.plot_interpolation(method_name, result, x_data, y_data, x_eval)
        
        return result
    
    def compare_all_methods(self, params):
        """Compara todos los métodos con los mismos parámetros"""
        x_str = params.get('x_data')
        y_str = params.get('y_data')
        
        x_data, y_data = self.parse_data(x_str, y_str)
        
        comparison = {}
        
        methods = ['vandermonde', 'newton_interpolating', 'lagrange', 'linear_spline', 'cubic_spline']
        
        for method_name in methods:
            try:
                # Ejecutar método directamente sin generar gráfico
                if method_name == 'vandermonde':
                    result = self.vandermonde(x_data, y_data)
                elif method_name == 'newton_interpolating':
                    result = self.newton_interpolating(x_data, y_data)
                elif method_name == 'lagrange':
                    result = self.lagrange(x_data, y_data)
                elif method_name == 'linear_spline':
                    result = self.linear_spline(x_data, y_data)
                elif method_name == 'cubic_spline':
                    result = self.cubic_spline(x_data, y_data)
                
                # Calcular errores
                errors = self.calculate_error(method_name, result, x_data, y_data)
                
                mean_abs_error = np.mean([e['absolute_error'] for e in errors])
                mean_rel_error = np.mean([e['relative_error'] for e in errors])
                max_abs_error = np.max([e['absolute_error'] for e in errors])
                max_rel_error = np.max([e['relative_error'] for e in errors])
                
                comparison[method_name] = {
                    'polynomial': result['polynomial'],
                    'mean_absolute_error': float(mean_abs_error),
                    'mean_relative_error': float(mean_rel_error),
                    'max_absolute_error': float(max_abs_error),
                    'max_relative_error': float(max_rel_error)
                }
            except Exception as e:
                comparison[method_name] = {'error': str(e)}
        
        # Encontrar el mejor método
        best_method = None
        best_score = float('inf')
        
        for method, data in comparison.items():
            if 'error' not in data:
                # Score basado en errores promedio
                score = data['mean_absolute_error'] * 0.7 + data['mean_relative_error'] * 0.3
                if score < best_score:
                    best_score = score
                    best_method = method
        
        comparison['best_method'] = best_method
        
        return comparison

