import numpy as np
import sympy as sp
from sympy import sympify, lambdify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class Chapter1Methods:
    """Métodos de búsqueda de raíces"""
    
    def __init__(self):
        self.x = sp.Symbol('x')
    
    def parse_function(self, expression):
        """Convierte una expresión string en función de SymPy"""
        if not expression or not expression.strip():
            raise ValueError("La expresión de la función no puede estar vacía")
        try:
            expr = sympify(expression)
            f = lambdify(self.x, expr, 'numpy')
            # Verificar que la función se puede evaluar
            test_val = f(0.0)
            if not isinstance(test_val, (int, float, complex)) and not hasattr(test_val, '__float__'):
                raise ValueError("La función no se puede evaluar correctamente")
            return f
        except Exception as e:
            raise ValueError(f"Error al parsear la función '{expression}': {str(e)}")
    
    def calculate_derivative(self, expression):
        """Calcula la derivada de una función"""
        expr = sympify(expression)
        derivative = sp.diff(expr, self.x)
        return str(derivative)
    
    def bisection(self, f, a, b, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de bisección"""
        # Validar que a y b sean números válidos
        try:
            a = float(a)
            b = float(b)
        except (ValueError, TypeError):
            raise ValueError("Los valores de a y b deben ser números válidos")
        
        if a >= b:
            raise ValueError("El valor de 'a' debe ser menor que 'b'")
        
        # Evaluar la función en los extremos
        try:
            fa = float(f(a))
            fb = float(f(b))
        except Exception as e:
            raise ValueError(f"Error al evaluar la función en los extremos: {str(e)}")
        
        # Verificar valores especiales
        if np.isnan(fa) or np.isnan(fb):
            raise ValueError(f"La función devuelve NaN en los extremos. f({a}) = {fa}, f({b}) = {fb}")
        if np.isinf(fa) or np.isinf(fb):
            raise ValueError(f"La función devuelve infinito en los extremos. f({a}) = {fa}, f({b}) = {fb}")
        
        if fa * fb > 0:
            raise ValueError(f"No hay raíz en el intervalo [a, b]. f({a}) = {fa}, f({b}) = {fb}. Los valores deben tener signos opuestos.")
        
        iterations = []
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            
            if error_type == 'relative':
                error = abs((b - a) / (2 * c)) if c != 0 else abs(b - a)
            elif error_type == 'absolute':
                error = abs(b - a) / 2
            else:  # condition
                error = abs(fc)
            
            iterations.append({
                'iteration': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })
            
            if abs(fc) < tol or error < tol:
                break
            
            if fa * fc < 0:
                b = c
            else:
                a = c
        
        return {
            'root': c,
            'iterations': iterations,
            'converged': abs(fc) < tol or error < tol
        }
    
    def false_position(self, f, a, b, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de regla falsa"""
        # Validar que a y b sean números válidos
        try:
            a = float(a)
            b = float(b)
        except (ValueError, TypeError):
            raise ValueError("Los valores de a y b deben ser números válidos")
        
        if a >= b:
            raise ValueError("El valor de 'a' debe ser menor que 'b'")
        
        # Evaluar la función en los extremos
        try:
            fa = float(f(a))
            fb = float(f(b))
        except Exception as e:
            raise ValueError(f"Error al evaluar la función en los extremos: {str(e)}")
        
        # Verificar valores especiales
        if np.isnan(fa) or np.isnan(fb):
            raise ValueError(f"La función devuelve NaN en los extremos. f({a}) = {fa}, f({b}) = {fb}")
        if np.isinf(fa) or np.isinf(fb):
            raise ValueError(f"La función devuelve infinito en los extremos. f({a}) = {fa}, f({b}) = {fb}")
        
        if fa * fb > 0:
            raise ValueError(f"No hay raíz en el intervalo [a, b]. f({a}) = {fa}, f({b}) = {fb}. Los valores deben tener signos opuestos.")
        
        iterations = []
        c_prev = a
        
        for i in range(max_iter):
            if fb - fa == 0:
                raise ValueError("f(b) - f(a) es cero. Regla falsa no aplicable")

            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)

            if error_type == 'relative':
                error = abs((c - c_prev) / c) if c != 0 else abs(c - c_prev)
            elif error_type == 'absolute':
                error = abs(c - c_prev)
            else:
                error = abs(fc)
            
            iterations.append({
                'iteration': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })
            
            if abs(fc) < tol or error < tol:
                break
            
            if f(a) * fc < 0:
                b = c
            else:
                a = c
            
            c_prev = c
        
        return {
            'root': c,
            'iterations': iterations,
            'converged': abs(fc) < tol or error < tol
        }
    
    def fixed_point(self, g, x0, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de punto fijo"""
        # Validar x0
        try:
            x0 = float(x0)
        except (ValueError, TypeError):
            raise ValueError("El valor inicial x0 debe ser un número válido")
        
        iterations = []
        x_prev = x0
        
        # Límite para detectar overflow
        MAX_VALUE = 1e10
        MIN_VALUE = -1e10
        
        for i in range(max_iter):
            try:
                # Configurar numpy para capturar warnings de overflow como excepciones
                with np.errstate(all='raise'):
                    x = g(x_prev)
                    # Convertir a float y verificar overflow
                    x = float(x)
                    
                    if np.isnan(x) or np.isinf(x):
                        raise ValueError(f"La función g(x) devuelve un valor no válido en x={x_prev}: {x}")
                    
                    if abs(x) > MAX_VALUE:
                        raise ValueError(f"Overflow: el valor de x se volvió demasiado grande ({x}) en la iteración {i+1}. La función g(x) puede no converger.")
                    
                    if abs(x) < MIN_VALUE:
                        raise ValueError(f"Underflow: el valor de x se volvió demasiado pequeño ({x}) en la iteración {i+1}.")
                
            except (OverflowError, FloatingPointError) as e:
                error_msg = str(e)
                if "Result too large" in error_msg or "34" in error_msg or isinstance(e, FloatingPointError):
                    raise ValueError(f"Overflow: el resultado es demasiado grande en la iteración {i+1}. La función g(x) puede no converger con x0={x0}. Intenta con un valor inicial diferente o verifica que la función g(x) sea contractiva (|g'(x)| < 1).")
                raise ValueError(f"Overflow en la iteración {i+1}: {error_msg}. La función g(x) puede no converger con x0={x0}.")
            except ValueError as e:
                # Re-lanzar ValueError sin modificar
                raise
            except Exception as e:
                error_msg = str(e)
                # Verificar si es un error de numpy overflow
                if "Result too large" in error_msg or "34" in error_msg or "overflow" in error_msg.lower():
                    raise ValueError(f"Overflow: el resultado es demasiado grande en la iteración {i+1}. La función g(x) puede no converger con x0={x0}. Intenta con un valor inicial diferente o verifica que la función g(x) sea contractiva (|g'(x)| < 1).")
                raise ValueError(f"Error al evaluar g(x) en x={x_prev}: {error_msg}")
            
            # Calcular g(x) solo si es necesario para el error de condición
            if error_type == 'condition':
                try:
                    gx = float(g(x))
                    if np.isnan(gx) or np.isinf(gx):
                        gx = x  # Usar x como fallback
                except:
                    gx = x  # Usar x como fallback
            else:
                gx = x  # Para otros tipos de error, usar x directamente
            
            if error_type == 'relative':
                error = abs((x - x_prev) / x) if x != 0 else abs(x - x_prev)
            elif error_type == 'absolute':
                error = abs(x - x_prev)
            else:  # condition
                error = abs(gx - x)
            
            iterations.append({
                'iteration': i + 1,
                'x': x,
                'g(x)': gx,
                'error': error
            })
            
            if abs(x - x_prev) < tol or error < tol:
                break
            
            x_prev = x
        
        return {
            'root': x,
            'iterations': iterations,
            'converged': abs(x - x_prev) < tol or error < tol
        }
    
    def newton(self, f, df, x0, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de Newton-Raphson"""
        iterations = []
        x_prev = x0
        
        for i in range(max_iter):
            fx = f(x_prev)
            dfx = df(x_prev)
            
            if abs(dfx) < 1e-10:
                raise ValueError("Derivada muy cercana a cero")
            
            x = x_prev - fx / dfx
            
            if error_type == 'relative':
                error = abs((x - x_prev) / x) if x != 0 else abs(x - x_prev)
            elif error_type == 'absolute':
                error = abs(x - x_prev)
            else:  # condition
                error = abs(fx)
            
            iterations.append({
                'iteration': i + 1,
                'x': x,
                'f(x)': fx,
                'f\'(x)': dfx,
                'error': error
            })
            
            if abs(fx) < tol or error < tol:
                break
            
            x_prev = x
        
        return {
            'root': x,
            'iterations': iterations,
            'converged': abs(fx) < tol or error < tol
        }
    
    def secant(self, f, x0, x1, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de la secante"""
        iterations = []
        x_prev = x0
        x = x1
        
        for i in range(max_iter):
            fx_prev = f(x_prev)
            fx = f(x)
            
            if abs(fx - fx_prev) < 1e-10:
                raise ValueError("Diferencia de funciones muy cercana a cero")
            
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            
            if error_type == 'relative':
                error = abs((x_new - x) / x_new) if x_new != 0 else abs(x_new - x)
            elif error_type == 'absolute':
                error = abs(x_new - x)
            else:  # condition
                error = abs(fx)
            
            iterations.append({
                'iteration': i + 1,
                'x': x_new,
                'f(x)': fx,
                'error': error
            })
            
            if abs(fx) < tol or error < tol:
                break
            
            x_prev = x
            x = x_new
        
        return {
            'root': x,
            'iterations': iterations,
            'converged': abs(fx) < tol or error < tol
        }
    
    def multiple_roots_newton(self, f, df, d2f, x0, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de Newton para raíces múltiples"""
        iterations = []
        x_prev = x0
        
        for i in range(max_iter):
            fx = f(x_prev)
            dfx = df(x_prev)
            d2fx = d2f(x_prev)
            
            # Método de Newton modificado: x = x - (f*f')/((f')^2 - f*f'')
            denominator = dfx**2 - fx * d2fx
            
            if abs(denominator) < 1e-10:
                raise ValueError("Denominador muy cercana a cero")
            
            x = x_prev - (fx * dfx) / denominator
            
            if error_type == 'relative':
                error = abs((x - x_prev) / x) if x != 0 else abs(x - x_prev)
            elif error_type == 'absolute':
                error = abs(x - x_prev)
            else:  # condition
                error = abs(fx)
            
            iterations.append({
                'iteration': i + 1,
                'x': x,
                'f(x)': fx,
                'f\'(x)': dfx,
                'f\'\'(x)': d2fx,
                'error': error
            })
            
            if abs(fx) < tol or error < tol:
                break
            
            x_prev = x
        
        return {
            'root': x,
            'iterations': iterations,
            'converged': abs(fx) < tol or error < tol
        }
    
    def plot_function(self, f, root, a=None, b=None, iterations=None):
        """Genera gráfico de la función y la raíz encontrada"""
        if a is None:
            a = root - 2
        if b is None:
            b = root + 2
        
        x = np.linspace(a, b, 1000)
        y = f(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='f(x)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=root, color='r', linestyle='--', label=f'Raíz ≈ {root:.6f}')
        plt.plot(root, f(root), 'ro', markersize=10)
        
        if iterations:
            for it in iterations:
                if 'x' in it:
                    plt.plot(it['x'], f(it['x']), 'go', markersize=5, alpha=0.5)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Búsqueda de Raíz')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    
    def execute_method(self, method_name, params):
        """Ejecuta un método específico"""
        expression = params.get('expression')
        error_type = params.get('error_type', 'relative')
        tol = float(params.get('tol', 1e-6))
        max_iter = int(params.get('max_iter', 100))
        
        f = self.parse_function(expression)
        
        if method_name == 'bisection':
            if 'a' not in params or 'b' not in params:
                raise ValueError("Los parámetros 'a' y 'b' son requeridos para el método de bisección")
            a = params.get('a')
            b = params.get('b')
            if a is None or a == '' or b is None or b == '':
                raise ValueError("Los parámetros 'a' y 'b' no pueden estar vacíos")
            result = self.bisection(f, a, b, tol, max_iter, error_type)
            result['plot'] = self.plot_function(f, result['root'], float(a), float(b))
        
        elif method_name == 'false_position':
            if 'a' not in params or 'b' not in params:
                raise ValueError("Los parámetros 'a' y 'b' son requeridos para el método de regla falsa")
            a = params.get('a')
            b = params.get('b')
            if a is None or a == '' or b is None or b == '':
                raise ValueError("Los parámetros 'a' y 'b' no pueden estar vacíos")
            result = self.false_position(f, a, b, tol, max_iter, error_type)
            result['plot'] = self.plot_function(f, result['root'], float(a), float(b))
        
        elif method_name == 'fixed_point':
            g_expr = params.get('g_expression', expression)
            g = self.parse_function(g_expr)
            x0 = float(params['x0'])
            result = self.fixed_point(g, x0, tol, max_iter, error_type)
            result['plot'] = self.plot_function(f, result['root'])
        
        elif method_name == 'newton':
            df_expr = params.get('df_expression')
            if not df_expr:
                df_expr = self.calculate_derivative(expression)
            df = self.parse_function(df_expr)
            x0 = float(params['x0'])
            result = self.newton(f, df, x0, tol, max_iter, error_type)
            result['plot'] = self.plot_function(f, result['root'])
        
        elif method_name == 'secant':
            x0 = float(params['x0'])
            x1 = float(params['x1'])
            result = self.secant(f, x0, x1, tol, max_iter, error_type)
            result['plot'] = self.plot_function(f, result['root'])
        
        elif method_name == 'multiple_roots':
            df_expr = params.get('df_expression')
            d2f_expr = params.get('d2f_expression')
            if not df_expr:
                df_expr = self.calculate_derivative(expression)
            if not d2f_expr:
                d2f_expr = self.calculate_derivative(df_expr)
            df = self.parse_function(df_expr)
            d2f = self.parse_function(d2f_expr)
            x0 = float(params['x0'])
            result = self.multiple_roots_newton(f, df, d2f, x0, tol, max_iter, error_type)
            result['plot'] = self.plot_function(f, result['root'])
        
        else:
            raise ValueError(f"Método desconocido: {method_name}")
        
        return result
    
    def compare_all_methods(self, params):
        """Compara todos los métodos con los mismos parámetros"""
        expression = params.get('expression')
        error_type = params.get('error_type', 'relative')
        tol = float(params.get('tol', 1e-6))
        max_iter = int(params.get('max_iter', 100))
        
        comparison = {}
        
        # Métodos que requieren intervalo
        if 'a' in params and 'b' in params:
            a = float(params['a'])
            b = float(params['b'])
            f = self.parse_function(expression)
            
            try:
                result = self.bisection(f, a, b, tol, max_iter, error_type)
                comparison['bisection'] = {
                    'root': result['root'],
                    'iterations': len(result['iterations']),
                    'converged': result['converged'],
                    'final_error': result['iterations'][-1]['error'] if result['iterations'] else None
                }
            except Exception as e:
                comparison['bisection'] = {'error': str(e)}
            
            try:
                result = self.false_position(f, a, b, tol, max_iter, error_type)
                comparison['false_position'] = {
                    'root': result['root'],
                    'iterations': len(result['iterations']),
                    'converged': result['converged'],
                    'final_error': result['iterations'][-1]['error'] if result['iterations'] else None
                }
            except Exception as e:
                comparison['false_position'] = {'error': str(e)}
        
        # Métodos que requieren punto inicial
        if 'x0' in params:
            x0 = float(params['x0'])
            f = self.parse_function(expression)
            
            try:
                if 'df_expression' in params:
                    df_expr = params['df_expression']
                else:
                    df_expr = self.calculate_derivative(expression)
                df = self.parse_function(df_expr)
                result = self.newton(f, df, x0, tol, max_iter, error_type)
                comparison['newton'] = {
                    'root': result['root'],
                    'iterations': len(result['iterations']),
                    'converged': result['converged'],
                    'final_error': result['iterations'][-1]['error'] if result['iterations'] else None
                }
            except Exception as e:
                comparison['newton'] = {'error': str(e)}
            
            try:
                g_expr = params.get('g_expression', expression)
                g = self.parse_function(g_expr)
                result = self.fixed_point(g, x0, tol, max_iter, error_type)
                comparison['fixed_point'] = {
                    'root': result['root'],
                    'iterations': len(result['iterations']),
                    'converged': result['converged'],
                    'final_error': result['iterations'][-1]['error'] if result['iterations'] else None
                }
            except Exception as e:
                comparison['fixed_point'] = {'error': str(e)}
        
        # Método de la secante
        if 'x0' in params and 'x1' in params:
            x0 = float(params['x0'])
            x1 = float(params['x1'])
            f = self.parse_function(expression)
            
            try:
                result = self.secant(f, x0, x1, tol, max_iter, error_type)
                comparison['secant'] = {
                    'root': result['root'],
                    'iterations': len(result['iterations']),
                    'converged': result['converged'],
                    'final_error': result['iterations'][-1]['error'] if result['iterations'] else None
                }
            except Exception as e:
                comparison['secant'] = {'error': str(e)}
        
        # Encontrar el mejor método
        best_method = None
        best_score = float('inf')
        
        for method, data in comparison.items():
            if 'error' not in data and data['converged']:
                # Score basado en iteraciones y error final
                score = data['iterations'] * 0.5 + data['final_error'] * 1000
                if score < best_score:
                    best_score = score
                    best_method = method
        
        comparison['best_method'] = best_method
        
        return comparison

