import numpy as np
from scipy.linalg import eigvals

class Chapter2Methods:
    """Métodos iterativos para sistemas lineales"""
    
    def spectral_radius(self, A):
        """Calcula el radio espectral de una matriz"""
        eigenvalues = eigvals(A)
        return np.max(np.abs(eigenvalues))
    
    def check_convergence(self, method, A, b, w=None):
        """Verifica si un método puede converger según el radio espectral"""
        if method == 'jacobi':
            # Para Jacobi: A = D + L + U
            D = np.diag(np.diag(A))
            D_inv = np.linalg.inv(D)
            T = -D_inv @ (A - D)
            rho = self.spectral_radius(T)
        
        elif method == 'gauss_seidel':
            # Para Gauss-Seidel: A = D + L + U
            D = np.diag(np.diag(A))
            L = np.tril(A, -1)
            U = np.triu(A, 1)
            DL_inv = np.linalg.inv(D + L)
            T = -DL_inv @ U
            rho = self.spectral_radius(T)
        
        elif method == 'sor':
            if w is None:
                w = 1.0
            D = np.diag(np.diag(A))
            L = np.tril(A, -1)
            U = np.triu(A, 1)
            DL_inv = np.linalg.inv(D + w * L)
            T = DL_inv @ ((1 - w) * D - w * U)
            rho = self.spectral_radius(T)
        
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        return rho, rho < 1
    
    def jacobi(self, A, b, x0=None, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de Jacobi"""
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)
        
        D = np.diag(np.diag(A))
        D_inv = np.linalg.inv(D)
        T = -D_inv @ (A - D)
        c = D_inv @ b
        
        iterations = []
        x = x0.copy()
        
        for i in range(max_iter):
            x_new = T @ x + c
            
            if error_type == 'relative':
                error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new) if np.linalg.norm(x_new) > 0 else np.linalg.norm(x_new - x)
            elif error_type == 'absolute':
                error = np.linalg.norm(x_new - x)
            else:  # condition
                error = np.linalg.norm(A @ x_new - b)
            
            iterations.append({
                'iteration': i + 1,
                'x': x_new.tolist(),
                'error': float(error),
                'residual': float(np.linalg.norm(A @ x_new - b))
            })
            
            if error < tol:
                break
            
            x = x_new
        
        rho, can_converge = self.check_convergence('jacobi', A, b)
        
        return {
            'solution': x.tolist(),
            'iterations': iterations,
            'converged': bool(error < tol),
            'spectral_radius': float(rho),
            'can_converge': bool(can_converge)
        }
    
    def gauss_seidel(self, A, b, x0=None, tol=1e-6, max_iter=100, error_type='relative'):
        """Método de Gauss-Seidel"""
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)
        
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        DL_inv = np.linalg.inv(D + L)
        T = -DL_inv @ U
        c = DL_inv @ b
        
        iterations = []
        x = x0.copy()
        
        for i in range(max_iter):
            x_new = T @ x + c
            
            if error_type == 'relative':
                error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new) if np.linalg.norm(x_new) > 0 else np.linalg.norm(x_new - x)
            elif error_type == 'absolute':
                error = np.linalg.norm(x_new - x)
            else:  # condition
                error = np.linalg.norm(A @ x_new - b)
            
            iterations.append({
                'iteration': i + 1,
                'x': x_new.tolist(),
                'error': float(error),
                'residual': float(np.linalg.norm(A @ x_new - b))
            })
            
            if error < tol:
                break
            
            x = x_new
        
        rho, can_converge = self.check_convergence('gauss_seidel', A, b)
        
        return {
            'solution': x.tolist(),
            'iterations': iterations,
            'converged': bool(error < tol),
            'spectral_radius': float(rho),
            'can_converge': bool(can_converge)
        }
    
    def sor(self, A, b, w=1.0, x0=None, tol=1e-6, max_iter=100, error_type='relative'):
        """Método SOR (Successive Over-Relaxation)"""
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)
        
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        DL_inv = np.linalg.inv(D + w * L)
        T = DL_inv @ ((1 - w) * D - w * U)
        c = w * DL_inv @ b
        
        iterations = []
        x = x0.copy()
        
        for i in range(max_iter):
            x_new = T @ x + c
            
            if error_type == 'relative':
                error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new) if np.linalg.norm(x_new) > 0 else np.linalg.norm(x_new - x)
            elif error_type == 'absolute':
                error = np.linalg.norm(x_new - x)
            else:  # condition
                error = np.linalg.norm(A @ x_new - b)
            
            iterations.append({
                'iteration': i + 1,
                'x': x_new.tolist(),
                'error': float(error),
                'residual': float(np.linalg.norm(A @ x_new - b))
            })
            
            if error < tol:
                break
            
            x = x_new
        
        rho, can_converge = self.check_convergence('sor', A, b, w)
        
        return {
            'solution': x.tolist(),
            'iterations': iterations,
            'converged': bool(error < tol),
            'spectral_radius': float(rho),
            'can_converge': bool(can_converge)
        }
    
    def parse_matrix(self, matrix_str):
        """Parsea una matriz desde string"""
        rows = matrix_str.strip().split('\n')
        matrix = []
        for row in rows:
            row = row.strip()
            if row:
                values = [float(x.strip()) for x in row.split()]
                matrix.append(values)
        return np.array(matrix)
    
    def execute_method(self, method_name, params):
        """Ejecuta un método específico"""
        A_str = params.get('A')
        b_str = params.get('b')
        error_type = params.get('error_type', 'relative')
        tol = float(params.get('tol', 1e-6))
        max_iter = int(params.get('max_iter', 100))
        
        A = self.parse_matrix(A_str)
        b = np.array([float(x.strip()) for x in b_str.strip().split()])
        
        x0_str = params.get('x0')
        x0 = None
        if x0_str:
            x0 = np.array([float(x.strip()) for x in x0_str.strip().split()])
        
        if method_name == 'jacobi':
            result = self.jacobi(A, b, x0, tol, max_iter, error_type)
        
        elif method_name == 'gauss_seidel':
            result = self.gauss_seidel(A, b, x0, tol, max_iter, error_type)
        
        elif method_name == 'sor':
            w = float(params.get('w', 1.0))
            result = self.sor(A, b, w, x0, tol, max_iter, error_type)
        
        else:
            raise ValueError(f"Método desconocido: {method_name}")
        
        return result
    
    def compare_all_methods(self, params):
        """Compara todos los métodos con los mismos parámetros"""
        A_str = params.get('A')
        b_str = params.get('b')
        error_type = params.get('error_type', 'relative')
        tol = float(params.get('tol', 1e-6))
        max_iter = int(params.get('max_iter', 100))
        
        A = self.parse_matrix(A_str)
        b = np.array([float(x.strip()) for x in b_str.strip().split()])
        
        x0_str = params.get('x0')
        x0 = None
        if x0_str:
            x0 = np.array([float(x.strip()) for x in x0_str.strip().split()])
        
        comparison = {}
        
        # Jacobi
        try:
            result = self.jacobi(A, b, x0, tol, max_iter, error_type)
            comparison['jacobi'] = {
                'solution': result['solution'],
                'iterations': len(result['iterations']),
                'converged': bool(result['converged']),
                'spectral_radius': result['spectral_radius'],
                'can_converge': bool(result['can_converge']),
                'final_error': result['iterations'][-1]['error'] if result['iterations'] else None,
                'final_residual': result['iterations'][-1]['residual'] if result['iterations'] else None
            }
        except Exception as e:
            comparison['jacobi'] = {'error': str(e)}
        
        # Gauss-Seidel
        try:
            result = self.gauss_seidel(A, b, x0, tol, max_iter, error_type)
            comparison['gauss_seidel'] = {
                'solution': result['solution'],
                'iterations': len(result['iterations']),
                'converged': bool(result['converged']),
                'spectral_radius': result['spectral_radius'],
                'can_converge': bool(result['can_converge']),
                'final_error': result['iterations'][-1]['error'] if result['iterations'] else None,
                'final_residual': result['iterations'][-1]['residual'] if result['iterations'] else None
            }
        except Exception as e:
            comparison['gauss_seidel'] = {'error': str(e)}
        
        # SOR con diferentes valores de w
        w_values = [0.5, 1.0, 1.5]
        for w in w_values:
            try:
                result = self.sor(A, b, w, x0, tol, max_iter, error_type)
                comparison[f'sor_w{w}'] = {
                    'solution': result['solution'],
                    'iterations': len(result['iterations']),
                    'converged': bool(result['converged']),
                    'spectral_radius': result['spectral_radius'],
                    'can_converge': bool(result['can_converge']),
                    'w': w,
                    'final_error': result['iterations'][-1]['error'] if result['iterations'] else None,
                    'final_residual': result['iterations'][-1]['residual'] if result['iterations'] else None
                }
            except Exception as e:
                comparison[f'sor_w{w}'] = {'error': str(e)}
        
        # Encontrar el mejor método
        best_method = None
        best_score = float('inf')
        
        for method, data in comparison.items():
            if 'error' not in data and data['converged'] and data['can_converge']:
                # Score basado en iteraciones, error final y residual
                score = data['iterations'] * 0.5 + data['final_error'] * 1000 + data['final_residual'] * 0.1
                if score < best_score:
                    best_score = score
                    best_method = method
        
        comparison['best_method'] = best_method
        
        return comparison

