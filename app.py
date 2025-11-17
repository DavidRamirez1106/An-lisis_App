from flask import Flask, render_template, request, jsonify
from methods.chapter1 import Chapter1Methods
from methods.chapter2 import Chapter2Methods
from methods.chapter3 import Chapter3Methods
import json

app = Flask(__name__)

# Instancias de los métodos
chapter1 = Chapter1Methods()
chapter2 = Chapter2Methods()
chapter3 = Chapter3Methods()

@app.route('/')
def index():
    """Menú principal"""
    return render_template('index.html')

@app.route('/chapter1')
def chapter1_page():
    """Página del Capítulo 1: Búsqueda de raíces"""
    return render_template('chapter1.html')

@app.route('/chapter2')
def chapter2_page():
    """Página del Capítulo 2: Sistemas lineales"""
    return render_template('chapter2.html')

@app.route('/chapter3')
def chapter3_page():
    """Página del Capítulo 3: Interpolación"""
    return render_template('chapter3.html')

# Rutas API para Capítulo 1
@app.route('/api/chapter1/execute', methods=['POST'])
def chapter1_execute():
    data = request.json
    method = data.get('method')
    params = data.get('params', {})
    generate_report = params.get('generate_report', False) or data.get('generate_report', False)
    
    try:
        result = chapter1.execute_method(method, params)
        if generate_report:
            comparison = chapter1.compare_all_methods(params)
            result['comparison'] = comparison
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chapter1/derivative', methods=['POST'])
def chapter1_derivative():
    data = request.json
    expression = data.get('expression')
    
    try:
        derivative = chapter1.calculate_derivative(expression)
        return jsonify({'success': True, 'derivative': derivative})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Rutas API para Capítulo 2
@app.route('/api/chapter2/execute', methods=['POST'])
def chapter2_execute():
    data = request.json
    method = data.get('method')
    params = data.get('params', {})
    generate_report = params.get('generate_report', False) or data.get('generate_report', False)
    
    try:
        result = chapter2.execute_method(method, params)
        if generate_report:
            comparison = chapter2.compare_all_methods(params)
            result['comparison'] = comparison
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Rutas API para Capítulo 3
@app.route('/api/chapter3/execute', methods=['POST'])
def chapter3_execute():
    data = request.json
    method = data.get('method')
    params = data.get('params', {})
    generate_report = params.get('generate_report', False) or data.get('generate_report', False)
    
    try:
        result = chapter3.execute_method(method, params)
        if generate_report:
            comparison = chapter3.compare_all_methods(params)
            result['comparison'] = comparison
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

