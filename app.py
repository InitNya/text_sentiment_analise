import joblib
import numpy as np
import re
from flask import Flask, request, jsonify
from nltk.corpus import stopwords

app = Flask(__name__)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    russian_stopwords = stopwords.words('russian')
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    
    return ' '.join(words).strip()

model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
preprocessor = preprocess_text
    
print("Модели успешно загружены!")
    
try:
    if hasattr(model, 'classes_'):
        classes = model.classes_
        if isinstance(classes[0], (int, np.integer)):
            class_names = ['Нейтральный','Позитивный','Негативный']
        else:
            class_names = [str(c) for c in classes]
    else:
        class_names = ['Нейтральный','Позитивный','Негативный']
except:
    class_names = ['Нейтральный','Позитивный','Негативный']

print(f"Классы модели: {class_names}")

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Анализатор тональности</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; max-width: 600px; margin: 40px auto; padding: 20px; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .negative { background: #ffcccc; }
            .neutral { background: #ffffcc; }
            .positive { background: #ccffcc; }
            .prob-bar { height: 20px; background: #ddd; margin: 5px 0; }
            .prob-fill { height: 100%; background: #007bff; }
        </style>
    </head>
    <body>
        <h1>Анализатор эмоциональной окраски текста</h1>
        <textarea id="textInput" placeholder="Введите текст для анализа..."></textarea>
        <br>
        <button onclick="analyze()">Анализировать</button>
        <div id="result"></div>
        
        <script>
            function analyze() {
                const text = document.getElementById('textInput').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<p>Анализируем...</p>';
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                })
                .then(r => r.json())
                .then(data => {
                    let html = `<div class="result ${data.sentiment.toLowerCase()}">`;
                    html += `<h3>Результат: ${data.sentiment}</h3>`;
                    html += '<h4>Вероятности:</h4>';
                    
                    for (const [label, prob] of Object.entries(data.probabilities)) {
                        html += `<div>${label}: ${prob}%</div>`;
                        html += `<div class="prob-bar"><div class="prob-fill" style="width: ${prob}%"></div></div>`;
                    }
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                })
                .catch(e => {
                    resultDiv.innerHTML = '<p style="color: red">Ошибка: ' + e + '</p>';
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Введите текст'}), 400
        
        processed_text = preprocessor(text)
        
        vectorized = vectorizer.transform([processed_text])
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized)[0]
        else:
            pred = model.predict(vectorized)[0]
            probabilities = [0, 0, 0]
            probabilities[int(pred)] = 1.0
        
        pred_idx = np.argmax(probabilities)
        
        if pred_idx < len(class_names):
            sentiment = class_names[pred_idx]
        else:
            sentiment = ['Нейтральный','Позитивный','Негативный'][pred_idx % 3]
        
        result = {
            'sentiment': sentiment,
            'probabilities': {}
        }
        
        for i, prob in enumerate(probabilities):
            label = class_names[i] if i < len(class_names) else f'Класс {i}'
            result['probabilities'][label] = round(float(prob) * 100, 2)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Откройте http://localhost:5000 в браузере")
    app.run(debug=True, port=5000)