from flask import Flask, render_template, request
import pickle

# Inicializar o aplicativo Flask
app = Flask(__name__)

# Carregar o modelo pickle
with open('nbModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Carregar o vetorizador
with open('vectorizerNB.pkl', 'rb') as f:
    vectorizerNB = pickle.load(f)

# Rota principal para renderizar a introdução HTML
@app.route('/')
def home():
    return render_template('index.html')

# Rota para acessar o formulário de predição
@app.route('/predict_form')
def predict_form():
    return render_template('predict.html')

# Rota para lidar com a predição
@app.route('/predict', methods=['POST'])
def predict():
    # Receber o texto da crítica do formulário
    review_text = request.form['review']
    
    # Vetorizar o texto da crítica
    review_vectorized = vectorizerNB.transform([review_text])
    
    # Fazer a predição com o modelo
    sentiment = model.predict(review_vectorized)
    

    
    # Renderizar o template com o resultado da predição
    return render_template('prediction.html', review=review_text, sentiment=sentiment)

# Rodar o aplicativo Flask
if __name__ == '__main__':
    app.run(debug=True)

