from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from prometheus_client import start_http_server, Summary, Counter, Histogram, generate_latest

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Prometheus metrics
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Histogram of HTTP request durations', ['method', 'endpoint'])
REQUEST_COUNT = Counter('http_requests_total', 'Total number of HTTP requests', ['method', 'endpoint'])
ERROR_COUNT = Counter('http_errors_total', 'Total number of HTTP errors', ['endpoint', 'method'])

# Load the saved model
model_filename = 'model/xgboost_final_model.pkl'
model = joblib.load(model_filename)

# Load the vectorizer
vectorizer_filename = 'vectorizer/vectorizer.pkl'
vectorizer = joblib.load(vectorizer_filename)

def clean_doc(doc):
    # Split document into individual words
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # Remove punctuation from each word
    tokens = [re_punc.sub(' ', w) for w in tokens]
    # Remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # Lowercase all words
    tokens = [word.lower() for word in tokens]
    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # Word stemming
    lm = WordNetLemmatizer()
    tokens = [lm.lemmatize(word) for word in tokens]

    return " ".join(tokens)

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.labels(method='POST', endpoint='/predict').time()
def predict():
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    try:
        data = request.json
        description = data.get('description', '')  # Get the description from the request
        if not description:
            ERROR_COUNT.labels(endpoint='/predict', method='POST').inc()
            return jsonify({'error': 'Description is required'}), 400

        description = clean_doc(description)

        # Transform the description using the same vectorizer
        features = vectorizer.transform([description])

        # Make prediction
        prediction = model.predict(features)

        # Convert prediction to a standard Python float
        prediction = float(prediction[0])

        return jsonify({'price': prediction})
    except Exception as e:
        ERROR_COUNT.labels(endpoint='/predict', method='POST').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest()

if __name__ == '__main__':
    # Start the Prometheus metrics server on port 8000
    start_http_server(8000)
    app.run(host='0.0.0.0', port=8080)
