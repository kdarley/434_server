from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model
model_filename = 'model/xgboost_final_model.pkl'
model = joblib.load(model_filename)

# Load the vectorizer
vectorizer_filename = 'vectorizer/vectorizer.pkl'
vectorizer = joblib.load(vectorizer_filename)

def clean_doc(doc):
    #split document into individual words
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub(' ', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # word stemming
    lm = WordNetLemmatizer()
    tokens = [lm.lemmatize(word) for word in tokens]

    return " ".join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = data.get('description', '')  # Get the description from the request
    if not description:
        return jsonify({'error': 'Description is required'}), 400
    
    description = clean_doc(description)

    # Transform the description using the same vectorizer
    features = vectorizer.transform([description])
    
    # Make prediction
    prediction = model.predict(features)

    # Convert prediction to a standard Python float
    prediction = float(prediction[0])

    return jsonify({'price': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
