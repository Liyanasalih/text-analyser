
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def analyze_text(text):
    # Create TextBlob object
    blob = TextBlob(text)
    
    # Basic statistics
    word_count = len(word_tokenize(text))
    sentence_count = len(sent_tokenize(text))
    paragraph_count = len([p for p in text.split('\n') if p.strip()])
    
    # Sentiment analysis
    sentiment = blob.sentiment
    
    # Word frequency analysis
    words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    word_freq = {word: words.count(word) for word in set(words)}
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'sentiment': {
            'polarity': round(sentiment.polarity, 2),
            'subjectivity': round(sentiment.subjectivity, 2)
        },
        'top_words': dict(top_words)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    result = analyze_text(data['text'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
