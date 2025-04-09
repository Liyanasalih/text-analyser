from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download all required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading punkt tokenizer...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading stopwords...")
    nltk.download('stopwords')

app = Flask(__name__)

def analyze_text(text):
    try:
        logger.debug(f"Analyzing text: {text[:100]}...")  # Log first 100 chars of input
        
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Basic statistics
        word_count = len(word_tokenize(text))
        # Use NLTK's sentence tokenizer for more accurate sentence counting
        sentence_count = len(sent_tokenize(text))
        # Count paragraphs (split by newlines and filter out empty ones)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Sentiment analysis
        sentiment = blob.sentiment
        
        # Word frequency analysis
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 most common words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'sentiment': {
                'polarity': round(sentiment.polarity, 2),
                'subjectivity': round(sentiment.subjectivity, 2)
            },
            'top_words': dict(top_words)
        }
        
        logger.debug(f"Analysis result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}", exc_info=True)
        raise

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}", exc_info=True)
        return str(e), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        text = request.json.get('text', '')
        if not text:
            logger.error("No text provided in request")
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Received text for analysis: {text[:100]}...")  # Log first 100 chars
        results = analyze_text(text)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)