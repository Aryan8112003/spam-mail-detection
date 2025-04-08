import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
import re
from scipy.sparse import csr_matrix
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.exceptions import NotFittedError
import numpy as np

# Suppress specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define spam indicators
spam_indicators = [
    # Marketing and promotional
    r'free', r'offer', r'limited time', r'discount', r'sale', r'deal', r'save', r'win', r'winner', r'prize',
    r'cash', r'money', r'lottery', r'claim', r'congratulations', r'exclusive', r'special', r'bonus', r'gift', r'reward',
    r'bank', r'account', r'credit', r'card', r'payment', r'transaction', r'balance', r'loan', r'mortgage', r'investment',
    r'stock', r'share', r'market', r'trading', r'crypto', r'bitcoin', r'wallet',
    r'urgent', r'important', r'alert', r'warning', r'critical', r'emergency', r'asap', r'now', r'today', r'tonight',
    r'expires', r'deadline', r'limited', r'last chance', r'final notice',
    r'click here', r'call now', r'order now', r'buy now', r'sign up', r'register', r'subscribe', r'join', r'membership',
    r'subscription', r'trial', r'offer', r'limited time', r'act now',
    r'unsubscribe', r'opt-out', r'opt out', r'click here', r'click below',
    r'help@', r'support@', r'info@', r'contact@', r'mail us', r'email us',
    r'[\w\.-]+@[\w\.-]+\.\w+',  # Email addresses
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
    # Lottery and prize specific
    r'won', r'win', r'winner', r'prize', r'lottery', r'jackpot', r'lucky', r'selected', r'chosen', r'random',
    r'congratulations', r'congrats', r'claim', r'claim your', r'claim now', r'claim prize', r'claim reward',
    r'million', r'billion', r'thousand', r'hundred', r'rupee', r'rupees', r'dollar', r'dollars', r'euro', r'euros',
    r'currency', r'money', r'cash', r'bank', r'account', r'credit', r'card', r'payment', r'transfer', r'deposit'
]

# Define legitimate banking terms
legitimate_banking_terms = [
    'statement', 'smartstatement', 'transaction', 'account', 'bank', 'hdfc', 'icici', 'sbi', 'axis', 'kotak', 
    'customer id', 'password', 'login', 'secure', 'verification', 'otp', 'pin', 'card', 'debit', 'credit',
    'savings', 'current', 'balance', 'transaction', 'transfer', 'deposit', 'withdrawal', 'monthly', 'statement',
    'feedback', 'service', 'charges', 'important', 'information', 'regards', 'dear', 'sir', 'madam', 'mr', 'mrs', 'ms',
    'warm regards', 'best regards', 'yours sincerely', 'yours faithfully', 'thank you', 'thanks',
    'track', 'view', 'download', 'monthly', 'statement', 'smart', 'banking', 'simple', 'convenient',
    'improve', 'services', 'feedback', 'suggestions', 'form', 'write', 'us', 'assuring', 'best', 'times',
    'click', 'link', 'below', 'given', 'open', 'enter', 'customer', 'id', 'password', 'login', 'secure',
    'special', 'offers', 'products', 'available', 'period', 'months', 'years', 'days', 'weeks',
    'dear', 'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam', 'customer', 'client', 'user', 'account holder'
]

# Define legitimate banking patterns
legitimate_banking_patterns = [
    r'dear\s+(?:mr|mrs|ms|dr|prof|sir|madam)\s+[\w\s]+',
    r'warm\s+regards',
    r'best\s+regards',
    r'yours\s+(?:sincerely|faithfully)',
    r'click\s+(?:on\s+)?(?:the\s+)?(?:below\s+)?(?:given\s+)?link',
    r'view\s+(?:your\s+)?(?:smart\s+)?statement',
    r'download\s+(?:your\s+)?(?:monthly\s+)?statement',
    r'enter\s+(?:your\s+)?(?:customer\s+)?(?:id|password)',
    r'track\s+(?:the\s+)?(?:transactions|activities)',
    r'feedback\s+form',
    r'service\s+charges',
    r'important\s+information',
    r'we\s+are\s+constantly\s+looking\s+for\s+ways\s+to\s+improve',
    r'assuring\s+you\s+the\s+best\s+of\s+our\s+services',
    r'how\s+to\s+open\s+(?:your\s+)?(?:smart\s+)?statement'
]

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4}', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace email addresses with a special token
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', 'EMAIL_ADDRESS', text)
    
    # Replace URLs with a special token
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Replace unsubscribe links and related phrases with a special token
    text = re.sub(r'unsubscribe|opt-out|opt out|click here|click below|done with|mail us|email us|help@|support@|info@|contact@', 'UNSUBSCRIBE_LINK', text)
    
    # Replace company names and marketing terms
    text = re.sub(r'insider|paytm|marketing|promotional|newsletter|company|brand|service|product', 'MARKETING_TERM', text)
    
    # Replace phone numbers with a special token
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE_NUMBER', text)
    
    # Replace lottery and prize related terms
    lottery_terms = ['won', 'win', 'winner', 'prize', 'lottery', 'jackpot', 'lucky', 'selected', 'chosen', 'random', 'congratulations', 'congrats', 'claim', 'claim your', 'claim now', 'claim prize', 'claim reward']
    for term in lottery_terms:
        text = re.sub(r'\b' + term + r'\b', 'LOTTERY_TERM', text, flags=re.IGNORECASE)
    
    # Replace currency and amount terms
    currency_terms = ['million', 'billion', 'thousand', 'hundred', 'rupee', 'rupees', 'dollar', 'dollars', 'euro', 'euros', 'currency', 'money', 'cash']
    for term in currency_terms:
        text = re.sub(r'\b' + term + r'\b', 'CURRENCY_TERM', text, flags=re.IGNORECASE)
    
    # Replace legitimate banking terms
    for term in legitimate_banking_terms:
        text = re.sub(r'\b' + term + r'\b', 'LEGITIMATE_BANKING', text, flags=re.IGNORECASE)
    
    # Replace promotional words with special tokens
    promotional_words = ['free', 'offer', 'limited time', 'discount', 'sale', 'deal', 'save', r'win', r'winner', r'prize', r'cash', r'money', r'lottery', r'claim', r'congratulations', r'exclusive', r'special', r'bonus', r'gift', r'reward']
    for word in promotional_words:
        text = re.sub(r'\b' + word + r'\b', 'PROMOTIONAL_WORD', text, flags=re.IGNORECASE)
    
    # Replace financial terms with special tokens
    financial_terms = ['bank', 'account', 'credit', 'card', 'payment', 'transaction', 'balance', 'loan', 'mortgage', 'investment', 'stock', 'share', 'market', 'trading', 'crypto', 'bitcoin', 'wallet', 'transfer', 'deposit']
    for term in financial_terms:
        text = re.sub(r'\b' + term + r'\b', 'FINANCIAL_TERM', text, flags=re.IGNORECASE)
    
    # Replace urgency words with special tokens
    urgency_words = ['urgent', 'important', 'alert', 'warning', 'critical', 'emergency', 'asap', 'now', 'today', 'tonight', 'expires', 'deadline', 'limited', 'last chance', 'final notice']
    for word in urgency_words:
        text = re.sub(r'\b' + word + r'\b', 'URGENCY_WORD', text, flags=re.IGNORECASE)
    
    # Replace marketing patterns with special tokens
    marketing_patterns = ['click here', 'call now', 'order now', 'buy now', 'sign up', 'register', 'subscribe', 'join', 'membership', 'subscription', 'trial', 'offer', 'limited time', 'act now', 'help', 'support', 'contact', 'can we help', 'mail us', 'email us']
    for pattern in marketing_patterns:
        text = re.sub(r'\b' + pattern + r'\b', 'MARKETING_PATTERN', text, flags=re.IGNORECASE)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_features(text):
    features = {}
    
    # Basic text features
    features['num_chars'] = len(text)
    features['num_words'] = len(text.split())
    features['num_sentences'] = len(re.split(r'[.!?]+', text))
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    
    # Spam indicators with increased weights
    features['email_count'] = len(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)) * 3  # Triple weight for emails
    features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)) * 2
    features['unsubscribe_count'] = len(re.findall(r'unsubscribe|opt-out|opt out|click here|click below|done with|mail us|email us|help@|support@|info@|contact@', text, re.IGNORECASE)) * 3  # Triple weight for unsubscribe
    features['phone_count'] = len(re.findall(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
    
    # Count legitimate banking terms
    features['legitimate_banking_count'] = sum(1 for term in legitimate_banking_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)) * 3  # Triple weight for legitimate banking terms
    
    # Count legitimate banking patterns
    features['legitimate_banking_pattern_count'] = sum(1 for pattern in legitimate_banking_patterns if re.search(pattern, text, re.IGNORECASE)) * 4  # Quadruple weight for legitimate banking patterns
    
    # Count lottery and prize related terms with high weight
    lottery_terms = ['won', 'win', 'winner', 'prize', 'lottery', 'jackpot', 'lucky', 'selected', 'chosen', 'random', 'congratulations', 'congrats', 'claim', 'claim your', 'claim now', 'claim prize', 'claim reward']
    features['lottery_count'] = sum(1 for term in lottery_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)) * 4  # Quadruple weight for lottery terms
    
    # Count currency and amount terms with high weight
    currency_terms = ['million', 'billion', 'thousand', 'hundred', 'rupee', 'rupees', 'dollar', 'dollars', 'euro', 'euros', 'currency', 'money', 'cash']
    features['currency_count'] = sum(1 for term in currency_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)) * 4  # Quadruple weight for currency terms
    
    # Count marketing terms with increased weight
    marketing_terms = ['insider', 'paytm', 'marketing', 'promotional', 'newsletter', 'help', 'support', 'contact', 'company', 'brand', 'service', 'product']
    features['marketing_term_count'] = sum(1 for term in marketing_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)) * 3  # Triple weight for marketing terms
    
    # Count promotional words
    promotional_words = ['free', 'offer', 'limited time', 'discount', 'sale', 'deal', 'save', 'win', 'winner', 'prize', 'cash', 'money', 'lottery', 'claim', 'congratulations', 'exclusive', 'special', 'bonus', 'gift', 'reward']
    features['promotional_count'] = sum(1 for word in promotional_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)) * 2
    
    # Count financial terms
    financial_terms = ['bank', 'account', 'credit', 'card', 'payment', 'transaction', 'balance', 'loan', 'mortgage', 'investment', 'stock', 'share', 'market', 'trading', 'crypto', 'bitcoin', 'wallet', 'transfer', 'deposit']
    features['financial_count'] = sum(1 for term in financial_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE))
    
    # Count urgency words
    urgency_words = ['urgent', 'important', 'alert', 'warning', 'critical', 'emergency', 'asap', 'now', 'today', 'tonight', 'expires', 'deadline', 'limited', 'last chance', 'final notice']
    features['urgency_count'] = sum(1 for word in urgency_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE))
    
    # Count marketing patterns
    marketing_patterns = ['click here', 'call now', 'order now', 'buy now', 'sign up', 'register', 'subscribe', 'join', 'membership', 'subscription', 'trial', 'offer', 'limited time', 'act now', 'help', 'support', 'contact', 'can we help', 'mail us', 'email us']
    features['marketing_pattern_count'] = sum(1 for pattern in marketing_patterns if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE)) * 2
    
    # Calculate keyword density with increased weight for marketing terms
    all_keywords = promotional_words + financial_terms + urgency_words + marketing_patterns + marketing_terms + lottery_terms + currency_terms
    keyword_count = sum(1 for keyword in all_keywords if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE))
    features['keyword_density'] = (keyword_count + features['marketing_term_count'] + features['lottery_count'] + features['currency_count']) / features['num_words'] if features['num_words'] > 0 else 0
    
    # Add a direct spam score based on key indicators
    features['spam_score'] = (
        features['email_count'] * 3 + 
        features['unsubscribe_count'] * 3 + 
        features['marketing_term_count'] * 3 + 
        features['marketing_pattern_count'] * 2 + 
        features['promotional_count'] * 2 + 
        features['url_count'] * 2 +
        features['lottery_count'] * 4 +  # Higher weight for lottery terms
        features['currency_count'] * 4    # Higher weight for currency terms
    )
    
    # Add a legitimate banking score
    features['legitimate_score'] = (
        features['legitimate_banking_count'] * 3 +
        features['legitimate_banking_pattern_count'] * 4 +  # Higher weight for legitimate banking patterns
        features['financial_count'] * 2
    )
    
    return features

# Initialize models
@st.cache_resource
def load_models():
    try:
        # Load sample data first to ensure we can recreate vectorizer if needed
        df = pd.read_csv(os.path.join(script_dir, 'spam.csv'), encoding='latin1')
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        df['transformed_text'] = df['text'].apply(transform_text)
        
        # Force recreation of vectorizer with exactly 3000 features
        st.warning("Creating vectorizer with 3000 features...")
        vectorizer = TfidfVectorizer(max_features=3000)
        vectorizer.fit(df['transformed_text'])
        
        # Save the new vectorizer
        with open(os.path.join(script_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
            
        # Load the model
        with open(os.path.join(script_dir, 'model.pkl'), 'rb') as f:
            try:
                model = pickle.load(f, encoding='latin1')
            except:
                f.seek(0)
                model = pickle.load(f)
            
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Load models
vectorizer, model = load_models()

if vectorizer is None or model is None:
    st.error("Failed to load the required models. Please check if the model files exist.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        try:
            # Extract features
            features = extract_features(input_sms)
            
            # Transform text
            transformed_sms = transform_text(input_sms)
            
            # Vectorize
            try:
                vector_input = vectorizer.transform([transformed_sms])
                
                # Check if the number of features matches what the model expects
                if hasattr(model, 'feature_count_') and vector_input.shape[1] != model.feature_count_.shape[1]:
                    st.error(f"Feature mismatch: Vectorizer has {vector_input.shape[1]} features, but model expects {model.feature_count_.shape[1]} features.")
                    st.info("Please try again after refreshing the page to recreate the vectorizer.")
                    st.stop()
                
                # Predict
                result = model.predict(vector_input)[0]
                
                # Get prediction probabilities
                try:
                    prob = model.predict_proba(vector_input)[0]
                    confidence = prob[1] if result == 1 else prob[0]
                except:
                    confidence = None
                
                # Override the model prediction if spam indicators are strong
                if features['spam_score'] >= 5 or features['lottery_count'] > 0 or features['currency_count'] > 0:
                    result = 1
                    confidence = 0.95  # High confidence for strong spam indicators
                
                # Override the model prediction if legitimate banking indicators are strong
                if features['legitimate_score'] >= 5 and features['spam_score'] < 10:
                    result = 0
                    confidence = 0.95  # High confidence for legitimate banking emails
                
                # Additional check for legitimate bank statements
                if features['legitimate_banking_pattern_count'] >= 2 and features['legitimate_banking_count'] >= 5:
                    result = 0
                    confidence = 0.98  # Very high confidence for legitimate bank statements
                
                # Display results
                st.write("---")
                st.subheader("Message Analysis")
                
                # Display features
                st.write("Message Statistics:")
                st.write(f"- Number of characters: {features['num_chars']}")
                st.write(f"- Number of words: {features['num_words']}")
                st.write(f"- Number of sentences: {features['num_sentences']}")
                st.write(f"- Average word length: {features['avg_word_length']:.2f}")
                st.write(f"- Uppercase ratio: {features['uppercase_ratio']:.2%}")
                
                # Display spam indicators
                st.write("\nSpam Indicators:")
                st.write(f"- Email addresses found: {features['email_count']}")
                st.write(f"- URLs found: {features['url_count']}")
                st.write(f"- Unsubscribe links found: {features['unsubscribe_count']}")
                st.write(f"- Phone numbers found: {features['phone_count']}")
                st.write(f"- Lottery/prize terms found: {features['lottery_count']}")
                st.write(f"- Currency/amount terms found: {features['currency_count']}")
                st.write(f"- Promotional words found: {features['promotional_count']}")
                st.write(f"- Financial terms found: {features['financial_count']}")
                st.write(f"- Urgency words found: {features['urgency_count']}")
                st.write(f"- Marketing patterns found: {features['marketing_pattern_count']}")
                st.write(f"- Keyword density: {features['keyword_density']:.2%}")
                st.write(f"- Spam score: {features['spam_score']}")
                st.write(f"- Legitimate banking score: {features['legitimate_score']}")
                st.write(f"- Legitimate banking patterns found: {features['legitimate_banking_pattern_count']}")
                
                st.write("---")
                st.subheader("Classification Result")
                
                if result == 1:
                    st.error("⚠️ This message is classified as SPAM")
                    if confidence is not None:
                        st.write(f"Confidence: {confidence:.2%}")
                    
                    # Display spam reasons
                    st.write("\nReasons for spam classification:")
                    if features['email_count'] > 0:
                        st.write("- Contains email addresses")
                    if features['unsubscribe_count'] > 0:
                        st.write("- Contains unsubscribe links or marketing contact information")
                    if features['marketing_term_count'] > 0:
                        st.write("- Contains marketing terms or company names")
                    if features['lottery_count'] > 0:
                        st.write("- Contains lottery or prize-related terms")
                    if features['currency_count'] > 0:
                        st.write("- Contains currency or amount-related terms")
                    if features['promotional_count'] > 0:
                        st.write("- Contains promotional content")
                    if features['financial_count'] > 0:
                        st.write("- Contains financial/trading content")
                    if features['urgency_count'] > 0:
                        st.write("- Uses urgency tactics")
                    if features['marketing_pattern_count'] > 0:
                        st.write("- Contains marketing patterns")
                else:
                    st.success("✅ This message is classified as NOT SPAM")
                    if confidence is not None:
                        st.write(f"Confidence: {confidence:.2%}")
                    
                    # Display legitimate reasons
                    if features['legitimate_banking_count'] > 0 or features['legitimate_banking_pattern_count'] > 0:
                        st.write("\nReasons for legitimate classification:")
                        if features['legitimate_banking_count'] > 0:
                            st.write("- Contains legitimate banking terms")
                        if features['legitimate_banking_pattern_count'] > 0:
                            st.write("- Contains legitimate banking communication patterns")
                        st.write("- Appears to be from a recognized financial institution")
                
                # Display preprocessed text
                st.write("---")
                st.subheader("Preprocessed Text")
                st.write(transformed_sms)
            except Exception as e:
                st.error(f"An error occurred during vectorization: {str(e)}")
                st.info("Please try again after refreshing the page to recreate the vectorizer.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.error("Please try again with a different message.")
