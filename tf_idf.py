import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
# Download the 'punkt_tab' resource
nltk.download('punkt_tab') # This line is added to download the missing resource.
# Download the missing 'averaged_perceptron_tagger_eng' resource
nltk.download('averaged_perceptron_tagger_eng') # This line downloads the necessary resource.

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to map POS tags for better lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if not found

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Sample corpus inspired by motivational books
corpus = [
    "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "Opportunities multiply as they are seized.",
    "The only limit to our realization of tomorrow is our doubts of today.",
    "Don’t wish it were easier, wish you were better.",
    "Your time is limited, so don’t waste it living someone else’s life.",
    "Great minds discuss ideas; average minds discuss events; small minds discuss people.",
]

# Apply text preprocessing
preprocessed_corpus = [preprocess_text(doc) for doc in corpus]

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)

# Display results
print("TF-IDF Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())