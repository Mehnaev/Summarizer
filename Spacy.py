import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to generate summary
def generate_summary(text, per=0.3):
    # Process the text with spaCy
    doc = nlp(text)
    
    # List to store sentences
    sentences = [sent for sent in doc.sents]
    
    # Token list without stop words and punctuation
    words = [token.text for token in doc if token.text.lower() not in STOP_WORDS and token.text not in punctuation]
    
    # Frequency distribution of words
    word_freq = Counter(words)
    
    # Maximum frequency
    max_freq = max(word_freq.values())
    
    # Normalize word frequencies
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq
    
    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_freq[word.text.lower()]
                else:
                    sentence_scores[sent] += word_freq[word.text.lower()]
    
    # Select the top sentences as the summary
    num_sentences = int(len(sentences) * per)
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join([sent.text for sent in summary_sentences])
    
    return summary

# Function to extract key components
def extract_key_components(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage
if __name__ == "__main__":
    text = """
    Your long text goes here. This is an example paragraph that contains multiple sentences.
    SpaCy is an open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors, and more.
    """
   
    
    # Generate summary
    summary = generate_summary(text)
    print("Summary of the text:")
    print(summary)
    
    # Extract key components
    key_components = extract_key_components(text)
    print("\nKey Components :")
    for component in key_components:
        print(component)
