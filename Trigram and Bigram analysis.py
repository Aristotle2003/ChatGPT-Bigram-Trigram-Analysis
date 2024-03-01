import re
from collections import defaultdict
import random


def clean_document(doc):
    """Keep only specified punctuation and convert to lower case."""
    cleaned = re.sub(r'[^\w\s,.?!]', '', doc).lower()
    return "<START> " + cleaned + " <END>"
# Function to generate n-grams
def generate_ngrams(doc, n=2):
    """Generate n-grams from a given document."""
    words = doc.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

# Calculate N-gram frequencies per class in the training corpus
def calculate_ngram_frequencies(data, n=2):
    """Calculate n-gram frequencies per class in the training corpus."""
    ngram_frequencies = {'human': defaultdict(int), 'gpt': defaultdict(int)}
    vocabulary = set()

    for doc, label in data:
        ngrams = generate_ngrams(doc, n)
        for ngram in ngrams:
            ngram_frequencies[label][ngram] += 1
            vocabulary.add(ngram)
    return ngram_frequencies, vocabulary
def calculate_oov_rate(test_data, frequencies, vocabulary, n=2):
    """Calculate the OOV rate for n-grams in the test set."""
    unseen_count = 0
    total_count = 0

    for doc, _ in test_data:
        ngrams = generate_ngrams(doc, n)
        total_count += len(ngrams)
        unseen_count += sum(1 for ngram in ngrams if ngram not in vocabulary)

    oov_rate = unseen_count / total_count * 100  # Convert to percentage
    return oov_rate

def load_data(hum_file, gpt_file):
    """Load and clean data from hum.txt and gpt.txt."""
    with open(hum_file, 'r', encoding='utf-8') as f:
        hum_docs = [clean_document(line.strip()) for line in f if line.strip()]

    with open(gpt_file, 'r', encoding='utf-8') as f:
        gpt_docs = [clean_document(line.strip()) for line in f if line.strip()]

    data = [(doc, 'human') for doc in hum_docs] + [(doc, 'gpt') for doc in gpt_docs]
    return data

def partition_data(data, test_ratio=0.1):
    """Partition the data into training and test sets without sklearn."""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def predict_with_bigrams(doc, bigram_frequencies):
    scores = {'human': 0, 'gpt': 0}
    bigrams = generate_ngrams(doc, 2)
    for ngram in bigrams:
        for label in scores.keys():
            scores[label] += bigram_frequencies[label].get(ngram, 0)
    return max(scores, key=scores.get)

def predict_with_trigrams(doc, trigram_frequencies):
    scores = {'human': 0, 'gpt': 0}
    trigrams = generate_ngrams(doc, 3)
    for ngram in trigrams:
        for label in scores.keys():
            scores[label] += trigram_frequencies[label].get(ngram, 0)
    return max(scores, key=scores.get)

def evaluate_model_accuracy(test_data, ngram_frequencies, ngram_type):
    correct_predictions = 0
    for doc, actual_label in test_data:
        if ngram_type == 'bigram':
            predicted_label = predict_with_bigrams(doc, ngram_frequencies)
        elif ngram_type == 'trigram':
            predicted_label = predict_with_trigrams(doc, ngram_frequencies)
        else:
            raise ValueError("ngram_type must be 'bigram' or 'trigram'")

        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data) * 100
    return accuracy

def augment_ngram_frequencies(ngram_frequencies):
    """Augment n-gram frequencies to ensure there's always at least one continuation."""
    for key in list(ngram_frequencies.keys()):
        if not any(ngram.startswith(key) for ngram in ngram_frequencies):
            # Assuming '<END>' as a safe default continuation
            ngram_frequencies[(key[-1], '<END>')] = 1
    return ngram_frequencies

def main():
    # Load and preprocess the data
    data = load_data('hum.txt', 'gpt.txt')

    # Partition the data into training and test sets
    train_data, test_data = partition_data(data, test_ratio=0.1)

    # Train bigram and trigram models
    bigram_frequencies, bigram_vocabulary = calculate_ngram_frequencies(train_data, 2)
    trigram_frequencies, trigram_vocabulary = calculate_ngram_frequencies(train_data, 3)

    # Calculate OOV rates
    bigram_oov_rate = calculate_oov_rate(test_data, bigram_frequencies, bigram_vocabulary, 2)
    trigram_oov_rate = calculate_oov_rate(test_data, trigram_frequencies, trigram_vocabulary, 3)

    print(f"Bigram OOV Rate: {bigram_oov_rate}%")
    print(f"Trigram OOV Rate: {trigram_oov_rate}%")

    bigram_accuracy = evaluate_model_accuracy(test_data, bigram_frequencies, 'bigram')
    print(f"Bigram Model Accuracy: {bigram_accuracy}%")

    trigram_accuracy = evaluate_model_accuracy(test_data, trigram_frequencies, 'trigram')
    print(f"Trigram Model Accuracy: {trigram_accuracy}%")



if __name__ == "__main__":
    main()
