import json
from collections import defaultdict
import re
import random
import pickle

def solve_text_completion(input_filename, candidate_filename, corpus_filename):
    """
    Solves a text completion problem by predicting missing words in the input text
    based on candidates and a given corpus.

    Parameters:
        input_filename (str): The filename of the input text with placeholders.
        candidate_filename (str): The filename containing candidate words.
        corpus_filename (str): The filename of the corpus used for prediction context.

    Returns:
        None
    """
    candidate_set = set()
    input_bigrams = set()
    results = []
    correct_order = []

    load_candidates(candidate_filename, candidate_set, correct_order)
    extract_input_bigrams(input_filename, input_bigrams)

    bigram_counts, trigram_counts = calculate_ngram_counts(corpus_filename, candidate_set, input_bigrams)
    # These dicts already in the coutns.pkl file.    
    # To use them and save time, use this code instead:

    # with open("counts.pkl", "rb") as f:
    # bigram_counts, trigram_counts = pickle.load(f) 
    
    predict_words(input_filename, candidate_set, bigram_counts, trigram_counts, results)
    correct_predictions, total_predictions = evaluate_predictions(results, correct_order)

    print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")

def load_candidates(filename, candidates_set, order_list):
    """
    Loads candidate words from a file into a set and list for further processing.

    Parameters:
        filename (str): The filename containing candidate words.
        candidates_set (set): A set to store unique candidate words.
        order_list (list): A list to maintain the original order of candidates.

    Returns:
        None
    """
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            for word in line.split():
                candidates_set.add(word)
                order_list.append(word)

def extract_input_bigrams(filename, bigrams_set):
    """
    Extracts bigrams from the input file and stores them in a set.

    Parameters:
        filename (str): The filename of the input text.
        bigrams_set (set): A set to store extracted bigrams.

    Returns:
        None
    """
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            for i in range(1, len(words)):
                if words[i] != "__________" and words[i - 1] != "__________":
                    bigram = (re.sub(r'[^a-zA-Z]', '', words[i - 1]).lower(),
                              re.sub(r'[^a-zA-Z]', '', words[i]).lower())
                    bigrams_set.add(bigram)

def calculate_ngram_counts(filename, candidates, input_bigrams):
    """
    Calculate counts of bigrams and trigrams from a corpus, filtered by candidate words and input bigrams.

    Parameters:
        filename (str): The filename of the corpus.
        candidates (set): A set of candidate words.
        input_bigrams (set): A set of bigrams extracted from the input.

    Returns:
        tuple: A tuple containing dictionaries of bigram and trigram counts.
    """
    bigram_counts = defaultdict(int)
    trigram_counts = defaultdict(int)

    with open(filename, 'r', encoding='utf-8') as file:
        print('Reading the text file...')
        w1, w2 = None, None  # Initialize context tokens

        for i, line in enumerate(file):
            for w3 in line.split():
                w3 = re.sub(r'[^a-zA-Z]', '', w3).lower()
                if w3 in candidates:
                    trigram_counts[(w1, w2, w3)] += 1
                if w2 in candidates:
                    trigram_counts[(w1, w2, w3)] += 1

                if w2:
                    if (w2, w3) in input_bigrams or w3 in candidates:
                        bigram_counts[(w2, w3)] += 1

                w1 = w2
                w2 = w3

            if (i + 1) % 100000 == 0:
                print(f'{(i + 1) / 100000}% Done.')

    with open("counts.pkl", "wb") as f:
        pickle.dump((bigram_counts, trigram_counts), f)

    return bigram_counts, trigram_counts

def predict_words(filename, candidates_set, bigram_counts, trigram_counts, results):
    """
    Predict missing words in the input file using calculated n-gram counts.

    Parameters:
        filename (str): The filename of the input text.
        candidates_set (set): A set of candidate words.
        bigram_counts (dict): A dictionary of bigram counts.
        trigram_counts (dict): A dictionary of trigram counts.
        results (list): A list to store the predicted words.

    Returns:
        None
    """
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            for i, word in enumerate(words):
                if word == "__________":
                    max_value = -1
                    top_candidate = None
                    for candidate in list(candidates_set):
                        if i < 2 or i + 1 >= len(words):
                            continue

                        w1, w2, w3 = words[i - 2], words[i - 1], words[i + 1]
                        denominator = bigram_counts.get((w1, w2), 0) * bigram_counts.get((w2, candidate), 0)

                        if denominator == 0:
                            continue

                        score = (trigram_counts.get((w1, w2, candidate), 0) * trigram_counts.get((w2, candidate, w3), 0)) / denominator

                        if score > max_value:
                            max_value = score
                            top_candidate = candidate

                    if top_candidate is None:
                        top_candidate = random.choice(list(candidates_set))

                    candidates_set.remove(top_candidate)
                    results.append(top_candidate)

def evaluate_predictions(predicted, true):
    """
    Evaluate the accuracy of predictions against the true order of words.

    Parameters:
        predicted (list): A list of predicted words.
        true (list): A list of the true order of words.

    Returns:
        tuple: A tuple containing the number of correct predictions and the total number of predictions.
    """
    correct_count = sum(1 for i, pred in enumerate(predicted) if pred == true[i])
    return correct_count, len(true)

def experiment(candidates_filename):
    """
    Conducts an experiment by generating multiple unique permutations of candidate words and evaluating the accuracy.

    Parameters:
        candidates_filename (str): The filename containing candidate words.

    Returns:
        float: The accuracy percentage of the random choice predictions.
    """
    true_order = []
    load_candidates(candidates_filename, set(), true_order)
    unique_permutations = set()

    while len(unique_permutations) < 100:
        shuffled = true_order[:]
        random.shuffle(shuffled)
        unique_permutations.add(tuple(shuffled))

    accuracy = sum(evaluate_predictions(perm, true_order)[0] / len(true_order) for perm in unique_permutations) / len(unique_permutations)

    return accuracy * 100

if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    experiment_results = experiment(config['candidates_filename'])
    print(f'{experiment_results}% Accuracy with random choice.')

    solve_text_completion(config['input_filename'], config['candidates_filename'], config['corpus'])
