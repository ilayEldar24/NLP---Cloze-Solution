import json
from collections import defaultdict
import re
import random
import pickle


def solve_cloze(input_filename, candidate_filename, corpus_filename):
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

    # Load the candidates, into a candidate_set, and a list (to have the original order of it).
    load_candidates(candidate_filename, candidate_set, correct_order)

    # Extract all the pairs of words in the input file.
    extract_input_bigrams(input_filename, input_bigrams)

    # Calculate the relevant bigram coutns and trigram coutns.
    bigram_counts, trigram_counts = calculate_ngram_counts(corpus_filename, candidate_set, input_bigrams)

    # Solve the cloze using the counts and the trigram formula.
    predict_words(input_filename, candidate_set, bigram_counts, trigram_counts, results)

    # Compare the results to the original order.
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
                if w3 in candidates or w2 in candidates or w1 in candidates:
                    trigram_counts[(w1, w2, w3)] += 1

                if w2:
                    if (w2, w3) in input_bigrams or w3 in candidates or w2 in candidates:
                        bigram_counts[(w2, w3)] += 1

                w1 = w2
                w2 = w3

            if (i + 1) % 100000 == 0:
                print(f'{(i + 1) / 100000}% Done.')

    with open("counts2.pkl", "wb") as f:
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
    k = 0.001
    vocab_size = 50000
    kv = k * vocab_size

    with open(filename, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            words = line.split()
            for i, word in enumerate(words):
                if word == "__________":
                    maxScore = float('-inf')
                    top = None
                    for c in list(candidates_set):  # Use a copy of candidates_set
                        if i < 2 or i + 2 >= len(words):  # Check index validity
                            continue

                        # Extract context words
                        w1, w2, w3, w4 = words[i - 2], words[i - 1], words[i + 1], words[i + 2]

                        factor1 = (trigram_counts.get((w1, w2, c), 0) + k) / (bigram_counts.get((w1, w2), 0) + kv)
                        factor2 = (trigram_counts.get((w2, c, w3), 0) + k) / (bigram_counts.get((w2, c), 0) + kv)
                        factor3 = (trigram_counts.get((c, w3, w4), 0) + k) / (bigram_counts.get((c, w3), 0) + kv)

                        curScore = factor1 * factor2 * factor3

                        if curScore > maxScore:
                            maxScore = curScore
                            top = c

                    candidates_set.remove(top)
                    results.append(top)


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

    accuracy = sum(evaluate_predictions(perm, true_order)[0] / len(true_order) for perm in unique_permutations) / len(
        unique_permutations)

    return accuracy * 100


if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solve_cloze(config['input_filename'], config['candidates_filename'], config['corpus'])

    experiment_results = experiment(config['candidates_filename'])
