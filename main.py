import json
from collections import defaultdict
import re
import random
import pickle


def solve_cloze(input_filename, candidate_filename, corpus_filename):
    """
    Solves a text completion problem by predicting missing words in the input text
    based on candidate words and a given corpus.

    The function reads an input text file with placeholders for missing words and
    uses a list of candidate words alongside a corpus to predict the most probable
    replacements. The predictions are evaluated against the original order of
    candidates.

    Parameters:
        input_filename (str): Path to the input text file containing placeholders.
        candidate_filename (str): Path to the file with candidate words.
        corpus_filename (str): Path to the corpus file used for n-gram probability calculations.

    Returns:
        None: Prints the evaluation metrics, specifically the number of correct
              predictions and total predictions made.
    """
    # Set to hold unique candidate words for fast lookup.
    candidate_set = set()

    # Set to store extracted bigrams from the input file.
    input_bigrams = set()

    # List to store the final predicted results for the placeholders.
    results = []

    # List to retain the original order of candidate words from the file.
    correct_order = []

    # Step 1: Load candidate words into a set and a list to preserve their order.
    load_candidates(candidate_filename, candidate_set, correct_order)

    # Step 2: Extract bigrams (two-word sequences) from the input text for context analysis.
    extract_input_bigrams(input_filename, input_bigrams)

    # Step 3: Compute n-gram counts (bigram and trigram) from the corpus to enable probability-based predictions.
    bigram_counts, trigram_counts = calculate_ngram_counts(
        corpus_filename, candidate_set, input_bigrams
    )

    # Step 4: Predict the missing words in the input text using n-gram probabilities.
    predict_words(input_filename, candidate_set, bigram_counts, trigram_counts, results)

    # Step 5: Evaluate predictions by comparing them against the original candidate order.
    correct_predictions, total_predictions = evaluate_predictions(results, correct_order)

    # Output the prediction evaluation metrics to the console.
    print(f"Correct Predictions: {correct_predictions}, Total Predictions: {total_predictions}")


def load_candidates(filename, candidates_set, order_list):
    """
    Loads candidate words from a file and populates both a set (for unique word storage)
    and a list (to preserve the original order).

    This function ensures all candidate words are added to a set for quick lookup
    and a list to retain the sequence in which they appear in the file.

    Parameters:
        filename (str): Path to the file containing candidate words, where each line
                        may contain one or more words separated by spaces.
        candidates_set (set): A set to store unique candidate words, allowing
                              fast existence checks.
        order_list (list): A list to maintain the original order of candidates
                           as they appear in the file.

    Returns:
        None: Populates the provided `candidates_set` and `order_list` in place.
    """
    try:
        # Open the file in read mode with UTF-8 encoding for compatibility.
        with open(filename, 'r', encoding='utf-8') as file:
            # Process each line in the file.
            for line in file:
                # Split the line into words and process each word.
                for word in line.split():
                    # Add the word to the set (automatically ensures uniqueness).
                    candidates_set.add(word)

                    # Append the word to the list to preserve its original order.
                    order_list.append(word)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading the file '{filename}': {e}")
        raise


def extract_input_bigrams(filename, bigrams_set):
    """
    Extracts bigrams (two consecutive words) from the input text file and stores them in a set,
    while ignoring placeholders and non-alphabetic characters.

    Bigrams are generated only for consecutive words that are not placeholders
    (denoted by "__________"). Words are converted to lowercase and stripped of
    non-alphabetic characters for consistency.

    Parameters:
        filename (str): Path to the input text file.
        bigrams_set (set): A set to store unique bigrams extracted from the text.

    Returns:
        None: Populates the `bigrams_set` with tuples of bigrams.
    """
    try:
        # Open the input file in read mode with UTF-8 encoding.
        with open(filename, 'r', encoding='utf-8') as file:
            # Process each line in the file.
            for line in file:
                # Split the line into words.
                words = line.split()

                # Iterate through word pairs to generate bigrams.
                for i in range(1, len(words)):
                    # Skip bigrams containing placeholders ("__________").
                    if words[i] != "__________" and words[i - 1] != "__________":
                        # Clean words: Remove non-alphabetic characters and convert to lowercase.
                        bigram = (
                            re.sub(r'[^a-zA-Z]', '', words[i - 1]).lower(),
                            re.sub(r'[^a-zA-Z]', '', words[i]).lower()
                        )
                        # Add the bigram to the set for uniqueness.
                        bigrams_set.add(bigram)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading the file '{filename}': {e}")
        raise


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
    Calculates bigram and trigram counts from a corpus, filtering counts based on
    candidate words and input bigrams for relevance.

    This function processes a large text corpus line by line, extracts bigrams and
    trigrams involving candidate words, and counts their occurrences. These counts
    are used to estimate probabilities for solving text completion problems.

    Parameters:
        filename (str): Path to the corpus file containing text.
        candidates (set): A set of candidate words to focus the counting process.
        input_bigrams (set): A set of input bigrams to filter bigram counts.

    Returns:
        tuple: A tuple containing two dictionaries:
               - `bigram_counts` (dict): Maps (word1, word2) bigrams to their counts.
               - `trigram_counts` (dict): Maps (word1, word2, word3) trigrams to their counts.
    """
    # Initialize dictionaries for bigram and trigram counts with default integer values.
    bigram_counts = defaultdict(int)
    trigram_counts = defaultdict(int)

    try:
        # Open the corpus file in read mode with UTF-8 encoding.
        with open(filename, 'r', encoding='utf-8') as file:
            print('Reading the text file...')

            # Initialize context tokens for trigram generation.
            w1, w2 = None, None

            # Process the file line by line.
            for i, line in enumerate(file):
                # Split each line into words and process.
                for w3 in line.split():
                    # Clean the word: remove non-alphabetic characters and convert to lowercase.
                    w3 = re.sub(r'[^a-zA-Z]', '', w3).lower()

                    # Count trigrams if any word in the trigram is a candidate.
                    if w3 in candidates or w2 in candidates or w1 in candidates:
                        trigram_counts[(w1, w2, w3)] += 1

                    # Count bigrams if relevant to input bigrams or candidates.
                    if w2:
                        if (w2, w3) in input_bigrams or w3 in candidates or w2 in candidates:
                            bigram_counts[(w2, w3)] += 1

                    # Update context tokens for the next iteration.
                    w1 = w2
                    w2 = w3

                # Periodically print progress for large files.
                if (i + 1) % 100000 == 0:
                    print(f'Processed {(i + 1):,} lines.')

        # Serialize the counts to a file for later use.
        with open("counts2.pkl", "wb") as f:
            pickle.dump((bigram_counts, trigram_counts), f)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while processing the file '{filename}': {e}")
        raise

    # Return the computed bigram and trigram counts.
    return bigram_counts, trigram_counts


def predict_words(filename, candidates_set, bigram_counts, trigram_counts, results):
    """
    Predicts missing words in the input text file using bigram and trigram probabilities.

    This function iterates through the input text, identifies placeholders (e.g., "__________"),
    and predicts the most probable replacement word from the candidate set based on
    n-gram probabilities. Predictions are appended to the results list.

    Parameters:
        filename (str): Path to the input text file with placeholders.
        candidates_set (set): A set of candidate words for prediction.
        bigram_counts (dict): Dictionary mapping bigrams to their counts.
        trigram_counts (dict): Dictionary mapping trigrams to their counts.
        results (list): A list to store the predicted words in the order of appearance.

    Returns:
        None: Populates the `results` list with predicted words.
    """
    # Additive smoothing constant and vocabulary size for probability calculation.
    k = 0.001
    vocab_size = 50000
    kv = k * vocab_size  # Smoothing factor adjusted for vocabulary size.

    try:
        # Open the input file in read mode with UTF-8 encoding.
        with open(filename, 'r', encoding='utf-8') as inputFile:
            # Process each line in the input file.
            for line in inputFile:
                words = line.split()

                # Iterate through each word in the line.
                for i, word in enumerate(words):
                    # Check for placeholders (denoted by "__________").
                    if word == "__________":
                        max_score = float('-inf')  # Initialize maximum score.
                        best_candidate = None     # Initialize the best candidate word.

                        # Iterate through a copy of the candidate set to avoid modifying it during iteration.
                        for candidate in list(candidates_set):
                            # Ensure sufficient context exists around the placeholder.
                            if i < 2 or i + 2 >= len(words):
                                continue

                            # Extract context words around the placeholder.
                            w1, w2, w3, w4 = words[i - 2], words[i - 1], words[i + 1], words[i + 2]

                            # Calculate trigram probabilities for the candidate.
                            factor1 = (trigram_counts.get((w1, w2, candidate), 0) + k) / (
                                bigram_counts.get((w1, w2), 0) + kv)
                            factor2 = (trigram_counts.get((w2, candidate, w3), 0) + k) / (
                                bigram_counts.get((w2, candidate), 0) + kv)
                            factor3 = (trigram_counts.get((candidate, w3, w4), 0) + k) / (
                                bigram_counts.get((candidate, w3), 0) + kv)

                            # Compute the overall score as the product of the factors.
                            current_score = factor1 * factor2 * factor3

                            # Update the best candidate if a higher score is found.
                            if current_score > max_score:
                                max_score = current_score
                                best_candidate = candidate

                        # Remove the best candidate from the set to avoid reuse.
                        if best_candidate:
                            candidates_set.remove(best_candidate)
                            results.append(best_candidate)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while processing the file '{filename}': {e}")
        raise


def experiment(candidates_filename):
    """
    Conducts an experiment to evaluate the accuracy of random permutations of candidate words
    in predicting the correct order.

    The function generates multiple unique permutations of the candidate words, compares each
    permutation to the true order, and calculates the average accuracy across all permutations.

    Parameters:
        candidates_filename (str): Path to the file containing candidate words.

    Returns:
        float: The average accuracy percentage of the random choice predictions.
    """
    # Load the true order of candidates from the file.
    true_order = []
    load_candidates(candidates_filename, set(), true_order)

    # Set to store unique permutations of the candidate words.
    unique_permutations = set()

    # Generate 100 unique permutations of the candidate words.
    while len(unique_permutations) < 100:
        shuffled = true_order[:]
        random.shuffle(shuffled)  # Randomly shuffle the true order.
        unique_permutations.add(tuple(shuffled))  # Add as a tuple to ensure immutability.

    # Evaluate each permutation and calculate the average accuracy.
    total_accuracy = 0
    for perm in unique_permutations:
        correct_predictions, _ = evaluate_predictions(perm, true_order)
        accuracy = correct_predictions / len(true_order)
        total_accuracy += accuracy

    # Calculate the average accuracy across all permutations.
    average_accuracy = (total_accuracy / len(unique_permutations)) * 100

    return average_accuracy

if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    print('Start experiment:')
    experiment_results = experiment(config['candidates_filename'])
    print(f'End of experiment. Avarage accuracy: {experiment_results}%')

    solve_cloze(config['input_filename'], config['candidates_filename'], config['corpus'])


