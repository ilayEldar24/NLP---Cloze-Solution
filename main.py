import json
from collections import defaultdict
import re
import random
import pickle


def solve_cloze(input, candidates, corpus):
    print(f'starting to solve the cloze {input} with {candidates} using {corpus}')
    candidates_set = set()
    input_bigrams = set()
    res = []
    rightOrder = []

    getCadnidatesAndOrder(candidates, candidates_set, rightOrder)
    getInputBigrams(input, input_bigrams)


    
    bigram_counts, trigram_counts = getCounts(corpus, candidates_set, input_bigrams)
    #These dicts already in the coutns.pkl file.    
    #To use them and save time, use this code instead:

    # with open("counts.pkl", "rb") as f:
        # bigram_counts, trigram_counts = pickle.load(f)        

    
    solve(input, candidates_set, bigram_counts, trigram_counts, res)
    rightPredictions, totalPredictions = getResults(res, rightOrder)

    print(f'Good Predictions: {rightPredictions}, Total predictions: {totalPredictions}')


def getCadnidatesAndOrder(candidatesFileName, candidates_set, candidates_list):
    with open(candidatesFileName, 'r', encoding='utf-8') as candidates:
        for line in candidates:
            for word in line.split():
                candidates_set.add(word)
                candidates_list.append(word)


def getInputBigrams(inputFileName, inputBigrams):
    with open(inputFileName, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            for i in range(1, len(line.split())):
                if line.split()[i] != "__________" and line.split()[i - 1] != "__________":
                    inputBigrams.add((re.sub(r'[^a-zA-Z]', '', line.split()[i - 1]).lower(),
                                      re.sub(r'[^a-zA-Z]', '', line.split()[i]).lower()))


def getCounts(fileName, candidates, input_bigrams):
    # Initialize counts as defaultdicts for automatic key handling
    bigram_counts = defaultdict(int)
    trigram_counts = defaultdict(int)

    with open(fileName, 'r', encoding='utf-8') as fin:
        print('Reading the text file...')
        w1, w2 = None, None  # Initialize context tokens

        for i, line in enumerate(fin):
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

            # Progress update every 1 million lines
            if (i + 1) % 100000 == 0:
                print(f'{(i + 1) / 100000}% Done.')

        print('Finished processing the text file.')

    # Save the dictionaries to a file
    with open("counts.pkl", "wb") as f:
        pickle.dump((bigram_counts, trigram_counts), f)

    return bigram_counts, trigram_counts


def solve(inputFileName, candidates_set, bigram_counts, trigram_counts, res):
    with open(inputFileName, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            words = line.split()
            for i, word in enumerate(words):
                if word == "__________":
                    max_val = -1  # Renamed variable to avoid conflict with `max` function
                    top = None
                    for c in list(candidates_set):  # Use a copy of candidates_set
                        if i < 2 or i + 1 >= len(words):  # Check index validity
                            continue

                        w1, w2, w3 = words[i - 2], words[i - 1], words[i + 1]
                        denominator = (
                                bigram_counts.get((w1, w2), 0) * bigram_counts.get((w2, c), 0)
                        )

                        if denominator == 0:
                            continue  # Skip this candidate

                        cur = (
                                      trigram_counts.get((w1, w2, c), 0) * trigram_counts.get((w2, c, w3), 0)
                              ) / denominator

                        if cur > max_val:
                            max_val = cur
                            top = c

                    # If no valid candidate is found, pick a random word
                    if top is None:
                        top = random.choice(list(candidates_set))

                    candidates_set.remove(top)
                    res.append(top)


def getResults(predicted, true):
    count = 0

    for i in range(len(true)):
        if predicted[i] == true[i]:
            count += 1

    return count, len(true)


if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['corpus'])

    print('cloze solution:', solution)
