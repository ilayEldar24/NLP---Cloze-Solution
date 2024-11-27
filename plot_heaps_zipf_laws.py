import json
import pickle
import os.path
from collections import defaultdict
from matplotlib import pyplot as plt
from math import log
import seaborn as sn

sn.set()


def read_data(filename):
    word2freq = defaultdict(int)

    i = 0
    with open(filename, 'r', encoding='utf-8') as fin:
        print('reading the text file...')
        for i, line in enumerate(fin):
            for word in line.split():
                word2freq[word] += 1
            if i % 100000 == 0:
                print(i)

    total_words = sum(word2freq.values())
    word2nfreq = {w: word2freq[w]/total_words for w in word2freq}

    return word2nfreq

def plot_heaps_law(filename):
    """
    Reads a file and calculates the relationship between total word count
    and unique word count (Heaps' Law). Plots the results after processing.
    """
    word2freq = defaultdict(int)
    unique_counter = 0
    word_counter = 0
    results = []

    print("Reading the text file...")
    with open(filename, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            # Update word counters
            for word in line.split():
                if word2freq[word] == 0:
                    unique_counter += 1
                word2freq[word] += 1
                word_counter += 1

                # Save intermediate results every 100,000 words
                if word_counter % 100_000 == 0:
                    results.append((word_counter, unique_counter))

            # Print progress every 100,000 lines
            if i % 100_000 == 0 and i > 0:
                percent_complete = (i / 10_000_000) * 100
                print(f"{percent_complete:.2f}% Complete")
                print(f"Unique Words: {unique_counter}, Total Words: {word_counter}")

    print("Finished processing the file.")

    # Extract word counts and unique counts
    word_counts, unique_counts = zip(*results)

    #Plot
    plt.plot(word_counts, unique_counts)
    plt.title("Heaps' Law: Unique Words vs Total Words")
    plt.xlabel("Total Words")
    plt.ylabel("Unique Words")
    plt.show()

    return results



















def plot_zipf_law(word2nfreq):
    y = sorted(word2nfreq.values(), reverse=True)
    x = list(range(1, len(y)+1))

    product = [a * b for a, b in zip(x, y)]
    print(product[:1000])  # todo: print and note the roughly constant value

    y = [log(e, 2) for e in y]
    x = [log(e, 2) for e in x]

    plt.plot(x, y)
    plt.xlabel('log(rank)')
    plt.ylabel('log(frequency)')
    plt.title("Zipf's law")
    plt.show()












if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    plot_heaps_law(config['corpus'])

    if not os.path.isfile('word2nfreq.pkl'):
        data = read_data(config['corpus'])
        pickle.dump(data, open('word2nfreq.pkl', 'wb'))

    plot_zipf_law(pickle.load(open('word2nfreq.pkl', 'rb')))

