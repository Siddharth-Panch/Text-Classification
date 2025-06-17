"""Module to explore data. Contains functions to help study, visualize, and understand datasets"""

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_num_classes(labels):
    """Gets the total number of classes.

    Args:
        labels (list): List of labels. There should be at least one sample for values in the range (0, num_classes-1).

    Returns:
        int: Total number of classes.

    Raises:
        ValueError: If any label value in the range (0, num_classes-1) is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError(
            f"Missing classes: {missing_classes}. All classes in the range (0, {num_classes-1}) must be present."
        )
    if num_classes <= 1:
        raise ValueError(
            f"Number of classes must be greater than 1, got {num_classes}."
        )
    return num_classes


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample.

    Args:
        sample_texts (list): List of sample texts.

    Returns:
        int: Median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def plot_frequency_distribution_of_ngrams(
    sample_texts, ngram_range=(1, 2), num_ngrams=50
):
    """Plots the frequency distribution of n-grams in the sample texts.
    Args:
        sample_texts (list): List of sample texts.
        ngram_range (tuple): Range of n-grams to consider (default is (1, 2)).
        num_ngrams (int): Number of top n-grams to plot (default is 50). Top 'num_ngrams' frequent n-grams will be plotted.
    """
    # Create args required for vectorizing
    kwargs = {
        "ngram_range": ngram_range,
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": "word",
    }

    vectorizer = CountVectorizer(**kwargs)
    vectorized_texts = vectorizer.fit_transform(sample_texts)
    all_ngrams = list(vectorizer.get_feature_names_out())
    num_ngrams = min(num_ngrams, len(all_ngrams))

    # Add up the counts per n-gram i.e. column-wise
    all_counts = vectorized_texts.sum(axis=0).A1

    # Sort n-grams and counts by frequency and get top 'num_ngrams' ngrams.
    all_counts, all_ngrams = zip(*sorted(zip(all_counts, all_ngrams), reverse=True))
    ngrams = list(all_ngrams[:num_ngrams])
    counts = list(all_counts[:num_ngrams])

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color="b")
    plt.xlabel("N-grams")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of N-grams")
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()


def plot_sample_length_distribution(sample_texts):
    """Plots the distribution of sample lengths in the sample texts.

    Args:
        sample_texts (list): List of sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel("Sample Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sample Lengths")
    plt.show()


def plot_class_distribution(labels):
    """Plots the distribution of classes in the labels.

    Args:
        labels (list): List of labels.
    """
    num_classes = get_num_classes(labels)
    count_map = Counter(labels)
    counts = [count_map[i] for i in range(num_classes)]
    idx = np.arange(num_classes)
    plt.bar(idx, counts, width=0.8, color="b")
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.xticks(idx, idx)
    plt.show()
