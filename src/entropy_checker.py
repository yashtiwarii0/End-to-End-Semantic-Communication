# src/entropy_checker.py

from collections import Counter
import math


def calculate_entropy(sentence):

    words = sentence.split()

    counts = Counter(words)

    total = len(words)

    entropy = 0

    for count in counts.values():

        p = count / total

        entropy -= p * math.log2(p)

    return entropy