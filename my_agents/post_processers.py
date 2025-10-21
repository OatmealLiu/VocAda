# Written by Mingxuan Liu
import os
import json
import re


def deduplicate_list(input_list):
    return list(set(input_list))


def remove_repetitive_phrases(text, min_words=2, max_words=10, threshold=5):
    """
    Removes repetitive phrases consisting of between min_words and max_words words.

    Args:
    text (str): The text to clean.
    min_words (int): Minimum number of words in a repeating phrase to consider.
    max_words (int): Maximum number of words in a repeating phrase to consider.
    threshold (int): The number of allowed repetitions of phrases. Phrases repeated more than this will be reduced.

    Returns:
    str: The cleaned text.
    """
    # Split text into words
    words = text.split()

    # Check for repetitive phrases of different lengths
    for num_words in range(min_words, max_words + 1):
        # Generate all possible combinations of start positions and phrase lengths
        for start in range(len(words) - num_words + 1):
            phrase = ' '.join(words[start:start + num_words])
            pattern = re.escape(phrase) + r'(?:\s+' + re.escape(phrase) + r'){' + str(threshold) + ',}'
            text = re.sub(pattern, phrase, text, flags=re.IGNORECASE)

    return text


def clean_text_by_word_repetition(text, repetition_threshold=2):
    """
    Cleans the input text by removing lines where any word appears more than the specified threshold.

    Parameters:
    - text (str): The input text to be cleaned.
    - repetition_threshold (int): The maximum allowed repetitions of any word in a single line. Lines with words exceeding this count will be removed.

    Returns:
    - str: The cleaned text with lines removed according to the repetition threshold.
    """
    cleaned_lines = []

    # Split the text into lines
    lines = text.split('\n')

    for line in lines:
        # Split each line into words
        words = line.split()

        # Count the occurrences of each word in the line
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Check if any word exceeds the repetition threshold
        if not any(count > repetition_threshold for count in word_counts.values()):
            cleaned_lines.append(line)  # Add line to cleaned lines if it meets criteria

    # Join the cleaned lines back into a single string
    return '\n'.join(cleaned_lines)


def clean_repetitive_text(text, char_threshold=3, word_threshold=2):
    """
    Cleans text from repetitive characters and words.

    Args:
    text (str): The text to clean.
    char_threshold (int): The number of allowed repetitive characters. Characters repeated more than this will be reduced.
    word_threshold (int): The number of allowed repetitive words. Words repeated more than this will be reduced.

    Returns:
    str: The cleaned text.
    """

    # First, reduce excessive character repetitions
    char_pattern = r'(.)\1{' + str(char_threshold) + ',}'
    text = re.sub(char_pattern, '\1', text)

    # Then, address word repetitions
    # The pattern looks for any sequence of word characters (\w+), followed by a space (optional),
    # and that whole group repeated {word_threshold,} times.
    word_pattern = r'(\b\w+\b)(?: \1){{{},}}'.format(word_threshold)
    text = re.sub(word_pattern, r'\1', text)

    return text.replace("\x01", "")


def postprocess_caption(text):
    text = clean_repetitive_text(text, char_threshold=10, word_threshold=5)
    text = remove_repetitive_phrases(text, min_words=2, max_words=10, threshold=5)
    # t = clean_text_by_word_repetition(t, repetition_threshold=5)
    return text

