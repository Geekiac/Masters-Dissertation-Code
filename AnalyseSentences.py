"""Analyses sentences in conclusions_with_fw_15_lines_or_less_post_cleanup.xml
and generates the conclusions_dataframe.pickle binary serialized sentences
DataFrame.  This DataFrame will then be used to create the features to train
and test the models.

Also outputs some summary statistics.
"""

#__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

import os
import pandas as pd
import numpy as np
import re
from Logging import LoggingWrapper
from lxml import etree
import nltk
from functools import partial
from sklearn.model_selection import train_test_split

continuation_phrases = [
    'it', 'this', 'these', 'those', 'they', 'and', 'also', 'but', 'because',
    'therefore', 'thus', 'hence', 'such', 'further', 'furthermore',
    'moreover', 'another', 'first', 'firstly', 'second', 'secondly',
    'thirdly', 'third', 'fourth', 'fourthly', 'finally', 'for example',
    'for instance', 'in particular', 'following this', 'we note though',
    'then', 'it is also likely', 'on the other hand', 'in addition'
]

re_starts_with_digit_or_roman_numeral = \
    re.compile('^([0-9]+|i|ii|iii|iv|v|vi|vii|viii|ix|x|' +
               'xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)[\. ]')


def is_a_question_sentence(sentence):
    """Checks if the sentence is a question.

    :param sentence: the sentence to check is a question
    :return: 1, if the sentence is a question, otherwise 0
    """
    return int(sentence.endswith('?'))


def is_a_continuation_sentence(sentence):
    """If the sentence starts with a digit or roman numeral or
    the sentence starts with any of the phrases in the continuation_phrases
    array then the sentence is a continuation sentence.

    :param sentence: the sentence to check is a continuation sentence
    :return: 1, if the sentence is a continuation sentence, otherwise 0
    """
    digit_or_roman_numeral = \
        re_starts_with_digit_or_roman_numeral.match(sentence) is not None
    starts_with_continuation_phrase = \
        any(map(sentence.startswith, continuation_phrases))
    return int(digit_or_roman_numeral or starts_with_continuation_phrase)


def get_bool_attribute(node, attr):
    """Gets an attribute from an XML node and determines if it has been
    given a boolean True value

    :param node: the xml node containing the attribute
    :param attr: the attribute to check
    :return: True, if the attribute contains a true value, otherwise False
    """
    return node is not None and node.get(attr,'').lower() == 'true'


def num_words(sentence):
    """Determines the number of words in a sentence.

    :param sentence: the sentence to count
    :return: the number of words in the sentence
    """
    if sentence:
        num_spaces = sum(map(str.isspace, sentence))
        return num_spaces + 1
    return 0


def create_conclusion_sentence_data_frame(xml_conclusions):
    """Creates the base sentence DataFrame from the conclusions xml

    :param xml_conclusions: the xml to create the DataFrame from
    :return: a DataFrame containing sentence data
    """
    df_sentences = pd.DataFrame([
        dict(key=doc.get('key'),
             doc_id=int(doc.get('id')),
             num_sentences=int(doc.get('num-sentences')),
             sent_id=int(sentence.get('id')),
             is_fw=int(get_bool_attribute(sentence, 'is-fw')),
             prev_fw=int(get_bool_attribute(sentence.getprevious(), 'is-fw')),
             next_fw=int(get_bool_attribute(sentence.getnext(), 'is-fw')),
             text=re.sub('\s+', ' ', sentence.text.lower()))
        for doc in xml_conclusions.findall("doc")
        for sentence in doc.findall("sentence")
        if doc.get('ignore', '').lower() != 'true'
           and re.match('[a-z]', sentence.text.lower())
    ])
    return df_sentences


def is_fw_first(row):
    """Is this sentence the first fw sentence in the group?

    :param row: sentence row from DataFrame
    :return: 1 if True otherwise 0
    """
    return int(row.is_fw and not row.prev_fw)


def is_fw_middle(row):
    """Is this sentence a middle fw sentence in the group?

    :param row: sentence row from DataFrame
    :return: 1 if True otherwise 0
    """
    return int(row.is_fw and row.prev_fw and row.next_fw)


def is_fw_last(row):
    """Is this sentence the last fw sentence in the group?

    :param row: sentence row from DataFrame
    :return: 1 if True otherwise 0
    """
    return int(row.is_fw and row.prev_fw and not row.next_fw)


def two_class_target(row):
    """Two class list containing one hot encoding for "not is fw" and "is fw"

    :param row: sentence row from DataFrame
    :return: one hot encoded list of length 2
    """
    return [int(not row.is_fw), row.is_fw]


def three_class_target(row):
    """Three class list containing one hot encoding for
    "not is fw"
    "first is fw"
    "middle or last is fw"

    :param row: sentence row from DataFrame
    :return: one hot encoded list of length 3
    """
    return list(map(int, [int(not row.is_fw),
                          row.is_fw_first,
                          row.is_fw_middle | row.is_fw_last]))


def four_class_target(row):
    """Four class list containing one hot encoding for
    "not is fw"
    "first is fw"
    "middle is fw"
    "last is fw"

    :param row: sentence row from DataFrame
    :return: one hot encoded list of length 4
    """
    return list(map(int, [int(not row.is_fw),
                          row.is_fw_first,
                          row.is_fw_middle,
                          row.is_fw_last]))


def extend_sentences_dataframe(df):
    """Extends the sentence data frame with additional
    calculated columns.

    :param df: the sentence DataFrame
    :return:
    """
    if not df.columns.contains('text_length'):
        df['text_length'] = df['text'].apply(len)

    if not df.columns.contains('text_num_words'):
        df['text_num_words'] = df['text'].apply(num_words)

    if not df.columns.contains('clean'):
        df['clean'] = df['text'].apply(partial(re.sub, '[^a-z ]', ''))

    if not df.columns.contains('clean_length'):
        df['clean_length'] = df['clean'].apply(len)

    if not df.columns.contains('clean_num_words'):
        df['clean_num_words'] = df['clean'].apply(num_words)

    if not df.columns.contains('is_question'):
        df['is_question'] = df['text'].apply(is_a_question_sentence)

    if not df.columns.contains('is_continuation'):
        df['is_continuation'] = df['text'].apply(is_a_continuation_sentence)

    if not df.columns.contains('is_fw_first'):
        df['is_fw_first'] = df.apply(is_fw_first, axis=1)

    if not df.columns.contains('is_fw_middle'):
        df['is_fw_middle'] = df.apply(is_fw_middle, axis=1)

    if not df.columns.contains('is_fw_last'):
        df['is_fw_last'] = df.apply(is_fw_last, axis=1)

    if not df.columns.contains('two_class_target'):
        df['two_class_target'] = df.apply(two_class_target, axis=1)

    if not df.columns.contains('three_class_target'):
        df['three_class_target'] = df.apply(three_class_target, axis=1)

    if not df.columns.contains('four_class_target'):
        df['four_class_target'] = df.apply(four_class_target, axis=1)

    if not df.columns.contains('clean_pos_tags'):
        tagger = nltk.StanfordPOSTagger(
                    './models/english-bidirectional-distsim.tagger',
                    'stanford-postagger.jar',
                    java_options='-mx8192m')
        df['clean_pos_tags'] = \
            [' '.join(tag for _, tag in s)
             for s in tagger.tag_sents(df['clean'].apply(str.split))]

    if not df.columns.contains('rowIndex'):
        df['row_index'] = df.apply(lambda row: row.name, axis=1)

    if not df.columns.contains('is_train'):
        np.random.seed(1234)
        train_indices, _ = train_test_split(np.array(range(0, len(df))),
                                            test_size=0.3)
        df['is_train'] = df['row_index'].apply(lambda idx:
                                               idx in train_indices)


def main():
    """The root function called by the script.
    Creates or updates the conclusions_dataframe.pickle file from
    conclusions_with_fw_15_lines_or_less_post_cleanup.xml.
    Also outputs some summary statistics.
    """
    with LoggingWrapper('AnalyseSentences'):
        pickle_filename = './conclusions_dataframe.pickle'
        if os.path.exists(pickle_filename):
            with LoggingWrapper('Load DataFrame from pickle'):
                df_sentences = pd.read_pickle(pickle_filename)
        else:
            with LoggingWrapper('Load conclusion xml'):
                xml_filename = \
                    './conclusions_with_fw_15_lines_or_less_post_cleanup.xml'
                xml_conclusions = etree.parse(xml_filename)
            with LoggingWrapper('Create sentences dataframe'):
                df_sentences = create_conclusion_sentence_data_frame(
                    xml_conclusions)

        with LoggingWrapper('Extending sentences dataframe'):
            extend_sentences_dataframe(df_sentences)
        with LoggingWrapper('Save sentences dataframe to pickle'):
            df_sentences.to_pickle(pickle_filename)

        corpus_all_words = [word
                            for sentence in df_sentences['clean']
                            for word in sentence.split(' ')]
        corpus_vocabulary = set(corpus_all_words)

        is_fw_words = df_sentences[df_sentences['is_fw'] == 1]['clean']
        corpus_is_fw_all_words = [word
                                  for sentence in is_fw_words
                                  for word in sentence.split(' ')]
        corpus_is_fw_vocabulary = set(corpus_is_fw_all_words)

        is_not_fw_words = df_sentences[df_sentences['is_fw'] == 0]['clean']
        corpus_is_not_fw_all_words = [word
                                      for sentence in is_not_fw_words
                                      for word in sentence.split(' ')]
        corpus_is_not_fw_vocabulary = set(corpus_is_not_fw_all_words)

        only_is_fw_words = \
            corpus_is_fw_vocabulary.difference(corpus_is_not_fw_vocabulary)
        only_is_not_fw_words = \
            corpus_is_not_fw_vocabulary.difference(corpus_is_fw_vocabulary)

        print('-' * 78)
        print("Total number of docs=", len(set(df_sentences['doc_id'])))
        print("Total number of sentences=", len(df_sentences))
        print('-' * 78)
        print("Total number of words in the corpus=", len(corpus_all_words))
        print("Total number of words in the vocabulary=",
              len(corpus_vocabulary))
        print('-' * 78)
        print("Total number of words in class is-fw=",
              len(corpus_is_fw_all_words))
        print("Total number of words in class is-fw vocabulary=",
              len(corpus_is_fw_vocabulary))
        print('-' * 78)
        print("Total number of words in class is-not-fw=",
              len(corpus_is_not_fw_all_words))
        print("Total number of words in class is-not-fw vocabulary=",
              len(corpus_is_not_fw_vocabulary))
        print('-' * 78)
        print("Number of words only in is-fw sentences=",
              len(only_is_fw_words))
        print("Number of words only in is-not-fw sentences=",
              len(only_is_not_fw_words))
        print('-' * 78)
        print("Most common words only in is-fw sentences=",
              nltk.FreqDist(word
                            for word in corpus_is_fw_all_words
                            if word in only_is_fw_words).most_common(50))
        print('-' * 78)


if __name__ == '__main__':
    main()
