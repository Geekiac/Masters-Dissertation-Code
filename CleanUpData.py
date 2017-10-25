"""Contains variables and functions to help clean the data set.
"""

__all__ = ['is_valid_conclusion_or_fw_heading', 'clean_up_heading',
           'repair_sentence']
__version__ = '0.1'
__author__ = "Steven Smith"

from nltk import regexp_tokenize, trigrams as get_trigrams
from nltk.corpus import words, stopwords
from nltk.stem import SnowballStemmer


# when this file is loaded the all_words, missing_pdf_word_parts, stemmer,
# heading_stop_words and valid_conclusion_or_fw_headings variables are all
# initialised for use in the functions.

# all_words is a vocabulary of words to act as valid words in a document
all_words = set(words.words())
words_to_remove = ['bene', 'c', 'cation', 'eld', 'con', 'cult',
                   'dent', 'er', 'ers', 'es',
                   'o',    # to allow fixing o[ff]
                   'ort',  # to allow fixing [e]ff[ort]
                   'ne', 're',
                   't']    # to allow fixing [bene]fi[t]
for word in words_to_remove:
    if word in all_words:
        all_words.remove(word)

words_to_add = ['afford', 'affordance', 'affordances',
                'antiunification',
                'benefit', 'benefitted', 'benefits',
                'clarified', 'clarifies',
                'classifiers', 'classification', 'classifications',
                'certificate', 'certificates', 'certification',
                'coefficients', 'coefficient',
                'confident',
                'configuration', 'configurations',
                'conflict', 'conflicts',
                'convexification',
                'define', 'defines', 'defined', 'defining',
                'definition', 'definitions',
                'difficult', 'difficulties',
                'differs', 'differing', 'differences',
                'effort', 'efforts',
                'field', 'fields',
                'findings',
                'first', 'first-order', 'first order',
                'fits',
                'exemplified',
                'identified', 'identifies', 'identifiability', 'identifiers',
                'influenced', 'influencer', 'influences', 'influencing',
                'offline',
                'justified', 'justifier', 'justifies',
                'justification', 'justifications',
                'magnified',
                'misclassification', 'misclassifications',
                'modified', 'modifier', 'modifiers', 'modifies',
                'modifications',
                'profit', 'profits', 'profitable',
                'quantifies', 'quantifier', 'quantifiers', 'quantified',
                'reconfiguration', 'reconfigurations',
                'refine', 'refines', 'refined',
                'reflect', 'reflects', 'reflected',
                'reflection', 'reflections',
                'sacrificed',
                'simplifications', 'simplification', 'simplifies',
                'specified', 'specifies', 'specifics',
                'specification', 'specifications',
                'traffic',
                'unifies', 'unifier', 'unifiers', 'unified',
                'unidentifiability',
                'verified', 'verifier', 'verification', 'verifiability',
                'workflows']
all_words.update(words_to_add)

# missing_pdf_word_parts - parts of words which are sometimes missing after
# text extraction by CERMINE.
missing_pdf_word_parts = ['ffl', 'ffi', 'ff', 'fl', 'fi']

# stemmer and stop words to help parse the headings
stemmer = SnowballStemmer('english')
heading_stop_words = stopwords.words("english")
heading_stop_words.extend(['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii',
                           'ix', 'x', 'xi', 'xii', 'xiii'])
heading_stop_words.remove('further')
heading_stop_words = set(heading_stop_words)

# valid_conclusion_or_fw_headings - the stemmed headings to look for that are
# conclusion/further work sections.
valid_conclusion_or_fw_headings = {
    'conclus',
    'conclus futur',
    'conclus further',
    'conclus relat',
    'discuss conclus',
    'conclus open',
    'chapter conclus',
    'futur work',
    'conclus futur work',
    'conclusion',
    'conclus perspect',
    'conclus futur research'
}


def need_to_fix_word(word):
    """Determines if a word is missing word parts from the
    missing_pdf_word_parts at the beginning or end of the word.

    :param word: word to determine if it needs repairing
    :return: tuple, either (True, corrected_word), if the word needed to be
             repaired or (False, None) otherwise
    """
    word_exists = word.lower() in all_words

    if not word_exists:
        for part in missing_pdf_word_parts:
            new_word = '%s%s' % (part, word)
            new_word_exists = new_word.lower() in all_words

            if new_word_exists:
                return True, new_word

        for part in missing_pdf_word_parts:
            new_word = '%s%s' % (word, part)
            new_word_exists = new_word.lower() in all_words

            if new_word_exists:
                return True, new_word

    return False, None


def need_to_fix_words(bigram):
    """Determines if a pair of words are missing word parts from the
    missing_pdf_word_parts between them.

    :param bigram: a tuple word pair
    :return: tuple, either (True, corrected_word), if the word needed to be
             repaired or (False, None) otherwise
    """
    word1, word2 = bigram

    word1_exists = word1.lower() in all_words
    word2_exists = word2.lower() in all_words

    if not word1_exists or not word2_exists:
        for part in missing_pdf_word_parts:
            new_word = '%s%s%s' % (word1, part, word2)
            new_word_exists = new_word.lower() in all_words

            if new_word_exists:
                return True, new_word

    return False, None


def tokenize(sentence):
    """break a sentence into a word list keeping letters, apostrophes and
    dashes, as well as the remaining individual characters.

    :param sentence: sentence to tokenize
    :return: a list of words
    """
    tokens, current = [], []
    len_words = len(sentence)

    for i, c in enumerate(sentence):
        if c.isalpha() or c == "'" or c == '-':
            current.append(c)
            if i == (len_words - 1):
                tokens.append(''.join(current))
        else:
            if current:
                tokens.append(''.join(current))
            tokens.append(c)
            current = []

    return tokens


def repair_sentence(sentence):
    """Repairs the sentence if it is found to have known missing letters.

    :param sentence: sentence to repair
    :return: a repaired sentence including it's missing letters
    """
    tokens = tokenize(sentence)
    len_tokens = len(tokens)

    if len_tokens < 3:
        return ''.join(tokens)

    new_tokens = []
    skip_next_trigram = 0
    trigrams = list(get_trigrams(tokens))
    last_trigram = trigrams[-1]
    for trigram in trigrams:
        if skip_next_trigram > 0:
            skip_next_trigram -= 1
            continue

        # if the first and last part of the trigram are words determine if it
        # is missing letters between the two.  If it is put them back in to
        # repair the word.
        word1, word2, word3 = trigram
        if word1[0].isalpha() \
                and not word2[0].isalpha() \
                and word3[0].isalpha():
            need_fix, new_word = need_to_fix_words((word1, word3))
            if need_fix:
                new_tokens.append(new_word)
                skip_next_trigram = 2
            else:
                new_tokens.append(word1)
        else:
            if trigram == last_trigram:
                new_tokens.append(''.join(trigram))
            else:
                new_tokens.append(word1)

    # fix single word issues such as [fl]oat and o[ff]
    new_tokens2 = []
    for word in tokenize(''.join(new_tokens)):
        need_fix, new_word = need_to_fix_word(word)
        if need_fix:
            new_tokens2.append(new_word)
        else:
            new_tokens2.append(word)

    return ''.join(new_tokens2)


def clean_up_heading(heading):
    """Repairs the heading and then only returns words containing at least two
    characters that are not in the stop word list.

    :param heading: the heading to clean up
    :return: a cleaned up heading
    """
    sentence = repair_sentence(heading)
    tokens = regexp_tokenize(sentence, '[a-zA-Z]{2,}')
    return ' '.join(stemmer.stem(word)
                    for word in tokens
                    if word not in heading_stop_words).lower()


def clean_up_sentence(sentence):
    """Returns a sentence only consisting of words containing at least two
    characters.

    :param sentence: sentence to clean up
    :return: a lower case sentence only consisting of words containing at
             least two characters.
    """
    tokens = regexp_tokenize(sentence, '[a-zA-Z]{2,}')
    return ' '.join(tokens).lower()


def is_valid_conclusion_or_fw_heading(heading):
    """Determine if a heading contains text identifying it as a conclusion or
    further work.

    :param heading: heading to check
    :return: True, if the heading is a conclusion or further work, otherwise
             False.
    """
    return clean_up_heading(heading) in valid_conclusion_or_fw_headings
