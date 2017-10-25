"""Executes the MNB and SVM experiments
"""

#__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from Logging import LoggingWrapper, log
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, \
                            precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import scipy.sparse as sp
from gensim.models import Word2Vec
from collections import Counter
import nltk
from itertools import chain, combinations_with_replacement, product, compress
import os
from enum import Enum

# To run tests with resampling
# HOWEVER: most of them will ERROR!
USE_SAMPLING_CLASSES_ON_BEST_RESULTS = False


class VectorizerType(Enum):
    """Enumeration that identifies the type of Vectorizer used."""
    COUNT = 1
    TFIDF = 2


class WordType(Enum):
    """Enumeration that determines whether to use words or POS tags."""
    WORD = 1
    POS_TAG = 2


class WordVectorizerType(Enum):
    """Combines both the Vectorizer and Word types into a single enumeration.
    """
    WORD_COUNT = 1
    WORD_TFIDF = 2
    POS_TAG_COUNT = 3
    POS_TAG_TFIDF = 4

    def word_type(self):
        """Gets the WordType associated with this class.

        :return: a WordType
        """
        if self.name.startswith('WORD_'):
            return WordType.WORD
        return WordType.POS_TAG

    def vectorizer_type(self):
        """Gets the VectorizerType associated with this class.

        :return: a VectorizerType
        """
        if self.name.endswith('_TFIDF'):
            return VectorizerType.TFIDF
        return VectorizerType.COUNT


class FeatureCreator(object):
    """Creates the sparse feature matrices for training and testing the
    models."""

    def __init__(self, df):
        """Initialises the FeatureCreator with the sentence DataFrame.

        :param df: the sentence DataFrame
        """
        self.df = df
        self.df_train = df[df['is_train']]
        self.df_test = df[~df['is_train']]
        self.y_test = np.array(self.df_test['is_fw'])
        self.train_sentences = list(self.df_train['clean'])
        self.test_sentences = list(self.df_test['clean'])
        self.train_pos_tag_sentences = list(self.df_train['clean_pos_tags'])
        self.test_pos_tag_sentences = list(self.df_test['clean_pos_tags'])
        self.vectorizers = {}

    def get_vectorizer_features(self, ngrams, word_vectorizer_type):
        """Gets the appropriate vectorizer, creating it if necessary and
        storing it in a caching dictionary.

        :param ngrams: an n-gram range tuple e.g. (1,1) or (2,3)
        :param word_vectorizer_type: a WordVectorizerType to create
        :return: a tuple containing (training_features, test_features)
        """

        key = "%s_%d_%d" % (word_vectorizer_type.name, ngrams[0], ngrams[1])
        features = self.vectorizers.get(key, None)
        if features is None:
            if word_vectorizer_type.vectorizer_type() is VectorizerType.COUNT:
                vectorizer = CountVectorizer(ngram_range=ngrams, min_df=1)
            else:
                vectorizer = TfidfVectorizer(ngram_range=ngrams, min_df=1)

            if word_vectorizer_type.word_type() is WordType.WORD:
                features = (vectorizer.fit_transform(self.train_sentences),
                            vectorizer.transform(self.test_sentences))
            else:
                features = (vectorizer.fit_transform(
                                self.train_pos_tag_sentences),
                            vectorizer.transform(self.test_pos_tag_sentences))
            self.vectorizers[key] = features
            log('Added "%s" key to the vectorizers dictionary' % key)
        return features

    def get_bag_of_words_features(self, train_features, test_features,
                                  ngrams, word_vectorizer_type):
        """Generates the bag of words or TF-IDF features.

        :param train_features: list of training features to append to
        :param test_features: list of test features to append to
        :param ngrams: an n-gram range tuple e.g. (1,1) or (2,3)
        :param word_vectorizer_type: a WordVectorizerType to create
        """
        train_feature, test_feature = \
            self.get_vectorizer_features(ngrams, word_vectorizer_type)
        train_features.append(train_feature)
        test_features.append(test_feature)

    def get_special_features(self, feature_name, weight_additional_features,
                             train_features, test_features):
        """

        :param feature_name: the name of the custom feature to append
        :param weight_additional_features: whether to weight the feature
        :param train_features: list of training features to append to
        :param test_features: list of test features to append to
        """
        for features, df_t in [(train_features, self.df_train),
                               (test_features, self.df_test)]:
            f = np.array(df_t[feature_name])
            if weight_additional_features:
                f *= len(df_t)  # multiply feature by the number of samples
            features.append(sp.csr_matrix(f).transpose())

    def get(self,
            count_ngrams=None,
            tfidf_ngrams=None,
            pos_tag_count_ngrams=None,
            pos_tag_tfidf_ngrams=None,
            nbsvm_ngrams=None,
            mean_w2v_size=None,
            use_prev_fw=False,
            use_is_question=False,
            use_is_continuation=False,
            use_additional_features=False,
            weight_additional_features=False,
            sampling_class=None,
            valid_names=[]):
        """Get four sparse matrices of features

        :param count_ngrams: n-gram range - None or a tuple e.g. (1,1)
        :param tfidf_ngrams: n-gram range - None or a tuple e.g. (1,1)
        :param pos_tag_count_ngrams: n-gram range - None or a tuple e.g. (1,1)
        :param pos_tag_tfidf_ngrams: n-gram range - None or a tuple e.g. (1,1)
        :param nbsvm_ngrams: n-gram range - None or a tuple e.g. (1,1)
        :param mean_w2v_size: Word2Vec dimension
        :param use_prev_fw: include the prev_fw feature
        :param use_is_question: include the is_question feature
        :param use_is_continuation: include the is_continuation feature
        :param use_additional_features: include all three custom features
        :param weight_additional_features: weight the custom features
        :param sampling_class: the resampling class
        :param valid_names: valid classifier for these features
        :return: a tuple of features (x_train, x_test, y_train, y_test)
        """

        y_train = np.array(self.df_train['is_fw'])
        train_features, test_features = [], []
        if count_ngrams is not None:
            self.get_bag_of_words_features(train_features, test_features,
                                           count_ngrams,
                                           WordVectorizerType.WORD_COUNT)

        if tfidf_ngrams is not None:
            self.get_bag_of_words_features(train_features, test_features,
                                           tfidf_ngrams,
                                           WordVectorizerType.WORD_TFIDF)

        if pos_tag_count_ngrams is not None:
            self.get_bag_of_words_features(train_features, test_features,
                                           pos_tag_count_ngrams,
                                           WordVectorizerType.POS_TAG_COUNT)

        if pos_tag_tfidf_ngrams is not None:
            self.get_bag_of_words_features(train_features, test_features,
                                           pos_tag_tfidf_ngrams,
                                           WordVectorizerType.POS_TAG_TFIDF)

        if mean_w2v_size is not None:
            w2v = Word2Vec(map(str.split, self.train_sentences),
                           size=mean_w2v_size, min_count=1).wv
            keys = w2v.vocab.keys()
            train_mean_w2v = np.array([np.array([w2v[w]
                                                 for w in s
                                                 if w in keys]).mean(axis=0)
                                       for s in self.train_sentences])
            test_mean_w2v = np.array([np.array([w2v[w]
                                                for w in s
                                                if w in keys]).mean(axis=0)
                                      for s in self.test_sentences])

            # Scale this feature between 0 and 1
            train_mm_scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled_mean_w2v = \
                train_mm_scaler.fit_transform(train_mean_w2v)
            train_features.append(sp.csr_matrix(train_scaled_mean_w2v))
            test_mm_scaler = MinMaxScaler(feature_range=(0, 1))
            test_scaled_mean_w2v = test_mm_scaler.fit_transform(test_mean_w2v)
            test_features.append(sp.csr_matrix(test_scaled_mean_w2v))

        if nbsvm_ngrams is not None:
            n_start, n_end = nbsvm_ngrams
            is_fws = Counter('_'.join(ngram)
                             for s in compress(self.train_sentences,
                                               y_train == 1)
                             for n in range(n_start, n_end + 1)
                             for ngram in nltk.ngrams(s.split(), n))
            not_fws = Counter('_'.join(ngram)
                              for s in compress(self.train_sentences,
                                                y_train == 0)
                              for n in range(n_start, n_end + 1)
                              for ngram in nltk.ngrams(s.split(), n))

            log('NBSVM - len(is_fws)=%d, len(not_fws)=%d' % (len(is_fws),
                                                             len(not_fws)))

            tokens = set(chain(is_fws.keys(), not_fws.keys()))
            token_idxs = dict((token, idx)
                              for idx, token in enumerate(tokens))
            len_tokens = len(tokens)

            alpha = 1
            p, q = np.ones(len_tokens) * alpha, np.ones(len_tokens) * alpha
            for token in tokens:
                p[token_idxs[token]] += is_fws[token]
                q[token_idxs[token]] += not_fws[token]

            p /= np.abs(p).sum()
            q /= np.abs(q).sum()
            r = np.log(p/q)

            for features, sentences in [(train_features, self.train_sentences),
                                        (test_features, self.test_sentences)]:
                indptr, indices, data = [0], [], []
                for s in sentences:
                    for token in ['_'.join(ngram)
                                  for n in range(n_start, n_end + 1)
                                  for ngram in nltk.ngrams(s.split(), n)]:
                        if token in tokens:
                            # this is necessary to build a sparse matrix
                            index = token_idxs[token]
                            indices.append(index)
                            data.append(r[index])
                    indptr.append(len(indices))
                f = sp.csr_matrix((data, indices, indptr),
                                  shape=[len(sentences), len_tokens],
                                  dtype=np.float64)
                features.append(f)

        if use_prev_fw or use_additional_features:
            self.get_special_features('prev_fw', weight_additional_features,
                                      train_features, test_features)

        if use_is_question or use_additional_features:
            self.get_special_features('is_question',
                                      weight_additional_features,
                                      train_features, test_features)

        if use_is_continuation or use_additional_features:
            self.get_special_features('is_continuation',
                                      weight_additional_features,
                                      train_features, test_features)

        if sampling_class is None:
            x_train = sp.hstack(train_features).tocsr()
            x_test = sp.hstack(test_features).tocsr()
        else:
            x_train = sp.hstack(train_features).toarray()
            x_test = sp.hstack(test_features).toarray()
            sampler = sampling_class(random_state=1234)
            x_train, y_train = sampler.fit_sample(x_train, y_train)
            x_train = x_train.astype(np.float32)

        return x_train, x_test, y_train, self.y_test


def create_feature_combinations(classifiers, df_results):
    """Creates all of the feature combinations for all of the experiments

    PLEASE NOTE: The "if USE_SAMPLING_CLASSES_ON_BEST_RESULTS:" section of
    the code is not reliable and will cause errors due to needing too much
    memory.

    :param classifiers: a list of classifiers
    :param df_results: the mnb and svm results DataFrame
    :return:
    """
    features = []
    if USE_SAMPLING_CLASSES_ON_BEST_RESULTS:
        features = [
            dict(count_ngrams=(2, 3), use_is_continuation=True,
                 use_prev_fw=True, weight_additional_features=True,
                 valid_names=['MNB']),
            dict(count_ngrams=(2, 3), use_is_continuation=True,
                 use_is_question=True, use_prev_fw=True,
                 weight_additional_features=True, valid_names=['MNB']),
            dict(count_ngrams=(2, 3), tfidf_ngrams=(2, 3),
                 use_is_continuation=True, use_is_question=True,
                 use_prev_fw=True, weight_additional_features=True,
                 valid_names=['MNB']),
            dict(tfidf_ngrams=(1, 2), use_is_continuation=True,
                 use_is_question=True, use_prev_fw=True,
                 valid_names=['SVM']),
            dict(pos_tag_tfidf_ngrams=(1, 2), tfidf_ngrams=(1, 2),
                 use_is_question=True, use_prev_fw=True,
                 valid_names=['SVM']),
            dict(pos_tag_tfidf_ngrams=(1, 2), tfidf_ngrams=(1, 2),
                 use_is_continuation=True, use_is_question=True,
                 use_prev_fw=True, valid_names=['SVM']),
        ]
        features_with_sampling_class = []
        sampling_classes = [RandomUnderSampler, RandomOverSampler, SMOTE]
        for sampling_class in sampling_classes:
            for feature_args in features:
                feature_args_with_sampling_class = feature_args.copy()
                feature_args_with_sampling_class['sampling_class'] = \
                    sampling_class
                features_with_sampling_class.append(
                    feature_args_with_sampling_class)
        features = features_with_sampling_class

    else:
        four_options = [[False, True]] * 4
        ngram_types = ['count_ngrams', 'tfidf_ngrams',
                       'pos_tag_count_ngrams', 'pos_tag_tfidf_ngrams']
        ngram_combinations = [opts
                              for opts in product(*four_options)
                              if any(opts)]  # don't want [False] * 4
        all_special_features = ['use_prev_fw', 'use_is_question',
                                'use_is_continuation',
                                'weight_additional_features']
        # we want to include [False] * 4
        special_feature_combinations = list(product(*four_options))
        # remove [False, False, False, True] which is only
        # weight_additional_features
        del special_feature_combinations[1]
        ngram_ranges = list(combinations_with_replacement([1, 2, 3, 4], 2))
        w2v_sizes = [100, 200, 300, 400, 500]

        features = []

        for ngram_range in ngram_ranges:
            for ng_comb in ngram_combinations:
                for sf_comb in special_feature_combinations:
                    feature = {}
                    feature.update({ngram_type: ngram_range
                                    for ngram_type in compress(ngram_types,
                                                               ng_comb)})
                    feature.update({special_feature: True
                                    for special_feature in compress(
                                                        all_special_features,
                                                        sf_comb)})
                    features.append(feature)

            for sf_comb in special_feature_combinations:
                feature = {'nbsvm_ngrams': ngram_range,
                           'valid_names': ['SVM']}
                feature.update({special_feature: True
                                for special_feature in compress(
                                                    all_special_features,
                                                    sf_comb)})
                features.append(feature)

        for sf_comb in special_feature_combinations:
            if any(sf_comb):  # need at least 1 feature enabled
                features.append({special_feature: True
                                 for special_feature in compress(
                                                     all_special_features,
                                                     sf_comb)})

        for w2v_size in w2v_sizes:
            for sf_comb in special_feature_combinations:
                feature = {'mean_w2v_size': w2v_size}
                feature.update({special_feature: True
                                for special_feature in compress(
                                                    all_special_features,
                                                    sf_comb)})
                features.append(feature)

    return features


def get_data():
    """Gets the conclusions DataFrame from the conclusions_dataframe.pickle

    :return: a conclusions DataFrame
    """
    return pd.read_pickle('conclusions_dataframe.pickle')


def get_results_dataframe(results_filename):
    """Loads the results DataFrame if possible or returns None.
    Will also add the sampling_class and sampling_class_name columns if they
    are missing.

    :param results_filename: the filename of the results pickle
    :return: the pandas DataFrame or None if it does not exist
    """
    if os.path.exists(results_filename):
        df = pd.read_pickle(results_filename)
        if 'sampling_class' not in df.columns:
            df['sampling_class'] = None
        if 'sampling_class_name' not in df.columns:
            df['sampling_class_name'] = df['sampling_class'].apply(
                lambda c: '' if c is None else c.__name__)
        return df
    return None


def create_classifiers():
    """Gets the list of classifiers

    :return: a list of classifier tuples
    """
    return [("MNB", MultinomialNB()), ("SVM", LinearSVC())]


def result_already_exists(df_results, df_errors, classifier_name,
                          feature_args):
    """Checks if the result already exists.

    :param df_results: the results DataFrame
    :param df_errors: an errors DataFrame (used when trying resampling)
    :param classifier_name: the classifier name
    :param feature_args: the feature arguments dictionary
    :return: True if the result already exists otherwise False
    """
    for df in [df_results, df_errors]:
        if df is None:
            continue
        data = df[(df['classifier_name'] == classifier_name) &
                  (df['features'] == feature_args)]
        if len(data) > 0:
            return True
    return False


def get_display_results_dataframe(df_results):
    """Gets a results DataFrame sorted for display

    :param df_results: the results DataFrame
    :return: a display results DataFrame
    """
    df = df_results[['classifier_name',
                     'feature_string',
                     'fit_time',
                     'predict_time',
                     'x_train_shape',
                     'accuracy',
                     'precision_0',
                     'precision_1',
                     'recall_0',
                     'recall_1',
                     'fscore_0',
                     'fscore_1',
                     'confusion']]
    df = df.sort_values(['fscore_1','fscore_0'], ascending=[False, False])

    if df is not None and len(df) > 0:
        df['true_neg'] = df.apply(lambda row: row.confusion[0, 0], axis=1)
        df['false_neg'] = df.apply(lambda row: row.confusion[1, 0], axis=1)
        df['false_pos'] = df.apply(lambda row: row.confusion[0, 1], axis=1)
        df['true_pos'] = df.apply(lambda row: row.confusion[1, 1], axis=1)
    return df


def display_results(df_results, top_n=10000):
    """Prints a table of results to the screen.

    :param df_results: a display results DataFrame
    :param top_n: the top n to restrict the results
    """
    df_display = get_display_results_dataframe(df_results)
    len_results = len(df_results)
    min_top_n = np.min([top_n, len_results])
    print(' ID CLSFR    FIT_T PRED_T   ACC  PR_0  PR_1  RL_0  RL_1  F1_0  '
          'F1_1 (%d of %d)' % (min_top_n, len_results))
    print('=== ======= ====== ====== ===== ===== ===== ===== ===== ===== '
          '=====')
    for i, row in enumerate(df_display[:top_n].itertuples()):
        print('%3d %-7.7s %06.2f %06.2f %04.3f %04.3f %04.3f %04.3f %04.3f '
              '%04.3f %04.3f F%r FS%r' %
              (i + 1, row.classifier_name, row.fit_time, row.predict_time,
               row.accuracy, row.precision_0, row.precision_1,
               row.recall_0, row.recall_1, row.fscore_0, row.fscore_1,
               row.feature_string, row.x_train_shape))


def get_feature_string(feature_args):
    """Gets a string representation of the feature arguments dictionary.

    :param feature_args: a feature arguments dictionary
    :return: a comma seperated string of features
    """
    bool_features = []
    features = []
    for key, value in feature_args.items():
        if type(value) is bool:
            bool_features.append(key.replace('use_', ''))
        else:
            features.append('%s=%r' % (key.replace('_ngrams',''), value))

    return ', '.join(chain.from_iterable([sorted(features),
                                          sorted(bool_features)]))


def create_result_dict(classifier, classifier_name, feature_args, fit_time,
                       fscores, precisions, predict_time, recalls, supports,
                       x_train, x_test, y_train, y_test, y_pred):
    """Creates a results dictionary to insert into the results DataFrame.

    :param classifier: the classifier
    :param classifier_name: the classifier name
    :param feature_args: the feature arguments dictionary
    :param fit_time: the time it took to fit the model
    :param fscores: the F1-Scores for the predictions
    :param precisions: the precisions for the predictions
    :param predict_time: the time it took for the model to predict
    :param recalls: the recalls for the predictions
    :param supports: the supports for the predictions
    :param x_train: the training features
    :param x_test: the test features
    :param y_train: the training classes
    :param y_test: the test classes
    :param y_pred: the predicted classes
    :return: a results dictionary to insert into the results DataFrame
    """
    sampling_class = feature_args.get('sampling_class', None)
    sampling_class_name = '' if sampling_class is None \
                          else sampling_class.__name__
    result = dict(classifier=classifier,
                  classifier_name=classifier_name,
                  features=feature_args,
                  feature_string=get_feature_string(feature_args),
                  x_train_shape=x_train.shape,
                  y_train_shape=y_train.shape,
                  x_test_shape=x_test.shape,
                  y_test_shape=y_test.shape,
                  created=pd.Timestamp.now(),
                  fit_time=fit_time,
                  predict_time=predict_time,
                  accuracy=accuracy_score(y_test, y_pred),
                  precision_0=precisions[0],
                  precision_1=precisions[1],
                  recall_0=recalls[0],
                  recall_1=recalls[1],
                  fscore_0=fscores[0],
                  fscore_1=fscores[1],
                  support_0=supports[0],
                  support_1=supports[1],
                  confusion=confusion_matrix(y_test, y_pred),
                  prfs_macro=precision_recall_fscore_support(y_test,
                                                             y_pred,
                                                             average='macro'),
                  prfs_micro=precision_recall_fscore_support(y_test,
                                                             y_pred,
                                                             average='micro'),
                  prfs_weighted=precision_recall_fscore_support(
                                y_test,
                                y_pred,
                                average='weighted'),
                  count_ngrams=feature_args.get('count_ngrams', None),
                  tfidf_ngrams=feature_args.get('tfidf_ngrams', None),
                  pos_tag_count_ngrams=feature_args.get(
                                       'pos_tag_count_ngrams', None),
                  pos_tag_tfidf_ngrams=feature_args.get(
                                       'pos_tag_tfidf_ngrams', None),
                  nbsvm_ngrams=feature_args.get('nbsvm_ngrams', None),
                  mean_w2v_size=feature_args.get('mean_w2v_size', None),
                  use_prev_fw=feature_args.get('use_prev_fw', None),
                  use_is_question=feature_args.get('use_is_question', None),
                  use_is_continuation=feature_args.get('use_is_continuation',
                                                       None),
                  weight_additional_features=feature_args.get(
                                             'weight_additional_features',
                                             None),
                  sampling_class=sampling_class,
                  sampling_class_name=sampling_class_name,
                  )
    return result


def get_errors_dataframe(errors_filename):
    """Gets the errors DataFrame or None if the file does not exist.

    :param errors_filename: the filename of the errors DataFrame pickle
    :return: the errors DataFrame or None if the file does not exist
    """
    if os.path.exists(errors_filename):
        return pd.read_pickle(errors_filename)
    return None


def write_to_error_table(errors_filename, classifier_name, feature_args):
    """Writes to the errors DataFrame.

    :param errors_filename: the errors DataFrame filename
    :param classifier_name: the classifier name
    :param feature_args: the feature arguments dictionary
    :return: the updated errors DataFrame
    """
    error = dict(classifier_name=classifier_name,
                 features=feature_args,
                 created=pd.datetime.now())
    if os.path.exists(errors_filename):
        df = pd.read_pickle(errors_filename)
        df = df.append(error, ignore_index=True)
    else:
        df = pd.DataFrame([error])
    df.to_pickle(errors_filename)
    return df


def main():
    """The root function to execute when the script executes"""
    with LoggingWrapper('NBandSVMTests'):
        np.random.seed(1234)
        df_sentences = get_data()
        errors_filename = './nb_and_svm_errors.pickle'
        results_filename = './nb_and_svm_results.pickle'
        df_results = get_results_dataframe(results_filename)
        df_errors = get_errors_dataframe(errors_filename)
        classifiers = create_classifiers()
        features = create_feature_combinations(classifiers, df_results)
        len_classifiers, len_features = len(classifiers), len(features)
        feature_creator = FeatureCreator(df_sentences)

        for classifier_idx, (classifier_name, classifier) in enumerate(
                                                             classifiers):
            with LoggingWrapper("Classifier %d of %d: %s"
                                % (classifier_idx+1, len_classifiers,
                                   classifier_name)):
                for feature_args_idx, feature_args in enumerate(features):
                    valid_names = feature_args.get('valid_names')
                    if valid_names is not None \
                            and classifier_name not in valid_names:
                        log("Not valid test %d of %d: %r"
                            % (feature_args_idx+1, len_features,
                               feature_args))
                        continue
                    if result_already_exists(df_results, df_errors,
                                             classifier_name, feature_args):
                        log("Already exists %d of %d: %r"
                            % (feature_args_idx+1, len_features,
                               feature_args))
                        continue

                    with LoggingWrapper("Creating Features %d of %d: %r"
                                        % (feature_args_idx+1, len_features,
                                           feature_args)):
                        #try:
                            memory_error_occurred = False
                            x_train, x_test, y_train, y_test = \
                                feature_creator.get(**feature_args)
                        #except MemoryError:
                        #    log("Memory Error Feature %d of %d: %r"
                        #        % (feature_args_idx+1, len_features,
                        #           feature_args))
                        #    memory_error_occurred = True

                    if memory_error_occurred:
                        df_errors = write_to_error_table(errors_filename,
                                                         classifier_name,
                                                         feature_args)

                        #sys.exit()
                    else:
                        with LoggingWrapper("Fitting Classifier %s"
                                            % classifier_name) as lw:
                            classifier.fit(x_train, y_train)
                            fit_time = lw.elapsed()

                        with LoggingWrapper("Predicting Classifier %s"
                                            % classifier_name) as lw:
                            y_pred = classifier.predict(x_test)
                            predict_time = lw.elapsed()

                        precisions, recalls, fscores, supports = \
                            precision_recall_fscore_support(y_test, y_pred)
                        log('%s-%r : f1_0=%3.2f, f1_1=%3.2f' %
                            (classifier_name, feature_args, fscores[0],
                             fscores[1]))
                        result = create_result_dict(classifier,
                                                    classifier_name,
                                                    feature_args, fit_time,
                                                    fscores, precisions,
                                                    predict_time, recalls,
                                                    supports, x_train, x_test,
                                                    y_train, y_test, y_pred)

                        if df_results is None:
                            df_results = pd.DataFrame([result])
                        else:
                            df_results = df_results.append(result,
                                                           ignore_index=True)
                        df_results.to_pickle(results_filename)

        display_results(df_results)


if __name__ == '__main__':
    main()
