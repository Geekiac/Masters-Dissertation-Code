"""Analyses the MNB and SVM results.

Outputs tables and box plots measuring the performance of the CNN.
Also outputs the sentences that were predicted incorrectly.
"""

#__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from NBandSVMTests import display_results, get_display_results_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.svm import LinearSVC


def print_title(title):
    """Prints the title to the screen underlined using # characters.

    :param title: the title to display
    """
    print(title)
    print('#' * len(title))


def frame_results(df, title, top_n=5):
    """Displays a results table framed with a title and a line of "=" above
    and below the table.

    :param df: the DataFrame containing the results
    :param title: the title to display
    :param top_n: how many results to display
    """
    print('=' * 100)
    print_title(title)
    display_results(df, top_n)
    print('=' * 100)


def output_to_csv(df_results, top_n=5):
    """Output results in CSV format.

    :param df_results: results DataFrame
    :param top_n: how many results to output
    :return: a string containing the results in CSV format
    """
    columns = ['classifier_name',
               'precision_0',
               'precision_1',
               'recall_0',
               'recall_1',
               'fscore_0',
               'fscore_1',
               'feature_string']
    df = get_display_results_dataframe(df_results)[:top_n]
    return df[columns].to_csv(float_format='%g')


def is_svm(row):
    """DataFrame filter to retrieve rows for the SVM classifier.

    :param row: the row to compare
    :return: True, if for the SVM classifier, otherwise False
    """
    return row.classifier_name == 'SVM'


def is_mnb(row):
    """DataFrame filter to retrieve rows for the MNB classifier.

    :param row: the row to compare
    :return: True, if for the MNB classifier, otherwise False
    """
    return row.classifier_name == 'MNB'


def is_ngrams(row):
    """DataFrame filter to retrieve rows for bag of words count or TF-IDF
    experiments.

    :param row: the row to compare
    :return: True, if for bag of words count or TF-IDF experiments, otherwise
             False
    """
    return ~row.count_ngrams.isnull() \
           | ~row.tfidf_ngrams.isnull() \
           | ~row.pos_tag_count_ngrams.isnull() \
           | ~row.pos_tag_tfidf_ngrams.isnull()


def is_mean_w2v(row):
    """DataFrame filter to retrieve rows for mean Word2Vec experiments.

    :param row: the row to compare
    :return: True, if for mean Word2Vec experiments, otherwise False
    """
    return ~row.mean_w2v_size.isnull()


def is_nbsvm(row):
    """DataFrame filter to retrieve rows for NBSVM experiments.

    :param row: the row to compare
    :return: True, if for NBSVM experiments, otherwise False
    """
    return ~row.nbsvm_ngrams.isnull()


def get_mnb_additional_features_only(row):
    """DataFrame filter to retrieve rows for MNB additional feature only
    experiments.

    :param row: the row to compare
    :return: True, if for MNB additional feature only experiments, otherwise
             False
    """
    return is_mnb(row) & ~is_ngrams(row) & ~is_mean_w2v(row) & ~is_nbsvm(row)


def get_mnb_ngrams(row):
    """DataFrame filter to retrieve rows for MNB bag of words count or TF-IDF
    experiments.

    :param row: the row to compare
    :return: True, if for MNB bag of words count or TF-IDF experiments,
             otherwise False
    """
    return is_mnb(row) & is_ngrams(row) & ~is_mean_w2v(row) & ~is_nbsvm(row)


def get_mnb_mean_w2v(row):
    """DataFrame filter to retrieve rows for MNB mean Word2Vec experiments

    :param row: the row to compare
    :return: True, if for MNB mean Word2Vec experiments, otherwise False
    """
    return is_mnb(row) & ~is_ngrams(row) & is_mean_w2v(row) & ~is_nbsvm(row)


def get_svm_additional_features_only(row):
    """DataFrame filter to retrieve rows for SVM additional feature only
    experiments.

    :param row: the row to compare
    :return: True, if for SVM additional feature only experiments, otherwise
             False
    """
    return is_svm(row) & ~is_ngrams(row) & ~is_mean_w2v(row) & ~is_nbsvm(row)


def get_svm_ngrams(row):
    """DataFrame filter to retrieve rows for SVM bag of words count or TF-IDF
    experiments.

    :param row: the row to compare
    :return: True, if for SVM bag of words count or TF-IDF experiments,
             otherwise False
    """
    return is_svm(row) & is_ngrams(row) & ~is_mean_w2v(row) & ~is_nbsvm(row)


def get_svm_mean_w2v(row):
    """DataFrame filter to retrieve rows for SVM mean Word2Vec experiments.

    :param row: the row to compare
    :return: True, if for SVM mean Word2Vec experiments, otherwise False
    """
    return is_svm(row) & ~is_ngrams(row) & is_mean_w2v(row) & ~is_nbsvm(row)


def get_svm_nbsvm(row):
    """DataFrame filter to retrieve rows for SVM NBSVM experiments.

    :param row: the row to compare
    :return: True, if for SVM NBSVM experiments, otherwise False
    """
    return is_svm(row) & ~is_ngrams(row) & ~is_mean_w2v(row) & is_nbsvm(row)


def plot_additional_features_only(df):
    """Plot a bar chart to an SVG file and the screen
    for additional features only.

    :param df: a results DataFrame
    """
    classifier_name = df.iloc[0]['classifier_name']
    features = [('weight_additional_features', 'WGT'),
                ('use_prev_fw', 'PFW'),
                ('use_is_question', 'QST'),
                ('use_is_continuation', 'CON')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df['abbrev_features'] = df.apply(abbreviate, axis=1)
    df = df.set_index('abbrev_features', drop=False)
    plot_f1score(df,
                 title='%s Additional Features Only' % classifier_name,
                 x_label='Abbreviated custom features',
                 filename='./plots/%s_additional_features_only.svg' %
                          classifier_name.lower())


def plot_mnb_count_and_tfidf_ngrams_2_4(df):
    """Plot a bar chart to an SVG file and the screen
    for MNB Count and TF-IDF N-Grams where
    ngrams=(2,4), Is Continuation, Previous Is Further Work,
    Weight Additional Features, NOT Is Question

    :param df: a results DataFrame
    """
    df2 = df[lambda r: ((r.count_ngrams == (2, 4))
                        | (r.tfidf_ngrams == (2, 4))
                        | (r.pos_tag_count_ngrams == (2, 4))
                        | (r.pos_tag_tfidf_ngrams == (2, 4)))
                       & r.use_is_continuation
                       & r.use_prev_fw
                       & r.weight_additional_features
                       & r.use_is_question.isnull()].copy()

    features = [('count_ngrams', 'CTW'),
                ('tfidf_ngrams', 'TIW'),
                ('pos_tag_count_ngrams', 'CTP'),
                ('pos_tag_tfidf_ngrams', 'TIP')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df2['abbrev_features'] = df2.apply(abbreviate, axis=1)
    df2 = df2.set_index('abbrev_features', drop=False)
    plot_f1score(df2,
                 title='MNB Count and TF-IDF N-Grams where\n' +
                       'ngrams=(2,4), Is Continuation, ' +
                       'Previous Is Further Work,\n' +
                       'Weight Additional Features, NOT Is Question',
                 x_label='Abbreviated n-gram types',
                 filename='./plots/mnb_count_and_tfidf_ngrams_2_4.svg')


def plot_mnb_count_ngrams(df):
    """Plot a bar chart to an SVG file and the screen
    for MNB Count N-Grams where
    Count n-grams only, Is Continuation, Previous Is Further Work,
    Weight Additional Features, NOT Is Question

    :param df: a results DataFrame
    """
    df2 = df[lambda r: ~r.count_ngrams.isnull()
                       & r.tfidf_ngrams.isnull()
                       & r.pos_tag_count_ngrams.isnull()
                       & r.pos_tag_tfidf_ngrams.isnull()
                       & r.use_is_continuation
                       & r.use_prev_fw
                       & r.weight_additional_features
                       & r.use_is_question.isnull()].copy()

    df2 = df2.set_index('count_ngrams', drop=False)
    plot_f1score(df2, title='MNB Count N-Grams where\n' +
                            'Count n-grams only, Is Continuation, ' +
                            'Previous Is Further Work,\n' +
                            'Weight Additional Features, NOT Is Question',
                 x_label='Count Words n-gram ranges',
                 filename='./plots/mnb_count_ngrams_only.svg')


def plot_mnb_count_ngrams_2_4(df):
    """Plot a bar chart to an SVG file and the screen
    for MNB Count N-Grams where Count n-grams=(2,4) only

    :param df: a results DataFrame
    """
    df2 = df[lambda r: (r.count_ngrams == (2,4))
                       & r.tfidf_ngrams.isnull()
                       & r.pos_tag_count_ngrams.isnull()
                       & r.pos_tag_tfidf_ngrams.isnull()].copy()
    features = [('weight_additional_features', 'WGT'),
                ('use_prev_fw', 'PFW'),
                ('use_is_question', 'QST'),
                ('use_is_continuation', 'CON')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df2['abbrev_features'] = df.apply(abbreviate, axis=1)
    df2 = df2.set_index('abbrev_features', drop=False)
    plot_f1score(df2, title='MNB Count N-Grams where\n' +
                            'Count n-grams=(2,4) only',
                 x_label='Abbreviated custom features',
                 filename='./plots/mnb_count_ngrams_2_4_only.svg')


def plot_mean_w2v(df):
    """Plot a bar chart to an SVG file and the screen
    for mean Word2Vec.  This plots all of the additional features
    against all of the mean Word2Vec sizes.

    :param df: a results DataFrame
    """
    classifier_name = df.iloc[0]['classifier_name']
    features = [('weight_additional_features', 'WGT'),
                ('use_prev_fw', 'PFW'),
                ('use_is_question', 'QST'),
                ('use_is_continuation', 'CON')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df['abbrev_features'] = df.apply(abbreviate, axis=1)
    df['mean_w2v_size_str'] = df['mean_w2v_size'].apply(str)
    df = pd.crosstab(index=df['abbrev_features'], values=df['fscore_1'],
                      aggfunc=lambda x: x, columns=df['mean_w2v_size_str'])
    ax = df.plot.bar()
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6,
              title="Word2Vec size", borderaxespad=0, mode='expand')
    ax.set_title('%s Mean Word2Vec' % classifier_name, y=1.3)
    ax.set_xlabel('Abbreviated custom features')
    ax.set_ylabel('F1-Score for "Is Further Work" class')
    plt.tight_layout()
    plt.savefig('./plots/%s_mean_w2v.svg' % classifier_name.lower())
    plt.show()


def plot_svm_count_and_tfidf_ngrams_1_2(df):
    """Plot a bar chart to an SVG file and the screen
    for SVM Count and TF-IDF N-Grams where
    ngrams=(1,2), Is Continuation, Is Question
    Previous Is Further Work, NOT Weight Additional Features

    :param df: a results DataFrame
    """
    df2 = df[lambda r: ((r.count_ngrams == (1, 2))
                        | (r.tfidf_ngrams == (1, 2))
                        | (r.pos_tag_count_ngrams == (1, 2))
                        | (r.pos_tag_tfidf_ngrams == (1, 2)))
                       & r.use_is_continuation
                       & r.use_is_question
                       & r.use_prev_fw
                       & r.weight_additional_features.isnull()].copy()

    features = [('count_ngrams', 'CTW'),
                ('tfidf_ngrams', 'TIW'),
                ('pos_tag_count_ngrams', 'CTP'),
                ('pos_tag_tfidf_ngrams', 'TIP')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df2['abbrev_features'] = df2.apply(abbreviate, axis=1)
    df2 = df2.set_index('abbrev_features', drop=False)
    plot_f1score(df2, title='SVM Count and TF-IDF N-Grams where\n' +
                            'ngrams=(1,2), Is Continuation, Is Question,\n' +
                            'Previous Is Further Work, '+
                            'NOT Weight Additional Features',
                 x_label='Abbreviated n-gram types',
                 filename='./plots/svm_count_and_tfidf_ngrams_1_2.svg')


def plot_svm_count_ngrams(df):
    """Plot a bar chart to an SVG file and the screen
    for SVM TF-IDF N-Grams where
    TF-IDF n-grams only, Is Continuation, Is Question,
    Previous Is Further Work, NOT Weight Additional Features

    :param df: a results DataFrame
    """
    df2 = df[lambda r: r.count_ngrams.isnull()
                       & ~r.tfidf_ngrams.isnull()
                       & r.pos_tag_count_ngrams.isnull()
                       & r.pos_tag_tfidf_ngrams.isnull()
                       & r.use_is_continuation
                       & r.use_is_question
                       & r.use_prev_fw
                       & r.weight_additional_features.isnull()].copy()

    df2 = df2.set_index('tfidf_ngrams', drop=False)
    plot_f1score(df2, title='SVM TF-IDF N-Grams where\n' +
                            'TF-IDF n-grams only, Is Continuation, '+
                            'Is Question,\n' +
                            'Previous Is Further Work, ' +
                            'NOT Weight Additional Features',
                 x_label='TF-IDF Words n-gram ranges',
                 filename='./plots/svm_count_ngrams_only.svg')


def plot_svm_tfidf_ngrams_1_2(df):
    """Plot a bar chart to an SVG file and the screen
    for SVM TF-IDF N-Grams where TF-IDF n-grams=(1,2) only

    :param df: a results DataFrame
    """
    df2 = df[lambda r: r.count_ngrams.isnull()
                       & (r.tfidf_ngrams == (1,2))
                       & r.pos_tag_count_ngrams.isnull()
                       & r.pos_tag_tfidf_ngrams.isnull()].copy()
    features = [('weight_additional_features', 'WGT'),
                ('use_prev_fw', 'PFW'),
                ('use_is_question', 'QST'),
                ('use_is_continuation', 'CON')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df2['abbrev_features'] = df.apply(abbreviate, axis=1)
    df2 = df2.set_index('abbrev_features', drop=False)
    plot_f1score(df2, title='SVM TF-IDF N-Grams where\n' +
                            'TF-IDF n-grams=(1,2) only',
                 x_label='Abbreviated custom features',
                 filename='./plots/svm_tfidf_ngrams_1_2.svg')


def plot_svm_nbsvm_ngrams(df):
    """Plot a bar chart to an SVG file and the screen
    for SVM NBSVM N-Grams where
    NBSVM n-grams only, Is Question, Previous Is Further Work
    NOT Is Continuation, NOT Weight Additional Features

    :param df: a results DataFrame
    """
    df2 = df[lambda r: r.use_is_continuation.isnull()
                       & r.use_is_question
                       & r.use_prev_fw
                       & r.weight_additional_features.isnull()].copy()

    df2 = df2.set_index('nbsvm_ngrams', drop=False)
    plot_f1score(df2, title='SVM NBSVM N-Grams where\n' +
                            'NBSVM n-grams only, Is Question, '+
                            'Previous Is Further Work\n' +
                            'NOT Is Continuation, ' +
                            'NOT Weight Additional Features',
                 x_label='NBSVM n-gram ranges',
                 filename='./plots/svm_nbsvm_ngrams_only.svg')


def plot_svm_nbsvm_ngrams_1_2(df):
    """Plot a bar chart to an SVG file and the screen
    for SVM NBSVM N-Grams where NBSVM n-grams=(1,2) only

    :param df: a results DataFrame
    """
    df2 = df[lambda r: (r.nbsvm_ngrams == (1,2))].copy()
    features = [('weight_additional_features', 'WGT'),
                ('use_prev_fw', 'PFW'),
                ('use_is_question', 'QST'),
                ('use_is_continuation', 'CON')]
    abbreviate = lambda r: ','.join(lbl for feat, lbl in features if r[feat])
    df2['abbrev_features'] = df.apply(abbreviate, axis=1)
    df2 = df2.set_index('abbrev_features', drop=False)
    plot_f1score(df2, title='SVM NBSVM N-Grams where\n' +
                            'NBSVM n-grams=(1,2) only',
                 x_label='Abbreviated custom features',
                 filename='./plots/svm_nbsvm_ngrams_1_2.svg')


def plot_f1score(df, title='', x_label='',
                 y_label='F1-Score for "Is Further Work" class',
                 filename=None, show=True):
    """Plots a bar chart, outputs to file and to screen if required

    :param df: results DataFrame
    :param title: title of the plot
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param filename: filename to export
    :param show: if True displays the plot to screen
    """
    fscore_1 = df['fscore_1']
    ax = fscore_1.plot.bar()
    max = fscore_1.max()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(np.max([fscore_1.min() - 0.01, 0]),
                np.min([fscore_1.max() + 0.01, 1]))
    plt.axhline(max, color='r', linestyle='dotted')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


def predict_best(df_results):
    """Outputs details of the best classifier including the
    incorrectly predicted sentences

    :param df_results: the results DataFrame
    """
    # get data to train the classifier
    df_sentences = pd.read_pickle('./conclusions_dataframe.pickle')
    df_train = df_sentences[df_sentences.is_train]
    df_test = df_sentences[~df_sentences.is_train]
    y_train = np.array(df_train.is_fw)
    y_test = np.array(df_test.is_fw)
    train_sentences = list(df_train.clean)
    test_sentences = list(df_test.clean)
    # create TF-IDF feature for training
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    train_features = [vectorizer.fit_transform(train_sentences)]
    # create additional custom features for training
    train_features.append(sp.csr_matrix(df_train.prev_fw).transpose())
    train_features.append(sp.csr_matrix(df_train.is_question).transpose())
    train_features.append(sp.csr_matrix(df_train.is_continuation).transpose())
    # create TF-IDF feature for testing
    test_features = [vectorizer.transform(test_sentences)]
    # create additional custom features for testing
    test_features.append(sp.csr_matrix(df_test.prev_fw).transpose())
    test_features.append(sp.csr_matrix(df_test.is_question).transpose())
    test_features.append(sp.csr_matrix(df_test.is_continuation).transpose())
    # create sparse matrix of training and testing features
    x_train = sp.hstack(train_features).tocsr()
    x_test = sp.hstack(test_features).tocsr()
    # fit and predict with the SVM
    classifier = LinearSVC()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print('=' * 100)
    print("BEST CLASSIFIER")
    print("Confusion Matrix=",confusion_matrix(y_test, y_pred))
    print("Precision,Recall,FScore,Support=",
          precision_recall_fscore_support(y_test, y_pred))

    incorrectly_classified = y_test != y_pred
    incorrectly_classified_sentences = df_test[incorrectly_classified]
    print("Number incorrectly_classified=",
          len(incorrectly_classified_sentences))
    print('=' * 100)

    # output incorrectly predicted sentences to the screen
    for row in incorrectly_classified_sentences.itertuples():
        print('%d,%d,%d,%d,%d,%d,%d,"%s"' %
              (row.row_index, row.doc_id, row.sent_id,
               row.prev_fw, row.is_question, row.is_continuation, row.is_fw,
               row.clean))
    print('=' * 100)


def main():
    """The root function called by the script.
    Outputs the results of the MNB and SVM experiments.
    """
    results_filename = './nb_and_svm_results.pickle'
    df_results = pd.read_pickle(results_filename)

    print("Total Count of MNB results=", len(df_results[is_mnb]))
    print("Total Count of SVM results=", len(df_results[is_svm]))

    df = df_results[get_mnb_additional_features_only].copy()
    frame_results(df, 'MNB Additional Features Only')
    plot_additional_features_only(df)
    print(output_to_csv(df, 6))

    df = df_results[get_mnb_ngrams].copy()
    frame_results(df, 'MNB Count and TF-IDF N-Grams')
    plot_mnb_count_and_tfidf_ngrams_2_4(df)
    plot_mnb_count_ngrams(df)
    plot_mnb_count_ngrams_2_4(df)
    print(output_to_csv(df))

    df = df_results[get_mnb_mean_w2v].copy()
    frame_results(df, 'MNB Mean Word2Vec')
    plot_mean_w2v(df)
    print(output_to_csv(df))

    df = df_results[get_svm_additional_features_only].copy()
    frame_results(df, 'SVM Additional Features Only')
    plot_additional_features_only(df)
    print(output_to_csv(df, 6))

    df = df_results[get_svm_ngrams].copy()
    frame_results(df, 'SVM Count and TF-IDF N-Grams')
    plot_svm_count_and_tfidf_ngrams_1_2(df)
    plot_svm_count_ngrams(df)
    plot_svm_tfidf_ngrams_1_2(df)
    print(output_to_csv(df))

    df = df_results[get_svm_mean_w2v].copy()
    frame_results(df, 'SVM Mean Word2Vec')
    plot_mean_w2v(df)
    print(output_to_csv(df))

    df = df_results[get_svm_nbsvm].copy()
    frame_results(df, 'SVM NBSVM')
    plot_svm_nbsvm_ngrams(df)
    plot_svm_nbsvm_ngrams_1_2(df)
    print(output_to_csv(df))

    predict_best(df_results)


if __name__ == '__main__':
    main()
