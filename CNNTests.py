"""Executes the CNN experiments

Adjust the TRAINING_STEPS = 50000 variable to 2000, 20000 or 50000 to execute
the different stages.
"""

#__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support, \
                            confusion_matrix, \
                            accuracy_score
from Logging import LoggingWrapper, log
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators \
    import model_fn as model_fn_lib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from itertools import product, chain

tf.logging.set_verbosity(tf.logging.INFO)

# This is used to determine how many training steps to execute for
# TRAINING_STEPS = 2000
# TRAINING_STEPS = 20000
TRAINING_STEPS = 50000


def cnn_model_fn(features, labels, mode, params):
    """Builds the model for the CNN

    :param features: the matrix containing the features
    :param labels: the class labels
    :param mode: training mode, evaluation mode or inference mode.
    :param params: additional parameters
    :return: a model_fn
    """
    # Input Layer
    num_additional_features = 0
    if 'use_additional_features' in params:
        num_additional_features = 3

    if num_additional_features == 0:
        word2vec_data = features
        additional_features_data = None
    else:
        word2vec_data = tf.slice(
            features,
            [0,0],
            [-1, params['max_sentence_len'] * params['w2v_size']])
        additional_features_data = tf.slice(
            features,
            [0, params['max_sentence_len'] * params['w2v_size']],
            [-1, num_additional_features])

    word2vec_reshape = tf.reshape(word2vec_data,
                                  [-1, params['max_sentence_len'],
                                   params['w2v_size'], 1])

    pools = []
    for filter_size in [3,4,5]:
        with tf.name_scope("Conv_%s_by_%s_Pool_" %
                                   (filter_size, filter_size)):
            # Convolutional Layer
            conv = tf.layers.conv2d(
                inputs=word2vec_reshape,
                filters=params['num_filters'],
                kernel_size=(filter_size, params['w2v_size']),
                activation=tf.nn.relu)

            # Pooling Layer
            pool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=(params['max_sentence_len'] - filter_size + 1, 1),
                strides=1,
                padding="valid")
            pools.append(pool)

    pools_concat = tf.concat(pools, 3)

    # Dense Layer
    # number of filters * number of filter types
    pool_flat = tf.reshape(pools_concat, [-1, params['num_filters'] * 3])
    if additional_features_data is None:
        dropout_input = pool_flat
    else:
        dropout_input = tf.concat(
            [pool_flat, additional_features_data], axis=1)

    dropout = tf.layers.dropout(inputs=dropout_input, rate=0.4,
                                training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer=tf.train.AdadeltaOptimizer())

    # Generate Predictions
    predictions = {
        learn.PredictionKey.CLASSES: tf.argmax(input=logits, axis=1),
        learn.PredictionKey.PROBABILITIES: tf.nn.softmax(
            logits, name="softmax_tensor"),
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def split_max_tokens(sentence, max_sentence_len=200):
    """Splits the sentence into tokens/words padded with '<PAD>' to
    max_sentence_len.

    :param sentence: sentence to split
    :param max_sentence_len: max number of words in sentence
    :return:
    """
    split_sentence = sentence.split()[:max_sentence_len]
    num_missing_words = max_sentence_len - len(split_sentence)
    split_sentence.extend(['<PAD>'] * num_missing_words)
    return split_sentence


def create_get_vector_func(w2v):
    """Creates a function to get a Word2Vec vector for a token

    :param w2v: the Word2Vec dictionary to bind to the function
    :return: returns a function to get a Word2Vec vector for a token
    """
    values = np.array(list(w2v.values()))
    std = values.std(axis=0)
    mean = values.mean(axis=0)

    def get_vec(token):
        if token in w2v:
            return w2v[token]
        else:
            # creates a random vector with mean and standard deviation
            # from the distribution of w2v vectors.
            new_vec = np.random.normal(mean, std)
            w2v[token] = new_vec
            return new_vec

    return get_vec


def create_features(df,
                    max_sentence_len=200,
                    w2v_size=300,
                    num_filters=100,
                    sampling_class=None,
                    use_additional_features=False):
    """Creates a feature matrix from a set of feature parameters.

    :param df: sentence DataFrame
    :param max_sentence_len: The maximum sentence length to trim or pad to
    :param w2v_size: The Word2Vec dimension
    :param num_filters: The number of filters to use
    :param sampling_class: The sampling class to use
    :param use_additional_features: include the three custom features
    :return: a feature vector
    """

    df_train = df[df['is_train']]
    df_test = df[~df['is_train']]
    y_train = np.array(df_train['is_fw'], np.int8)
    y_test = np.array(df_test['is_fw'], np.int8)
    train_sentences = list(df_train['clean'])
    test_sentences = list(df_test['clean'])
    train_sentences_max_tokens = list(map(
        lambda s: split_max_tokens(s, max_sentence_len), train_sentences))
    test_sentences_max_tokens = list(map(
        lambda s: split_max_tokens(s, max_sentence_len), test_sentences))
    w2v = Word2Vec(train_sentences_max_tokens,
                   size=w2v_size, min_count=1, seed=1234)
    w2v = {token: w2v.wv.syn0[vocab.index]
           for token, vocab in w2v.wv.vocab.items()}

    get_vec_func = create_get_vector_func(w2v)
    x_train = np.array([list(map(get_vec_func, s))
                        for s in train_sentences_max_tokens], np.float32)
    x_test = np.array([list(map(get_vec_func, s))
                       for s in test_sentences_max_tokens], np.float32)

    # convert the word2vec matrices to a vectors
    x_train = x_train.reshape([-1, max_sentence_len * w2v_size])
    x_test = x_test.reshape([-1, max_sentence_len * w2v_size])

    if use_additional_features:
        columns = ['prev_fw', 'is_question', 'is_continuation']
        additional_features_train = np.array(df_train[columns], np.float32)
        additional_features_test = np.array(df_test[columns], np.float32)
        x_train = np.concatenate((x_train, additional_features_train), axis=1)
        x_test = np.concatenate((x_test, additional_features_test), axis=1)

    sampling_class_name = 'None'
    if sampling_class is not None:
        sampler = sampling_class(random_state=1234)
        sampling_class_name = sampling_class.__name__
        x_train, y_train = sampler.fit_sample(x_train, y_train)
        x_train = x_train.astype(np.float32)

    model_name = "cnn_TS_%d_SL_%d_W2V_%d_FLT_%d_AF_%d_SC_%s" % \
                 (TRAINING_STEPS, max_sentence_len, w2v_size,
                  num_filters, use_additional_features,
                  sampling_class_name)

    return x_train, x_test, y_train, y_test, model_name


def get_results_dataframe(results_filename):
    """Get the results DataFrame.

    :param results_filename: the results filename
    :return: A pandas DataFrame or None if the file cannot be found
    """
    if os.path.exists(results_filename):
        return pd.read_pickle(results_filename)
    return None


def get_feature_string(feature_args):
    """Converts the feature_args dictionary to a pretty string

    :param feature_args: the feature arguments dictionary
    :return: a pretty string representation of the feature arguments
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


def create_result_dict(model_name, feature_args, fit_time, fscores, loss,
                       precisions, predict_time, recalls, supports,
                       x_train, y_pred, y_test, y_train):
    """Creates a result dictionary which will then be appended to the results
    pandas DataFrame.

    :param model_name: the name of the model
    :param feature_args: the feature args dictionary
    :param fit_time: the time taken to fit the model
    :param fscores: the f1-scores for the model
    :param loss: the loss at the end of training the model
    :param precisions: the precisions for the model
    :param predict_time: the time taken to predict using the model
    :param recalls: the recalls for the model
    :param supports: the supports for the model
    :param x_train: the x training features
    :param y_pred: the predicted values
    :param y_test: the y test features
    :param y_train: the y training features
    :return: a results dictionary
    """
    result = dict(classifier_name='CNN',
                  model_name=model_name,
                  features=feature_args,
                  feature_string=get_feature_string(feature_args),
                  training_steps=TRAINING_STEPS,
                  x_train_shape=x_train.shape,
                  y_train_shape=y_train.shape,
                  x_test_shape=y_test.shape,
                  y_test_shape=y_test.shape,
                  created=pd.Timestamp.now(),
                  fit_time=fit_time,
                  predict_time=predict_time,
                  loss=loss,
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
                  prfs_macro=precision_recall_fscore_support(y_test, y_pred, average='macro'),
                  prfs_micro=precision_recall_fscore_support(y_test, y_pred, average='micro'),
                  prfs_weighted=precision_recall_fscore_support(y_test, y_pred, average='weighted'),
                  use_additional_features=feature_args.get('use_additional_features', None),
                  weight_additional_features=feature_args.get('weight_additional_features', None),
                  sampling_class=feature_args.get('sampling_class', None)
                  )
    return result


def result_already_exists(df, feature_args):
    """Does the result already exist in the results DataFrame.

    :param df: results DataFrame
    :param feature_args: the feature arguments dictionary for the model
    :return: True if the model exists otherwise False
    """
    if df is None:
        return False
    # Checks for the existence of the feature arguments and for the current
    # training steps
    results = df[(df['features'] == feature_args) &
                 (df['training_steps'] == TRAINING_STEPS)]
    if len(results) > 0:
        return True
    return False


def get_display_results_dataframe(df_results):
    """Gets an abbreviated results DataFrame for sorted by F1-Score for
    display to the screen.

    :param df_results: the results DataFrame
    :return: an abbreviated results DataFrame for sorted by F1-Score
    """
    return df_results[['model_name',
                       'features',
                       'feature_string',
                       'training_steps',
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
                       'loss']].sort_values(['fscore_1',
                                             'fscore_0'], ascending=[False,
                                                                     False])


def display_results(df_results, top_n=10000):
    """Displays results to the screen.

    :param df_results: the results DataFrame
    :param top_n: the top n results to show
    """
    df_display = get_display_results_dataframe(df_results)
    len_results = len(df_results)
    min_top_n = np.min([top_n, len_results])
    print(' ID  FIT_T PRED_T   ACC  PR_0  PR_1  RL_0  RL_1  F1_0  F1_1  '
          'Loss (%d of %d)' % (min_top_n, len_results))
    print('=== ====== ====== ===== ===== ===== ===== ===== ===== ===== =====')
    for i, row in enumerate(df_display[:top_n].itertuples()):
        print(row.x_train_shape)
        print('%3d %06.2f %06.2f %04.3f %04.3f %04.3f %04.3f %04.3f %04.3f '
              '%04.3f %04.3f %d %r %r' %
              (i + 1, row.fit_time, row.predict_time,
               row.accuracy, row.precision_0, row.precision_1,
               row.recall_0, row.recall_1, row.fscore_0, row.fscore_1,
               row.loss, row.training_steps,
               row.model_name, row.x_train_shape))


def create_all_features(df):
    """Creates all of the feature combinations to experiment with

    :param df: the results DataFrame
    :return: the feature combinations to experiment with
    """
    features = []
    if TRAINING_STEPS == 2000:
        sampling_classes = [None, RandomUnderSampler, RandomOverSampler,
                            SMOTE]
        max_sentence_lens = [100, 150, 200]
        w2v_sizes = [100, 300, 500]
        num_filters = [100, 200, 300]
        use_additional_features = [False, True]
        all_feature_combinations = list(product(max_sentence_lens, w2v_sizes,
                                                num_filters,
                                                use_additional_features,
                                                sampling_classes))

        for max_sentence_len, w2v_size, num_filter, use_additional_feature, \
            sampling_class in all_feature_combinations:
            f = {'max_sentence_len': max_sentence_len,
                 'w2v_size': w2v_size,
                 'num_filters': num_filter}
            if use_additional_feature:
                f['use_additional_features'] = True
            if sampling_class is not None:
                f['sampling_class'] = sampling_class
            features.append(f)
    elif TRAINING_STEPS == 20000:
        # top 20 - 2000 step results
        features = list(df[lambda r:r.training_steps == 2000]
                        .sort_values(['fscore_1'],
                                     ascending=[False])[:20]['features'])
    elif TRAINING_STEPS == 50000:
        # top 20 - 20000 step results
        features = list(df[lambda r:r.training_steps == 20000]
                        .sort_values(['fscore_1'],
                                     ascending=[False])[:3]['features'])

    len_features = len(features)
    return features, len_features


def main(unused_argv):
    """ The main function to execute the experiments

    :param unused_argv: an unused argument needed for tf.app.run()
    """
    with LoggingWrapper("CNNTests"):
        np.random.seed(1234)
        results_filename = './cnn_results.pickle'
        df_results = get_results_dataframe(results_filename)
        with LoggingWrapper("Loading Data"):
            df_sentences = pd.read_pickle('./conclusions_dataframe.pickle')

        features, len_features = create_all_features(df_results)
        for feature_args_idx, feature_args in enumerate(features):
            with LoggingWrapper("Training features %d of %d: %r" %
                                (feature_args_idx+1,
                                 len_features, feature_args)):
                if result_already_exists(df_results, feature_args):
                    log("Already exists %d of %d: %r"
                        % (feature_args_idx + 1, len_features, feature_args))
                    continue

                with LoggingWrapper("Creating features"):
                    x_train, x_test, y_train, y_test, model_name = \
                        create_features(df_sentences, **feature_args)

                for data in [x_train, x_test, y_train, y_test]:
                    print(data.shape)

                # with LoggingWrapper("Create the estimator"):
                model_dir = './logs/%s' % model_name
                mnist_classifier = learn.Estimator(model_fn=cnn_model_fn,
                                                   model_dir=model_dir,
                                                   params=feature_args)

                # Set up logging for predictions
                # tensors_to_log = {"probabilities": "softmax_tensor"}
                tensors_to_log = {}
                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=50)

                validation_metrics = {
                    "accuracy":
                        learn.MetricSpec(
                            metric_fn=tf.contrib.metrics.streaming_accuracy,
                            prediction_key=learn.PredictionKey.CLASSES),
                    "precision":
                        learn.MetricSpec(
                            metric_fn=tf.contrib.metrics.streaming_precision,
                            prediction_key=learn.PredictionKey.CLASSES),
                    "recall":
                        learn.MetricSpec(
                            metric_fn=tf.contrib.metrics.streaming_recall,
                            prediction_key=learn.PredictionKey.CLASSES),
                }
                validation_monitor = learn.monitors.ValidationMonitor(
                    x_test, y_test, batch_size=50,every_n_steps=50,
                    metrics=validation_metrics)
                with LoggingWrapper("Fit the model to the training data") \
                        as lw:
                    mnist_classifier.fit(
                        x=x_train,
                        y=y_train,
                        batch_size=50,
                        steps=TRAINING_STEPS,
                        monitors=[logging_hook, validation_monitor])
                    fit_time = lw.elapsed()

                # Configure the accuracy metric for evaluation
                evaluation_metrics = {
                    "accuracy":
                        learn.MetricSpec(
                            metric_fn=tf.metrics.accuracy,
                            prediction_key=learn.PredictionKey.CLASSES),
                    "precision":
                        learn.MetricSpec(
                            metric_fn=tf.metrics.precision,
                            prediction_key=learn.PredictionKey.CLASSES),
                    "recall":
                        learn.MetricSpec(
                            metric_fn=tf.metrics.recall,
                            prediction_key=learn.PredictionKey.CLASSES),
                }

                with LoggingWrapper("Evaluate the model"):
                    eval_results = mnist_classifier.evaluate(
                        x=x_test, y=y_test, batch_size=50,
                        metrics=evaluation_metrics)
                    print('eval_results', eval_results)

                with LoggingWrapper("Predict using the model") as lw:
                    pred_results = mnist_classifier.predict(
                        x=x_test, batch_size=50, as_iterable=False)
                    predict_time = lw.elapsed()
                    y_pred = pred_results[learn.PredictionKey.CLASSES]
                    conf_mat = confusion_matrix(y_test, y_pred)
                    precisions, recalls, fscores, supports = \
                        precision_recall_fscore_support(y_test, y_pred)

                print('\n%s\n' % feature_args)
                print(' ID  PREC RECAL F1SCR  CM_0  CM_1')
                print('=== ===== ===== ===== ===== =====')
                metrics = enumerate(zip(precisions, recalls,
                                        fscores, conf_mat))
                for i, (precision, recall, fscore, cm) in metrics:
                    print('%3d %04.3f %04.3f %04.3f %5d %5d' %
                          (i, precision, recall, fscore, cm[0], cm[1]))

                result = create_result_dict(
                    model_name, feature_args, fit_time, fscores,
                    eval_results['loss'], precisions, predict_time, recalls,
                    supports, x_train, y_pred, y_test, y_train)

                if df_results is None:
                    df_results = pd.DataFrame([result])
                else:
                    df_results = df_results.append(result, ignore_index=True)
                df_results.to_pickle(results_filename)


if __name__ == '__main__':
    tf.app.run()