"""Analyses the CNN results.

Outputs tables and box plots measuring the performance of the CNN.
"""

#__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

import matplotlib.pyplot as plt
import pandas as pd
from CNNTests import display_results, get_display_results_dataframe


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
    """Output results in CSV format

    :param df_results: results DataFrame
    :param top_n: how many results to output
    :return: a string containing the results in CSV format
    """
    columns = ['training_steps',
               'precision_0',
               'precision_1',
               'recall_0',
               'recall_1',
               'fscore_0',
               'fscore_1',
               'feature_string']
    df = get_display_results_dataframe(df_results)[:top_n]
    return df[columns].to_csv(float_format='%g')


def get_2000_training_steps(row):
    """DataFrame filter to retrieve rows where training steps is 2000

    :param row: the row to compare
    :return: True, if training step is 2000, otherwise False
    """
    return row.training_steps == 2000


def get_20000_training_steps(row):
    """DataFrame filter to retrieve rows where training steps is 20000

    :param row: the row to compare
    :return: True, if training step is 20000, otherwise False
    """
    return row.training_steps == 20000


def get_50000_training_steps(row):
    """DataFrame filter to retrieve rows where training steps is 50000

    :param row: the row to compare
    :return: True, if training step is 50000, otherwise False
    """
    return row.training_steps == 50000


def get_class_name(obj):
    """Gets an abbreviated class name containing the upper case letters from
    the class name

    :param obj: the object to get an abbreviated class name for
    :return: the upper case letters from the class name
    """
    if obj is None:
        return 'None'
    else:
        class_name = ''.join(char for char in obj.__name__ if char.isupper())
        return class_name


def create_box_plots(df, features):
    """Create a plot containing box plots for the five features
    in a 2 by 3 grid.
    Outputs the plot to the screen and an SVG file.

    :param df: the results DataFrame
    :param features: list of feature names to plot
    """
    training_steps = df.iloc[0]['training_steps']
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True)
    axes[1, 2].axis('off')  # hides the sixth empty boxplot
    i = 0
    for feature in features:
        x = []
        labels = []
        df2 = df[[feature, 'fscore_1']]

        # create a list of F1-Scores for each distinct value of the feature
        for group, fscore_1_list in df2.groupby(feature)['fscore_1']:
            x.append(fscore_1_list)
            labels.append(group)

        ax = axes[i//3, i % 3]
        ax.boxplot(x=x, labels=labels)
        ax.set_ylabel('F1-Score')
        ax.set_xlabel(feature)
        i += 1

    # outputs the plot to an svg format file and displays it to the screen
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Boxplot of F1-Score for %d training steps" % training_steps)
    plt.savefig("./plots/cnn_%d_training_steps.svg" % training_steps)
    plt.show()


def main():
    """This is the root function call of the script.
    Generates three plots, for 2000, 20000 and 50000 training steps.
    Outputs the plots to the screen and to SVG files.
    """
    results_filename = './cnn_results.pickle'
    df_results = pd.read_pickle(results_filename)
    features = ['sampling_class', 'max_sentence_len', 'num_filters',
                'use_additional_features', 'w2v_size']
    # creates a new feature column with a printable representation of the
    # feature.
    for feature in features:
        if feature == 'sampling_class':
            df_results[feature] = \
                df_results['features'].apply(lambda x:
                                             get_class_name(x.get(feature,
                                                                  None)))
        elif feature == 'use_additional_features':
            df_results[feature] = \
                df_results['features'].apply(lambda x: x.get(feature, False))
        else:
            df_results[feature] = \
                df_results['features'].apply(lambda x: x.get(feature, None))

    print("Total Count of CNN results=", len(df_results))

    df = df_results[get_2000_training_steps]
    frame_results(df, 'CNN trained for 2000 steps')
    create_box_plots(df, features)
    print(output_to_csv(df))

    df = df_results[get_20000_training_steps]
    frame_results(df, 'CNN trained for 20000 steps')
    create_box_plots(df, features)
    print(output_to_csv(df))

    df = df_results[get_50000_training_steps]
    frame_results(df, 'CNN trained for 50000 steps')
    create_box_plots(df, features)
    print(output_to_csv(df))


if __name__ == '__main__':
    main()
