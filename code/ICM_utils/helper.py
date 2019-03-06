from pathlib import Path
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_path = str(Path(os.getcwd()).parent) + "/data/"


def write_to_pickle(dataframe, name):
    dataframe.to_pickle(data_path + name + ".pickle")


def read_from_pickle(name):
    return pd.read_pickle(data_path + name + ".pickle")


def binning(col, cut_points, labels=None):
    # Define min and max values:
    minval = col.min()
    maxval = col.max()

    # create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]

    # if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points) + 1)

    # Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin


def label_encoder(df1, labels, cut_points):
    """Process the dataframe into integer categories using 
    LabelEncoder plus some previously defined logic

    Args:
        df1 (pandas.df): Dataframe to transform
        labels (list): Labels
        cut_points (list): Binning points

    Returns:
        pandas.df, dict: Transformed dataframe and its associated dictionary
    """
    df = df1.copy(deep=True)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    df.loc[:, "life_expectancy_bin"] = binning(
        df.life_expectancy, cut_points, labels)

    df['life_expectancy_bin'] = LabelEncoder(
    ).fit_transform(df['life_expectancy_bin'])

    # Initialise an empty dictionary to keep all LabelEncoders
    le_dict = dict()
    # LabelEncoders
    df_categories = df.copy(deep=True)
    # Loop over attributes by excluding the continuous oness
    for column in df_categories.drop(
            ['Age_surgery', 'life_expectancy', 'Tumor_grade', 'IDH_TERT',
             'IK'], axis=1):
        le = LabelEncoder().fit(
            df_categories[column])  # Initialise the LabelEncoder and fit
        df_categories[column] = le.transform(df_categories[
            column])
        # Transform data and save in credit_clean DataFrame
        le_dict[column] = le  # Store the LabelEncdoer in dictionary

    return df_categories, le_dict


def process_dataset(df_amelia, labels, cut_points):
    """Process any dataset given a name and some binning cutting points
    The binning is used for classification purposes.
    Args:
        df_amelia (pandas.df): Dataframe to process
        labels (list): Labels
        cut_points (list): bin points

    Returns:
        dataframe,labels: The processed dataframe and labels
    """
    df = df_amelia.copy(deep=True)
    df.drop("Unnamed: 0", axis=1, inplace=True)

    df.loc[:, "life_expectancy_bin"] = binning(
        df.life_expectancy, cut_points, labels)

    non_dummy_cols = ['Tumor_grade', 'IDH_TERT', 'life_expectancy',
                      'life_expectancy_bin', 'Gender', 'IK', 'Age_surgery', 'TERT']
    dummy_cols = list(set(df.columns) - set(non_dummy_cols))

    df = pd.get_dummies(df, columns=dummy_cols)

    df.Gender.replace(to_replace={'M': 1, 'F': 0}, inplace=True)
    df.TERT.replace(to_replace={'wt': 0, 'mutant': 1}, inplace=True)

    return df, labels


# This needs to go away eventually
def process_amelia(amelia_csv_fn):
    # Load the dataset
    df_amelia = pd.read_csv(amelia_csv_fn)
    df_amelia.drop("Unnamed: 0", axis=1, inplace=True)
    labels = ["1.5year", "4years", "more"]
    cut_points = [500, 1500]

    # labels = ["3_months","6_months","9_months","12_months","15_months",
    # "18_months","2_years","3_years","4_years","5_years","10_years",
    # "10_plus_years"] cut_points = [90,180,270,360,450,540,720,1095,1460,
    # 1825,3650]
    df_amelia.loc[:, "life_expectancy_bin"] = binning(
        df_amelia.life_expectancy, cut_points, labels)

    df_amelia['life_expectancy_bin'] = LabelEncoder(
    ).fit_transform(df_amelia['life_expectancy_bin'])

    le_dict = dict()  # Initialise an empty dictionary to keep all
    # LabelEncoders
    df_categories = df_amelia.copy(deep=True)
    # Loop over attributes by excluding the continuous oness
    for column in df_categories.drop(
            ['Age_surgery', 'life_expectancy', 'Tumor_grade', 'IDH_TERT',
             'IK'], axis=1):
        le = LabelEncoder().fit(
            df_categories[column])  # Initialise the LabelEncoder and fit
        df_categories[column] = le.transform(df_categories[
            column])
        # Transform data and save in credit_clean DataFrame
        le_dict[column] = le  # Store the LabelEncdoer in dictionary

    df = df_amelia.copy(deep=True)
    non_dummy_cols = ['Tumor_grade', 'IDH_TERT', 'life_expectancy',
                      'life_expectancy_bin', 'Gender', 'IK', 'Age_surgery', 'TERT']
    dummy_cols = list(set(df.columns) - set(non_dummy_cols))

    df = pd.get_dummies(df, columns=dummy_cols)

    df.Gender.replace(to_replace={'M': 1, 'F': 0}, inplace=True)
    df.TERT.replace(to_replace={'wt': 0, 'mutant': 1}, inplace=True)

    return df, labels


def display_values(df_amelia):
    for column in df_amelia:
        unique_vals = np.unique(df_amelia[column])
        nr_vals = len(unique_vals)
        if nr_vals < 20:
            print(
                'Number of values for attribute {}: {} -- {}'.format(column, nr_vals, unique_vals))
        else:
            print('Number of values for attribute {}: {}'.format(column, nr_vals))


def get_train_test_data(data_df, regression=False, train_size=0.8, random_state=1232):
    """TODO: Paul to approve, it was too complicated before

    Args:
        data_df ([type]): [description]
        regression (bool, optional): Defaults to False. [description]
        train_size (float, optional): Defaults to 0.8. [description]
        random_state (int, optional): Defaults to 1232. [description]

    Returns:
        [type]: [description]
    """

    df = data_df.copy(deep=True)

    if regression:
        Y = df.life_expectancy
    else:
        Y = df.life_expectancy_bin

    # Remove columns
    X = df.drop(["life_expectancy", "life_expectancy_bin"], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        train_size=train_size, random_state=random_state)

    return X_train, Y_train, X_test, Y_test


from collections import defaultdict


def from_dummies(df_dummies):
    """Super specialized method for reverting from dummies dataframe to original dataframe for our tumor dataset
    from_dummies(pd.get_dummies(X)) == X

    Args:
        df_dummies (pd.dataframe): The dummies dataframe

    Returns:
        pd.dataframe: The original dataframe before dummies was applied
    """

    # This works mostly for our dataset
    pos = defaultdict(list)
    vals = defaultdict(list)

    for i, c in enumerate(df_dummies.columns):
        if "_" in c:
            # Split at second one only, if it doesn't have three don't join
            values = c.split("_", -1)
            if len(values) > 2:
                k = "_".join(values[:2])
                v = values[-1]
                pos[k].append(i)
                vals[k].append(v)
            elif values[1] in ["mutant", "wt", "NC"]:
                k = values[0]
                v = values[1]
                pos[k].append(i)
                vals[k].append(v)
            else:
                # The continuous variables that have _ in the name
                k = "_".join(values)
                pos["_"].append(i)
        else:
            # Continuous variables with no _ in the name
            k = c
            pos["_"].append(i)

    df = pd.DataFrame({k: pd.Categorical.from_codes(
        np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
        vals[k])
        for k in vals})
    # Copy the Continuous non-dummies and keep them the same
    indexes = df_dummies.columns[pos["_"]]
    all_vals = df_dummies.iloc[:, pos["_"]]
    all_vals = all_vals.reset_index(drop=True)
    # df.reset_index(drop=True)
    df[indexes] = all_vals

    return df


def dropnull(df):
    """Drop any row that could contain NA values
    """

    for col in df.columns:
        print('{0}\n  {1}\n'.format(col, df[col].isnull().value_counts()))
    df_1 = df.dropna(axis=0, how='any')

# Scale and visualize the embedding vectors


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
