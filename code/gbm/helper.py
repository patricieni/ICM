from pathlib import Path
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


data_path = str(Path(os.getcwd())) + "/data/"


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

    df_amelia['life_expectancy_bin'] = LabelEncoder().fit_transform(
        df_amelia['life_expectancy_bin'])
    # df_amelia.drop("life_expectancy", axis=1, inplace =True)

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
                      'life_expectancy_bin', 'Gender', 'IK', 'Age_surgery']
    dummy_cols = list(set(df.columns) - set(non_dummy_cols))

    df = pd.get_dummies(df, columns=dummy_cols)

    df.Gender.replace(to_replace={'M': 1, 'F': 0}, inplace=True)


def get_train_test_data(data_df, regression=False, train_size=0.8):
    df_tensorflow = data_df.copy(deep=True)
    X_train, X_test = train_test_split(df_tensorflow,
                                       train_size=train_size,
                                       test_size=1 - train_size)

    if regression:
        Y_train = X_train['life_expectancy']
        Y_test = X_test['life_expectancy']
    else:
        Y_train = X_train['life_expectancy_bin']
        Y_test = X_test['life_expectancy_bin']

    # remove columns
    X_train.drop('life_expectancy', axis=1, inplace=True)
    X_test.drop('life_expectancy', axis=1, inplace=True)

    X_train.drop('life_expectancy_bin', axis=1, inplace=True)
    X_test.drop('life_expectancy_bin', axis=1, inplace=True)

    return X_train, Y_train, X_test, Y_test
