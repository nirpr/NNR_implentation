import time
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import List


def data_init(data_trn, data_vld, df_tst):
    """
    the function initializes the data sets and dataframes given and returns them scaled, also returns y_arrays.
    """
    df_trn = pd.read_csv(data_trn)
    df_vld = pd.read_csv(data_vld)

    y_trn_np = df_trn['class'].to_numpy()
    y_vld_np = df_vld['class'].to_numpy()

    scaler = StandardScaler()
    x_trn_scaled = scaler.fit_transform(df_trn.drop('class', axis=1))
    x_vld_scaled = scaler.transform(df_vld.drop('class', axis=1))
    x_tst_scaled = scaler.transform(df_tst)

    return x_trn_scaled, x_vld_scaled, x_tst_scaled, y_trn_np, y_vld_np


def best_radius(distances, y_vld_np, y_train_np, radius_range):
    """
    the function finds the radius with the best accuracy in the provided range.
    """
    best_accuracy = 0.0
    best_rad = -1

    for radius in radius_range:
        predictions = predict(distances, y_train_np, radius)
        accuracy = accuracy_score(y_vld_np, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_rad = radius

    return best_rad


def predict(distances, class_array, radius):
    """
    the function gets an npArray of distances, a class array and radius and returns a list of predicted classes.
    """
    num_instances = distances.shape[1]
    predictions = np.full(num_instances, "", dtype=object)

    for test_ind in range(num_instances):
        test_distances = distances[:, test_ind]
        closest_indices = np.where(test_distances <= radius)[0]

        if closest_indices.size > 0:
            closest_class = class_array[closest_indices]
            unique_classes, class_counts = np.unique(closest_class, return_counts=True)
            max_count = np.max(class_counts)
            most_frequent_classes = unique_classes[class_counts == max_count]
            predicted_class = np.random.choice(
                most_frequent_classes)  # Choose randomly if multiple classes have the same count
            predictions[test_ind] = predicted_class

    return predictions


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')
    # initialize data
    x_trn_scaled, x_vld_scaled, x_tst_scaled, y_trn_np, y_vld_np = data_init(data_trn, data_vld, df_tst)

    # calculate all the distances and enter them to numpy arrays for easy calculations.
    distances_train_val = np.sqrt(np.sum((x_trn_scaled[:, np.newaxis] - x_vld_scaled) ** 2, axis=2))
    distances_train_test = np.sqrt(np.sum((x_trn_scaled[:, np.newaxis] - x_tst_scaled) ** 2, axis=2))

    min_radius = np.min(distances_train_val, 1)  # set the range min
    max_radius = np.max(distances_train_val)  # set the range max
    num_points = int(np.sqrt(distances_train_val.shape[0] / 2))  # after a lot of tries this had the best results
    radius_range = np.linspace(min_radius, max_radius, num_points)

    radius = best_radius(distances_train_val, y_vld_np, y_trn_np, radius_range)
    predictions = predict(distances_train_test, y_trn_np, radius)  # predict
    return list(predictions)


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
