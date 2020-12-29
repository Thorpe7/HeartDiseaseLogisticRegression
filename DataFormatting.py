import numpy as np
from sklearn.datasets.base import Bunch
import csv
import pandas

# Convert dataset into correct formatting for sklearn features
def convert_my_data(csv_name_str):
    with open(csv_name_str) as csv_file:
        data_file = csv.reader(csv_file)
        pd_data_file = pandas.read_csv(csv_name_str)
        temp = next(data_file)
        n_samples = 299
        n_features = (len(list(pd_data_file.columns))-1) # Need the -1 to exclude the target feature
        names = list(pd_data_file.columns)
        name_target = ['DEATH_EVENT']
        data = np.empty((n_samples,n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float128)
            target[i] = np.asarray(sample[-1], dtype=np.int)
    return Bunch(data=data, target=target, feature_names = names, target_names = name_target)

