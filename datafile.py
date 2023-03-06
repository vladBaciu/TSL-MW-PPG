import os
import pandas as pd
from formats.format import *
import config

TIMESTAMP = 'Timestamp'


class DataFile:
    def __init__(self, filename, labels):
        self.filename = filename

        self.df = None
        self.labels_list = []

        ext = os.path.splitext(filename)[1]
        self.io = get_format(ext)

        if self.io is None:
            config.logger.error("Unrecognized format for file: {}".format(self.filename))
            raise UnrecognizedFormatError

        self.read()
        self.update_labels_list(labels)

    def read(self):
        self.df = self.io.read(self.filename)
        if self.df is None:
            config.logger.error("Cannot read file {}, is it structured correctly?".format(self.filename))
            raise BadFileError

    def get_shape(self):
        return self.df.shape[0]

    def get_data_columns(self):
        data_col = []
        for i, key in enumerate(self.df):
            if key != TIMESTAMP:
                data_col.append(i)
        return data_col

    def get_original_columns(self):
        functions = config.get_functions()

        orig_col = []
        for i, key in enumerate(self.df):
            if key not in functions:
                orig_col.append(i)
        return orig_col

    def get_function_columns(self):
        functions = config.get_functions()

        func_col = []
        for i, key in enumerate(self.df):
            if key in functions:
                func_col.append(i)
        return func_col

    def get_data_header(self):
        col_names = []
        for key in list(self.df):
            if key != TIMESTAMP:
                col_names.append(key)
        return col_names

    def get_timestamp(self):
        if TIMESTAMP not in list(self.df):
            return []
        return pd.to_datetime(self.df[TIMESTAMP])

    @staticmethod
    def get_label_range(label_col):
        indexes = [i for i, val in enumerate(label_col) if val == 1.0]
        return indexes[0], indexes[-1]

    def update_labels_list(self, labels):
        self.labels_list = []
        for i, key in enumerate(list(self.df)):
            if ''.join(labels) in key:
                data = self.df.iloc[:, i]
                self.labels_list.append([key, self.get_label_range(data)])

        # Labels are removed from DataFrame to avoid mistakes
        for label in self.labels_list:
            if label[0] in list(self.df):
                del self.df[label[0]]

    def get_label_series(self, label):
        a = label[1][0]
        b = label[1][1] + 1
        n_rows = self.get_shape()
        s = pd.Series(n_rows * ['0'], name=label[0])
        for i in range(a, b):
            s.iat[i] = '1'
        return s

    def merge_same_label_types(self, columns):
        available_labels, _       = config.get_labels_info()
        _, is_channel_independent = config.get_additional_options()
        plots_number, _           = config.get_plot_info()
        n_rows                    = self.get_shape()
        new_columns = []
        new_labels  = []

        if is_channel_independent == 'true':
            #create new labels for each channel
            for _, label in enumerate(available_labels):
                for j in range(len(plots_number)):
                    new_labels.append(label + '_ch' + str(j))
        else:
            #do not modify labels
            new_labels = available_labels

        for label in new_labels:
            new_column = pd.Series(n_rows * ['0'], name=label)
            #get columns for the same label type - to be merged
            column_index = [j for j, entry in enumerate(self.labels_list) if entry[0] == label]
            #if multiple columns for the same label, merge to a new column
            if len(column_index):
                for row in range(0, len(new_column)):
                    #loop through each row in the column matrix
                    for index in column_index:
                        if columns[index][row] == '1':
                            new_column[row] = '1'
                            break
            new_columns.append(new_column)
        return pd.concat(new_columns, axis=1)

    def labels_list_to_df(self):
        if not self.labels_list:
            return None

        columns = [self.get_label_series(l) for l in self.labels_list]
        all_columns = pd.concat(columns, axis=1)
        return all_columns

    def labels_merged_list_to_df(self):
        if not self.labels_list:
            return None

        columns = [self.get_label_series(l) for l in self.labels_list]
        all_columns = self.merge_same_label_types(columns)
        return all_columns

    def save(self):
        label_df = self.labels_list_to_df()
        func_df = self.df.iloc[:, self.get_function_columns()]
        all_data = self.df.iloc[:, self.get_original_columns()]

        if func_df is not None:
            all_data_final = pd.concat([all_data, func_df], axis=1)
        if label_df is not None:
            all_data_final = pd.concat([all_data, label_df], axis=1)

        self.io.save(all_data_final, self.filename)

        # merge labels if is an anomaly detection project
        anomaly_detection_options = config.get_additional_options()
        if(anomaly_detection_options[0] == 'true'):
            #merge columns together
            label_df = self.labels_merged_list_to_df()
            all_data_merged_labels = pd.concat([all_data, label_df], axis=1)

            #create new directory for labeled data
            new_dir_path = os.path.join(os.path.dirname(self.filename), "ad_labeled_files")
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_path = self.filename.replace(os.path.dirname(self.filename), new_dir_path)

            #save labeled data
            self.io.save(all_data_merged_labels, new_file_path)

    def get_series_to_process(self, column, name):
        data = self.df.iloc[:, column]
        index = pd.DatetimeIndex(self.df[TIMESTAMP]) if TIMESTAMP in self.df else self.df.index
        return pd.Series(data.values, index=index, name=name)

    def add_function(self, series):
        self.df = pd.concat([self.df, series], axis=1)

    def remove_function(self, f_name):
        del self.df[f_name]
