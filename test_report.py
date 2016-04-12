#! /usr/bin/env python2
""" Utilities to calculate and write the confusion matrix and the
    classification report"""

import os
from glob import glob
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize

from code import classification


TEST_DIR = os.path.join(os.getcwd(), 'for_testing')


def populate(file_ext):
    """ Read file_ext kind of files """
    file_list = list()
    for filename in glob(os.path.join(TEST_DIR, file_ext)):
        file_list.append(filename)
    return file_list


def get_real_y(filename, inv_target_names):
    """ Get real target """
    filename = os.path.split(filename)[1]
    filename = os.path.splitext(filename)[0]
    label = ''.join(
        char for char in filename if char.isalpha())
    return inv_target_names[label]


def main(args):
    """ Write out classification report and confusion matrix """
    _, file_ext, sample_rate, package, out_name = args
    package = pickle.load(open(package, 'rb'))
    target_names = package[1]
    inv_target_names = {val: key for key, val in target_names.items()}
    y_true, y_pred = [list() for i in range(0, 2)]
    file_list = populate(file_ext)
    for filename in file_list:
        y_pred_temp, _, _ = classification.query(filename,
                                                 sample_rate, package)
        y_true_temp = get_real_y(filename, inv_target_names)
        y_true.append(y_true_temp)
        y_pred.append(y_pred_temp[0])
    report = classification_report(y_true, y_pred,
                                   target_names=target_names.values())
    conf_matrix = normalize(
        np.array(
            confusion_matrix(y_true, y_pred)).astype(np.float64),
        axis=1, norm='l1')
    np.set_printoptions(precision=2)
    plt.figure(figsize=(9, 9))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greys)
    plt.colorbar()
    tick_marks = np.arange(len(target_names.values()))
    plt.xticks(tick_marks, target_names.values(), rotation=45)
    plt.yticks(tick_marks, target_names.values())
    plt.tight_layout()
    plt.subplots_adjust(left=0.175)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(out_name)
    with open("{}.cm.txt".format(out_name), 'wb') as text_file:
        text_file.write("Confusion Matrix\n---------\n" + str(conf_matrix))
        text_file.write("Categories\n----------\n" +
                        str(target_names.values()))

    with open("{}.txt".format(out_name), 'wb') as text_file:
        text_file.write(report)


if __name__ == '__main__':
    main(sys.argv)
