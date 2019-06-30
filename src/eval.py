import glob
import os
import pickle
import sys
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import tensorflow as tf
from matplotlib.ticker import MultipleLocator
from sklearn import model_selection as sk_ms
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVC
import gen_plots as ut

global LAB

matplotlib.style.use('classic')
PARAMS = {'legend.fontsize': 'xx-large',
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'xx-large'}
plt.rcParams.update(PARAMS)

def cross_validate_all(model, data, labels):
    """Pass the data and model to the cross validation function"""

    print("Running Cross Validation")
    model = model
    dum_unifrom = DummyClassifier(strategy='uniform') #uniform most_frequent stratified
    dum_stratified = DummyClassifier(strategy='stratified')
    dum_freq = DummyClassifier(strategy='most_frequent')

    scoring = ['accuracy', 'f1_micro', 'f1_macro']

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, data, labels, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1)
    dum_uniform_scores = cross_validate(dum_unifrom, data, labels, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1)
    dum_stratified_scores = cross_validate(dum_stratified, data, labels, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1)
    dum_freq_scores = cross_validate(dum_freq, data, labels, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1)

    acc_mean = np.around(scores['test_accuracy'].mean(), decimals=3)
    acc_std = np.around(scores['test_accuracy'].std(), decimals=3)

    micro_mean = np.around(scores['test_f1_micro'].mean(), decimals=3)
    micro_std = np.around(scores['test_f1_micro'].std(), decimals=3)

    macro_mean = np.around(scores['test_f1_macro'].mean(), decimals=3)
    macro_std = np.around(scores['test_f1_macro'].std(), decimals=3)

    # calculate the lift over random score
    dum_mean = dum_uniform_scores['test_f1_micro'].mean()
    strat_mean = dum_stratified_scores['test_f1_micro'].mean()
    freq_mean = dum_freq_scores['test_f1_micro'].mean()

    dum_lift_mean = np.around(((acc_mean-dum_mean)/dum_mean)*100, decimals=2)
    strat_lift_mean = np.around(((acc_mean-strat_mean)/strat_mean)*100, decimals=2)
    freq_lift_mean = np.around(((acc_mean-freq_mean)/freq_mean)*100, decimals=2)

    return acc_mean, acc_std, micro_mean, micro_std, macro_mean, macro_std, dum_lift_mean, strat_lift_mean, freq_lift_mean

    #acc = sk_ms.cross_val_score(model, data, labels, cv=10, scoring='accuracy', n_jobs=-1)
    #f1_macro = sk_ms.cross_val_score(model, data, labels, cv=10, scoring='f1_macro', n_jobs=-1)
    #f1_micro = sk_ms.cross_val_score(model, data, labels, cv=10, scoring='f1_micro', n_jobs=-1)

    #return acc, f1_macro, f1_micro

def run_linear_reg(X, Y, test_ratio=0.3):

    #http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html#sphx-glr-auto-examples-plot-cv-predict-py
    # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor

    #Y = preprocessing.scale(Y, axis=0)

    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, train_size=test_ratio)

    # Create the logreg with balanced class weights 
    nn_model = MLPRegressor()
    #Fit the model
    fit_nn = nn_model.fit(X_train, Y_train)

    # Predict on the test set
    pred = fit_nn.predict(X_test)

    print(pred)

    print(nn_model.loss_curve_)

    print(mean_squared_error(Y_test, pred))
    print(r2_score(Y_test, pred))


    # Plot outputs
    plt.scatter(Y_test, pred,  color='black')

    plt.xticks(())
    plt.yticks(())

    plt.show()

    return 0

def balanced_log_reg(X, Y, test_ratio=0.2, class_weight=None):
    """Run the logreg model"""

    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_ratio)

    # Create the logreg with balanced class weights 
    logreg = lr(class_weight=class_weight)
    #Fit the model
    fit_logreg = logreg.fit(X_train, Y_train)
    # Get the model acccuracy
    score = fit_logreg.score(X_test, Y_test)
    # Predict on the test set
    pred = fit_logreg.predict(X_test)

    micro = f1_score(Y_test, pred, average='micro')
    macro = f1_score(Y_test, pred, average='macro')

    a, mic, mac = dummy_classification(X_train, X_test, Y_train, Y_test, 'most_frequent')

    print(a)
    print(mic)
    print(mac)

    return score, micro, macro

#TODO: put back to trainsplit for generating figs
def balanced_SVM(X, Y, test_ratio=0.2, class_weight=None):
    """Run the SVM model with cbf kernel"""

    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_ratio)

    # USE THIS! to calculate the random guess level and level for just predicting the majority class
    print(np.unique(Y_test, return_counts=True))

    # Create the logreg with balanced class weights 
    svm_model = SVC(class_weight=class_weight)
    #Fit the model
    fit_svm = svm_model.fit(X_train, Y_train)
    # Get the model acccuracy
    score = fit_svm.score(X_test, Y_test)
    # Predict on the test set
    pred = fit_svm.predict(X_test)

    micro = f1_score(Y_test, pred, average='micro')
    macro = f1_score(Y_test, pred, average='macro')

    a, mic, mac = dummy_classification(X_train, X_test, Y_train, Y_test, 'most_frequent')

    print(a)
    print(mic)
    print(mac)

    return score, micro, macro

def balanced_NN(X, Y, train_ratio=0.9):
    """Run the NN model"""

    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, train_size=train_ratio)

    # Create the logreg with balanced class weights 
    nn_model = MLPClassifier(hidden_layer_sizes=(500,))
    #Fit the model
    fit_nn = nn_model.fit(X_train, Y_train)
    # Get the model acccuracy
    score = fit_nn.score(X_test, Y_test)
    # Predict on the test set
    pred = fit_nn.predict(X_test)

    micro = f1_score(Y_test, pred, average='micro')
    macro = f1_score(Y_test, pred, average='macro')

    return score, micro, macro

def dummy_classification(X_train, X_test, Y_train, Y_test, mode):

    dum = DummyClassifier(strategy=mode)
    fit_dum = dum.fit(X_train, Y_train)
    score = fit_dum.score(X_test, Y_test)
    pred = fit_dum.predict(X_test)

    micro = f1_score(Y_test, pred, average='micro')
    macro = f1_score(Y_test, pred, average='macro')

    return score, micro, macro


def bins_values_and_create_class_labels(labels, num_bins):
    """Bin the labels based on a histogram"""

    n, bins = np.histogram(labels, bins=num_bins)
    class_array = []

    print("Bin edges and counts")
    print(bins)
    print(n)

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html

    # Loop through labels and bin based on histogram
    for val in labels:
        for i in range(len(bins)-1):
            if val >= bins[i] and val <= bins[i+1]:
                class_array.append(i)
                break

    labels = class_array

    return labels

def load_and_preprocess_data(filename, feature, num_bins, norm=True):
    """Load and process the label data -- including the binning process for the features"""

    total_graph = nx.read_gpickle(filename)
    print("Graph Loaded")

    # extract graph features/labels
    if feature == 1:
        labels = nx.get_node_attributes(total_graph, "DEG")
        print('Degree ======================================')
    elif feature == 2:
        labels = nx.get_node_attributes(total_graph, "TR")
        print('Triangle Count ======================================')
    elif feature == 3:
        labels = nx.get_node_attributes(total_graph, "PR")
        print('Page Rank ======================================')
    elif feature == 4:
        labels = nx.get_node_attributes(total_graph, "DC")
        print('Degree Centrality ======================================')
    elif feature == 5:
        labels = nx.get_node_attributes(total_graph, "CLU")
        print('Clustering Coefficient ======================================')
    elif feature == 6:
        labels = nx.get_node_attributes(total_graph, 'EC')
        print('Eigen Vector Centrality ======================================')
    elif feature == 7:
        labels = nx.get_node_attributes(total_graph, 'BC')
        print('Betweenness Centrality ======================================')
    else:
        raise ValueError('Invalid Feature Selection')

    print("Labels Loaded")
    labels = np.absolute(np.asarray(list(labels.values())))

   # Get the zero elements if any
    zero_elements = np.where(labels == 0)[0]
    non_zeros = np.where(labels != 0)[0]

    # Take the log of the none zero elements
    if norm:
        print("Using Log Norm on Vertex Features")
        labels[non_zeros] = np.log(labels[non_zeros])

    # Bin the non-zero labels
    # labels[non_zeros] = bins_values_and_create_class_labels(labels[non_zeros], num_bins)
    # # Add one to increase the class number of the none zero elements
    # labels = labels+1
    # # Create the zero labels as class zero (although should be zero anyway)
    # labels[zero_elements] = 0
    # print("Label Binning Completed")

    return labels.astype(float)

def createErrorMatrix():
    """Create an error matrix for the multiclass model """

    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn.metrics import confusion_matrix
    import itertools

    #dataset_location = '../data/embeddings/facebook_combined-node2vec-hom.emb'
    dataset_location = '../data/embeddings/soc-sign-bitcoinotc-autoenc.emb'
    labels_location = '../data/features/soc-sign-bitcoinotc.ini.gz'
    feature = 2

    # Load the dataset in the required format
    if dataset_location.endswith('.emb'):
        print("Loading .emb file  -  " + dataset_location.split('/')[-1])
        data = np.loadtxt(dataset_location)
    else:
        raise ValueError('Invalid file format')
    
    labels = load_and_preprocess_data(labels_location, feature, 6, norm=True)
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(data, labels, train_size=0.6)

    # Create the NN with balanced class weights 
    nn_model = MLPClassifier(hidden_layer_sizes=(500,))
    #Fit the model
    fit_nn = nn_model.fit(X_train, Y_train)
    y_pred = fit_nn.predict(X_test)

    print(np.unique(Y_test))
    class_names = np.unique(Y_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, y_pred, labels=class_names)
    np.set_printoptions(precision=2)

    # Normalise
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({'font.size': 16})

    # Plot normalized confusion matrix
    plt.figure()
    ut.plot_error_matrix(cnf_matrix, classes=class_names)
    plt.tight_layout()
    plt.savefig('EM-facebook_combined-node2vec-hom.pdf', format='PDF', bbox_inches='tight')
    plt.show()

def generate_classification_alg_choice_results(bins=5, norm=True, balance=None):

    dataset_location = '../data/embeddings/facebook_combined-autoenc.emb'
    labels_location = '../data/features/facebook_combined.ini.gz'

    # Load the dataset in the required format
    if dataset_location.endswith('.emb'):
        print("Loading .emb file  -  " + dataset_location.split('/')[-1])
        data = np.loadtxt(dataset_location)
    else:
        raise ValueError('Invalid file format')

    test_feats = [1,2,6]

    # loop through all the features
    for i in test_feats:

        # Load the labels
        labels = load_and_preprocess_data(labels_location, i, bins, norm=norm)

        svm_lin = SVC(kernel='linear', class_weight=balance)
        svm = SVC(class_weight=balance)
        logr = lr(class_weight=balance)
        nn = MLPClassifier()
        complex_nn = MLPClassifier(hidden_layer_sizes=(1000, 500))
        models = [logr, svm_lin, svm, nn, complex_nn]
        
        for model in models:
            acc_mean, acc_std, micro_mean, micro_std, macro_mean, macro_std, dum_lift_mean, strat_lift_mean, freq_lift_mean = cross_validate_all(model, data, labels)
            print(model)
            print('Micro', micro_mean, '+', micro_std)
            print('Macro', macro_mean, '+', macro_std)
            print('Uniform Lift', dum_lift_mean)
            print('Stratified Lift', strat_lift_mean)
            print('Freq Lift', freq_lift_mean)
            print('---------------------------------------------------------------')

    return 0

def run_multi_class(dataset_location, labels_location, feature=1):
    """Load the data and labels and run the logreg classification"""

    # Load the dataset in the required format
    if dataset_location.endswith('.emb'):
        print("Loading .emb file  -  " + dataset_location.split('/')[-1])
        data = np.loadtxt(dataset_location)
    else:
        raise ValueError('Invalid file format')
    
    # Load datasets
    labels = load_and_preprocess_data(labels_location, feature, 5)

    labelling_faction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Loop through all the labelling fractions
    mean_score = []
    mean_micro = []
    mean_macro = []
    std_score = []
    std_micro = []
    std_macro = []
    results = {}
    for frac in labelling_faction:
        print(frac)

        average_score = []
        average_macro = []
        average_micro = []
        # Run the logreg for 10 repeats
        for i in range(10):

            # create a new model with the required split
            score, micro, macro = balanced_log_reg(data, labels, frac)
            average_score.append(score)
            average_micro.append(micro)
            average_macro.append(macro)

        #print(np.mean(average_score))
        #print(np.mean(average_micro))
        #print(np.mean(average_macro))
        #print(np.std(average_score))
        #print(np.std(average_micro))
        #print(np.std(average_macro))
        mean_score.append(np.mean(average_score))
        mean_micro.append(np.mean(average_micro))
        mean_macro.append(np.mean(average_macro))
        std_score.append(np.std(average_score))
        std_micro.append(np.std(average_micro))
        std_macro.append(np.std(average_macro))
    
    results['avg_score'] = mean_score
    results['avg_micro'] = mean_micro
    results['avg_macro'] = mean_macro
    results['std_score'] = std_score
    results['std_micro'] = std_micro
    results['std_macro'] = std_macro
    #print (results)

    return results

if __name__ == "__main__":

    generate_classification_alg_choice_results(bins=5, norm=True, balance=None)