#!/usr/bin/env python

# ___________________________________________________ Libraries ___________________________________________________ #


# Algorithm libraries
from sklearn.metrics import \
    classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import \
    train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Visualization libraries
from PIL import Image
from sklearn.tree import export_graphviz
from plot_metrics import \
    plot_classification_report, plot_feature_importances
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pydot

# Text processing libraries
from assembly_preprocessing import Assembly_Preprocessor
import pandas as pd

# Other libraries
import time
import numpy as np

# ___________________________________________ Constants and Definitions ___________________________________________ #


DATASET_PATH = r'INSERT_DATASET_PATH_HERE'
TEST_PATH = r'INSERT_TEST_PATH_HERE'
TEST_SIZE = 0.2
TREE_TO_SAVE = 5
N_TREES = 500

# _____________________________________________ Classes and Functions _____________________________________________ #


class RandomForestAgent:

    def __init__(self, n_trees):
        self.Classifier = \
            RandomForestClassifier(n_estimators=n_trees, random_state=0)

    def build_model(self, X, Y, test_size):
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=test_size, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        self.Classifier.fit(X_train, Y_train)
        return X_test, Y_test

    def dataset_predict(self, X_test, Y_test):
        prediction = self.Classifier.predict(X_test)
        return (prediction,
                confusion_matrix(Y_test, prediction),
                classification_report(Y_test, prediction),
                accuracy_score(Y_test, prediction))

    def test_predict(self, X):
        sc = StandardScaler()
        X_train = sc.fit_transform(X)
        X_test = sc.transform(X)
        prediction = self.Classifier.predict(X_test)
        return prediction

    def evaluate_score(self, X, Y):
        kf = KFold(random_state=42, shuffle=True)
        score = cross_val_score(self.Classifier, X, Y, cv=kf, scoring="accuracy")
        print("K-Fold Cross Validation score: ", score, "\n[Mean: ", np.mean(score), "]")
        kf = StratifiedKFold(random_state=42, shuffle=True)
        score = cross_val_score(self.Classifier, X, Y, cv=kf, scoring="accuracy")
        print("Stratified K-Fold score: ", score, "\n[Mean: ", np.mean(score), "]")

    def tree_to_png(self, features, labels):
        estimator = self.Classifier.estimators_[TREE_TO_SAVE]
        export_graphviz(estimator, out_file='tree.dot', 
                feature_names = features,
                class_names = labels,
                precision = 1, filled = True)
        (graph, ) = pydot.graph_from_dot_file('tree.dot')
        graph.write_png('tree.png')

    def render(self, X_test, Y_test, classification_report, features):
        plot_confusion_matrix(self.Classifier, X_test, Y_test)
        plt.title("Confusion Matrix")
        plt.show()
        plt.close()
        plot_classification_report(classification_report)
        plt.savefig('Classification_Report.png', dpi=200, format='png', bbox_inches='tight')
        plt.show()
        plt.close()
        plot_feature_importances(features, self.Classifier.feature_importances_)
        plt.show()
        
# _____________________________________________________ Main _____________________________________________________ #


if __name__ == '__main__':

    # Object initializing
    print("Initialization...")
    test_size = TEST_SIZE
    agent = RandomForestAgent(N_TREES)
    preprocessor = Assembly_Preprocessor()

    # Data processing
    print("Data processing...")
    dataset = (pd.read_json(DATASET_PATH, lines=True)).drop(columns="cfg")
    dataset_function_list = (dataset[['lista_asm']]).lista_asm
    X = preprocessor.complete_preprocessing(dataset_function_list)
    Y = dataset['semantic']

    # Training Algorithm
    print("Training and testing...")
    X_test, Y_test = agent.build_model(X, Y, test_size)
    dataset_prediction, confusion_matrix, classification_report, accuracy_score = \
        agent.dataset_predict(X_test, Y_test)
        
    # Show Algorithm results
    print("\n_____________________ Original DataFrame _____________________\n\n", dataset)
    time.sleep(2)
    print("\n_____________________ Processed DataFrame _____________________\n\n", X)
    time.sleep(2)
    print("\n\n\n _____________________ Algorithm Results _____________________\n")
    print("  - Confusion Matrix:\n\n", confusion_matrix, "\n\n")
    print("  - Classification Report:\n\n", classification_report)
    print("  - Accuracy: ", accuracy_score)
    choice = ""
    while choice != "y" and choice != "n":
        choice = input("Do you want to see results images? (y/n): ")
    if choice == "y":
        print("Good choice! Close windows to advance through the program\n")
        time.sleep(2)
        features = [feature for feature in X]
        labels = []
        for label in Y:
            if label not in labels:
                labels.append(label)
        agent.render(X_test, Y_test, classification_report, features)
        ## optionally save a tree as png
        #print("Saving tree...")
        #agent.tree_to_png(features, labels)

    # K-Fold evaluation
    choice = ""
    while choice != "y" and choice != "n":
        choice = input("Do you want to evaluate algorithm accuracy? (y/n): ")
    if choice == "y":
        agent.evaluate_score(X, Y)
    
    # Blind test
    print("Blind test...")
    test_set = (pd.read_json(TEST_PATH, lines=True)).drop(columns="cfg")
    test_function_list = (test_set[['lista_asm']]).lista_asm
    X = preprocessor.complete_preprocessing(test_function_list)
    test_prediction = agent.test_predict(X)
    ## optionally show blind test prediction
    #print("Test prediction: \n")
    #time.sleep(1)
    #print(test_prediction)

# ________________________________________________________________________________________________________________ #
