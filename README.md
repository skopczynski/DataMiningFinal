# DataMiningFinal
## Running the classifiers
To run the classifiers an anaconda distribution of python is required, it is then as simple as running python3 main.py to utililze the program.
## Program Structure
main.py reads in all the data and then stores it in a 2D list, following that all the data is cleaned, discretized, and encoded(for svm) in main.py. In main.py is also where the function calls are made to generate all of the figures in the Figures folder. Finally in main.py is where the KNN, Naive Bayes, and Ensemble Random Decision forest classifiers are imported and used. 

The file knn_naive_classifier.py contains all of the necessary source code to fit a KNN and Naive Bayes classifier to the current set of data. The results of both classifiers are then tested in main.py using K-Fold cross validation.

The file ensemble_classifier.py contains all of the necessary source code to fit a random decision forest ot the current set of data. The results of this classifier are then tested in main.py using K-Fold cross validation.
