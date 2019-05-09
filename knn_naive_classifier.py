import random
import numpy as np
import math
import operator
from tabulate import tabulate
import copy
import random

'''
Programmer: Scott Kopczynski
Class: CPCS 310
Programming Assignment #5
3/20/2019

Description: This program creates a naive bayes classifier with/without using gauss distribution
for continuous attribes, and then uses it on multiple datasets Auto MPG, and Titanic. It also 
compares the naive bayes classifier accuracy to kNN, Zero R, and random classfier.
'''
global accuracies
accuracies = []
# Reads in a csv and stores it in a 2D array
def read_file_to_array(file_name):
    file = open(file_name, "r")
    data = file.read().splitlines()
    initial_processed_data = []
    for i in data:
        initial_processed_data.append(i.split(","))
    return initial_processed_data
# Given a 2D array will get all the columns for a provided index.
def get_column(table, column_index):
    column = []
    for i in table:
        if i[column_index] != "NA":
            column.append(i[column_index])
    return column
# Calculates the priors, for a given dataset. Table is all the data, and the class_index
# is the index in the table of the column that is trying to be predicted.
def calc_priors(table, class_index):
    classifications = get_column(table, class_index)
    unique_classes = set(classifications)
    prior_dictionary = {}
    for i in unique_classes:
        count = 0
        for j in classifications:
            if i == j:
                count+=1
        prior_dictionary[i] = count / len(classifications)
    return prior_dictionary
# Calculates the posteriors for a given set of data. Table is all the data, and the class_index
# is the index in the table of the column that is trying to be predicted. If gaussian is true a
# gaussian distribution is used to caculate the posterior for the gauss index.
def calc_posteriors(table, class_index, col_names, gaussian, gauss_index):
    priors = calc_priors(table, class_index)
    classifications = get_column(table, class_index)
    unique_classes = set(classifications)
    column_names = []
    for i in unique_classes:
        column_names.append(i)
    columns = []
    for i in range(0, class_index):
        columns.append(get_column(table, i))
    row_dict = {}
    count = 0
    for row in columns:
        unique_row = set(row)
        row_names = []
        for val in unique_row:
            row_names.append(val)
        row_dict[col_names[count]] = row_names
        count+=1
    posteriors = []
    for value in unique_classes:
        for key in row_dict:
            for val in row_dict[key]:
                count = 0
                for i in range(0,len(columns[col_names.index(key)])):
                    if(columns[col_names.index(key)][i] == val and classifications[i] == value):
                        if(gaussian):
                            if(col_names.index(key) != gauss_index):
                                count+=1
                            else:
                                posteriors.append([key, value, val, perform_gaussian(value, val, table, gauss_index)])
                        else:
                            count+=1
                posteriors.append([key,value, val, (count / len(columns[0]))/priors[value]])

    return posteriors
# Provided gaussian method from class.
def gaussian(x, mean, sdev):
  first, second = 0, 0
  if sdev > 0:
      first = 1 / (math.sqrt(2 * math.pi) * sdev)
      second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
  return first * second
# Caculates the posterior using a gaussian distribution.
def perform_gaussian(value, val, table, index):
    #value = class val = test instance, table, index that matters in table
    gauss_table = []
    for row in table:
        if(row[len(row)-1] == value):
            gauss_table.append(row[index])
    mean = np.mean(gauss_table)
    stdev = np.std(gauss_table)
    result = gaussian(val, mean ,stdev)
    return result
# Given the previously calculated priors, posteriors, and instance to predict, it will
# predict what that instance should be classified as.
def predict_instance(priors, posteriors, instance, possible_classes, header):
    results = []
    for val in possible_classes:
        result = priors[val]
        for i in range(0, len(instance)):
            for value in posteriors:
                if(value[0] == header[i] and instance[i] == value[2] and value[1] == val):
                    result*=value[3]
        results.append(result)
    final_result = possible_classes[results.index(max(results))]
    return final_result
# Generic function used to complete the first step, reused by steps 2 and 3.
def step_one(table, col_names, class_index, prediction_instance, is_gaussian, gauss_index):
    priors = calc_priors(table, class_index)
    posteriors = calc_posteriors(table,class_index, col_names[:class_index], is_gaussian, gauss_index)
    classifications = get_column(table, class_index)
    unique_classes = set(classifications)
    classes = []
    for val in unique_classes:
        classes.append(val)
    return predict_instance(priors, posteriors, prediction_instance, classes,col_names[:class_index])

# Makes weight categorical.
def categorical_weight(val):
    if val >= 3500:
        return 5
    elif val >= 3000 and val <= 3499:
        return 4
    elif val >= 2500 and val <= 2999:
        return 3
    elif val >= 2000 and val <= 2499:
        return 2
    elif val <= 1999:
        return 1
# Given a list of data, will pick 5 random instances.
def pick_rand_instance(data):
    rand_instances = []
    for i in range(0,5):
        rand_instances.append(data[random.randint(0,len(data))])
    return rand_instances
# Makes mpg categorical.
def put_in_mpg_bin(value):
    if(value <= 13):
        return 1
    elif value == 14:
        return 2
    elif (value >= 15 and value <= 16):
        return 3
    elif (value >= 17 and value <= 19):
        return 4
    elif (value >= 20 and value <= 23):
        return 5
    elif (value >= 24 and value <= 26):
        return 6
    elif (value >= 27 and value <= 30):
        return 7
    elif (value >= 31 and value <= 36):
        return 8
    elif (value >= 37 and value <= 44):
        return 9
    else:
        return 10
# Given a list will convert it to a string
def list_to_string(data):
    list_string = ""
    for i in data:
        list_string += (str(i) + " ")
    return list_string
# Performs a random 2:1 split on a the give data set, provided in class
def compute_holdout_partitions(table):
    # 2:1 split... 2/3 in training set 1/3 in test set
    # first lets randomized our table so that each
    # time we call this function we "should" get
    # a different test set 
    randomized = table[:] # shallow copy
    n = len(randomized)
    for i in range(n):
        # generate a random index to swap with i
        rand_index = random.randrange(0, n) # [0, n)
        # task: do the swap
        randomized[i], randomized[rand_index] = randomized[rand_index], randomized[i]

    # compute the split index
    split_index = int(2 / 3 * n)
    train_set = randomized[:split_index]
    test_set = randomized[split_index:]
    return train_set, test_set
# Partitions the data into k-folds.
# Round up partitions, if 31.5, then 32.
def determine_stratified_k_fold(data, k):
    partition_size = len(data) / k
    partition_size = round(partition_size)
    chunks = [data[x:x+partition_size] for x in range(0, len(data), partition_size)]
    return chunks
# Completes everything detailed in step 2 of the PA, also reused for step 3.
def step_two(is_gauss, gauss_index, is_step_three):
    auto_table = read_file_to_array("auto-data.txt")
    #Need Cylinders(1), Weight(4), Model year(6), and MPG(0)
    prepared_data = []
    for row in auto_table:
        if(not is_gauss):
            prepared_data.append([row[1], categorical_weight(int(row[4])), row[6] , row[0]])
        else:
            prepared_data.append([row[1], int(row[4]), row[6] , row[0]])
    header = ["cylinders", "weight", "model_year", "mpg"]
    rand_instances = pick_rand_instance(auto_table)
    if(is_step_three):
        print("===========================================")
        print("STEP 3: Random instances Naive Bayes(Gaussian)")
        print("===========================================")
    else:
        print("===========================================")
        print("STEP 2: Random instances Naive Bayes")
        print("===========================================")
    for row in rand_instances:
        print("Instance: ", list_to_string(row))
        print("Class: "+str(put_in_mpg_bin(int(float(step_one(prepared_data, header, 3, [row[1], categorical_weight(int(row[4])), row[6]], is_gauss, gauss_index))))))
        print("Actual: "+str(put_in_mpg_bin(int(float(rand_instances[rand_instances.index(row)][0])))))
    correct_counts = []
    for i in range(0, 10):
        training_set, test_set = compute_holdout_partitions(auto_table)
        training_set_prepped = []
        test_set_prepped = []
        for row in training_set:
            training_set_prepped.append([row[1], categorical_weight(int(row[4])), row[6] , row[0]])
        correct_count = 0
        for row in test_set:
            result = put_in_mpg_bin(int(float(step_one(training_set_prepped, header, 3, [row[1], categorical_weight(int(row[4])), row[6]], is_gauss, gauss_index))))
            if(result == put_in_mpg_bin(int(float(row[0])))):
                correct_count += 1
        correct_counts.append(correct_count)
    accuracy = (sum(correct_counts) / len(correct_counts))/len(test_set)
    if(is_step_three):
        print("===========================================")
        print("STEP 3: Predictive Accuracy(Gaussian)")
        print("===========================================")
        print("Random Subsample (k=10, 2:1 Train/Test)")
        print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    else:
        print("===========================================")
        print("STEP 2: Predictive Accuracy")
        print("===========================================")
        print("Random Subsample (k=10, 2:1 Train/Test)")
        print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    k_folds = determine_stratified_k_fold(auto_table, 10)
    correct_counts = []
    for i in range(0, len(k_folds)):
        train_data = []
        for j in range(0, len(k_folds)):
            if j != i:
                train_data+= k_folds[j]
        training_set_prep = []
        for row in train_data:
            training_set_prep.append([row[1], categorical_weight(int(row[4])), row[6] , row[0]])
        correct_count = 0
        for row in k_folds[i]:
            result = put_in_mpg_bin(int(float(step_one(training_set_prep, header, 3, [row[1], categorical_weight(int(row[4])), row[6]], is_gauss, gauss_index))))
            if(result == put_in_mpg_bin(int(float(row[0])))):
                correct_count += 1
        correct_counts.append(correct_count) 
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    if(is_step_three):
        print("===========================================")
        print("STEP 3: Predictive Accuracy(Gaussian)")
        print("===========================================")
        print("Stratified 10-Fold Cross Validation")
        print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    else:
        print("===========================================")
        print("STEP 2: Predictive Accuracy")
        print("===========================================")
        print("Stratified 10-Fold Cross Validation")
        print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3))) 
# Properly calls step two in order to complete step 3.
def step_three():
    step_two(True, 1, True)
# Converts the titanic data to numbers instead of strings so kNN can be used.
def prepare_data_for_knn(data):
    knn_list = copy.deepcopy(data)
    class_dict = {'first':1, 'second':2, 'third':3, 'crew':4}
    age_dict = {'child':1, 'adult':2}
    sex_dict = {'male':1, 'female':2}
    survided_dict = {'no':1, 'yes':2}
    for row in knn_list:
        row[0] = class_dict[row[0]]
        row[1] = age_dict[row[1]]
        row[2] = sex_dict[row[2]]
        row[3] = survided_dict[row[3]]
    return knn_list
# Calculates the euclidean distance between two points.
def compute_distance(v1, v2):
    assert(len(v1) == len(v2))
    dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist
# Perfomrs k nearest on the mpg data set, for the k = 5 vote takes the average of all
# the mpg values since there may be 5 different mpg values
def find_k_nearest_and_calculate(k, instance, data):
    temp_data = copy.deepcopy(data)
    for row in temp_data:
        row.append(compute_distance(row[:-2], instance[:-2]))
    temp_data.sort(key=operator.itemgetter(len(row) -1))
    count = 0
    class_vals = [val[len(val)-2] for val in temp_data[:5]]
    temp_list = []
    unique_vals = set(class_vals)
    for val in unique_vals:
        temp_list.append(class_vals.count(val))
    unique_vals = list(unique_vals)
    return unique_vals[temp_list.index(max(temp_list))]
# performs knn on an instance give a training data set.
def perform_knn(train_data, instance):
    return find_k_nearest_and_calculate(5, instance, train_data)
# Completes step 4 as detailed in the PA write up.
def step_four():
    titanic_data = read_file_to_array("titanic.txt")
    header = titanic_data[0]
    titanic_data = titanic_data[1:]
    knn_data = prepare_data_for_knn(titanic_data)
    k_folds = determine_stratified_k_fold(titanic_data, 10)
    k_folds_knn = determine_stratified_k_fold(knn_data, 10)
    correct_counts = []
    predicted = []
    actual = []
    for i in range(0, len(k_folds)):
        train_data = []
        for j in range(0, len(k_folds)):
            if j != i:
                train_data+= k_folds[j]
        correct_count = 0
        for row in k_folds[i]:
            #table, col_names, class_index, prediction_instance, is_gaussian, gauss_index
            result = step_one(titanic_data, header, 3, [row[0], row[1], row[2]], False, 0)
            predicted.append(result)
            actual.append(row[3])
            if(result == row[3]):
                correct_count += 1
        correct_counts.append(correct_count) 
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    accuracies.append(accuracy)
    print("===========================================")
    print("STEP 4: Predictive Accuracy(Naive Bayes)")
    print("===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    print("==========================================================")
    print("STEP 4: Predictive Accuracy(Naive Bayes) Confusion Matrix")
    print("==========================================================")
    confusion_vals = [["yes", 0, 0, 0, 0], ["no", 0, 0, 0, 0]]
    for i in range(0, len(predicted)):
        if(predicted[i] == "yes" and actual[i] == "yes"):
            confusion_vals[0][1] += 1
        elif(predicted[i] == 'no' and actual[i] =='no'):
            confusion_vals[1][2]+=1
        elif(predicted[i] == "yes" and actual[i] == "no"):
            confusion_vals[1][1]+=1
        elif(predicted[i] == "no" and actual[i] == "yes"):
            confusion_vals[0][2]+=1
    confusion_vals[0][3] = confusion_vals[0][1] + confusion_vals[0][2]
    confusion_vals[1][3] = confusion_vals[1][1] + confusion_vals[1][2]
    confusion_vals[0][4] = (confusion_vals[0][1] / confusion_vals[0][3])*100
    confusion_vals[1][4] = (confusion_vals[1][2] / confusion_vals[1][3])*100
    print (tabulate(confusion_vals, headers=['Survived','yes', 'no', 'Total', 'Recognition (%)'], tablefmt='orgtbl'))
    correct_counts = []
    predicted = []
    actual = []
    for i in range(0, len(k_folds_knn)):
        train_data = []
        for j in range(0, len(k_folds_knn)):
            if j != i:
                train_data+= k_folds_knn[j]
        correct_count = 0
        for row in k_folds_knn[i]:
            #table, col_names, class_index, prediction_instance, is_gaussian, gauss_index
            result = perform_knn(train_data, row)
            predicted.append(result)
            actual.append(row[3])
            if(result == row[3]):
                correct_count += 1
        correct_counts.append(correct_count) 
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    accuracies.append(accuracy)
    print("===========================================")
    print("STEP 4: Predictive Accuracy(kNN k = 5)")
    print("===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    survided_dict = {1:'no', 2:'yes'}
    confusion_vals = [["yes", 0, 0, 0, 0], ["no", 0, 0, 0, 0]]
    for i in range(0, len(predicted)):
        if(survided_dict[predicted[i]] == "yes" and survided_dict[actual[i]] == "yes"):
            confusion_vals[0][1] += 1
        elif(survided_dict[predicted[i]] == 'no' and survided_dict[actual[i]] =='no'):
            confusion_vals[1][2]+=1
        elif(survided_dict[predicted[i]] == "yes" and survided_dict[actual[i]] == "no"):
            confusion_vals[1][1]+=1
        elif(survided_dict[predicted[i]] == "no" and survided_dict[actual[i]] == "yes"):
            confusion_vals[0][2]+=1
    confusion_vals[0][3] = confusion_vals[0][1] + confusion_vals[0][2]
    confusion_vals[1][3] = confusion_vals[1][1] + confusion_vals[1][2]
    confusion_vals[0][4] = (confusion_vals[0][1] / confusion_vals[0][3])*100
    confusion_vals[1][4] = (confusion_vals[1][2] / confusion_vals[1][3])*100
    print("==========================================================")
    print("STEP 4: Predictive Accuracy(kNN) Confusion Matrix")
    print("==========================================================")
    print (tabulate(confusion_vals, headers=['Survived','yes', 'no', 'Total', 'Recognition (%)'], tablefmt='orgtbl'))
# Zero r classifier used for step 5, always predicts the most common class label in the training set.
def zero_r_classifier(train_data, instance):
    count_yes = 0
    count_no = 0
    for row in train_data:
        if(row[3] == 'yes'):
            count_yes += 1
        else:
            count_no += 1
    if count_yes > count_no:
        return 'yes'
    else:
        return 'no'
# Random classifier for step 5, classifies an instance by randomly choosing a class label 
# (class labels probabilities of being chosen are weighted based on their frequency in the training set)
def random_classifier(train_data, instance):
    count_yes = 0
    count_no = 0
    for row in train_data:
        if(row[3] == 'yes'):
            count_yes += 1
        else:
            count_no += 1
    yes_prob = count_yes / len(train_data)
    yes_prob = yes_prob * 100
    rand_num = random.randint(1,101)
    if(float(rand_num) <= yes_prob):
        return "yes"
    else:
        return 'no'
# Completes step 5 of the PA as detailed in the write up
def step_five():
    titanic_data = read_file_to_array("titanic.txt")
    header = titanic_data[0]
    titanic_data = titanic_data[1:]
    k_folds = determine_stratified_k_fold(titanic_data, 10)
    predicted = []
    actual = []
    correct_counts = []
    for i in range(0, len(k_folds)):
        train_data = []
        for j in range(0, len(k_folds)):
            if j != i:
                train_data+= k_folds[j]
        correct_count = 0
        for row in k_folds[i]:
            #table, col_names, class_index, prediction_instance, is_gaussian, gauss_index
            result = zero_r_classifier(train_data, row)
            predicted.append(result)
            actual.append(row[3])
            if(result == row[3]):
                correct_count += 1
        correct_counts.append(correct_count) 
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    print("===========================================")
    print("STEP 5: Predictive Accuracy(Zero R)")
    print("===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    k_folds = determine_stratified_k_fold(titanic_data, 10)
    predicted = []
    actual = []
    correct_counts = []
    for i in range(0, len(k_folds)):
        train_data = []
        for j in range(0, len(k_folds)):
            if j != i:
                train_data+= k_folds[j]
        correct_count = 0
        for row in k_folds[i]:
            #table, col_names, class_index, prediction_instance, is_gaussian, gauss_index
            result = random_classifier(train_data, row)
            predicted.append(result)
            actual.append(row[3])
            if(result == row[3]):
                correct_count += 1
        correct_counts.append(correct_count) 
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    print("===========================================")
    print("STEP 5: Predictive Accuracy(Random_Classifier)")
    print("===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("Accuracy: " + str(round(accuracy, 3)) + " Error Rate: " + str(round(1-accuracy, 3)))
    print("===========================================")
    print("STEP 5: Predictive Accuracy(Naive Bayes)")
    print("===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("Accuracy: " + str(round(accuracies[0], 3)) + " Error Rate: " + str(round(1-accuracies[0], 3)))
    print("===========================================")
    print("STEP 5: Predictive Accuracy(kNN k = 5)")
    print("===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("Accuracy: " + str(round(accuracies[1], 3)) + " Error Rate: " + str(round(1-accuracies[1], 3)))


