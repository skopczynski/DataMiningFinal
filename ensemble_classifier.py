##############################################
# Programmer: Nathan Vanos
# Class: CPSC 310, Spring 2019
# Programming Assignment # 7 (Bonus)
# 4/18/2019
#
# Description: Implements Ensemble Classification with
#               Random Forests of Decision Trees
# Usage: python3.7 pa7.py
##############################################
import copy
import math
import numpy as np
import random as rand
import sys

# main program
def main():
    # Note: step 5 is integrated within other steps
    # step 1: implement basic random forest algorithm
    """classify_interviews()
    # step 2: run ensemble over auto and titanic datasets"""
    classify_titanic(20, 7, 2)
    classify_auto(20, 7, 2)
    # step 3: run ensemble over auto and titanic with different N, M, and F
    classify_titanic(80, 20, 20)
    classify_auto(80, 20, 20)
    classify_titanic(10, 4, 10)
    classify_auto(10, 4, 10)
    # step 4: run ensemble over wisconsin cancer data
    print("***********************Step 4:")
    classify_wisconsin(50, 10, 5)
    classify_wisconsin(100, 10, 2)
    classify_wisconsin(20, 10, 5)
    classify_wisconsin(50, 10, 20)

# performs step 1 of pa7 
def classify_interviews():
    col_names = ["level", "lang", "tweets", "phd", "interviewed_well"]
    table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
        ]
    att_domains = {0: ["Senior", "Mid", "Junior"],
        1: ["R", "Python", "Java"],
        2: ["yes", "no"],
        3:["yes", "no"]}
    att_indices = list(range(len(col_names) - 1))
    class_index = len(col_names) - 1
    remainder_set, test_set = compute_holdout_partitions(table)
    predictions, accuracy, test_set = ensemble(table, col_names, 20, 7, 2, att_domains, 
                                        class_index, att_indices, remainder_set, test_set)
    print("********* STEP 1: ")
    print_accuracy("Ensemble on Interview Dataset: ", accuracy)

# performs step 2 & 3 of pa7 (for titanic data)
def classify_titanic(N, M, F):
    # obtain data
    titanic_header = ["Class", "Age", "Sex", "Survived"]
    titanic_data = get_data("titanic.txt")
    titanic_table = get_table(titanic_data)
    titanic_table = titanic_table[1:]
    # prepare for ensemble
    att_indices = list(range(len(titanic_header) - 1))
    att_domains = get_att_domains(titanic_table, att_indices)
    class_index = titanic_header.index("Survived")
    remainder_set, test_set = compute_holdout_partitions(titanic_table)
    # create ensemble classifier
    predictions, accuracy, test_set = ensemble(titanic_table, titanic_header, N, M, F, att_domains, 
                                        class_index, att_indices, remainder_set, test_set)
    print("********* STEP 2 & 3: ")
    print_accuracy("Ensemble on Titanic Dataset: ", accuracy)
    class_names, class_groups = group_by(test_set, class_index)
    titanic_c_matrix = calc_conf_matrix(class_index, test_set, predictions, class_names)
    print_matrix("Ensemble on Titanic Dataset:", titanic_c_matrix)
    # now perform same process with 1 "normal" tree
    tree = tdidt(remainder_set, att_indices, att_domains, class_index, titanic_header, F)
    accuracy, predictions = do_normal_predictions(test_set, titanic_header,
                                                     tree, class_index)
    print_accuracy("Normal Tree on Titanic Dataset: ", accuracy)
    titanic_c_matrix = calc_conf_matrix(class_index, test_set, predictions, class_names)
    print_matrix("Normal Tree on Titanic Dataset:", titanic_c_matrix)

# performs step 2 & 3 of pa7 (for auto data) 
def classify_auto(N, M, F):
    auto_header = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight",
                   "Acceleration", "Model Year", "Origin", "Model Name", "MSRP"]
    auto_data = get_data("auto-data.txt")
    auto_table = get_table(auto_data)
    remove_na(auto_header, auto_table)
    # get rid of unneeded attributes
    required_attributes = ["MPG", "Cylinders", "Weight", "Model Year"]
    weight_classes = [1999, 2499, 2999, 3499, 10000]
    mpg_ratings = [14.0, 15.0, 17.0, 20.0, 24.0, 27.0, 31.0, 37.0, 45.0, 50.0]
    mpg_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    auto_table = parse_table(required_attributes, auto_header, auto_table)
    convert_to_categorical(weight_classes, required_attributes.index("Weight"),
                           auto_table)
    convert_to_categorical(mpg_ratings, required_attributes.index("MPG"),
                           auto_table)
    # prepare data for ensemble
    att_indices = list(range(1, len(required_attributes)))
    att_domains = get_att_domains(auto_table, att_indices)
    class_index = required_attributes.index("MPG")
    remainder_set, test_set = compute_holdout_partitions(auto_table)
    # create ensemble classifier
    predictions, accuracy, test_set = ensemble(auto_table, required_attributes, N, M, F, 
                                        att_domains, class_index, att_indices, remainder_set, test_set)
    print_accuracy("Ensemble on Automobile Dataset: ", accuracy)
    auto_c_matrix = calc_conf_matrix(class_index, test_set, predictions, mpg_labels)
    print_matrix("Ensemble on Automobile Dataset:" , auto_c_matrix)
    # now perform same process with 1 "normal" tree
    tree = tdidt(remainder_set, att_indices, att_domains, class_index, auto_header, F)
    accuracy, predictions = do_normal_predictions(test_set, auto_header,
                                                     tree, class_index)
    print_accuracy("Normal Tree on Titanic Dataset: ", accuracy)
    auto_c_matrix = calc_conf_matrix(class_index, test_set, predictions, mpg_labels)
    print_matrix("Normal Tree on Titanic Dataset:", auto_c_matrix)

# performs step 4 of pa7
def classify_wisconsin(N, M, F):
    # obtain data
    wisconsin_header = ["Clump Thickness", "Cell Size", "Cell Shape", "Marginal Adhesion",
                        "Epithelial Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                        "Mitoses", "Tumor"]
    wisconsin_data = get_data("wisconsin.txt")
    wisconsin_table = get_table(wisconsin_data)
    # prepare for ensemble
    att_indices = list(range(len(wisconsin_header) - 1))
    att_domains = get_att_domains(wisconsin_table, att_indices)
    class_index = wisconsin_header.index("Tumor")
    remainder_set, test_set = compute_holdout_partitions(wisconsin_table)
    # create ensemble classifier
    predictions, accuracy, test_set = ensemble(wisconsin_table, wisconsin_header, N, M, F, 
                                        att_domains, class_index, att_indices, 
                                        remainder_set, test_set)
    print_accuracy("Ensemble on Wisconsin Cancer Dataset: ", accuracy)
    class_names, class_groups = group_by(test_set, class_index)
    wisc_c_matrix = calc_conf_matrix(class_index, test_set, predictions, class_names)
    print_matrix("Ensemble on Wisconsin Cancer Dataset:", wisc_c_matrix)
    # now perform the same process with 1 "normal" tree
    tree = tdidt(remainder_set, att_indices, att_domains, class_index, wisconsin_header, F)
    accuracy, predictions = do_normal_predictions(test_set, wisconsin_header,
                                                     tree, class_index)
    print_accuracy("Normal Tree on Wisconsin Cancer Dataset: ", accuracy)
    wisc_c_matrix = calc_conf_matrix(class_index, test_set, predictions, class_names)
    print_matrix("Normal Tree on Wisconsin Cancer Dataset:", wisc_c_matrix)
    

# an ensemble classifier
def ensemble(table, header, N, M, F, att_domains, class_index, att_indices, remainder_set, test_set):
    # list to hold random forest
    decision_trees= []
    # list to hold accuracies
    tree_accuracies = []
    tree_track_records = []
    for t in range(0, N):
        # need a new att_indices each time
        att_indices_copy = copy.deepcopy(att_indices)
        # for each tree, generate a random training and validation set
        training_set, validation_set = compute_holdout_partitions(remainder_set)
        # use tdidt algorithm w/ f subset modification to generate a tree
        tree = tdidt(training_set, att_indices_copy, att_domains, class_index, header, F)
        # add tree to list
        decision_trees.append(tree)
        # make predictions and compute accuracy
        current_pred = []
        for i in range(len(validation_set)):
            prediction = classify_tdidt(tree, validation_set[i], header)
            current_pred.append(prediction)
        current_accuracy = compute_accuracy(header, validation_set, 
                                            current_pred, class_index)
        tree_accuracies.append(current_accuracy)
        # step 5
        track_record = compute_track_record(header, current_pred, validation_set, class_index, table)
        tree_track_records.append(track_record)
    # now, get the M most accurate classifiers
    ensemble_classifier, track_records = compute_most_accurate(tree_accuracies, tree_track_records, 
                                                                decision_trees, M)
    # now, use m most accurate to vote on class label
    ensemble_predictions = []
    class_names, class_groups = group_by(table, class_index)
    for i in range(len(test_set)):
        current_instance_preds = []
        for tree in ensemble_classifier:
            t = ensemble_classifier.index(tree)
            pred = classify_tdidt(tree, test_set[i], header)
            weight = track_records[t][class_names.index(pred)]
            current_instance_preds.append((pred, weight))
        current_instance_prediction = make_prediction(current_instance_preds, test_set, 
                                                        class_index)
        ensemble_predictions.append(current_instance_prediction)
    # lastly, compute the accuracy of the random forest approach
    ensemble_accuracy = compute_accuracy(header, test_set, ensemble_predictions, class_index)
    return ensemble_predictions, ensemble_accuracy, test_set

# calculates the probability of each attribute value
def calc_att_entropies(att_groups, class_index):
    att_probs = []
    for group in att_groups:
        class_names, class_groups = group_by(group, class_index)
        class_probs = calc_prior_probs(class_groups, len(group))
        e_att = calc_entropy(class_probs)
        att_probs.append(e_att)
    return att_probs

# generates a confusion matrix for predictive accuracies
def calc_conf_matrix(class_index, test_set, predictions, classifications):
    # first, get actual classes
    actual_classes = compute_actual_classes(class_index, test_set)
    matrix = []
    label_row = ["Class"] + classifications + ["Total", "Recognition %"]
    matrix.append(label_row)
    # next, prepopulate an n x n matrix with 0s
    for c in range(len(classifications)):
        row_label = [classifications[c]]
        row_body = [0] * (len(classifications) + 2)
        new_row = row_label + row_body
        matrix.append(new_row)
    # now count the TPs, TNs, FPs, and FNs
    for c in range(len(predictions)):
        # prediction is correct
        if predictions[c] == actual_classes[c]:
            index = classifications.index(predictions[c]) + 1
            matrix[index][index] += 1
        else:
            row_index = classifications.index(actual_classes[c]) + 1
            column_index = classifications.index(predictions[c]) + 1
            matrix[row_index][column_index] += 1
    matrix = compute_totals(matrix)
    return matrix

# calculates the entropy of an attribute
def calc_entropy(probs):
    entropy = 0
    for prob in probs:
        current_entropy = -(prob * math.log(prob, 2))
        entropy += current_entropy
    return np.round(entropy, 4)

# calculates the information gain of splitting on a specific attribute
def calc_info_gain(instances, att_index, class_index):
    # first, calculate e_start
    class_names, class_groups = group_by(instances, class_index)
    prior_probs = calc_prior_probs(class_groups, len(instances))
    e_start = calc_entropy(prior_probs)
    # next, calculate entropies for each attribute in the class
    att_names, att_groups = group_by(instances, att_index)
    att_entropies = calc_att_entropies(att_groups, class_index)
    # get new entropy
    e_new = calc_new_entropy(att_groups, att_entropies, len(instances))
    info_gain = np.round(e_start - e_new, 4)
    return info_gain

# calculates the new entropy
def calc_new_entropy(att_groups, att_entropies, total_instances):
    e_new = 0
    for group in range(len(att_groups)):
        value_probability = len(att_groups[group]) / total_instances
        current_entropy = value_probability * att_entropies[group]
        e_new += current_entropy
    return np.round(e_new, 4)

# calculates the prior probability of a class
def calc_prior_probs(instances, num_instances):
    prior_probs = []
    for instance in instances:
        prob = np.round(len(instance) / num_instances, 4)
        prior_probs.append(prob)
    return prior_probs

# classifies an instance using a decision tree
def classify_tdidt(decision_tree, instance, header):
    if decision_tree[0] == "Leaves":
        return decision_tree[1][0]
    else:
        a = header.index(decision_tree[1])
        v = 2
        while v < len(decision_tree) - 1 and instance[a] != decision_tree[v][1]:
            v += 1
        return classify_tdidt(decision_tree[v][2], instance, header)

# computes accuracy given a list of predictions
def compute_accuracy(header, test_set, predictions, class_index):
    num_correct = 0
    actual_classes = compute_actual_classes(class_index, test_set)
    c = 0
    while c < len(actual_classes) and c < len(predictions):
        if actual_classes[c] == predictions[c]:
            num_correct += 1
        c += 1
    accuracy = np.round(num_correct / len(actual_classes), 2)
    return accuracy

# gets the actual classes of the test set
def compute_actual_classes(class_index, test_set):
    actual_classes = []
    for instance in test_set:
        actual_classes.append(instance[class_index])
    return actual_classes

# does holdout method
def compute_holdout_partitions(table):
    # 2:1 split...2/3 in training set and 1/3 in test set
    # first lets randomize our table so that each
    # time we call this function we should get a different test set
    randomized = table[:] # shallow copy (this is fine here)
    n = len(randomized)   # n = number of rows in table
    for i in range(n):
        # generate a random index to swap with i
        rand_index = rand.randrange(0, n) # [0, n)
        # task: do the swap
        randomized[i], randomized[rand_index] = randomized[rand_index], randomized[i]
    # compute split index
    split_index = int(n * (2 / 3))
    train_set = randomized[:split_index]
    test_set = randomized[split_index:]
    return train_set, test_set

# computes the M most accurate decision trees
def compute_most_accurate(accuracies, track_records, trees, M):
    # list to hold the M most accurate classifiers
    most_accurate_trees = []
    track_records_of_most_accurate = []
    # list of tuples containing both the highest accuracies and the indices of the trees 
    # with those accuracies
    accuracies_of_most_accurate = [(0, 0)] * M
    # now get the highest accuracies and their indices
    for i in range(0, len(accuracies)):
        if greater_than_prev(accuracies[i], accuracies_of_most_accurate):
            index_of_smallest = get_smallest_accuracy(accuracies_of_most_accurate)
            accuracies_of_most_accurate[index_of_smallest] = (accuracies[i], i)
    # now map the most accurate trees to the new list using the 
    # indices in accuracies_of_most_accurate
    for a in accuracies_of_most_accurate:
        most_accurate_trees.append(trees[a[1]])
        track_records_of_most_accurate.append(track_records[a[1]])
    return most_accurate_trees, track_records_of_most_accurate

# implements majority voting
def compute_partition_stats(class_groups):
    class_votes = []
    for group in class_groups:
        class_votes.append(len(group))
    return class_votes

# computes the totals and recognition columns in matrix
def compute_totals(matrix):
    for row in range(1, len(matrix)):
        r_col = len(matrix[row]) - 1
        t_col = len(matrix[row]) - 2
        num_correct = matrix[row][row]
        for col in range(1, len(matrix[row]) - 2):
            matrix[row][len(matrix[row]) - 2] += matrix[row][col]
        if matrix[row][t_col] != 0:
            matrix[row][r_col] = np.round(num_correct / matrix[row][t_col], 2)
        else:
            matrix[row][r_col] = "NA"
    return matrix

# computes the track record of a tree for each class
def compute_track_record(header, predictions, validation_set, class_index, table):
    # first, get all of the class names
    class_names, class_groups, = group_by(table, class_index)
    # next, create a list of the track records for each class label
    track_records = [0] * len(class_names)
    actual_classes = compute_actual_classes(class_index, validation_set)
    # count number of each correctly predicted class
    for p in range(len(predictions)):
        if predictions[p] == validation_set[p][class_index]:
            class_label_index = class_names.index(predictions[p])
            track_records[class_label_index] += 1
    # divide each count by length of validation set to get percentage
    for r in range(len(track_records)):
        track_records[r] = track_records[r] / len(validation_set)
    return track_records

# converts a continuous attribute to a categorical one
def convert_to_categorical(cutoffs, column_index, table):
    for row in table:
        current_att_val = float(row[column_index])
        i = 0
        classified = False
        while i < len(cutoffs) and not classified:
            if current_att_val < cutoffs[i]:
                classified = True
            i += 1
        # new classification number
        row[column_index] = str(i)

# performs stratified k fold cross validation technique of computing predictor's accuracy
def do_normal_predictions(test_set, header, tree, class_index):
    predictions = []
    for i in range(len(test_set)):
        prediction = classify_tdidt(tree, test_set[i], header)
        predictions.append(prediction)
        # compute current accuracy
    accuracy = compute_accuracy(header, test_set, predictions, class_index)
    return accuracy, predictions

# creates the attribute domains of a dataset
def get_att_domains(table, att_indices):
    att_domains = {}
    # loop over each attribute in table
    for att in att_indices:
        attribute_values = sorted(get_column(table, att))
        current_val = attribute_values[0]
        att_labels = []
        att_labels.append(current_val)
        for val in attribute_values:
            if val != current_val:
                current_val = val
                att_labels.append(val)
        att_domains.update({att: att_labels})
    return att_domains

# gets column
def get_column(table, column_index):
    column = []
    for row in table:
        column.append(row[column_index])
    return column

# gets integer type casted version of the current column
def get_column_float(table, column_index):
    column = []
    for row in table:
        if row[column_index] != "NA":
            column.append(float(row[column_index]))
    return column

# reads the data from input file
def get_data(filename):
    try:
        input_stream = open(filename, 'r')
        instances = input_stream.readlines()
        instances = [i.strip() for i in instances]
        return instances
    except FileNotFoundError:
        sys.exit("File %s not found" % filename)

# returns subset of F random elements
def get_f_subset(value_list, num_values):
    shuffled = value_list[:]
    rand.shuffle(shuffled)
    return shuffled[:num_values]

# calculates which vote in the list is the highest
def get_highest_vote(names, votes):
    highest_vote = 0
    index_of_highest_vote = 0
    for v in range(len(votes)):
        if votes[v] > highest_vote:
            highest_vote = votes[v]
            index_of_highest_vote = v
    return names[index_of_highest_vote]

# retrieves the smallest accuracy in a list
def get_smallest_accuracy(accuracies):
    # initially set the smallest accuracy to the very first accuracy in the list
    smallest_accuracy = accuracies[0][0]
    smallest_accuracy_index = 0
    for a in range(len(accuracies)):
        if accuracies[a][0] < smallest_accuracy:
            smallest_accuracy = accuracies[a][0]
            smallest_accuracy_index = a
    return smallest_accuracy_index

# converts the data in a list into a table
def get_table(data):
    table = []
    for row in data:
        current_row = row.split(",")
        table.append(current_row)
    return table

# checks to see if a new accuracy is greater than any of the accuracies in the list
def greater_than_prev(accuracy, prev_accuracies):
    for p in range(0, len(prev_accuracies)):
        if accuracy > prev_accuracies[p][0]:
            return True
    return False

# groups table by certain attribute
def group_by(table, column_index):
    # first get unique vals in column
    group_names = sorted(list(set(get_column(table, column_index))))
    # now we need a list of subtables
    groups = [[] for name in group_names]
    for row in table:
        if row[column_index] != "NA":
            group_by_value = row[column_index]
            index = group_names.index(group_by_value)
            groups[index].append(row)
    return group_names, groups

# returns true if all instances have the same class label
def has_same_class_label(instances, class_index):
    # start at the first instance
    if instances != []:
        class_label = instances[0][class_index]
        for instance in instances:
            if instance[class_index] != class_label:
                return False
        return True
    return False

# creates a leaf node in the tree
def make_leaf(label, group, class_index, num_instances):
    num_with_label = 0
    for instance in group:
        if instance[class_index] == label:
            num_with_label += 1
    probability = np.round((num_with_label / num_instances) * 100, 1)
    leaf = [label, num_with_label, num_instances, probability]
    return leaf

# predicts the class label using majority voting
def make_prediction(predictions, test_set, class_index):
    # need sorted list of predictions to make counting votes easier
    sorted_predictions = sorted(predictions)
    overall_prediction = None
    current_prediction = sorted_predictions[0][0]
    highest_count = 0
    current_count = 0
    for i in range(len(sorted_predictions)):
        # weight for part 5
        weight = sorted_predictions[i][1]
        if sorted_predictions[i][0] == current_prediction:
            current_count += weight
            if current_count > highest_count:
                overall_prediction = current_prediction
                highest_count = current_count
        else:
            current_prediction = sorted_predictions[i][0]
            current_count = 0
    return overall_prediction

# removes unused attributes from the table
def parse_table(required_attributes, header, table):
    new_table = []
    for row in table:
        new_row = []
        for r in required_attributes:
            column_index = header.index(r)
            new_row.append(str(row[column_index]))
        new_table.append(new_row)
    return new_table

# partitions instances in a class
def partition_instances(instances, att_index, att_domain):
    # this is a group by att_domain, not by att_values in instances
    partition = {}
    for att_value in att_domain:
        subinstances = []
        for instance in instances:
            # check if this instance has att_value at att_index
            if instance[att_index] == att_value:
                subinstances.append(instance)
        partition[att_value] = subinstances
    return partition

# prints the resulting accuracies of the predictors
def print_accuracy(predictor_name, accuracy):
    print(predictor_name + "Accuracy = %.2f" % accuracy)
    error_rate = 1 - accuracy
    print("Error Rate = %.2f" % error_rate)

# prints a specified number of lines
def print_lines(n):
    sys.stdout.write("===== ")
    for i in range(n):
        sys.stdout.write("=== ")
    sys.stdout.write("==== ======= =================\n")

# prints out a confusion matrix
def print_matrix(predictor_name, matrix):
    print(predictor_name)
    print_lines(len(matrix) - 1)
    for row in matrix:
        index = matrix.index(row)
        spaces = "      "
        if index == 0:
            spaces = "  "
        sys.stdout.write(row[0] + spaces)
        for col in range(1, len(row)):
            spaces = "  "
            if (col == len(row) - 2 or col == len(row) - 1) and index != 0:
                spaces = "     "
            sys.stdout.write(spaces + str(row[col]))
        print()
    print_lines(len(matrix) - 1)
    

# prints out a decision tree
def print_tree(tree, indent):
    spaces = ''
    for i in range(0, indent):
        spaces += ' '
    sys.stdout.write(spaces + tree[0] + ', ')
    print(tree[1])
    if tree[0] != "Leaves":
        for branch in range(2, len(tree)):
            print_tree(tree[branch], indent + 3)

# removes na values in the table
def remove_na(header, table):
    for att in range(len(header)):
        if att != 8:
            vals = get_column_float(table, att)
            if att == 0 or att == 2 or att == 3 or att == 5:
                mean = np.round(sum(vals) / len(vals))
            else:
                mean = int(sum(vals) / len(vals))
            for row in table:
                if row[att] == "NA":
                    row[att] = str(mean)

# uses entropy based attribute selection to select an attribute to split on
def select_attribute(instances, att_indices, class_index):
    # find attribute with greatest info gain
    greatest_info_gain = 0
    selected_index = 0
    for att_index in att_indices:
        current_info_gain = calc_info_gain(instances, att_index, class_index)
        # selected index should be index with highest info gain
        if current_info_gain >= greatest_info_gain:
            greatest_info_gain = current_info_gain
            selected_index = att_index
    return selected_index

# creates a decision tree using top down induction method
def tdidt(instances, att_indices, att_domains, class_index, header, F):
    # Basic Approach (uses recursion!):
    # At each step, pick an attribute ("attribute selection")
    att_index = select_attribute(instances, att_indices, class_index)
    # can't choose the same attribute twice in a branch!!
    att_indices.remove(att_index) # remember: Python is pass
    # by object reference!!
    # Partition data by attribute values ... this creates pairwise disjoint partitions
    partition = partition_instances(instances, att_index, att_domains[att_index])
    attribute_node = ["Attribute", header[att_index]]
    # Repeat until one of the following occurs (base cases):
    for att_val, att_group in partition.items():
        value_branch = ["Value", att_val]
        leaves = ["Leaves"]
        # CASE 1: Partition has only class labels that are the same ...
        # no clashes, make a leaf node
        if has_same_class_label(att_group, class_index):
            class_label = att_group[0][class_index]
            leaf = make_leaf(class_label, att_group, class_index, len(instances))
            leaves.append(leaf)
            value_branch.append(leaves)
        # CASE 2: No more attributes to partition ...
        # reached the end of a branch and there may be clashes, majority voting
        # if we are here, then case 1's Boolean Condition failed
        elif len(att_indices) == 0 and len(att_group) != 0:
            class_names, class_groups = group_by(instances, class_index)
            class_votes = compute_partition_stats(class_groups)
            class_label = get_highest_vote(class_names, class_votes)
            leaf = make_leaf(class_label, att_group, class_index, len(instances))
            leaves.append(leaf)
            value_branch.append(leaves)
        # CASE 3: No more instances to partition ...
        # "backtrack" to replace the attribute node with a leaf node
        # can make use of compute_partition_stats() to find the class
        elif len(att_group) == 0:
            class_names, class_groups = group_by(instances, class_index)
            class_votes = compute_partition_stats(class_groups)
            class_label = get_highest_vote(class_names, class_votes)
            leaf = make_leaf(class_label, instances, class_index, len(instances))
            leaves.append(leaf)
            value_branch.append(leaves)
            attribute_node.append(value_branch)
            return attribute_node
        # if none of these cases evaluate to true, then recurse!!
        else:
            local_instances = []
            for instance in instances:
                if instance[att_index] == att_val:
                    local_instances.append(instance)
            # f subset modification
            f_subset = get_f_subset(local_instances, F)
            value = tdidt(f_subset, att_indices, att_domains, class_index, header, F)
            value_branch.append(value)
        attribute_node.append(value_branch)
    
    return attribute_node
