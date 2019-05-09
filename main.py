##############################################
# Programmer: Nathan Vanos, Scott Kopczynski
# Class: CPSC 310, Spring 2019
# Final Project
# 5/3/2019
#
# Description: Classifies a housing dataset with a support vector machine
#               and with other classifiers we used in class this year.
#               We will then compare the efficiency of the SVM to our classifiers.
# Usage: python3.7 main.py
##############################################
import numpy as np
import ensemble_classifier
import knn_naive_classifier
import sys
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import matplotlib.pyplot as plt
# main program
def main():
    # obtain data
    training_data = get_data("train.csv")
    header, training_table = get_table(training_data)
    #visualize_data(header, training_table)
    # ** clean data
    training_table = clean_alley(training_table, header)
    training_table  = clean_garage(training_table, header)
    training_table  = clean_basement(training_table, header)
    training_table  = fill_with_best_val(training_table, header)
    training_table  = replace_with_mode(training_table, header)
    training_table  = remove_nas(training_table)
    # discretize continuous attributes
    discretize_table(header, training_table)
    # classify data
    
    print("SVM Accuracy k-fold cross validation (k=10): ")
    print(str(round(classify_with_svm(training_table),4) * 100)+ "%")
    print("Ensemble Accuracy(Random Decision Forest) k-fold cross validation (k=10): ")
    print(str(round(classify_with_ensemble(header, training_table),4) * 100)+ "%")
    training_table = encode_data(training_table)
    print("KNN Accuracy k-fold cross validation (k=10):")
    classify_with_knn(training_table, header)
    print("Naive-Bayes Accuracy k-fold cross validation (k=10):")
    classify_with_naive_bayes(training_table, header)

def visualize_data(header, data):
    xs = get_column_float(data, header.index('SalePrice'))
    plt.title("Housing prices distribution")
    plt.xlabel("House Prices")
    plt.hist(xs, bins=20, alpha=0.75, color="b")
    plt.grid(True)
    plt.savefig('price_distribution.png')
    plt.figure()
    xs = np.log1p(xs)
    plt.title("Housing prices distribution (After normalizing)")
    plt.xlabel("House Prices (np.log1p)")
    plt.hist(xs, bins=20, alpha=0.75, color="b")
    plt.grid(True)
    plt.savefig('price_distribution_normalize.png')
    plt.figure()
    perform_neighborhood_visualization(header, data)
    visualize_bedrooms(header, data)
    visualize_sqftg(header, data)
    visualize_year_built(header, data)
    visualize_garages(header, data)
    visualize_baths(header, data)
def visualize_baths(header, data):
    baths = get_column_float(data, header.index('FullBath'))
    half_baths = get_column_float(data, header.index('HalfBath'))
    xs = []
    for i in range(len(baths)):
        xs.append(baths[i] + half_baths[i])
    ys = get_column_float(data, header.index('SalePrice'))
    plt.title("Bathrooms vs. Sale Price")
    plt.xlabel("Bathrooms")
    plt.ylabel('SalePrice')
    plt.plot(xs, ys,"b.",alpha=0.2, markersize=16)
    plt.savefig('baths_vs_sale_price.png')
    plt.show()
    plt.figure()
def visualize_garages(header, data):
    xs = get_column_float(data, header.index('GarageCars'))
    ys = get_column_float(data, header.index('SalePrice'))
    plt.title("Garage vs. Sale Price")
    plt.xlabel("Garage's")
    plt.ylabel('SalePrice')
    plt.plot(xs, ys,"b.", alpha=0.2, markersize=16)
    plt.savefig('garage_vs_saleprice.png')
    plt.show()
    plt.figure()
def visualize_year_built(header, data):
    xs = get_column_float(data, header.index('YearBuilt'))
    ys = get_column_float(data, header.index('SalePrice'))
    plt.title("Year Built vs. Sale Price")
    plt.xlabel("Year Built")
    plt.ylabel('SalePrice')
    plt.plot(xs, ys,"b.")
    plt.savefig('year_built_vs_saleprice.png')
    plt.figure()
def visualize_sqftg(header, data):
    bsmt = get_column_float(data, header.index('TotalBsmtSF'))
    upper1 = get_column_float(data, header.index('1stFlrSF'))
    upper2 = get_column_float(data, header.index('2ndFlrSF'))
    xs = []
    for i in range(len(bsmt)):
        xs.append(bsmt[i] + upper1[i] + upper2[i])
    ys = get_column_float(data, header.index('SalePrice'))
    plt.title("Square Footage vs. Sale Price")
    plt.xlabel("Square Footage")
    plt.ylabel('SalePrice')
    plt.plot(xs, ys,"b.")
    plt.savefig('square_footage_vs_saleprice.pdf')
    plt.figure()
def visualize_bedrooms(header, data):
    xs = get_column_float(data, header.index('BedroomAbvGr'))
    ys = get_column_float(data, header.index('SalePrice'))
    plt.title("Bedrooms vs. Sale Price")
    plt.xlabel("Bedrooms")
    plt.ylabel('SalePrice')
    plt.plot(xs, ys,"b.", alpha=0.2, markersize=16)
    plt.savefig('beds_vs_saleprice.pdf')
    plt.figure()
def perform_neighborhood_visualization(header, data):
    xs = get_column(data, header.index('Neighborhood'))
    ys = get_column_float(data, header.index('SalePrice'))
    plt.xticks(rotation='vertical')
    ytick_label = [str(int(i)) for i in range(100000, 1000000000, 100000)]
    ytick_vals = [i for i in range(100000, 1000000000, 100000)]
    plt.yticks(ytick_vals,ytick_label)
    plt.plot(xs, ys,"b.", alpha=0.2, markersize=16)
    plt.savefig("neighborhood_vs_price.pdf")
    plt.figure()
def classify_with_knn(training_table, header):
    accuracies = []
    k_folds = knn_naive_classifier.determine_stratified_k_fold(training_table, 10)
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
            result = knn_naive_classifier.perform_knn(train_data, row)
            predicted.append(result)
            actual.append(row[len(row) - 1])
            if(result == row[len(row) -1]):
                correct_count += 1
        correct_counts.append(correct_count)
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    accuracies.append(accuracy)
    print(str(round(accuracies[0],4) * 100)+ "%")

def classify_with_naive_bayes(training_table, header):
    accuracies = []
    training_table, header = select_attributes(training_table, header)
    k_folds = knn_naive_classifier.determine_stratified_k_fold(training_table, 10)
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
            result = knn_naive_classifier.step_one(training_table, header, len(header)-1, [row[:-2]], False, 0)
            predicted.append(result)
            actual.append(row[len(row)-1])
            if(result == row[len(row)-1]):
                correct_count += 1
        correct_counts.append(correct_count)
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    accuracies.append(accuracy)
    print(str(round(accuracies[0],4)*100) + "%")

def select_attributes(table, header):
    for i in range(len(table)):
        table[i] = [table[i][header.index('Neighborhood')],
                    table[i][header.index('YearBuilt')],
                    table[i][header.index('TotalBsmtSF')],
                    table[i][header.index('1stFlrSF')],
                    table[i][header.index('2ndFlrSF')],
                    table[i][header.index('FullBath')],
                    table[i][header.index('HalfBath')],
                    table[i][header.index('YearBuilt')],
                    table[i][header.index('GarageCars')],
                    table[i][header.index('SalePrice')]]
    header = [header[header.index('Neighborhood')],
               header[header.index('YearBuilt')],
               header[header.index('TotalBsmtSF')],
               header[header.index('1stFlrSF')],
               header[header.index('2ndFlrSF')],
               header[header.index('FullBath')],
               header[header.index('HalfBath')],
               header[header.index('YearBuilt')],
               header[header.index('GarageCars')],
               header[header.index('SalePrice')]]
    return table, header

# classify the dataset with support vector machine
def classify_with_svm(table):
    accuracies = []
    table = encode_data(table)
    k_folds = knn_naive_classifier.determine_stratified_k_fold(table, 10)
    correct_counts = []
    predicted = []
    actual = []
    for i in range(0, len(k_folds)):
        train_data = []
        for j in range(0, len(k_folds)):
            if j != i:
                train_data+= k_folds[j]
        y_vals = get_column(train_data, len(table[0])-1)
        x_vals = get_attribs(train_data)
        #x_vals = encode_data(x_vals)
        clf = svm.SVC(C=10000000000000, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=80, gamma=0.00000000000001, kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        clf.fit(x_vals, y_vals)
        correct_count = 0
        for row in k_folds[i]:
            result = clf.predict([row[:-2]])
            predicted.append(result)
            actual.append(row[len(row)-1])
            if(result == row[len(row)-1]):
                correct_count += 1
        correct_counts.append(correct_count)
    accuracy = (sum(correct_counts) / len(correct_counts))/len(k_folds[0])
    accuracies.append(accuracy)
    return accuracies[0]

# classify the data with ensemble
def classify_with_ensemble(header, table):
    # prepare data for ensemble
    att_indices = list(range(len(header) - 1))
    table = encode_data(table)
    att_domains = ensemble_classifier.get_att_domains(table, att_indices)
    class_index = header.index("SalePrice")
    remainder_set, test_set = ensemble_classifier.compute_holdout_partitions(table)
    # use ensemble
    predictions, accuracy, test_set = ensemble_classifier.ensemble(table, header, 50, 20, 20, att_domains,
                                        class_index, att_indices, remainder_set, test_set)
    return accuracy

# discretizes continuous attributes
def discretize_table(header, table):
    sale_prices = get_column_float(table, header.index("SalePrice"))
    sale_prices = np.log1p(sale_prices)
    for row in table:
        row[header.index("SalePrice")] = sale_prices[table.index(row)]
    # convert all continuous attributes to categorical ones
    for att in range(0, len(header)):
        if is_continuous(header[att]):
            values = get_column_float(table, att)
            cutoffs = compute_equal_widths_cutoffs(values, 10)
            convert_to_categorical(cutoffs, att, table)
            #if header[att] == "SalePrice":
                #print(cutoffs)

# divides list of values into a specified number of equal width bins
def compute_equal_widths_cutoffs(values, num_bins):
    values_range = max(values) - min(values)
    width = values_range / num_bins
    cutoffs = []
    n = 0
    cutoff = values[0] + width
    while n < num_bins:
        cutoffs.append(cutoff)
        cutoff += width
        n += 1
    
    cutoffs = [round(cutoff, 1) for cutoff in cutoffs]
    return cutoffs

# computes the frequencies of values in each bin
def compute_frequencies(values, cutoffs):
    freqs = [0] * len(cutoffs)
    for val in values:
        for i, cutoff in enumerate(cutoffs):
            if val <= cutoff:
                freqs[i] += 1
                break
    return freqs

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

# gets integer type casted version of the current column
def get_column(table, column_index):
    column = []
    for row in table:
        if row[column_index] != "NA":
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

# converts the data in a list into a table
def get_table(data):
    table = []
    for row in data:
        current_row = row.split(",")
        current_row = current_row[1:]
        table.append(current_row)
    header = table[0]
    table = table[1:]
    return header, table

# cleans alley attribute
def clean_alley(data, header):
    idx = header.index('Alley')
    for row in data:
        if(row[idx] == 'NA'):
            row[idx] = 'None'
    return data

# cleans garage attributes
def clean_garage(data, header):
    idx_year_built = header.index('GarageYrBlt')
    for row in data:
        if(row[idx_year_built] == 'NA'):
            row[idx_year_built] = 0
            row[header.index('GarageArea')] = 0
            row[header.index('GarageCars')] = 0
            row[header.index('GarageType')] = 'None'
            row[header.index('GarageFinish')] = 'None'
            row[header.index('GarageQual')] = 'None'
            row[header.index('GarageCond')] = 'None'
    return data

# cleans basement attributes
def clean_basement(data, header):
    idx_basement = header.index('BsmtQual')
    for row in data:
        if(row[idx_basement] == 'NA'):
            row[idx_basement] = 'None'
            row[header.index('BsmtCond')] = 'None'
            row[header.index('BsmtExposure')] = 'None'
            row[header.index('BsmtFinType1')] = 'None'
            row[header.index('BsmtFinType2')] = 'None'
    return data

# fills rows with best values
def fill_with_best_val(data, header):
    for row in data:
        if(row[header.index('Functional')] == 'NA'):
            row[header.index('Functional')] == 'Typ'
        if(row[header.index('Electrical')] == 'NA'):
            row[header.index('Electrical')] = 'SBrkr'
        if(row[header.index('KitchenQual')] == 'NA'):
            row[header.index('KitchenQual')] = 'TA'
        if(row[header.index('PoolQC')] == 'NA'):
            row[header.index('PoolQC')] = 'None'
        if(row[header.index('Fence')] == 'NA'):
            row[header.index('Fence')] = 'None'
        if(row[header.index('MiscFeature')] == 'NA'):
            row[header.index('MiscFeature')] = 'None'
        if(row[header.index('FireplaceQu')] == 'NA'):
            row[header.index('FireplaceQu')] = 'None'
    return data

# replaces NA values with mode
def replace_with_mode(data, header):
    for row in data:
        if(row[header.index('Exterior1st')] == 'NA'):
            row[header.index('Exterior1st')] = mode(get_column(data, header.index('Exterior1st')))
        if(row[header.index('Exterior2nd')] == 'NA'):
            row[header.index('Exterior2nd')] = mode(get_column(data, header.index('Exterior2nd')))
        if(row[header.index('SaleType')] == 'NA'):
            row[header.index('SaleType')] = mode(get_column(data, header.index('SaleType')))
    return data

# removes NA values from table
def remove_nas(data):
    result = []
    for row in data:
        is_clean = True
        for val in row:
            if val == 'NA':
                is_clean = False
        if(is_clean):
            result.append(row)
    return result

# obtains all attributes but the class
def get_attribs(data):
    result = []
    for row in data:
        result.append(row[:-2])
    return result

# encodes data as dictionary
def encode_data(data):
    for i in range(len(data[0])):
        x = get_column(data, i)
        x_unique = list(set(x))
        encoding = {}
        x = 0
        for val in x_unique:
            encoding[val] = x
            x+=1
        for row in data:
            row[i] = encoding[row[i]]
    return data

# checks if an attribute is continuous
def is_continuous(column):
    if column == "LotArea":
        return True
    if column == "BsmtFinSF1":
        return True
    if column == "BsmtFinSF2":
        return True
    if column == "BsmtUnfSF":
        return True
    if column == "TotalBsmtSF":
        return True
    if column == "1stFlrSF":
        return True
    if column == "2ndFlrSF":
        return True
    if column == "LowQualFinSF":
        return True
    if column == "GrLivArea":
        return True
    if column == "GarageArea":
        return True
    if column == "WoodDeckSF":
        return True
    if column == "OpenPorchSF":
        return True
    if column == "EnclosedPorch":
        return True
    if column == "3SsnPorch":
        return True
    if column == "ScreenPorch":
        return True
    if column == "PoolArea":
        return True
    if column == "MiscVal":
        return True
    if column == "SalePrice":
        return True
    return False

# call main
if __name__ == "__main__":
    main()
