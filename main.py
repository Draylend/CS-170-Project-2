# Works Cited:
# https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
#   - Use numpy library documentation for learning how to read in text files

import numpy as np      # For distance calc and reading text file
import time             # Timer for timting steps
import random as rand   # For random evaluation values

# Evaluate the accuracy of the classifier
def evalFunc(features, training_data):
    acc = 0     # Accuracy of classifier

    # Add index 0 to include labels of classes
    featuresWithLabel = [0] + features

    # Narrow down training data to only include subset of features
    reduced_training_data = training_data[:, featuresWithLabel]

    # Loop through all instances (and leave one out)
    for i in range(len(reduced_training_data)):
        # Get instance we are using as validation
        leave_one_out = reduced_training_data[i]
        # Make new feature vector with one left out
        one_reduced_training_data = np.delete(reduced_training_data, i, axis=0)

        # Call Nearest Neighbor
        label = nn_classifier(one_reduced_training_data, leave_one_out)

        # If label is correct, add to total correctness
        if leave_one_out[0] == label:
            acc += 1

    # Calculate overall accuracy
    acc = acc / len(reduced_training_data)

    return acc

# Greedy Algorithm: Add one feature at a time
def forwardSelection(numFeatures, training_data):
    globalBestfeatures = []   # Global best set of features
    globalHighAcc = 0   # Global highest accuracy

    # List accuracy using no features
    accNoFeat = evalFunc([], training_data)
    print(f"Using no features (default rate) and using \"leave-one-out\", I get an accuracy of {accNoFeat:.2%}")
    globalHighAcc = accNoFeat

    # This simulates the array of features we have
    features = [i for i in range(1, numFeatures+1)]
    comboFeatures = []   # Local best set of features

    # Loop through amount of features
    print("Beginning search")
    for i in range(numFeatures):
        highestAcc = 0      # Local highest accuracy
        bestFeatIndex = 0   # Best feature for this iteration

        # Loop through all combinations
        for j in range(len(features)):
            # Pass this current feature + previous best into the eval function
            currFeatures = comboFeatures + [features[j]]
            acc = evalFunc(currFeatures, training_data)

            # Print trace
            print(f"\tUsing feature(s) {currFeatures} accuracy is {acc:.2%}")

            # If a new high accuracy, record feature index (column)
            if acc > highestAcc:
                highestAcc = acc
                bestFeatIndex = j
        
        # Print trace
        currFeatures = comboFeatures + [features[bestFeatIndex]]
        print(f"Feature set {currFeatures} was best, accuracy is {highestAcc:.2%}")

        # If this combination of feature yielded global highest accuracy, record it
        if highestAcc > globalHighAcc:
            globalHighAcc = highestAcc
            globalBestfeatures = currFeatures
        # Else warn that accuracy is decreasing
        else:
            print("(Warning! Accuracy has decreased!)")

        # After all combinations, record best feature
        comboFeatures.append(features[bestFeatIndex])
        # Remove feature/column from possible combinations
        del features[bestFeatIndex]

    print(f"\nFinished search!!! The best feature subset is {globalBestfeatures}, which has an accuracy of {globalHighAcc:.2%}")
    return globalBestfeatures

# Greedy Algorithm: Remove one feature at a time
def backwardElimination(numFeatures, training_data):
    globalBestfeatures = []   # Global best set of features
    globalHighAcc = 0   # Global highest accuracy

    # List accuracy using no features
    accNoFeat = evalFunc([], training_data)
    print(f"Using no features (default rate) and using \"leave-one-out\", I get an accuracy of {accNoFeat:.2%}")
    globalHighAcc = accNoFeat

    # This simulates the array of features we have
    featuresToRemove = [i for i in range(1, numFeatures+1)]
    comboFeatures = [i for i in range(1, numFeatures+1)]   # Start with all features
    globalBestfeatures = comboFeatures

    # Loop through amount of features
    print("Beginning search")
    for i in range(numFeatures):
        highestAcc = 0      # Local highest accuracy
        bestFeatIndex = 0   # Best feature to remove this iteration

        # If this is the full set (first iteration), then don't remove any elements yet
        if i == 0:
            currFeatures = comboFeatures[:]
            # Pass this set of features into the eval function
            acc = evalFunc(currFeatures, training_data)

            if acc > highestAcc:
                highestAcc = acc

            # Print trace
            print(f"\tUsing feature(s) {currFeatures} accuracy is {acc:.2%}")
            print(f"Feature set {currFeatures} was best, accuracy is {highestAcc:.2%}")
        else:
            # Loop through all combinations
            for j in range(len(featuresToRemove)):
                # Copy current combination of features and remove one
                currFeatures = comboFeatures[:]
                del currFeatures[j]
                # Pass this set of features into the eval function
                acc = evalFunc(currFeatures, training_data)

                # Print trace
                print(f"\tUsing feature(s) {currFeatures} accuracy is {acc:.2%}")

                # If a new high accuracy, record feature index (column)
                if acc > highestAcc:
                    highestAcc = acc
                    bestFeatIndex = j
            
            # Print trace
            currFeatures = comboFeatures[:]
            del currFeatures[bestFeatIndex]
            print(f"Feature set {currFeatures} was best, accuracy is {highestAcc:.2%}")

            # After all combinations, record best feature grouping
            del comboFeatures[bestFeatIndex]
            # Remove feature/column from possible eliminations next time
            del featuresToRemove[bestFeatIndex]

        # If this combination of feature yielded global highest accuracy, record it
        if highestAcc > globalHighAcc:
            globalHighAcc = highestAcc
            globalBestfeatures = currFeatures
        # Else warn that accuracy is decreasing
        else:
            print("(Warning! Accuracy has decreased!)")

    print(f"\nFinished search!!! The best feature subset is {globalBestfeatures}, which has an accuracy of {globalHighAcc:.2%}")
    return globalBestfeatures

# Custom Search for Extra Credit (implement at the end)
def customSearch():
    return

# Select the set of features that yield best results based on accuracy
def featureSearch(numFeatures, algChoice, training_data):
    # Choose search algorithm based on user choice
    if algChoice == 1:
        features = forwardSelection(numFeatures, training_data)
    elif algChoice == 2:
        features = backwardElimination(numFeatures, training_data)
    elif algChoice == 3:
        features = customSearch(numFeatures, training_data)
    else:
        print("Error. Not a valid algorithm choice.")

    return features

# Train the model (however NN is a lazy learner so there's not much to train)
def train(training_data):
    # Normalize feature vector
    # Loop for as many columns
    for i in range(1, len(training_data[0])):
        featureCol = training_data[:,i]
        
        # Normalize (Min-Max Normalization) to a range of [0-1]
        normalizedCol = (featureCol - featureCol.min()) / (featureCol.max() - featureCol.min())
        # Assign changes
        training_data[:,i] = normalizedCol

    return

# Use the model
def test(test_instance, training_data):
    # Calculate distances of current point from all neighbors
    pred_class_label = None
    dists = []  # Store distances

    test_instance_no_label = test_instance[1:].copy()
    
    index = 0
    # Loop through all instances of training data
    for x_i in training_data:
        # Remove class label
        x_i_no_label = x_i[1:].copy()

        # Calculate distance of current point from test instance
        dist = np.linalg.norm(test_instance_no_label - x_i_no_label, ord=2)

        # Add distance and current index to list
        dists.append((dist, index))
        index += 1

    # Sort to find shortest distance
    dists.sort()
    
    # Get index of nearest neighbor
    nn_index = dists[0][1]
    pred_class_label = training_data[nn_index][0]

    return pred_class_label

# Nearest-Neighbor Classifier: Classifies points based on distance from neighbors
def nn_classifier(training_data, test_instance):
    # Test the model to find unknown point
    pred_label = test(test_instance, training_data)

    return pred_label

# Main driver code
def main():
    drayNetID = "dchow001"
    yangNetID = "ywang1245"

    print(f"Welcome to {drayNetID} and {yangNetID} Feature Selection Algorithm.")

    # Retrieve file name
    file_name = str(input("Type in the name of the file to test: "))

    # Load in data
    data = np.loadtxt(file_name, dtype=np.float32)

    # Get the number of features
    numFeatures = len(data[0]) - 1
    # Number of instances
    numInstances = len(data)

    # Ask user to choose algorithm choice for feature selection
    print("\nType the number of the algorithm you want to run:")
    print("\t1. Forward Selection")
    print("\t2. Backward Elimination")
    print("\t3. Custom Search (not implemented yet)")
    algChoice = int(input("\nChoice: "))

    # Print trace of dataset statistics
    print(f"\nThis dataset has {numFeatures} (not including class label) with {numInstances} instances.")
    # Normalize data and print trace
    print("Normalizing data, please wait... done!")
    train(data)

    print("\nRunning Nearest Neighbor now...")
    featureSearch(numFeatures, algChoice, data)

# Calls main
if __name__ == "__main__":
    main()

# Reporting Results
# Group: Draylend Chow - dchow001 - Session 21, Yang Wang - ywang1245 - Session 21
# DatasetID: ???
# Small Dataset Results:
#   - Forward: Feature Subset: {5, 3}, Acc: 0.92
#   - Backward:  Feature Subset: {3, 5}, Acc: 0.92
# Large Dataset Results:
#   - Forward:  Feature Subset: {27, 1}, Acc: 0.955
#   - Backward:  Feature Subset: {27}, Acc: 0.847