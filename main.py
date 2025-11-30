# Works Cited:
# https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
#   - Use numpy library documentation for learning how to read in text files

import numpy as np      # For distance calc and reading text file
import random as rand   # For random evaluation values

# Evaluation Function: Leave-one-out validation for checking accuracy
def featEvalFunc():
    # Stub function - only returns random value
    val = rand.random() * 100
    accuracy = round(val, 2)
    return accuracy

# Greedy Algorithm: Add one feature at a time
def forwardSelection(numFeatures):
    globalBestfeatures = []   # Global best set of features
    globalHighAcc = 0   # Global highest accuracy

    # List accuracy using no features
    accNoFeat = featEvalFunc()
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {accNoFeat}%")
    globalHighAcc = accNoFeat

    # This simulates the array of features we have (replace later)
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
            acc = featEvalFunc()

            # Print trace
            print(f"\tUsing feature(s) {currFeatures} accuracy is {acc}%")

            # If a new high accuracy, record feature index (column)
            if acc > highestAcc:
                highestAcc = acc
                bestFeatIndex = j
        
        # Print trace
        currFeatures = comboFeatures + [features[bestFeatIndex]]
        print(f"Feature set {currFeatures} was best, accuracy is {highestAcc}%")

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

    print(f"\nFinished search!!! The best feature subset is {globalBestfeatures}, which has an accuracy of {globalHighAcc}%")
    return globalBestfeatures

# Greedy Algorithm: Remove one feature at a time
def backwardElimination(numFeatures):
    globalBestfeatures = []   # Global best set of features
    globalHighAcc = 0   # Global highest accuracy

    # List accuracy using no features
    accNoFeat = featEvalFunc()
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {accNoFeat}%")
    globalHighAcc = accNoFeat

    # This simulates the array of features we have (replace later)
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
            acc = featEvalFunc()

            if acc > highestAcc:
                highestAcc = acc

            # Print trace
            print(f"\tUsing feature(s) {currFeatures} accuracy is {acc}%")
            print(f"Feature set {currFeatures} was best, accuracy is {highestAcc}%")
        else:
            # Loop through all combinations
            for j in range(len(featuresToRemove)):
                # Copy current combination of features and remove one
                currFeatures = comboFeatures[:]
                del currFeatures[j]
                # Pass this set of features into the eval function
                acc = featEvalFunc()

                # Print trace
                print(f"\tUsing feature(s) {currFeatures} accuracy is {acc}%")

                # If a new high accuracy, record feature index (column)
                if acc > highestAcc:
                    highestAcc = acc
                    bestFeatIndex = j
            
            # Print trace
            currFeatures = comboFeatures[:]
            del currFeatures[bestFeatIndex]
            print(f"Feature set {currFeatures} was best, accuracy is {highestAcc}%")

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

    print(f"\nFinished search!!! The best feature subset is {globalBestfeatures}, which has an accuracy of {globalHighAcc}%")
    return globalBestfeatures

# Custom Search for Extra Credit (implement at the end)
def customSearch():
    return

# Select the set of features that yield best results based on accuracy
def featureSearch(numFeatures, algChoice):
    # Choose search algorithm based on user choice
    if algChoice == 1:
        features = forwardSelection(numFeatures)
    elif algChoice == 2:
        features = backwardElimination(numFeatures)
    elif algChoice == 3:
        features = customSearch(numFeatures)
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

    return pred_class_label

# Evaluate the accuracy of the classifier (Nearest Neighbor)
def classEvalFunc(features, classifier, training_data):
    acc = 0

    return acc

# Nearest-Neighbor Classifier: Classifies points based on distance from neighbors
def nn_classifier(training_data, numFeatures, algChoice, test_instance):
    # Call feature search function and get best features and accuracy
    # Finding best features are not being used yet, still pass old code of numFeatures
    features = featureSearch(numFeatures, algChoice)

    # Train and test the model to find unknown point
    train(training_data)
    print(training_data)
    pred_label = test(test_instance, training_data)

    return pred_label

# Main driver code
def main():
    drayNetID = "dchow001"
    yangNetID = "ywang1245"

    # Use numpy library to read text file, convert to 32-bit Float
    large_data = np.loadtxt("large-test-dataset-2.txt", dtype=np.float32)
    small_data = np.loadtxt("small-test-dataset-2-2.txt", dtype=np.float32)

    print(f"Welcome to {drayNetID} and {yangNetID} Feature Selection Algorithm.")
    numFeatures = int(input("\nPlease enter total number of features: "))

    # Ask user to choose algorithm choice for feature selection
    print("Type the number of the algorithm you want to run:")
    print("\t1. Forward Selection")
    print("\t2. Backward Elimination")
    print("\t3. Custom Search (not implemented yet)")
    algChoice = int(input("\nChoice: "))

    # Call Classifier Evaluation Function (for now just testing classifier accuracy with specific feature subset)
    # classEvalFunc()
    nn_classifier(small_data, numFeatures, algChoice, 1)

# Calls main
if __name__ == "__main__":
    main()