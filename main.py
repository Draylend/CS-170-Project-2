import random as rand   # For random evaluation values

# Evaluation Function: Leave-one-out validation for checking accuracy
def evalFunc():
    # Stub function - only returns random value
    val = rand.random() * 100
    accuracy = round(val, 2)
    return accuracy

# Greedy Algorithm: Add one feature at a time
def forwardSelection(numFeatures):
    globalBestfeatures = []   # Global best set of features
    globalHighAcc = 0   # Global highest accuracy

    # List accuracy using no features
    accNoFeat = evalFunc()
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
            acc = evalFunc()

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
    accNoFeat = evalFunc()
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
            acc = evalFunc()

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
                acc = evalFunc()

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

# Nearest-Neighbor Classifier: Classifies points based on distance from neighbors
def nn_classifier(numFeatures, algChoice):
    # Call feature search function and get best features and accuracy
    features = featureSearch(numFeatures, algChoice)
    return

# Main driver code
def main():
    drayNetID = "dchow001"
    yangNetID = "ywang1245"

    print(f"Welcome to {drayNetID} and {yangNetID} Feature Selection Algorithm.")
    numFeatures = int(input("\nPlease enter total number of features: "))

    # Ask user to choose algorithm choice for feature selection
    print("Type the number of the algorithm you want to run:")
    print("\t1. Forward Selection")
    print("\t2. Backward Elimination")
    print("\t3. Custom Search (not implemented yet)")
    algChoice = int(input("\nChoice: "))

    # Call Nearest Neighbor (for now just testing search functions)
    nn_classifier(numFeatures, algChoice)

# Calls main
if __name__ == "__main__":
    main()