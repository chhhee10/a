import pandas as pd

data = pd.read_csv('training_data.csv')

print(data)

def find_s_algorithm(data):

    attributes = data.iloc[:, :-1].values  # All columns except last
    target = data.iloc[:, -1].values

    # Initialize hypothesis with the first positive example
    for i in range(len(target)):

        if target[i].lower() == "yes":  # Make sure we cover both Yes/yes

            hypothesis = attributes[i].copy()

            break

    # Refine hypothesis based on other positive examples
    for i in range(len(target)):

        if target[i].lower() == "yes":

            for j in range(len(hypothesis)):

                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = '?'

    return hypothesis

# Example usage

final_hypothesis = find_s_algorithm(data)

print("Most Specific Hypothesis:", final_hypothesis)