import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# import dataset
df = pd.read_csv('/Users/keru/Downloads/classification.csv')

# define feature and target
x = df.drop('success', axis=1)
y = df.success.values

# insert a value 1 column for intercept term
x = np.insert(np.array(x), 0, np.ones(df.shape[0]), axis=1)

# normalize feature by its vector size
x_normalizer = np.sqrt(sum(x**2))
x_norm = x/x_normalizer

# train and test split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=0)

# gradient ascent
# help function to get gradient
def gradient_logistic(x: np.array,
                      y: np.array,
                      weights: np.array) -> np.array:

    score = np.matmul(x, weights)
    proba_class1 = 1 / (1 + np.exp(-1 * score))
    gradient_matrix = np.zeros((x.shape[0], x.shape[1]))

    for i in range(len(weights)):  # loop over feature - column
        for j in range(x.shape[0]):  # loop over sample - row
            if y[j] == 1:
                gradient_matrix[j][i] = x[j, i] * (1 - proba_class1[j])
            if y[j] == 0:
                gradient_matrix[j][i] = x[j, i] * (0 - proba_class1[j])

    # sum over row
    gradient = gradient_matrix.sum(axis=0)
    return (gradient)

# initial weights, gradient, threshold and stepsize
weights_all = dict()
gradient_vector_size = dict()
threshold = 0.01
stepsize_list = [0.01, 0.1, 1, 5]

# write down iteration
for stepsize in stepsize_list:

    # initialling
    weights = np.zeros(x.shape[1])
    gradient = np.ones(x.shape[1])
    iter = 0

    # iteration starts for each stepsize
    gradient_vector_size_iteration = []
    while np.sqrt(np.dot(gradient, gradient)) > threshold:
        gradient_vector_size_iteration.append(np.sqrt(np.dot(gradient, gradient)))
        gradient = gradient_logistic(x_train, y_train, weights)
        weights = weights + stepsize * gradient
        iter += 1

    gradient_vector_size[str(stepsize)] = gradient_vector_size_iteration
    weights_all[str(stepsize)] = weights

# import matplotlib.pyplot as plt
# figure, ax = plt.subplots(1,4, figsize = (10,5), sharey=True)
# i = 0
# for key in gradient_vector_size.keys():
#     ax[i].plot(gradient_vector_size[key])
#     ax[i].set_title('stepsize ' + key)
#     ax[i].set_xlabel('iteration')
#     if i ==0:
#         ax[i].set_ylabel('gradient vector size')
#     i +=1
#
# plt.show()


# predicting on test data
# 1. get score
test_score = np.matmul(x_test, weights)
# 2. get proba
proba_class1 = 1/(1+np.exp(-1*test_score))
# 3. get pred label
pred_label = [1 if x>=0.5 else 0 for x in proba_class1]

# evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_label)
print(cm)

from sklearn.metrics import classification_report
cr = classification_report(y_test, pred_label)
print(cr)

# generate sigmoid function
score = list(range(-6, 7, 1))
sigmoid_score = [1/(1+np.exp(-1*x)) for x in score]
y = [0,0,0,0,0,0,0,1,1,1,1,1,1]
import matplotlib.pyplot as plt
plt.plot(score, sigmoid_score)
plt.scatter(score, y)
plt.vlines(x=0, ymin=0, ymax=1, color = 'red', linestyles='--')
plt.xlabel('score')
plt.ylabel('Sigmoid(score)')
plt.show()