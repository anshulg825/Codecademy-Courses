import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0,0],[0,1],[1,0],[1,1]]

labels = [0,0,0,1]

#plt.scatter((point[0] for point in data),   (point[1] for point in data), c = labels)

classifier =Perceptron(max_iter = 40)
classifier.fit(data,labels)

print(classifier.score(data,labels))


data_2 = [[0,0],[0,1],[1,0],[1,1]]

labels_2 = [0,1,1,1]

classifier_2 =Perceptron(max_iter = 40)
classifier_2.fit(data_2,labels_2)

print(classifier_2.score(data_2,labels_2))

plt.show()