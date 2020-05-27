import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

#print(digits)
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

#plt.gray() 
#plt.matshow(digits.images[100])
#print(digits.target[100])

model = KMeans(n_clusters = 10, random_state =2)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

new_samples = np.array( [
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.42,3.20,4.21,3.87,0.08,0.00,0.00,0.00,6.07,8.44,8.01,8.44,2.02,0.00,0.00,0.00,1.35,1.85,2.27,8.44,2.52,0.00,0.00,0.00,0.50,6.06,8.26,8.01,1.26,0.00,0.00,0.00,4.80,8.35,4.54,1.01,0.00,0.00,0.00,0.00,8.18,8.18,6.75,6.75,5.90,0.00,0.00,0.00,2.86,4.98,5.06,5.06,4.22,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,4.80,7.34,5.65,1.68,0.00,0.00,0.00,2.94,8.43,6.24,7.92,7.76,1.17,0.00,0.00,5.31,7.67,0.08,1.59,7.92,6.49,0.00,0.00,5.90,7.25,0.51,0.00,5.81,7.42,0.00,0.00,4.04,8.44,8.01,5.89,8.43,4.72,0.00,0.00,0.00,1.60,5.14,6.74,5.80,0.33,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.34,0.59,0.00,0.00,0.00,0.00,0.00,3.87,8.10,8.27,0.34,0.00,0.00,0.00,1.77,8.43,5.90,2.19,0.00,0.00,0.00,0.08,6.57,8.43,3.12,2.53,2.36,0.00,0.00,0.42,8.01,8.44,8.44,8.43,8.44,1.51,0.00,0.00,0.34,0.84,0.84,3.87,8.43,1.01,0.00,0.00,0.00,0.00,0.00,2.87,8.27,0.51,0.00,0.00,0.00,0.00,0.00,0.00,0.59,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.36,8.35,8.44,7.67,3.37,0.34,0.00,0.00,3.96,8.43,8.43,8.34,8.43,6.48,0.00,0.00,0.33,4.71,8.35,7.75,5.73,8.43,0.00,0.00,0.50,5.30,8.09,8.43,8.43,6.14,0.00,0.00,2.53,8.43,8.42,8.44,8.18,1.76,0.00,0.00,1.18,8.34,8.43,8.43,8.43,4.97,0.00,0.00,0.00,2.34,6.24,6.75,6.66,2.01,0.00,0.00]
]     )



new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(4, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(8, end='')
  elif new_labels[i] == 4:
    print(0, end='')
  elif new_labels[i] == 5:
    print(3, end='')
  elif new_labels[i] == 6:
    print(6, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(1, end='')
  elif new_labels[i] == 9:
    print(3, end='')

print(new_labels) 

plt.show()
