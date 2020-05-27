import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
df = pd.read_csv('passengers.csv')

# Update sex column to numerical
for i in df.index:
  if df.at[i,"Sex"] == 'male':
    df.at[i,"Sex"]=0
  else:
    df.at[i,"Sex"]=1
    
# Fill the nan values in the age column
average_age=df["Age"].mean()
print(average_age)
df['Age'].fillna(value=average_age,inplace=True)

# Create a first class column
ls = []
for i in df.index:
  if df.at[i,"Pclass"] == 1:
    ls.append(1)
  else:
    ls.append(0)
df.insert(3, "FirstClass", ls, True) 

# Create a second class column
ls2 = []
for i in df.index:
  if df.at[i,"Pclass"] == 2:
    ls2.append(1)
  else:
    ls2.append(0)
df.insert(4, "SecondClass", ls2, True) 

# Select the desired features
features = df[['Sex','Age','FirstClass','SecondClass']]
survival = df[['Survived']]
#print(features)
#print(survival)

# Perform train, test, split
x_train,x_test,y_train,y_test = train_test_split(features,survival,test_size=0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train the model
classifier= LogisticRegression()
classifier.fit(x_train,y_train)

# Score the model on the train data
print(classifier.score(x_train,y_train))

# Score the model on the test data
print(classifier.score(x_test,y_test))

# Analyze the coefficients
print(classifier.coef_)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([0.0,19.0,0.0,1.0])

# Combine passenger arrays
sample_passengers =np.array([Jack,Rose,You])
#print(sample_passengers)

# Scale the sample passenger features
sample_passengers=scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
print(classifier.predict(sample_passengers))
print(classifier.predict_proba(sample_passengers))