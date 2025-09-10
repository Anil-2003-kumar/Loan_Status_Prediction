import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#uploading the dataset
dataset = pd.read_csv('/content/sample_data/train_u6lujuX_CVtuZ9i (1).csv')

#sizeof the dataset
dataset.shape

#missing values
dataset.isnull().sum()

#it can be replaced or else completly droped
dataset = dataset.dropna()

#now check
dataset.isnull().sum()


#converting categerical into numirical values for better generalization
dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

#Graphs
#loan status education vs Loan_status
sns.countplot(x='Education',hue='Loan_Status',data=dataset)

#converting all the categerical value to numirical value
dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

#splitting the data into training and testing data
X = dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = dataset['Loan_Status']

print(x)
print(y)

#Training
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

#support vector machine model

model = svm.SVC(kernel='linear')

#training the support Vector Macine model
model.fit(X_train,Y_train)

#Evluation
X_train_predi = model.predict(X_train)
training_data_accuray = accuracy_score(X_train_predi,Y_train)

#accuracy training data
print('Accuracy on training data : ', training_data_accuray)

#accuracy on testing data
X_test_predi = model.predict(X_test)
test_data_accuray = accuracy_score(X_test_predi,Y_test)

print('Accuracy on test data : ', test_data_accuray)


------------The End------