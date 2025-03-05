
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing


dataset = pd.read_csv("Churn_Modelling.csv")

dataset.columns

X = dataset.drop(columns =['RowNumber', 'CustomerId', 'Surname','Exited'])
y = dataset[['Exited']]  #2D array

# Encoding categorical data
#"Gender" column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])


#"Geography" column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(),['Geography'])],remainder ='passthrough')
X = np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the output layer

ann.add(tf.keras.layers.Dense(units =1, activation = 'sigmoid'))

# Compiling the ANN


ann.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics= ['accuracy'] )

#Training the ANN
ann.fit(X_train, y_train, batch_size = 64, epochs = 130)


# Predicting the Test set results

y_pred = ann.predict(X_test)
y_pred = (y_pred )

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
