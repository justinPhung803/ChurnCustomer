import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import sklearn
import matplotlib
import matplotlib.pyplot as plt

from keras import layers
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


#Read and clean the data:
dataset=pd.read_csv("/Users/josephphung/churnPrediction/Project-Customer_Churn_Dataset.csv")
dataset.info()
feature=dataset.iloc[:,4:-1]
label=dataset.iloc[:,-1]
print(pd.unique(dataset['Churn?']))
feature=pd.get_dummies(feature)


#Split the data into training set and test set:
feature_train,feature_test,label_train,label_test=train_test_split(feature,label,test_size=0.33,random_state=42)


#Rescale the data and change string type data into numeric type with onehotencoder:
ct=ColumnTransformer([("numeric", StandardScaler(), ['VMail Message','Day Mins','Day Calls','Day Charge','Eve Mins','Eve Calls','Eve Charge','Night Mins','Night Calls','Night Charge','Intl Mins','Intl Calls','Intl Charge','CustServ Calls'])])

feature_train=ct.fit_transform(feature_train)
feature_test=ct.transform(feature_test)

le=LabelEncoder()

label_train=le.fit_transform(label_train)
label_test=le.transform(label_test)


#Resize the imbalanced data with SMOTE:
smote = SMOTE(random_state = 14)

feature_train, label_train= smote.fit_resample(feature_train, label_train)

unique, counts = np.unique(label_train, return_counts=True)
print(dict(zip(unique, counts)))


label_train = np.asarray(label_train).astype('float32')

label_test = np.asarray(label_test).astype('float32')


#Function to design learning model:
def model_design(feature, my_metrics, my_lr):
    model=Sequential()
    model.add(layers.InputLayer(input_shape=(feature.shape[1], )))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=my_metrics, optimizer=Adam(learning_rate=my_lr))
    return model

#Model Hyperparameters:
EPOCHS=100
BATCH_SIZE=32
VALIDATION_SPLIT=0.25
learning_rate=0.001
es=EarlyStopping(monitor='val_loss', mode='min', patience=20)
classification_threshold = 0.52
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy', 
                                      threshold=classification_threshold),
      tf.keras.metrics.Precision(thresholds=classification_threshold,
                                 name='precision' 
                                 ),
      tf.keras.metrics.Recall(thresholds=classification_threshold,name='recall'),
      tf.keras.metrics.AUC(num_thresholds=100, name='auc')
]


#Create and train model:
model=model_design(feature_train,METRICS, learning_rate)
history=model.fit(feature_train,label_train,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=VALIDATION_SPLIT,verbose=1, callbacks=[es])


metric_val=model.evaluate(feature_test,label_test)
print("metrics_val: ", metric_val)


#Calculate F1_score
label_predict=(model.predict(feature_test) > 0.5).astype("int32")
f1=f1_score(label_test,label_predict, average='weighted')
print("f1_score: ",f1)


#Plotting function
def plot_curve(epochs, hist, list_of_metrics):
    #Plot Metrics:
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()

    #Plot Confusion Matrix
    clf=SVC(random_state=0)
    clf.fit(feature_train,label_train)
    plot_confusion_matrix(clf, feature_test,label_test)
    plt.show()

print("Defined the plot_curve function.")
epochs=history.epoch
hist=pd.DataFrame(history.history)
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall','auc'] 
plot_curve(epochs, hist, list_of_metrics_to_plot)



