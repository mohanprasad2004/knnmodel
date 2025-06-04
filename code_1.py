import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



col = ["fLength","fWidth","fSize","fCone","fCone1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
x= pd.read_csv("magic04.data",names=col)
print(x)
x["class"]= (x["class"]=="g").astype(int)
#print(x.head())

for label in col[:-1]:
    plt.hist(x[x["class"]==1][label], color='blue',label='gamma',alpha= 0.7 , density = True) 
    plt.hist(x[x["class"]==0][label] , color='red',label='hadron',alpha= 0.7 , density = True) 
    plt.title(f"Distribution of {label}")
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# train data , validation , test dataset 

train , valid , test = np.split(x.sample(frac=1), [int(0.6*len(x)),int(0.8*len(x))])

def scale_dataset(df , oversample = False):
    features = df[df.columns[:-1]].values    # All columns except the last (features)
    labels = df[df.columns[-1]].values       # The last column (label)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if oversample:
         features, labels = RandomOverSampler().fit_resample(features, labels)


    # Combine the scaled features and original labels into one array
    data = np.hstack((features, np.reshape(labels, (-1, 1))))
    
    return data , features , labels


#print(scale_dataset(x))

print(len(train[train['class']==1]))
print(len(train[train['class']==0]))

train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=True)
test, x_test, y_test = scale_dataset(test, oversample=False)




# knn 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model= KNeighborsClassifier(n_neighbors=100)
knn_model.fit(x_train,y_train)

y_pred = knn_model.predict(x_test)

print(classification_report(y_test, y_pred))  



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Hadron", "Gamma"], yticklabels=["Hadron", "Gamma"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10,8))
corr_matrix = x[col[:-1]].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

