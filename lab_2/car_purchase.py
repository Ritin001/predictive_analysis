import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"C:\Users\RITIN\OneDrive\Desktop\5th sem\predictive_analysis\lab_2\car_data.csv")
x= df.iloc[:400,[2,3]].values
y= df.iloc[:400,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


sc=StandardScaler()
reg = LogisticRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()


#random forst classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print(y_pred_rf)