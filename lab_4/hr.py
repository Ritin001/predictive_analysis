import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns

df =pd.read_csv(r"C:\Users\RITIN\OneDrive\Desktop\5th sem\predictive_analysis\lab_4\HR_comma_sep.csv")
print(df.head())
print(df.columns)
x=df.drop('left',axis=1)
x = pd.get_dummies(x, drop_first=True)
y=df['left']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
reg = LogisticRegression(max_iter=1000)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)   
print(y_pred)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()
print("Accuracy:", accuracy_score(y_test, y_pred))