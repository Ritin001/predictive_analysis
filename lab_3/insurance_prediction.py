import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns

df=pd.read_csv(r"C:\Users\RITIN\OneDrive\Desktop\5th sem\predictive_analysis\lab_3\insurance_data.csv")
plt.scatter(df['age'], df['bought_insurance'])
plt.show()
x_train,x_test,y_train,y_test = train_test_split(df[['age']],df['bought_insurance'],test_size=0.2,random_state=0)
reg = LogisticRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()
print("Accuracy:", accuracy_score(y_test, y_pred))