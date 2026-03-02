import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv(r"C:\Users\RITIN\OneDrive\Desktop\5th sem\predictive_analysis\lab_5\salaries.csv")
inputs = ['company', 'job', 'degree']
target = ['salary_more_then_100k']
le_company = LabelEncoder()
le_job = LabelEncoder() 
le_degree = LabelEncoder()
df['company'] = le_company.fit_transform(df['company'])
df['job'] = le_job.fit_transform(df['job'])
df['degree'] = le_degree.fit_transform(df['degree'])
model = DecisionTreeClassifier()
model.fit(df[inputs],df[target])
y=model.predict([[2, 0, 0]])
print(y)
plot_tree(model, feature_names=inputs, filled=True)
plt.show()