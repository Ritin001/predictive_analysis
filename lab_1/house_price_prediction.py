import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\RITIN\OneDrive\Desktop\5th sem\predictive_analysis\lab_1\Houseprice1.csv")
plt.scatter(df.area,df.price,color='red',marker='+')
plt.show()

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df['price'])
reg.predict([[1500,3,40]])
new_df = df.drop('price',axis=1)
p=reg.predict(new_df)
new_df['price']=p
print(new_df)
