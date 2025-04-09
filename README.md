## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
 Name: D.B.V. SAI GANESH
 Reg No: 212223240025
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/b8726d13-1b16-43fe-92e6-afd12882c02b)

## Original Encoder
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/02b8f785-d322-406d-9016-4c52227d497c)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/150f2a0a-d635-420e-a439-7ff29c2de541)

## Label Encoder

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/df78b7e6-3660-4ebd-86d4-2cf98baa70f1)

## One Hot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/170df4ef-d238-448a-a693-9fc6ad61202a)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/5c2c692f-0d2c-404a-b590-20299f39f3ff)

## Binary Encoder

```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/c3aeab3c-f7c0-4dd2-b34a-1b7680128581)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/9cc8e1ba-7876-4aba-a436-8331571185d9)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/116b2db2-e2be-4d90-bbe0-f2b05c4037c0)

## Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/2eb89c66-4460-47da-a531-e252dd1a56af)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/9a9b4bb4-a66c-43b8-b60a-47b187c60ada)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/aa16d8bc-59ad-4687-b420-20711d0cccc8)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/cab29a93-eab9-45a8-8059-9f8d066436bf)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/5ff3395c-13d8-4b8d-a5fe-bb9596f98982)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d9f2ee81-418d-48fa-9497-722eaf00586d)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/74acb41a-b88f-4d82-987e-5c138f17a524)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/37026769-38d5-4d01-be15-29eddebdc998)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/edf03f3e-a4ae-4db9-b206-7af342e774d5)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a8835022-9b7e-4564-962c-6c10ecd5af9d)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/43d22987-4960-466c-982d-338d8f049ffb)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4172dad3-0df7-4f5a-b74c-f68d60e15161)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f07a1f66-90d6-49e5-8637-00b5878a33ce)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/822d90b0-bf58-48d4-b618-8dc827df92e7)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/77f542f1-b956-4f91-bcf0-6ca0bed85e99)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/fe0bacd3-33e3-4542-a2b8-c0baf9747f54)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/cc1c1a2b-2453-4007-a259-731b08581af0)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/002189f2-9439-4cb7-ada0-447632a32431)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/d3b77c6e-6464-4141-85a9-7b3f55e1cf88)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c4c00eb8-7ff7-49b5-81d4-d084e46bf895)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/af032fa5-84bf-44c1-b3e3-f74bdc22fd63)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
