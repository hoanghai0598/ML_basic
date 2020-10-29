import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import MLTSVM
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing as pp
from time import process_time

t1_start = process_time()

df = pd.read_csv(r"E:\Doan\FinalDataset\All.csv")

#print(df.info())

nan_select = df.columns[df.isna().any()].tolist()
null_select = df.columns[df.isnull().any()].tolist()

print(nan_select)
#print(null_select)
#print(df.isin([-1]))
#print(df.info())
#print("###############################")

#xoa cac cot chua gia tri nan,-1
for value in nan_select:
    df.drop(value, axis=1, inplace=True)

#print("###############################")
#print(df.info())

#thay doi gia tri cua label
df['URL_Type_obf_Type'].replace('Defacement', 1,inplace=True)
df['URL_Type_obf_Type'].replace('benign', 0,inplace=True)
df['URL_Type_obf_Type'].replace('malware', 2,inplace=True)
df['URL_Type_obf_Type'].replace('phishing', 3,inplace=True)
df['URL_Type_obf_Type'].replace('spam', 4,inplace=True)

y = LabelBinarizer().fit_transform(df['URL_Type_obf_Type']) #One hot encoder
df.drop('URL_Type_obf_Type', axis=1, inplace=True)
data = df

std = pp.MinMaxScaler()
data_std = std.fit_transform(data)

print(data_std.info())
X_train, X_test, y_train, y_test = train_test_split(data_std, y, test_size=0.3, random_state=42)

"""neigh = MLkNN()
neigh.fit(X_train, y_train)

print("Accuracy ",neigh.score(X_test, y_test))
t1_stop = process_time()
print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)"""