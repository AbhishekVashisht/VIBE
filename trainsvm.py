import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
raw_data = pd.read_csv('./fullcsvs/FullCSVatt.csv',header=None)
dataset = raw_data.values
X = dataset[0:1610, 0:4320].astype(float)
Y = dataset[0:1610, 4320]
print("YP")
print(raw_data.columns)
it=0
# itt=0
for itt in range(0,190):
    if Y[itt]==0:
        it=it+1

print(it)
# print(raw_data[[4320]==0].count())
# print(pd.count(Y=1))


# x=dataset
# kmeans5 = KMeans(n_clusters=2)
# y_kmeans5 = kmeans5.fit_predict(X)
# print(y_kmeans5)
# y=dataset[:,4320]
# kmeans5.cluster_centers_
#
# print(confusion_matrix(Y,y_kmeans5))
# print(classification_report(Y,y_kmeans5))



# plt.scatter(x[:,0],x[:,1],c=Y,cmap='rainbow')
# plt.scatter(x[:,0],x[:,1],c=y_kmeans5,cmap='rainbow')


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print("YP")
svclassifier = SVC(kernel='linear',max_iter=2000)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("g")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))