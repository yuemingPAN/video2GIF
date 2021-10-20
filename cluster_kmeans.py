import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt





dataset = pd.read_csv('C:/Users/v-yuemingpan/Desktop/features/JUstinGhost_resnet.csv')

#print(dataset.head())

col = dataset.columns.values.tolist()
data = np.array(dataset[col])

#b = data[1]
#print(np.shape(b))
print(np.shape(data))


k = 3
model = KMeans(n_clusters=k)
model.fit(data)

centers = model.cluster_centers_

labels = model.labels_

distance0 = []
distance1 = []
distance2 = []
cnt = -1
cnt1 = -1
cnt2 = -1
for i in range(data.shape[0]):
    if(labels[i]==0):
        cnt = cnt + 1
        dist = np.linalg.norm(data[i]-centers[0]) 
        distance0.append(dist)
        print((i+1)*30 , cnt)
        print(dist)
print(distance0)
print(np.min(distance0))
print(np.argmin(distance0))
print("第二类------------------------------------------------")
for i in range(data.shape[0]):
    if(labels[i]==1):
        cnt1 = cnt1 + 1
        dist = np.linalg.norm(data[i]-centers[1]) 
        distance1.append(dist)
        print((i+1)*30 , cnt1)
        print(dist)

print(distance1)
print(np.min(distance1))
print(np.argmin(distance1))
print("第三类----------------------------------------------------")
for i in range(data.shape[0]):
    if(labels[i]==2):
        cnt2 = cnt2 + 1
        dist = np.linalg.norm(data[i]-centers[2]) 
        distance2.append(dist)
        print((i+1)*30 , cnt2)
        print(dist)
print(distance2)
print(np.min(distance2))
print(np.argmin(distance2))
# min = np.min(distance0)
# print(min)
#打印出列表中最小值的索引
#print(np.shape(distance0)


#print(labels)
#print(np.shape(centers[1]))

#print(model.inertia_)
# inertia = []

# for i in range(1,10,1):
#     clf = KMeans(n_clusters=i)
#     s = clf.fit(data)
#     inertia.append(clf.inertia_)

# plt.plot(inertia)
# plt.show()
# print(np.shape(centers[1]))
# print(np.argmin(distance0))
# print(np.shape(distance0))
# print(np.min(distance0))
# print(distance0[1])

