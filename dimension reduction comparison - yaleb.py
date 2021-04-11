#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries

import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np


# We write a function to read .pgm files which is format of YaleB dataset.

# In[2]:


# converting the image format to a readable version

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    #print (pgmf.readline())
    assert pgmf.readline() == b'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster


# In[3]:


# reading the Extended Yale B dataset from the local repository
YALEB_DATASET = 'data/YaleB'


# We read the dataset and visualize some images. We discard the persons which have less than 50 face images to reduce noise and variance.

# In[4]:


# prepcrocssing of images that includes grey scale, resize and reading images and labels into arrays

n_imgs = 2425
width, height = 64,64
yaleb_imgs = np.zeros((n_imgs, height, width ,), dtype=np.float32)
yaleb_labels = np.zeros((n_imgs, ))
fig = plt.figure(figsize=(16, 6))
i = 1
unique_labels = {}
c = 0
for root, dirs, files in os.walk(YALEB_DATASET):
    for file in files:
        if file.endswith(".pgm"):
            img_path = os.path.join(root, file)
            with open(img_path, 'rb') as f:
                try:
                    img = read_pgm(f)
                except:
                    continue
            label = root.split('/')[-1]
            if label not in unique_labels:
                unique_labels[label] = c
                c+=1
            img = np.array(img, dtype=np.uint8)[:,:,np.newaxis]
            img = cv2.resize(img[:,:,0], (width, height))
            yaleb_imgs[i-1] = img[:,:]
            yaleb_labels[i-1] = unique_labels[label]
            if i <= 8:
                plt.subplot(2, 4, i)
                plt.imshow(img[:,:], cmap='gray', vmin=0, vmax=255) 
            i+=1
plt.show()
print ("Read %s images of %s persons from YaleB dataset"%(i,c))


# In[5]:


# test and train split after which test and train data are reshaped

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(yaleb_imgs, yaleb_labels, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1,width*height)
X_test = X_test.reshape(-1,width*height)


# ## Principal Comoponent Analysis

# We now compute Principal Component Analysis which extracts eigenfaces of the train dataset. The PCA's transformation matrix will be later used for testing.

# In[6]:


from sklearn.decomposition import PCA
n_components = 128
print("Extracting the top {} eigenfaces from {} faces".format(n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# Transform data into principal components representation
print("Transforming the test data using the the orthonormal basis of PCA")
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train_pca, y_train,)


# In[8]:


# Evaluation of the model
y_pred = model.predict(X_test_pca)


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))


# In[10]:


import sklearn
pca_acc = sklearn.metrics.accuracy_score(y_test, y_pred, )


# ## Linear Discriminant Analysis

# In[11]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)


# In[12]:


# Evaluation of the model
y_pred = model.predict(X_test)


# In[13]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))


# In[14]:


import sklearn
lda_acc = sklearn.metrics.accuracy_score(y_test, y_pred, )


# ## Non-Negative Matrix Factorization

# In[15]:


from sklearn.decomposition import NMF
model = NMF(n_components=128, init='random', random_state=0)
X_train_nmf = model.fit_transform(X_train)
X_test_nmf = model.transform(X_test)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train_nmf, y_train,)


# In[17]:


# Evaluation of the model
y_pred = model.predict(X_test_nmf)


# In[18]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))


# In[19]:


import sklearn
nmf_acc = sklearn.metrics.accuracy_score(y_test, y_pred, )


# ## Independent Component Analysis

# In[20]:


from sklearn.decomposition import FastICA
model = FastICA(n_components=128, random_state=0)
X_train_ica = model.fit_transform(X_train)
X_test_ica = model.transform(X_test)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train_ica, y_train,)


# In[22]:


# Evaluation of the model
y_pred = model.predict(X_test_ica)


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))


# In[24]:


import sklearn
ica_acc = sklearn.metrics.accuracy_score(y_test, y_pred, )


# ## Singular Value Decomposition

# In[25]:


from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=128, random_state=0)
X_train_svd = model.fit_transform(X_train)
X_test_svd = model.transform(X_test)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train_svd, y_train,)


# In[27]:


# Evaluation of the model
y_pred = model.predict(X_test_svd)


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))


# In[29]:


import sklearn
svd_acc = sklearn.metrics.accuracy_score(y_test, y_pred, )


# ## Locality Preserving Projection

# In[30]:


get_ipython().system('pip3 install lpproj')
from lpproj import LocalityPreservingProjection 
model = LocalityPreservingProjection(n_components=128, )
X_train_lpp = model.fit_transform(X_train)
X_test_lpp = model.transform(X_test)


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_train_lpp, y_train,)


# In[32]:


# Evaluation of the model
y_pred = model.predict(X_test_lpp)


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))


# In[34]:


import sklearn
lpp_acc = sklearn.metrics.accuracy_score(y_test, y_pred, )


# ## Comparing different techniques

# In[35]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
tech = ['PCA', 'LDA', 'NMF', 'ICA', 'SVD', 'LPP']
accs = [pca_acc, lda_acc, nmf_acc, ica_acc, svd_acc, lpp_acc]
bars = ax.bar(tech, accs)
bars[0].set_color('red')
bars[1].set_color('blue')
bars[2].set_color('green')
bars[3].set_color('black')
bars[4].set_color('yellow')
bars[5].set_color('purple')
plt.show()


# The best two methods are LDA and PCA. Now we combine both techniques by first applying PCA over data, then apply LDA on the concatenation of PCA and raw input features.

# ## LDA and PCA Combination

# In[36]:


from sklearn.decomposition import PCA
n_components = 128
print("Extracting the top {} eigenfaces from {} faces".format(n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# Transform data into principal components representation
print("Transforming the test data using the the orthonormal basis of PCA")
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[37]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
X_train_lda = model.fit_transform(np.concatenate((X_train, X_train_pca), axis=1), y_train)


# In[38]:


# Evaluation of the model
y_pred = model.predict(np.concatenate((X_test, X_test_pca), axis=1))


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(classification_report(y_test, y_pred, ))

