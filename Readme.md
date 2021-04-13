**Facial Recognition Using An Ensemble Model of Dimension Reduction Techniques and Convolutional Neural Network**

The project proposes an ensemble model of different dimensionality reduction techniques that can help improve the accuracy of a facial recognition model. The next subsection shows how the proposed ensemble model works, followed by the few dimensionality techniques that will be used in this research. The datasets lfw and yaleb are used in order to compare the different dimension reduction techniques. Once the comparison is made, a hybrid of two best dimension reduction techniques is developed and used to extract feature vectors which is passed through dense cnn layers for predicting faces. 

Principal Component Analysis (PCA) 

It creates new variables that are a linear combination from existing uncorrelated variables. PCA makes use of information that lies within the multi-dimensional data and projects onto lower d-dimensional subspace so that sum of squared error is minimized or variance is maximized and provides uncorrelated distributions. Dense expressions that get generated is a consequence of eigen decomposition of covariance matrix. The decomposition of the covariance matrix,
Σ=U ∧U^T


Where ∑ = covariance matrix, U = matrix with eigen vectors.

A facial image when reduced by PCA has training vectors along the largest variance to compute the eigenvectors. These are eigen faces and each one of them represent a feature. Importance of each feature can be understood when vector of a face is projected onto a face space. Eigenface coefficients are unique for each face projected onto its face space. So, a recognition takes place when an facial image is projected onto the face subspace and its position is compared with position of already known faces.

Linear Discriminant Analysis (LDA)

It reduces dimensionality of multi-class data by generating linear projection which in turn helps in maximizing the separation among classes while preserving most of the class discriminatory data. Dataset undergoes eigen decomposition and the computed eigen vectors are located in within class scatter and between-class scatter matrix. The data is projected on a good feature space when all eignevalues have more or less the same magnitude. Significant information about data distribution is captured by the eigenvector represented by largest eigenvalue and therefore is selected.

Unlike PCA, LDA uses bases that are not orthogonal to encode discriminating data in a linear distinguishable subspace. It reduces the rate of within-class and between-class scatter. It emphasizes on segregating faces with varying features into groups of facial images. Face vectors are referred to as fisher faces.

Locally Linear Embedding (LLE)

It tries to get a non-linear structure in the provided data. The technique assumes that objects are probably flat on smaller scale. It works by finding nearest neighbours. It then computes weight that helps reconstruc the datapoints, inturn minimizing the equation of cost with constrained linear fits. Finally, the vectors are computed that are best reconstructed by given weights. 

Ensemble Model

The ensemble model would make use of multiple dimension reduction techniques to improve the performance of the facial recognition algorithm as represented in algorithm 1. Once the data is preprocessed, and split for train and test, it is then passed through two dimension reduction techniques. These techniques reduce the misleading data, noise and redundant features, that helps in the final prediction. Making use of multiple dimension reduction technique helps reduce the noise/external factors that exists in the images such as occlusion, illumination. Once dimensions are reduced by using both the techniques, feature vectors are extracted from the images. These feature vectors are passed through the dense cnn layer which helps train the model on given images and inturn predict the face as shown in figure 3. 

Flask WebApp

The final stage of the proposed system is a flask based web app module that gets trained using a dataset on the local machine. The webapp component as shown in figure 4 is integrated with the ensemble model and predicts the face that is given as live video input from the webcam. The model predicts the face most similar to the one present in the local dataset. Inorder to have the model perform better, there is a functionality added that lets images to be uploaded into the lcal dataset. The webapp is run on the local host.
The webapp was tested for different external factors such as occlusion, orientation of the head tilt and varying illuminations, as well as its recognition rate.


Data dimensionality is the total number of variables being measured in every observation. With increasing trends in technology, there is a huge volume of data that is being created. One such field of technology is computer vision. Human beings are able to detect and recognize faces with ease even with external conditions such as expressions, illuminations or viewing angle affecting the sight when compared to the machines. This is because of high dimensions associated with it. The way forward is by reducing the dimensions that in turn helps in minimizing the with-in class distances. This project aims to compare different applicable dimensional reduction techniques suitable for facial recognition system and propose an ensemble model of such techniques that will help improve the accuracy of the model and gauge the performance by testing it with different datasets consisting of facial images with varying illuminations, complex backgrounds, and expressions. The proposed ensemble model extracts feature vectors using a hybrid of two dimensional reduction techniques – principal component analysis and locally linear embedding, and pass them through dense convolutional neural network to predict faces. The model performs with a testing accuracy of 0.95 and a testing F1 score of 0.94. on labelled faces in the wild dataset. 

**Test Cases**

Test case 1

![testcase1](https://user-images.githubusercontent.com/16033184/111923913-05073880-8a78-11eb-8cac-baa1f917519a.png)

Test case 2 -- Occlusion

![testcase2_occlusion](https://user-images.githubusercontent.com/16033184/111923919-0fc1cd80-8a78-11eb-8f6e-1589044be440.png)

Test case 3 -- illumination

![testcase3-illumination](https://user-images.githubusercontent.com/16033184/111923927-194b3580-8a78-11eb-9111-b166a20e59f8.png)

Test case 4 -- orientation

![testcase4-orientation](https://user-images.githubusercontent.com/16033184/111923930-21a37080-8a78-11eb-80e4-5059e966e723.png)

Test case 5 -- image uploader

![testcase5-image uploader](https://user-images.githubusercontent.com/16033184/111923939-2b2cd880-8a78-11eb-913c-45b25960f867.png)

Test case 6 -- personality recognition using phone

![testcase6-famous personality](https://user-images.githubusercontent.com/16033184/111923968-54e5ff80-8a78-11eb-814e-278caefb6ac3.png)


