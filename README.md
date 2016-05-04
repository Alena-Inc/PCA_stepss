# PCA DIY

Principal Component Analysis

In the following project we set out to understand how the principal component analysis worked (PCA), to understand the basic of this method we read the tutorial by Lindsay I Smith, where she states step by step what PCA does to the data and how it changed from being four variables to it being tree variables. 

To begin we should mention that the data set used in this work is the IRIS dataset used in Fisher´s paper, in the following image you can see that we obtain different data from this set, one being the actual measurement from the follower and the other the labeling of the different species. 
 
The first step to the PCA algorithm is to obtain and subtract the mean from the different measurements preformed. 
 
We then set out to obtain the covariance matrix which can be easily obtain by multiplying the data set obtained by subtracting the mean and the transpose of that same matrix. To multiply we use the numpy.dot function, the following image shows the obtained covariance matrix. 
 
After obtaining the covariance matrix we found the eigenvalues and vectors with the help of the linalg.eig function to which there are two results; w known as the values and v which stand for the eigenvectors, this both are presented in the following image. 
 
The next step is to start thinking about the new matrix to be able to do this we needed to remove the smaller value from the eigenvalues and from that remove the corresponding eigenvector. And then we multiplied the transpose of the original data and the transpose of the reduce eigenvector to obtain a matrix size 3X150. 
To plot the resulting data we adapted the code done by Gael Varoquaux about the same problem. The last image presents the new data set after the PCA. 
 
To make a quick and final comparison, the figure bellow is the plot of the data without PCA, it is easy to observe the differences, how virginica and versicolour are almost interpolated, and the three are too disperse. 
