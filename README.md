# PCA-based face recognition

## About the project

Face recognition is a fundamental task in computer vision with applications ranging from security systems to human-computer interaction.
While modern deep learning techniques have achieved state-of-the-art accuracy, this project explores a classical mathematical approach based on Principal Component Analysis (PCA) and the Eigenfaces method. Using linear algebra concepts such as covariance matrices, eigenvalues, and eigenvectors, we aim to develop a face recognition system. We plan on testing and comparing our model with the pre-made OpenCV designs to see how is the purely mathematical approach different form the modern techniques.

## Structure
eigenfaces.ipyng has implementation of our algorithm with explanations and test for one person.<br>

eigenfaces\_model.py has all of implementation structured in a class which can be easily used for training and testing data.<br>

eigenfaces\_model\_optimised.py has optimised version, but for know it has worse accuracy(83\%).<br>

accuracy\_testing.py this script is used for testing implementations.<br>

## Set up the project

```
pip install -r Requirements.txt
```

## Run the testing of our work

```
jupyter imlementations_testing.ipynb
```
