import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_train, X_test, y_train, y_test, include_training = False):
	""" Evaluate score and draw confusion matrix of a model
	
	Parameters
	----------
	model : any_model
		Your trained model.
	X_train, X_test, y_train, y_test : scipy.sparse._csr.csr_matrix, scipy.sparse._csr.csr_matrix, numpy.ndarray, numpy.ndarray
		The dataset used for evaluating the model.
	include_training : bool
		Trigger True if you want to output the training metrics. On default, false.
	
	Return
	------
	None

	"""
	# [EVALUATE ON THE TRAIN]
	if include_training == True:
		# Predict from the train
		y_pred = model.predict(X_train)

		# Calculate score
		print("Score of on train are:")
		print("\t- Accuracy score: {:.2f}".format(accuracy_score(y_pred, y_train)))
		print("\t- Micro F1 score: {:.2f}".format(f1_score(y_pred, y_train, average = 'micro')))
		print("\t- Macro F1 score: {:.2f}".format(f1_score(y_pred, y_train, average = 'macro')))
		
		# Draw confusion matrix
		cm = confusion_matrix(y_pred, y_train)
		cm_plt = ConfusionMatrixDisplay(cm)
		cm_plt.plot()
		cm_plt.ax_.set_title("Confusion matrix on Train")

	# [EVALUATE ON THE TEST]
	# Predict from the test
	y_pred = model.predict(X_test)

	# Calculate score
	print("Score of on test are:")
	print("\t- Accuracy score: {:.2f}".format(accuracy_score(y_pred, y_test)))
	print("\t- Micro F1 score: {:.2f}".format(f1_score(y_pred, y_test, average = 'micro')))
	print("\t- Macro F1 score: {:.2f}".format(f1_score(y_pred, y_test, average = 'macro')))

	# Draw confusion matrix
	cm = confusion_matrix(y_pred, y_test)
	cm_plt = ConfusionMatrixDisplay(cm)
	cm_plt.plot()
	cm_plt.ax_.set_title("Confusion matrix on Test")


def draw_learning_curve(model, X_train, y_train, cv = 5, train_sizes = np.linspace(0.2, 1, 5), scoring = 'accuracy'):
	""" Draw the learning curve for a model

	Parameters
	----------
	model : any_model
		Your trained model.
	X_train, y_train : scipy.sparse._csr.csr_matrix, numpy.ndarray
		The datasets on which you want to draw the learning curve.
	cv : int
		Number of cross-validation folds you want to commit. On default, 5.
	train_sizes : np.ndarray
		The threshold list for the learning curve to be evaluated. On default, [.2, .4, .6, .8, 1]
	scoring : str
		Scoring metric used. On default, accuracy score is used.
	
	Return
	------
	None
	
	"""
	# Calculate list score of cross validation
	_, train_score, test_score = learning_curve(model, X_train, y_train, n_jobs = -1, cv = cv,
												train_sizes = train_sizes, scoring = scoring)

	# Calculate mean and standard deviation of the scores
	train_mean_score = np.mean(train_score, axis = 1)
	train_std_score = np.std(train_score, axis = 1)
	test_mean_score = np.mean(test_score, axis = 1)
	test_std_score = np.std(test_score, axis = 1)

	# Draw the plot
	plt.fill_between(train_sizes, train_mean_score - train_std_score, train_mean_score + train_std_score, alpha = 0.1, color = 'g')
	plt.fill_between(train_sizes, test_mean_score - test_std_score, test_mean_score + test_std_score, alpha = 0.1, color = 'r')

	plt.plot(train_sizes, train_mean_score, color = 'g')
	plt.plot(train_sizes, test_mean_score, color = 'r')

def load_processed_data():
	""" Return the preprocessed data in order: X_train_bow, X_test_bow, X_train_tfidf, X_test_tfidf, y_train, y_test
	
	Parameters
	----------
	None
	
	Return
	------
	X_train_bow, X_test_bow, X_train_tfidf, X_test_tfidf, y_train, y_test

	"""
	directory = "data/dataset/processed/"
	return (
		sparse.load_npz(directory + "X_train_bow.npz"),
		sparse.load_npz(directory + "X_test_bow.npz"),
		sparse.load_npz(directory + "X_train_tfidf.npz"),
		sparse.load_npz(directory + "X_test_tfidf.npz"),
		np.loadtxt(directory + "y_train.txt", dtype='str'),
		np.loadtxt(directory + "y_test.txt", dtype='str')
	)
