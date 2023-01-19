#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import this python file for preprocessing extra dataset
import preprocess

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV

## Define class for Task A1
class SVM_A1:
	def __init__(self):
		self.param_RBF = {'C': stats.uniform(0.1, 10),
						 'gamma': stats.uniform(0.001, 0.01),
						 'kernel': ['rbf']}
		self.param_poly = {'C': stats.uniform(0.1, 10),
						  'coef0': stats.uniform(1, 5),
						  'degree': stats.uniform(1, 5),
						  'kernel': ['poly']}
		self.param_linear = {'C': stats.uniform(0.1, 10),
							'kernel': ['linear']}


	def SVC_randomSearch(self, X, y, param_kernel):
		# number of jobs = -1 mean using all processors
		rand_search = RandomizedSearchCV(SVC(), param_kernel, n_iter=10, n_jobs=-1, refit=True, verbose=3)
		rand_search.fit(X, y)
		
		return rand_search.best_params_, rand_search.best_estimator_
						
	def train(self, X, y, test_X, test_Y):
		# Obtaining optimum hyperparameters and classifier for different kernel
		print('Tuning optimum hyper parameter for SVM with polynomial kernel...')
		polySVC_param, clf_polySVC = self.SVC_randomSearch(X, y, self.param_poly)

		print('Tuning optimum hyper parameter for SVM with RBF kernel...')
		rbfSVC_param, clf_rbfSVC = self.SVC_randomSearch(X, y, self.param_RBF)

		print('Tuning optimum hyper parameter for SVM with linear kernel...')
		linearSVC_param, clf_linearSVC = self.SVC_randomSearch(X, y, self.param_linear)

		# Predict with the best linear SVM classifier
		pred1 = clf_linearSVC.predict(test_X)
		score1 = accuracy_score(test_Y,pred1)

		# Predict with the best polynomial SVM classifier
		pred2 = clf_polySVC.predict(test_X)
		score2 = accuracy_score(test_Y,pred2)

		# Predict with the best RBF SVM classifier
		pred3 = clf_rbfSVC.predict(test_X)
		score3 = accuracy_score(test_Y,pred3)

		# Return the score as a dictionary
		train_acc = {'Linear SVM': score1, 'Polynomial SVM': score2, 'RBF SVM': score3}
		classifier = [clf_linearSVC, clf_polySVC, clf_rbfSVC]

		return train_acc, classifier

	# Predict the output of extra test dataset
	def test(self, classifier, X, y):
		# Predict with the best linear SVM classifier
		pred1 = classifier[0].predict(X)
		score1 = accuracy_score(y,pred1)

		# Predict with the best polynomial SVM classifier
		pred2 = classifier[1].predict(X)
		score2 = accuracy_score(y,pred2)

		# Predict with the best RBF SVM classifier
		pred3 = classifier[2].predict(X)
		score3 = accuracy_score(y,pred3)

		# Return the score as a dictionary
		test_acc = {'Linear SVM': score1, 'Polynomial SVM': score2, 'RBF SVM': score3}

		return test_acc




