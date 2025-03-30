import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
)

class LogRegCCD:
    def __init__(self, lambdas):
        self.lambdas = lambdas
        self.coef_path_ = []
        self.intercept_path_ = []
        self.best_lambda_ = None
        self.best_coef_ = None
        self.best_intercept_ = None

    def fit(
        self, X_train, y_train, alpha=0.2, tol=1e-8, max_iter=100
    ):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        alpha (float): regularization strength parameter

        tol (float): convergence tolerance

        max_iter (int): maximum number of iterations allowed if the algorithm does not converge

        Returns
        -------
        None
        """
        X = np.array(X_train)
        y = np.array(y_train)
        N, d = X.shape

        for l in self.lambdas:  
            b = np.zeros(d)  # intercept and coefs initialzied to zero
            b_0 = 0  

            for _ in range(max_iter):
                b_old = b.copy()
                b0_old = b_0
                
                linear_comb = b_0 + X @ b
                p = expit(linear_comb) # sigmoid
                w = p * (1 - p)
                z = linear_comb + (y - p) / (w + 1e-10)
                # update each coordinate j independently
                for j in range(d):
                    X_j = X[:, j]  # feature col
                    numerator = np.sum(
                        w * X_j * (z - (X @ b - X_j * b[j]))
                    )  # with partial res
                    denominator = np.sum(w * X_j**2) + l * (1 - alpha)

                    denominator = max(denominator, 1e-10)  # to prevent division by 0
                    b[j] = self.soft_threshold(numerator, l * alpha) / denominator

                # b_0 = np.mean(y - p)
                eps = 1e-10
                b_0 = np.sum(w * (z - X @ b)) / (np.sum(w) + eps)

                if np.max(np.abs(b - b_old)) < tol and np.abs(b_0 - b0_old) < tol:
                    break

            self.coef_path_.append(b.copy())
            self.intercept_path_.append(b_0)

    def validate(self, X_valid, y_valid, measure="roc_auc"):
        """
        Select the best lambda from the solution path using validation data.

        The method evaluates the model's performance for each lambda in `self.lambdas`
        on the validation set using a specified metric. 
        The best lambda (and its corresponding coefficients) are stored in:
        - `self.best_lambda_`
        - `self.best_coef_`
        - `self.best_intercept_`

        Parameters
        ----------
        X_valid : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y_valid : array-like of shape (n_samples,)
            Target vector relative to X.
        measure : str, default='roc_auc'

        Returns
        -------
        None
        """
        best_score = -np.inf
        best_lambda = None
        best_index = None

        for i, l in enumerate(self.lambdas):
            b = self.coef_path_[i]
            b_0 = self.intercept_path_[i]
            # probas = 1 / (1 + np.exp(-(b_0 + X_valid @ b)))
            probas = expit(b_0 + X_valid @ b)
            if measure in ["recall", "precision", "f_measure", "balanced_accuracy"]:
                predictions = (probas >= 0.5).astype(int)
                score = self.compute_measure(y_valid, predictions, measure)
            elif measure == "roc_auc":
                score = roc_auc_score(y_valid, probas)
            elif measure == "sensitivity_precision_auc":
                score = average_precision_score(y_valid, probas)

            if score > best_score:
                best_score = score
                best_lambda = l
                best_index = i
        self.best_lambda_ = best_lambda
        self.best_coef_ = self.coef_path_[best_index]
        self.best_intercept_ = self.intercept_path_[best_index]

    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in `X_test` using the best model.

        Parameters
        ----------
        X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Probability of the sample belonging to class 1 (range [0, 1]).

        """
        # 1 / (1 + np.exp(-(self.best_intercept_ + X_test @ self.best_coef_)))
        return expit(self.best_intercept_ + X_test @ self.best_coef_)

    def predict(self, X_test):
        """
        Predict class labels for samples in `X_test` using the best model.

        Parameters
        ----------
        X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        return (self.predict_proba(X_test) >= 0.5).astype(int)

    def plot(self, X_valid, y_valid, measure="roc_auc"):
        """Plot performance measure vs lambda"""
        scores = []

        for i, l in enumerate(self.lambdas):
            b = self.coef_path_[i]
            b_0 = self.intercept_path_[i]
            # probas = 1 / (1 + np.exp(-(b_0 + X_valid @ b)))
            probas = expit(b_0 + X_valid @ b)

            if measure in ["recall", "precision", "f_measure", "balanced_accuracy"]:
                predictions = (probas >= 0.5).astype(int)
                score = self.compute_measure(y_valid, predictions, measure)
            elif measure == "roc_auc":
                score = roc_auc_score(y_valid, probas)
            elif measure == "sensitivity_precision_auc":
                score = average_precision_score(y_valid, probas)

            scores.append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(self.lambdas, scores, marker="o", linestyle="-")
        plt.xscale("log")
        plt.xlabel("Lambda")
        plt.ylabel(measure)
        plt.title(f"Performance Measure ({measure}) vs Lambda")
        plt.grid(True)
        plt.show()

    def plot_coefficients(self):
        """Plot coefficient paths vs lambda"""
        coef_paths = np.array(self.coef_path_).T

        plt.figure(figsize=(10, 6))
        for coef in coef_paths:
            plt.plot(self.lambdas, coef, linestyle="-", marker=".")
        plt.xscale("log")
        plt.xlabel("Lambda")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficient Paths vs Lambda")
        plt.grid(True)
        plt.show()

    def soft_threshold(self, z, gamma):
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0

    def compute_measure(self, y_true, y_pred, measure):
        if measure == "recall":
            return recall_score(y_true, y_pred)
        elif measure == "precision":
            return precision_score(y_true, y_pred)
        elif measure == "f_measure":
            return f1_score(y_true, y_pred)
        elif measure == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown measure: {measure}")