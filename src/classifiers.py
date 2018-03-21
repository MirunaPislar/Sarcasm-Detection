import utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE


def get_regularization_params(a=-1, b=1, c=3, d=1, e=5):
    reg_range = np.outer(np.logspace(a, b, c), np.array([d, e]))
    reg_range = reg_range.flatten()
    return reg_range


def grid_classifier(x_train, y_train, x_test, y_test, model, parameters,
                    make_feature_analysis=False, feature_names=None, top_features=0, plot_name="coeff"):
    grid = GridSearchCV(estimator=model, param_grid=parameters, verbose=0)
    grid.fit(x_train, y_train)
    sorted(grid.cv_results_.keys())
    classifier = grid.best_estimator_
    if make_feature_analysis:
        utils.plot_coefficients(classifier, feature_names, top_features, plot_name)
    y_hat = classifier.predict(x_test)
    utils.print_statistics(y_test, y_hat)


def linear_svm_grid(x_train, y_train, x_test, y_test, class_ratio,
               make_feature_analysis=False, feature_names=None, top_features=0, plot_name="coeff"):
    utils.print_model_title("Linear SVM")
    C_range = get_regularization_params()
    parameters = {'C': C_range}
    linear_svm = LinearSVC(C=1.0, class_weight=class_ratio, penalty='l2')
    grid_classifier(x_train, y_train, x_test, y_test, linear_svm, parameters,
                    make_feature_analysis, feature_names, top_features, plot_name)


def nonlinear_svm_grid(x_train, y_train, x_test, y_test, class_ratio,
                  make_feature_analysis=False, feature_names=None, top_features=0, plot_name="coeff"):
    utils.print_model_title("Nonlinear SVM")
    C_range = get_regularization_params(a=-1, b=0, c=2, d=1, e=5)
    gamma_range = get_regularization_params(a=-2, b=-1, c=2, d=1, e=5)
    parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}
    nonlinear_svm = SVC(class_weight=class_ratio)
    grid_classifier(x_train, y_train, x_test, y_test, nonlinear_svm, parameters,
                    make_feature_analysis, feature_names, top_features, plot_name)


def logistic_regression_grid(x_train, y_train, x_test, y_test, class_ratio,
                        make_feature_analysis=False, feature_names=None, top_features=0, plot_name="coeff"):
    utils.print_model_title("Logistic Regression")
    C_range = [0.001, 0.01, 0.1, 1, 10, 100]
    parameters = {'C': C_range}
    log_regr = LogisticRegression(C=1.0, class_weight=class_ratio, penalty='l2')
    grid_classifier(x_train, y_train, x_test, y_test, log_regr, parameters,
                    make_feature_analysis, feature_names, top_features, plot_name)


def linear_svm(x_train, y_train, x_test, y_test, class_ratio='balanced'):
    utils.print_model_title("Linear SVM")
    svm = LinearSVC(C=0.01, class_weight=class_ratio, penalty='l2')
    svm.fit(x_train, y_train)
    y_hat = svm.predict(x_test)
    utils.print_statistics(y_test, y_hat)


def logistic_regression(x_train, y_train, x_test, y_test, class_ratio='balanced'):
    utils.print_model_title("Logistic Regression")
    regr = LogisticRegression(C=0.01, class_weight=class_ratio, penalty='l2')
    regr.fit(x_train, y_train)
    y_hat = regr.predict(x_test)
    utils.print_statistics(y_test, y_hat)


def feature_selection(x_train, y_train, x_test, y_test):
    print("Feature selection with LinearSVC")
    model = LinearSVC(C=0.1, penalty='l2')
    rfe = RFE(model, 5)
    best_features_model = rfe.fit(x_train, y_train)
    y_hat = best_features_model.predict(x_test)
    utils.print_statistics(y_test, y_hat)
