import datetime
import pandas as pd
import numpy as np
import os
import time

# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, accuracy_score, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

CSV_HEADER = "Description; feature count; accuracy; true positive; false negative; false positive; true negative; false negative rate [%]; false positive rate [%]; AUC\n"

class FS:
  FS_KBEST = "KBEST"
  FS_RFE = "RFE"
  FS_PCA = "PCA"

FS_TYPES = [FS.FS_KBEST, FS.FS_RFE, FS.FS_PCA]

classifiers_params = [
    (DecisionTreeClassifier(), {
      "criterion": ["gini", "entropy", "log_loss"],
      "splitter": ["best", "random"],
      "max_depth": [None, 5, 10, 15], 
      "min_samples_split":[2, 4, 6],
      "min_samples_leaf": [1, 2, 4, 6],
      "max_features": ["sqrt", "log2", None, 1, 5, 25, 100],
      "class_weight": [None, "balanced", {0:1, 1:2}, {0:2, 1:1}, {0:1, 1:5}, {0:5, 1:1}]
      }),
    (SVC(), {
      "C": [0.1, 1, 10],
      "kernel": ["linear", "rbf", "poly", "sigmoid"],
      "degree": [2, 3, 4],
      "gamma": ["scale", "auto", 0.1, 1],
      "class_weight": [None, "balanced", {0:1, 1:2}, {0:2, 1:1}, {0:1, 1:5}, {0:5, 1:1}]
      }),
    (LinearRegression(), {
      "fit_intercept": [True, False],
      "positive": [True, False],
      "n_jobs": [None, -1]
      }),
    (KNeighborsClassifier(), {
      "n_neighbors": [3, 5, 7],
      "weights": ["uniform", "distance"],
      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
      "p": [1, 2]
      }),
    (BernoulliNB(), {
      "alpha": [0.1, 0.5, 1.0],
      "binarize": [0.5], # because input is 0/1 but not saved as boolean
      "fit_prior": [True, False]
      }),
    (MLPClassifier(), {
      "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
      "activation": ["relu", "logistic", "tanh"],
      "solver": ["lbfgs", "adam"],
      "alpha": [0.0001, 0.001, 0.01],
      "learning_rate": ["constant", "invscaling", "adaptive"],
      "max_iter": [100, 200, 300]
      }),
    (RandomForestClassifier(), {
      "n_estimators": [50, 100, 150],
      "criterion": ["gini", "entropy"],
      "max_depth": [None, 5, 10, 15],
      "min_samples_split": [2, 4, 6],
      "min_samples_leaf": [1, 2, 4, 6],
      "max_features": ["sqrt", "log2", None, 25, 100],
      "class_weight": [None, "balanced", {0:1, 1:2}, {0:1, 1:5}]
      }),
  ]

class ExperimentResults:
  def __init__(self, description, feature_count, accuracy, tp, fn, fp, tn, auc) -> None:
    self.description = description
    self.feature_count = feature_count
    self.accuracy = accuracy
    self.tp = tp
    self.fn = fn
    self.tn = tn
    self.fp = fp
    self.auc = auc
    self.fnr = fn / (fn + tp)
    self.fpr = fp / (fp + tn)
  
  def show(self):
    print("---")
    print(self.description)
    print("---")
    print("Features count:", self.feature_count)
    print("Accuracy:", self.accuracy)
    print(f"tp: {self.tp}, fn: {self.fn}, fp: {self.fp}, tn: {self.tn}")
    print(f"FNR: {self.fnr*100:.2f}%, FPR: {self.fpr*100:.2f}%")
    print(f"AUC: {self.auc:.4f}")
  
  def __str__(self) -> str:
    return f"Description: {self.description} " +\
           f"Features count: {self.feature_count} "+\
           f"Accuracy: {self.accuracy} "+\
           f"tp: {self.tp}, fn: {self.fn}, fp: {self.fp}, tn: {self.tn} " +\
           f"FNR: {self.fnr*100:.2f}%, FPR: {self.fpr*100:.2f}% " +\
           f"AUC: {self.auc:.4f}"
  
  def csv(self):
    return f"{self.description}; {self.feature_count}; {self.accuracy:.4f}; {self.tp}; {self.fn}; {self.fp}; {self.tn}; {self.fnr*100:.2f}; {self.fpr*100:.2f}; {self.auc:.4f}\n"
  
def write_to_csv(results, file_name_prefix = "experiment"):
  current_time = datetime.datetime.now()
  file_name = f"results\\{file_name_prefix}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

  lines = [CSV_HEADER, *[r.csv() for r in results]]
  with open(file_name, mode='w', encoding='utf8') as file:
    file.writelines(lines)
  print("Results saved to file:", file_name)
    
def summarize(test, pred):
  tn, fp, fn, tp = confusion_matrix(test, pred).ravel()

  # print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
  # FPR = fp / (fp + tn)
  # FNR = fn / (fn + tp)
  
  auc = roc_auc_score(test, pred)
  # print(f"FNR: {FNR*100:.2f}%, FPR: {FPR*100:.2f}%")
  # print(f"AUC: {auc:.4f}")
  return tn, fp, fn, tp, auc

def do_feature_selection(classifier, X_train, X_test, Y_train, feature_selection, fs_k):
  match feature_selection:
    case FS.FS_KBEST:
      selector = SelectKBest(score_func=f_classif, k=fs_k)
      X_train_fs = selector.fit_transform(X_train, Y_train)
      X_test_fs = selector.transform(X_test)
    case FS.FS_RFE:
      rfe = RFE(classifier, n_features_to_select=fs_k)
      X_train_fs = rfe.fit_transform(X_train, Y_train)
      X_test_fs = rfe.transform(X_test)
      pass
    case FS.FS_PCA:
      pca = PCA(n_components=fs_k)
      X_train_fs = pca.fit_transform(X_train, Y_train)
      X_test_fs = pca.transform(X_test)
    case _:
      X_train_fs = X_train
      X_test_fs = X_test
  return X_train_fs, X_test_fs

def get_default_classifier(classifier):
    return classifier.__class__()

def run_test(classifier, X_train, X_test, Y_train, Y_test, map_preds=None, feature_selection:str=None, fs_k=10, param_grid=None, grid_search=None, label_suffix=None):
  X_train_fs, X_test_fs = do_feature_selection(classifier, X_train, X_test, Y_train, feature_selection, fs_k)

  if param_grid:
    if grid_search == "custom":
      custom_scorer = make_scorer(custom_scoring_fn, greater_is_better=False)
      grid_search = GridSearchCV(classifier, param_grid, scoring=custom_scorer)
    elif grid_search == "roc_auc":
      grid_search = GridSearchCV(classifier, param_grid, scoring='roc_auc')
    else:
      grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy')
      
    grid_search.fit(X_train, Y_train)
    best_params = grid_search.best_params_
    classifier.set_params(**best_params)
  
  classifier.fit(X_train_fs,Y_train)
  Y_pred = classifier.predict(X_test_fs)
  acc = classifier.score(X_test_fs, Y_test)

  if(map_preds):
    Y_pred_mapped = map_preds(Y_pred)
    acc = accuracy_score(Y_test, Y_pred_mapped)
    Y_pred = Y_pred_mapped
  tn, fp, fn, tp, auc = summarize(Y_test, Y_pred)
  label = str(classifier) + (label_suffix if label_suffix is not None else "")
  result = ExperimentResults(label, X_train_fs.shape[1], acc, tp, fn, fp, tn, auc)
  result.show()
  return result

def custom_scoring_fn(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  tn, fp, fn, tp = cm.ravel()
  
  # print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
  # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  # disp.plot()
  # plt.show()

  FNR = fn / (fn + tp)
  FPR = fp / (fp + tn)
  
  penalty = 10
  return FNR + penalty*max(0, FPR-0.005)

def get_data():
  dataframe = pd.read_csv("./data_ml/spam.dat")

  X = dataframe.drop(['target'],axis=1)

  mapping = {'yes': 1, 'no': 0}
  Y = dataframe.target.map(mapping)
  # print(Y)
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=331, test_size=0.33)
  print("train rows: {}, test rows: {}".format(X_train.shape[0], X_test.shape[0]))

  scaler = StandardScaler()
  
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled, Y_train, Y_test

def main(X_train, X_test, Y_train, Y_test):
  # X_train, X_test, Y_train, Y_test = get_data()

  from sklearn.tree import DecisionTreeClassifier
  dt = DecisionTreeClassifier()
  run_test(
    "DecisionTreeClassifier: default",
    dt,
    X_train, X_test, Y_train, Y_test
  )
  
  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier()
  run_test(
    "RandomForestClassifier: default",
    rf,
    X_train, X_test, Y_train, Y_test
  )
  
  from sklearn.neural_network import MLPClassifier
  mlp = MLPClassifier()
  run_test(
    "MLPClassifier: default",
    mlp,
    X_train, X_test, Y_train, Y_test
  )

  from sklearn.svm import SVC
  svm = SVC()
  run_test(
    "Support Vector Classifier: default",
    svm,
    X_train, X_test, Y_train, Y_test
  )

  from sklearn.linear_model import LinearRegression
  lr = LinearRegression()
  run_test(
    "LinearRegression: default",
    lr,
    X_train, X_test, Y_train, Y_test,
    lambda y: (y > 0.5).astype(int)
  )
  
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier()
  run_test(
    "KNeighborsClassifier: default",
    knn,
    X_train, X_test, Y_train, Y_test
  )

  from sklearn.naive_bayes import BernoulliNB
  nb = BernoulliNB()
  run_test(
    "Bernoulli Naive Bayes: default",
    nb,
    X_train, X_test, Y_train, Y_test
  )

def main_FS(X_train, X_test, Y_train, Y_test):
  # X_train, X_test, Y_train, Y_test = get_data()
  
  from sklearn.tree import DecisionTreeClassifier
  dt = DecisionTreeClassifier()
  run_test(
    "DecisionTreeClassifier: No feature selection",
    dt,
    X_train, X_test, Y_train, Y_test, 
    feature_selection=None
  )
  
  from sklearn.tree import DecisionTreeClassifier
  dt = DecisionTreeClassifier()
  run_test(
    "DecisionTreeClassifier: Feature selection: SelectKBest",
    dt,
    X_train, X_test, Y_train, Y_test, 
    feature_selection=FS.FS_KBEST
  )
  
  from sklearn.tree import DecisionTreeClassifier
  dt = DecisionTreeClassifier()
  run_test(
    "DecisionTreeClassifier: Feature selection: RFE",
    dt,
    X_train, X_test, Y_train, Y_test, 
    feature_selection=FS.FS_RFE
  )
  
  from sklearn.tree import DecisionTreeClassifier
  dt = DecisionTreeClassifier()
  run_test(
    "DecisionTreeClassifier: Feature selection: PCA",
    dt,
    X_train, X_test, Y_train, Y_test, 
    feature_selection=FS.FS_PCA
  )
  
def parameter_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  for classifier, params in classifiers_params:
    map_predictions = None
    lr = isinstance(classifier, LinearRegression)
    if lr:
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    result = run_test(classifier, 
              X_train, X_test, Y_train, Y_test, map_predictions)
    results.append(result)
    for p_name, p_vals in params.items():
      for val in p_vals:
        default_classifier = get_default_classifier(classifier)
        default_classifier.set_params(**{p_name: val})
        result = run_test(default_classifier, 
                  X_train, X_test, Y_train, Y_test, map_predictions)
        results.append(result)
  return results

def gridsearch_custom_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  for classifier, params in classifiers_params:
    map_predictions = None
    lr = isinstance(classifier, LinearRegression)
    if lr:
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    params_set_len = 1
    for param in params.values():
      params_set_len *= len(param)

    if lr:
      print("GridSearch causes problems with LinearRegression, skipping...")
      continue
    
    print(f"Searching for best classifier through {params_set_len} parameter sets...")
    start = time.time()
    result = run_test(classifier, 
              X_train, X_test, Y_train, Y_test, 
              map_preds=map_predictions,
              param_grid=params,
              grid_search="custom",
              label_suffix="+GridSearchCV")
    end = time.time()
    print("elapsed time [s]:", end-start)
    results.append(result)

  return results

def gridsearch_accuracy_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  for classifier, params in classifiers_params:
    map_predictions = None
    lr = isinstance(classifier, LinearRegression)
    if lr:
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    params_set_len = 1
    for param in params.values():
      params_set_len *= len(param)

    if lr:
      print("GridSearch causes problems with LinearRegression, skipping...")
      continue
    
    print(f"Searching for best classifier through {params_set_len} parameter sets...")
    start = time.time()
    result = run_test(classifier, 
              X_train, X_test, Y_train, Y_test, 
              map_preds=map_predictions,
              param_grid=params,
              grid_search="accuracy",
              label_suffix="+GridSearchCV")
    end = time.time()
    print("elapsed time [s]:", end-start)
    results.append(result)

  return results

def gridsearch_rocauc_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  for classifier, params in classifiers_params:
    map_predictions = None
    lr = isinstance(classifier, LinearRegression)
    if lr:
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    params_set_len = 1
    for param in params.values():
      params_set_len *= len(param)

    if lr:
      print("GridSearch causes problems with LinearRegression, skipping...")
      continue
    
    print(f"Searching for best classifier through {params_set_len} parameter sets...")
    start = time.time()
    result = run_test(classifier, 
              X_train, X_test, Y_train, Y_test, 
              map_preds=map_predictions,
              param_grid=params,
              grid_search="roc_auc",
              label_suffix="+GridSearchCV")
    end = time.time()
    print("elapsed time [s]:", end-start)
    results.append(result)

  return results

def default_feature_selection_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  # TODO: fill with best parameters
  classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    SVC(),
    LinearRegression(),
    KNeighborsClassifier(),
    BernoulliNB()
  ]

  for classifier in classifiers:
    map_predictions = None
    if isinstance(classifier, LinearRegression):
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    for fs in FS_TYPES:
      if fs == FS.FS_RFE and (isinstance(classifier, MLPClassifier) or isinstance(classifier, SVC) or isinstance(classifier, KNeighborsClassifier) or isinstance(classifier, BernoulliNB)):
        print("MLPClassifier, SVC and KNeighborsClassifier are incompatible with RFE, skipping...")
        continue
      for k in (2, 5, 10, 20, 50, 100):
        # result = run_test(classifier, 
        #         X_train, X_test, Y_train, Y_test, map_predictions, feature_selection=fs, label_suffix=f"+FS.{fs}")
        # results.append(result)
        result = run_test(get_default_classifier(classifier), 
                X_train, X_test, Y_train, Y_test, map_predictions, feature_selection=fs, fs_k=k, label_suffix=f"+FS.{fs}")
        results.append(result)
    
  return results

def ensamble_experiments(X_train, X_test, Y_train, Y_test):
  results = []
  ensamble_classifiers = [
      (AdaBoostClassifier(), {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 1.0]
      }),
      (XGBClassifier(), {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
        "max_depth": [3, 5, 7],
        })
    ]

  for classifier, params in ensamble_classifiers:
    params_set_len = 1
    for param in params.values():
      params_set_len *= len(param)
    
    for search_scorer in ("custom", "accuracy", "roc_auc"):
      print(f"Searching for best {str(classifier)} through {params_set_len} parameter space...")
      start = time.time()
      result = run_test(classifier, 
                X_train, X_test, Y_train, Y_test,
                param_grid=params,
                grid_search=search_scorer,
                label_suffix=f"+GridSearchCV_{search_scorer}")
      end = time.time()
      print("elapsed time [s]:", end-start)
      results.append(result)

  return results

def fs_gs_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  for classifier, params in classifiers_params:
    lr = isinstance(classifier, LinearRegression)
    if lr:
      print("GridSearch causes problems with LinearRegression, skipping...")
      continue

    params_set_len = 1
    for param in params.values():
      params_set_len *= len(param)
    
    for search_scorer in ("custom", "accuracy", "roc_auc"):
      for fs in FS_TYPES:
        for k in [10, 20, 50]:
          print(f"Searching for best {str(classifier)} through {params_set_len} parameter space...")
          print(f"with {k} features...")
          start = time.time()
          result = run_test(classifier, 
                    X_train, X_test, Y_train, Y_test,
                    param_grid=params,
                    grid_search=search_scorer,
                    feature_selection=fs,
                    fs_k=k,
                    label_suffix=f"+GridSearchCV_{search_scorer}+FS.{fs}({k})")
          end = time.time()
          print("elapsed time [s]:", end-start)
          results.append(result)

  return results

def manual_parameter_experiments(X_train, X_test, Y_train, Y_test):
  results = []

  manual_parameters = [
    DecisionTreeClassifier(max_depth=15, min_samples_leaf=2, criterion="entropy"),
    DecisionTreeClassifier(max_depth=20, min_samples_leaf=2, criterion="entropy"),
    DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, criterion="entropy"),
    SVC(C=0.1, class_weight={0: 5, 1: 1}, degree=2, kernel='linear'),
    SVC(C=0.1, class_weight={0: 5, 1: 1}, degree=2, kernel='sigmoid'),
    SVC(C=0.1, class_weight={0: 10, 1: 1}, degree=2, kernel='linear'),
    SVC(C=0.1, class_weight={0: 10, 1: 1}, degree=2, kernel='sigmoid'),
    SVC(C=1, class_weight={0: 5, 1: 1}, degree=2, kernel='linear'),
    SVC(C=1, class_weight={0: 5, 1: 1}, degree=2, kernel='sigmoid'),
    SVC(C=10, class_weight={0: 5, 1: 1}, degree=2, kernel='linear'),
    SVC(C=10, class_weight={0: 5, 1: 1}, degree=2, kernel='sigmoid'),
    LinearRegression(positive=True, n_jobs=-1),
    RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_leaf=2, criterion="entropy"),
    RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_leaf=2, criterion="entropy"),
    RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=2, criterion="entropy"),
    RandomForestClassifier(n_estimators=50, max_depth=25, min_samples_leaf=2, criterion="entropy"),
  ]

  for classifier in manual_parameters:
    map_predictions = None
    lr = isinstance(classifier, LinearRegression)
    if lr:
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    for fs in FS_TYPES:
      if fs == FS.FS_RFE and (isinstance(classifier, MLPClassifier) or isinstance(classifier, SVC) or isinstance(classifier, KNeighborsClassifier) or isinstance(classifier, BernoulliNB)):
        print("MLPClassifier, SVC and KNeighborsClassifier are incompatible with RFE, skipping...")
        continue
      for k in (2, 5, 10, 20, 50, 100):
        result = run_test(classifier, 
                  X_train, X_test, Y_train, Y_test, map_predictions, 
                  feature_selection=fs, fs_k=k, label_suffix=f"+FS.{fs}")
        results.append(result)
  return results

if __name__ == "__main__":
  X_train, X_test, Y_train, Y_test = get_data()

  results = parameter_experiments(X_train, X_test, Y_train, Y_test)
  write_to_csv(results, "parameters")
  
  results = gridsearch_custom_experiments(X_train, X_test, Y_train, Y_test)
  write_to_csv(results, "gridsearch_custom")
  
  # results = gridsearch_accuracy_experiments(X_train, X_test, Y_train, Y_test)
  # write_to_csv(results, "gridsearch_accuracy")
  
  # results = gridsearch_rocauc_experiments(X_train, X_test, Y_train, Y_test)
  # write_to_csv(results, "gridsearch_rocauc")

  # results = ensamble_experiments(X_train, X_test, Y_train, Y_test)
  # write_to_csv(results, "ensamble")

  # !too many tests to run in a reasonable time
  # results = fs_gs_experiments(X_train, X_test, Y_train, Y_test)
  # write_to_csv(results, "feature_selection-gridsearch")

  # results = default_feature_selection_experiments(X_train, X_test, Y_train, Y_test)
  # write_to_csv(results, "default_feature_selection")
  
  # results = manual_parameter_experiments(X_train, X_test, Y_train, Y_test)
  # write_to_csv(results, "manual")
  

