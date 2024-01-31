import datetime
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, accuracy_score

CSV_HEADER = "Description; feature count; accuracy; true positive; false negative; false positive; true negative; false negative rate [%]; false positive rate [%]; AUC\n"

class FS:
  FS_KBEST = "KBEST"
  FS_RFE = "RFE"
  FS_PCA = "PCA"

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
    return f"{self.description}; {self.feature_count}; {self.accuracy}; {self.tp}; {self.fn}; {self.fp}; {self.tn}; {self.fnr*100:.2f}; {self.fpr*100:.2f}; {self.auc}\n"
  
def write_to_csv(results, file_name_prefix = "experiment"):
  current_time = datetime.datetime.now()
  file_name = f"results\\{file_name_prefix}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

  lines = [CSV_HEADER, *[r.csv() for r in results]]
  with open(file_name, mode='w', encoding='utf8') as file:
    file.writelines(lines)
  print("Results saved to file:", file_name)
    

def summarize(test, pred):
  tp, fn, fp, tn = confusion_matrix(test, pred).ravel()

  # print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
  # FPR = fp / (fp + tn)
  # FNR = fn / (fn + tp)
  
  auc = roc_auc_score(test, pred)
  # print(f"FNR: {FNR*100:.2f}%, FPR: {FPR*100:.2f}%")
  # print(f"AUC: {auc:.4f}")
  return tp, fn, fp, tn, auc

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


def run_test(label, classifier, X_train, X_test, Y_train, Y_test, map_preds=None, feature_selection:str=None, fs_k=10, param_grid=None, cv=5):
  X_train_fs, X_test_fs = do_feature_selection(classifier, X_train, X_test, Y_train, feature_selection, fs_k)

  if param_grid:
    custom_scorer = make_scorer(custom_scoring_fn, greater_is_better=False)
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring=custom_scorer)
    # grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy')
    # grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='roc_auc')
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
  tp, fn, fp, tn, auc = summarize(Y_test, Y_pred)
  result = ExperimentResults(label, X_train_fs.shape[1], acc, tp, fn, fp, tn, auc)
  result.show()
  return result

def custom_scoring_fn(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  FNR = fn / (fn + tp)
  FPR = fp / (fp + tn)
  
  penalty = 10
  return FNR + penalty * max(0, FPR - 0.005)

def get_data():
  dataframe = pd.read_csv("./data_ml/spam.dat")

  X = dataframe.drop(['target'],axis=1)

  mapping = {'yes': 1, 'no': 0}
  Y = dataframe.target.map(mapping)
  # print(Y)
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=331, test_size=0.33)
  print("train rows: {}, test rows: {}".format(X_train.shape[0], X_test.shape[0]))
  return X_train, X_test, Y_train, Y_test

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
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.neural_network import MLPClassifier
  from sklearn.svm import SVC
  from sklearn.linear_model import LinearRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import BernoulliNB
  
  results = []

  classifiers = [
    (DecisionTreeClassifier(), {"min_samples_split":[4, 6, 8, 10]}),
    (RandomForestClassifier(), {}),
    (MLPClassifier(), {}),
    (SVC(), {}),
    (LinearRegression(), {}),
    (KNeighborsClassifier(), {}),
    (BernoulliNB(), {})
  ]

  for classifier, params in classifiers:
    map_predictions = None
    if isinstance(classifier, LinearRegression):
      map_predictions = lambda y: (y > 0.5).astype(int)
    
    result = run_test(str(classifier), classifier, 
              X_train, X_test, Y_train, Y_test, map_predictions)
    results.append(result)
    for p_name, p_vals in params.items():
      for val in p_vals:
        classifier.set_params(**{p_name: val})
        result = run_test(str(classifier), classifier, 
                  X_train, X_test, Y_train, Y_test, map_predictions)
        results.append(result)
  return results

def feature_selection_experiments(X_train, X_test, Y_train, Y_test):
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.neural_network import MLPClassifier
  from sklearn.svm import SVC
  from sklearn.linear_model import LinearRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import BernoulliNB
  
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
    
    result = run_test(str(classifier), classifier, 
              X_train, X_test, Y_train, Y_test, map_predictions, feature_selection=FS.FS_KBEST)
    results.append(result)
    
  return results

if __name__ == "__main__":
  X_train, X_test, Y_train, Y_test = get_data()

  results = parameter_experiments(X_train, X_test, Y_train, Y_test)
  write_to_csv(results, "parameters")
  

