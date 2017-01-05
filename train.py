# -*- coding:utf-8 -*-

import numpy as np
import csv
import sys
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib

if __name__ == '__main__':
    """
    分類器作成
    実行コマンド
    python train.py xxx.csv
    xxx.csv     特徴を保存したファイル
    """

    param = sys.argv
    data = np.genfromtxt(param[1], delimiter=",")
    col_num = len(data[0][:])

    train = []
    label = []

    for i in range(len(data)):
        train.append(data[i][0:col_num-1])
        label.append(data[i][col_num-1])

    # 訓練データ
    trainData = np.array(train)
    # 教師データ
    labelData = np.array(label)
    # 学習用データと評価用データへ分割
    data_train, data_test, label_train, label_test = train_test_split(trainData, labelData, test_size=0.2)
    """
    classifier = LinearSVC()
    # 学習
    classifier.fit(data_train, label_train)
    # 学習結果保存
    #joblib.dump(classifier, "svc.model")

    result1 = classifier.predict(data_test)
    # 混合行列で評価
    cmat = confusion_matrix(label_test, result1)
    print("confusionMatrix : \n", cmat)
    # 全体の認識率
    accuracy1 = accuracy_score(label_test, result1)
    print("Accuracy1 : ", accuracy1)
    """
    # チューニングパラメータ
    tuned_parameters = [{"kernel":["rbf"],
                         "gamma" :np.logspace(-15,  3, 19, base=2),
                         "C"     :np.logspace( -5, 15, 21, base=2)},
                        {"kernel":["linear"],
                         "C"     :np.logspace( -5, 15, 21, base=2)}]

    # グリッドサーチ(cv:fold数, n_jobs:並列計算のためのコア数 -1では自動)
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=2, n_jobs = -1)
    # 学習
    clf.fit(data_train, label_train)
    # パラメータ, 平均認識率, 各foldの認識率
    for params, mean_score, all_scores in clf.grid_scores_:
        print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std()/2, params))
    print("Best Parameter:")
    print(clf.best_estimator_)
    # 学習結果保存
    joblib.dump(clf.best_estimator_, "trainModel/svc.model")
    result2 = clf.best_estimator_.predict(data_test)
    accuracy2 = accuracy_score(label_test, result2)
    print("Accuracy : ", accuracy2)
