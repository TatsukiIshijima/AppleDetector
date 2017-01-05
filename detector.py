# -*- coding:utf-8 -*-

import numpy as np
import csv
import sys
import cv2
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib
from skimage.feature import greycomatrix, greycoprops

import feature
#import patch

if __name__ == '__main__':
    """
    検出
    実行コマンド
    python detector.py xxx.model xxx.jpg size
    xxx.model   訓練済みのモデルファイル
    xxx.jpg     画像ファイル
    size        パッチサイズ
    """

    feature = feature.feature()
    #patch = patch.patch()
    param = sys.argv

    # 分類器
    classifier = joblib.load("trainModel/" + param[1])

    # 対象画像
    image = cv2.imread(param[2])
    patch_size = int(param[3])

    width = image.shape[1]
    height = image.shape[0]

    # 画像をパッチサイズで走査
    for row in range(0, height, patch_size * 2):
        for col in range(0, width, patch_size * 2):
            if row <  patch_size or col < patch_size or row > height - patch_size or col > width - patch_size:
                continue

            patch = image[row - patch_size : row + patch_size, col - patch_size : col + patch_size]

            # 特徴抽出
            texture = feature.extractTexture(patch, 1, 0, 90, True)
            texture = np.array(texture)

            # 分類
            result = classifier.predict(texture)

            if result[0] == 1.0:
                cv2.circle(image, (col, row), 2, (0, 255, 0), -1)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
