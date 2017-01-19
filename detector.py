# -*- coding:utf-8 -*-

import numpy as np
import csv
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib
from skimage.feature import greycomatrix, greycoprops

import feature

matplotlib.rcParams['font.size'] = 8

if __name__ == '__main__':
    """
    検出
    実行コマンド
    python detector.py xxx.model xxx.jpg size
    xxx.model   訓練済みのモデルファイル
    xxx.jpg     画像ファイル
    size        パッチサイズ(例：20x20なら10を指定)
    """

    feature = feature.feature()
    #patch = patch.patch()
    param = sys.argv

    # 分類器
    classifier = joblib.load("trainModel/" + param[1])

    # 対象画像
    image = cv2.imread(param[2])
    patch_size = int(param[3])

    image = cv2.resize(image, (972, 726))
    height, width = image.shape[:2]

    # マスク
    mask = np.zeros((height, width, 1), dtype=np.uint8)

    # 画像をパッチサイズで走査
    for row in range(0, height, patch_size * 2):
        for col in range(0, width, patch_size * 2):
            if row <  patch_size or col < patch_size or row > height - patch_size or col > width - patch_size:
                continue

            patch = image[row-patch_size : row+patch_size, col-patch_size : col+patch_size]

            # 特徴抽出
            texture = feature.extractTexture(patch, 1, 0, 90, True)
            texture = np.array(texture)

            # 分類
            result = classifier.predict(texture)

            if result[0] == 1.0:
                #cv2.circle(image, (col, row), 2, (0, 255, 0), -1)

                # マスク作成
                patch_mask = mask[row-patch_size : row+patch_size, col-patch_size : col+patch_size]
                patch_mask[:] = 255
                mask[row-patch_size : row+patch_size, col-patch_size : col+patch_size] = patch_mask

    # 描画用
    mask_img = mask.copy()
    mask_img = cv2.bitwise_and(image, image, mask=mask)

    # ラベリング
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(mask)
    for i in range(1, labelnum):
        x, y, w, h, area = contours[i]
        # ラベル面積による閾値判定(パッチの大きさx個数)
        if area <= pow(patch_size*2, 2) * 3:
            continue
        cv2.rectangle(image, (x,y), (x+w,y+h), (255, 255 ,0), 3)

    fig = plt.figure(figsize=(14, 7))
    ax_Result = fig.add_subplot(1, 2, 1)
    ax_Mask = fig.add_subplot(1, 2, 2)
    ax_Result.set_title("Result")
    ax_Mask.set_title("Mask")
    ax_Result.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_Mask.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))

    #cv2.imwrite("Result.jpg", image)
    #cv2.imwrite("Mask.jpg", mask)

    plt.show()
