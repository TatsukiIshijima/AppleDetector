# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

class feature:
    """特徴クラス"""

    def extractTexture(self, image, d, dig1, dig2, f_normed):
        """ テクスチャ特徴抽出
        @ param1[in] image              入力画像
        @ param2[in] d                  距離
        @ param3[in] dig1               角度1
        @ param4[in] dig2               角度2
        @ param5[in] f_normed(bool型)   特徴の正規化
        @ param[out] feature(list)      特徴
        """

        feature = []

        image_hsv = image.copy()
        image_lab = image.copy()

        width = image.shape[1]
        height = image.shape[0]

        # 色空間変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)
        image_lab = cv2.cvtColor(image_lab, cv2.COLOR_BGR2LAB)

        # 変位
        glcm_x = greycomatrix(image, [d], [dig1], normed=True, symmetric=True)
        glcm_y = greycomatrix(image, [d], [dig2], normed=True, symmetric=True)

        # 特徴
        contrast_x = greycoprops(glcm_x, "contrast")            # コントラスト
        contrast_y = greycoprops(glcm_y, "contrast")
        dissimilarity_x = greycoprops(glcm_x, "dissimilarity")  # 相違性
        dissimilarity_y = greycoprops(glcm_y, "dissimilarity")
        homogeneity_x = greycoprops(glcm_x, "homogeneity")      # 均質性
        homogeneity_y = greycoprops(glcm_y, "homogeneity")
        ASM_x = greycoprops(glcm_x, "ASM")                      # 角2次モーメント
        ASM_y = greycoprops(glcm_y, "ASM")
        correlation_x = greycoprops(glcm_x, "correlation")      # 相関
        correlation_y = greycoprops(glcm_y, "correlation")

        contrast = (contrast_x[0][0] + contrast_y[0][0]) / 2
        dissimilarity = (dissimilarity_x[0][0] + dissimilarity_y[0][0]) / 2
        homogeneity = (homogeneity_x[0][0] + homogeneity_y[0][0]) / 2
        ASM = (ASM_x[0][0] + ASM_y[0][0]) / 2
        correlation = (correlation_x[0][0] + correlation_y[0][0]) / 2

        H_list = []
        a_list = []

        for row in range(height):
            for col in range(width):
                pixelHSVValue = image_hsv[row][col]
                pixelLabValue = image_lab[row][col]

                H_list.append(pixelHSVValue[0])
                a_list.append(pixelLabValue[1])

        H_array = np.array(H_list)
        a_array = np.array(a_list)
        H_ave = np.average(H_array)     # Hの平均
        a_ave = np.average(a_array)     # a*の平均
        H_std = np.std(H_array)         # Hの標準偏差
        a_std = np.std(a_array)         # a*の標準偏差

        feature.append(contrast)
        feature.append(dissimilarity)
        feature.append(homogeneity)
        feature.append(ASM)
        feature.append(correlation)
        feature.append(H_ave)
        feature.append(a_ave)
        feature.append(H_std)
        feature.append(a_std)

        if f_normed:
            feature_array = np.array(feature)
            feature_ave = np.average(feature_array)     # 特徴の平均
            feature_std = np.std(feature_array)         # 特徴の標準偏差

            # 正規化
            normed_contrast = (contrast - feature_ave) / feature_std
            normed_diss = (dissimilarity - feature_ave) / feature_std
            normed_homo = (homogeneity - feature_ave) / feature_std
            normed_ASM = (ASM - feature_ave) / feature_std
            normed_corre = (correlation - feature_ave) / feature_std
            normed_H_ave = (H_ave - feature_ave) / feature_std
            normed_a_ave = (a_ave - feature_ave) / feature_std
            normed_H_std = (H_std - feature_ave) / feature_std
            normed_a_std = (a_std - feature_ave) / feature_std

            del feature[:]
            feature.append(normed_contrast)
            feature.append(normed_diss)
            feature.append(normed_homo)
            feature.append(normed_ASM)
            feature.append(normed_corre)
            feature.append(normed_H_ave)
            feature.append(normed_a_ave)
            feature.append(normed_H_std)
            feature.append(normed_a_std)

        return feature

if __name__ == '__main__':
    """
    サンプルコード
    実行コマンド
    python feature.py xxx.jpg
    """
    param = sys.argv
    feature = feature()

    image = cv2.imread(param[1])
    texture = feature.extractTexture(image, 1, 0, 90, True)
    print(texture)
