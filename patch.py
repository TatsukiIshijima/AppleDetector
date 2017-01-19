# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2

class patch:
    """パッチ作成クラス"""

    def makePatch(self, image, patch_size):
        """パッチ作成
        @ param1[in] image              入力画像
        @ param2[in] patch_size         パッチサイズ(例：20x20なら10を指定)
        @ param[out] patch_list(list)   パッチリスト

        """
        patch_list = []

        width = image.shape[1]
        height = image.shape[0]

        for row in range(0, height, patch_size * 2):
            for col in range(0, width, patch_size * 2):
                if row < patch_size or col < patch_size or row > height - patch_size or col > width - patch_size:
                    continue
                patch = image[row - patch_size : row + patch_size, col - patch_size : col + patch_size]
                patch_list.append(patch)

        return patch_list

if __name__ == '__main__':
    """
    サンプルコード
    実行コマンド
    python patch.py xxx.jpg
    """
    param = sys.argv
    patch = patch()

    image = cv2.imread(param[1])
    patch_list = patch.makePatch(image, 10)

    print(patch_list)
    """
    for i in range(len(patch_list)):
        cv2.imwrite("patch" + str(i) + ".jpg", patch_list[i])
    """
