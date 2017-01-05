# -*- coding:utf-8 -*-

import numpy as np
import cv2
import sys
import csv
import codecs
from bs4 import BeautifulSoup

import feature
import patch

class CustomFormat(csv.excel):
    quoting = csv.QUOTE_ALL

if __name__ == '__main__':
    """
    訓練データ作成
    実行コマンド
    python extract.py image_dir xxx.xml xxx.csv label
    image_dir   画像フォルダ
    xxx.xml     imglabで作成したXMLファイル
    xxx.csv     特徴保存ファイル(任意ファイル名)
    label       教師データ
    """

    feature = feature.feature()
    patch = patch.patch()
    param = sys.argv
    soup = BeautifulSoup(open(param[1] + "/" + param[2]), "lxml")
    f = codecs.open(param[3], "w", "shift_jis")
    dataWriter = csv.writer(f)

    # XMLファイル読み込み
    for img_tag in soup.findAll("image"):
        image_file = img_tag.get("file")
        image = cv2.imread(str(param[1]) + "/" + image_file)
        print(image_file)

        for box_tag in img_tag.findAll("box"):
            top = int(box_tag.get("top"))
            left = int(box_tag.get("left"))
            width = int(box_tag.get("width"))
            height = int(box_tag.get("height"))
            img_roi = image[top:top+height, left:left+width]
            #print(top, left, width, height)

            # パッチ作成
            patch_list = patch.makePatch(img_roi, 10)

            # 特徴抽出
            for i in range(len(patch_list)):
                texture = feature.extractTexture(patch_list[i], 1, 0, 90, True)
                texture.append(param[4])
                dataWriter.writerow(texture)
                print(texture)

    f.close()
