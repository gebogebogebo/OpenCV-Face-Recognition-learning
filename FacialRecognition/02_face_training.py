import cv2
import numpy as np
from PIL import Image
import os

# フォルダ作成
if( os.path.exists("trainer") == False):
    os.mkdir("trainer")

# Path for face image database
path = 'dataset'

# LBPH顔認識器の構築
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Haar-like特徴分類器クラス
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    # pathにあるファイルをリストで取得
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    # 全ての画像について
    for imagePath in imagePaths:

        # グレースケールで画像を読み込む
        PIL_img = Image.open(imagePath).convert('L')
        # uint8型NumPyの配列に変換する
        img_numpy = np.array(PIL_img,'uint8')

        # ex dataset\User.2.1.jpgの2をとる
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Haar-like特徴分類器で顔を検知
        # 引数で指定された画像の中から顔を検出する
        faces = detector.detectMultiScale(img_numpy)

        # 検出した顔の処理
        # x,y=位置,w,h=領域
        for (x,y,w,h) in faces:
            # faceSamplesリストに追加
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # idsリストに追加
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

# トレーニング画像を取得
# faces=画像データ,ids=画像データのユーザーID
# - faces[0]=画像データ,idx[0]=faces[0]のユーザーID
# - facesとidsの配列数は同じ
faces,ids = getImagesAndLabels(path)

# トレーニング実施
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

# ここからトレーニングの結果をテストする
print("\n ここからトレーニングの結果をテストする")

# テスト画像を取得
testfaces,testids = getImagesAndLabels("testdata")
for testface in testfaces:
    # テスト画像に対して予測実施
    label, confidence = recognizer.predict(testface)
    confidencestr = "  {0}%".format(round(100 - confidence))

    # 予測結果をコンソール出力
    print("Test Image: {}, Predicted Label: {}, Confidence: {}, Per: {}".format("test_files[i]", label, confidence,confidencestr))
    # テスト画像を表示
    #cv2.imshow("test image", test_images[i])
    #cv2.waitKey(300)

