import cv2
import numpy as np
import os 
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

names = ['None', 'gebo', 'micoco', 'user3', 'user4', 'user5'] 

# Define min window size to be recognized as a face
minW = 0.1*640
minH = 0.1*480

# pathにあるファイルをリストで取得
test_path="testdata"
image_paths = [os.path.join(test_path,f) for f in os.listdir(test_path)]     

# 全ての画像について
for image_path in image_paths:
    image_pil = Image.open(image_path).convert('L')
    gray = np.array(image_pil, 'uint8')

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        # 顔を 200x200 サイズにリサイズ
        face = cv2.resize(gray[y: y + h, x: x + w], (200, 200), interpolation=cv2.INTER_LINEAR)

        #id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        id, confidence = recognizer.predict(face)

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # 予測結果をコンソール出力
        print("Test Image: {}, Predicted id: {}, Confidence: {}".format(image_path, id, confidence))

        # テスト画像を表示
        cv2.imshow("test face", face)
        cv2.waitKey(1000)

cv2.destroyAllWindows()
