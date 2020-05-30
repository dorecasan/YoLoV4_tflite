# %%
import tensorflow as tf

# %%
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git

# %%
%cd /content/tensorflow-yolov4-tflite/data
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# %%
%cd /content/tensorflow-yolov4-tflite
!python convert_tflite.py --weights ./data/yolov4.weights --output ./data/detect.tflite --input_size=608

# %%
def upload():
  from google.colab import files
  uploaded = files.upload() 
  names = []
  for name, data in uploaded.items():
    names.append(name)
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
  return names


# %%
upload()

# %%
cd '/content'

# %%
!git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git

# %%
!mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite1
!cd tflite1

# %%
%cd tflite1
!mkdir yolov4

# %%
!mv '/content/tensorflow-yolov4-tflite/data/yolov4.tflite' '/content/tflite1/yolov4'

# %%
pwd

# %%
!wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# %%
!unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model

# %%
!mv '/content/tensorflow-yolov4-tflite/data/classes/coco.names' '/content/tflite1/yolov4/labelmap.txt'

# %%
cd /content/tflite1/

# %%
%matplotlib inline

!python3 TFLite_detection_video.py --modeldir=yolov4 --video=test.mp4 

# %%
pwd

# %%
!cp '/content/tflite1/TFLite_detection_webcam.py' '/content/drive/My Drive/YOLO Series'