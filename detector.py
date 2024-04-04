#importing stuff
import numpy as np
import cv2

#defining paths
image_path = "<image_file_name>"
# path to models
prototxt_path = "models/MobileNetSSD_deploy.prototxt" 
model_path = "models/MobileNetSSD_deploy.caffemodel" 

#setting up parameters - minimum confidence level,

min_confidence = 0.1
classes = ["background", "aeroplanes", "bicycle", "bird", "boat", "bottle", "bus", "cat", "chair", "cow", "diningtable" ,"dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#loading the data

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image = cv2.imread(image_path)

height, weight = image.shape[0], image.shape[1]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

net.setInput(blob)

detected_objects = net.forward() #returns a list of objects which are detected

for i in range(detected_objects.shape[2]):
  confidence = detected_objects[0][0][i][2]
  if confidence > min_confidence:
    class_index = int(detected_objects[0, 0, i ,1])
    if classes[class_index] == "bird" :
      print("bird detected")



