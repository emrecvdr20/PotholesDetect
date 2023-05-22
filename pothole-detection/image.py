import cv2
import os

img = cv2.imread("3.jpg") 

with open(os.path.join("project_files",'obj.names'), 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0,0,150), thickness=2)
 
cv2.imshow("pothole",img)
cv2.imwrite("result2"+".jpg",img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
