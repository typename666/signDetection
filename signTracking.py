from random import randint
import cv2
import torch

import sys
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

MIN_CONF= 0.7
INPUT_SIZE= (1280, 1280)

class Tracker():
    curentObjects= [] #[id, class, box, color, framesNotSeen]
    lastId= -1

    colors= [] 

    def __init__(self, threshold= 0.1, maxFramesNotSeem= 7):
        self.THRESHOLD= threshold
        self.MAX_FRAMES_NOT_SEEM= maxFramesNotSeem

    def addDetections(self, detections): #[class, box]
        def findIntersection(rectangle1, rectangle2):
            left = max(rectangle1[0], rectangle2[0])
            top = min(rectangle1[3], rectangle2[3])
            right = min(rectangle1[2], rectangle2[2])
            bottom = max(rectangle1[1], rectangle2[1])

            width = right - left
            height = top - bottom

            if width < 0 or height < 0:
                return 0

            intersection= width * height

            area1= (rectangle1[2]-rectangle1[0])*(rectangle1[3]-rectangle1[1])
            area2= (rectangle2[2]-rectangle2[0])*(rectangle2[3]-rectangle2[1])

            if area1>area2:
                return intersection/area1
            return intersection/area2

        result= [[], []] #[ind, track_id, box]

        for i, obj in enumerate(self.curentObjects):
            if obj[4]>self.MAX_FRAMES_NOT_SEEM:
                self.curentObjects.pop(i)
            else:
                obj[4]+=1

        for j, detection in enumerate(detections):
            if detection[4]<MIN_CONF:
                continue

            obj_class, box= detection[5].item(), (int(detection[0].item()), int(detection[1].item()), int(detection[2].item()), int(detection[3].item()))

            bestInd= -1
            bestArea= 0

            for i, obj in enumerate(self.curentObjects):
                if obj_class == obj[1]:
                    curentArea= findIntersection(obj[2], box)

                    if curentArea>bestArea:
                        bestArea= curentArea
                        bestInd= i

            if bestInd== -1 or bestArea< self.THRESHOLD:
                self.lastId+= 1
                self.curentObjects.append([self.lastId, obj_class, box, (randint(0, 255), randint(0, 255), randint(0, 255)), 0])
                result[0].append([obj_class, self.lastId, box])
            else:
                self.curentObjects[bestInd][2]= box
                self.curentObjects[bestInd][4]= 0
                result[1].append([obj_class, self.curentObjects[bestInd][0], box])

        return result

def drawBoxes(tracker, model, nextFrame):
    #Подгатавливаем кадр для YOLO
    origH, origW= nextFrame.shape[:2]
    resizedFrame= cv2.resize(nextFrame, INPUT_SIZE)
    frame= cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)

    frame = torch.from_numpy(frame).to(device)
    frame = frame.permute((2, 0, 1))

    frame = frame.half() if model.fp16 else frame.float()
    frame /= 255
    if len(frame.shape) == 3:
        frame = frame[None]

    pred = model(frame, augment= False, visualize= False) #Прогоняем кадр через YOLO
    pred = non_max_suppression(pred, MIN_CONF, 0.25, None, False, max_det= 1000)[0]
    pred.numpy()

    res= tracker.addDetections(pred)
    
    for obj in tracker.curentObjects:
        resizedFrame = cv2.rectangle(resizedFrame, obj[2][0:2], obj[2][2:4], obj[3], 2) #Выделяем знак прямоугольником

    return cv2.resize(resizedFrame, (origW, origH))

MODEL_PATH= '.\\weights\\YOLOv5.pt'
DATA_PATH= 'custom_data.yaml'

if __name__ == "__main__":
    #Инициализируем YOLOv5
    device= select_device('')
    model= DetectMultiBackend(MODEL_PATH, device= device, dnn=False, data= DATA_PATH, fp16= False)
    names= model.names
    model.eval()

    tracker= Tracker()

    cap = cv2.VideoCapture('.\\tracks\\040220_104824__1_p000.mp4')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        frame = drawBoxes(tracker, model, frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()