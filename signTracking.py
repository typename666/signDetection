from random import randint
import cv2
import torch

class Tracker():
    curentObjects= [] #[id, class, box, color, framesNotSeen]
    lastId= -1

    def __init__(self, minConf= 0.8, threshold= 0.1, maxFramesNotSeem= 7):
        self.MIN_CONF= minConf
        self.THRESHOLD= threshold
        self.MAX_FRAMES_NOT_SEEM= maxFramesNotSeem

    def addDetections(self, detections):
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

        result= [] #[ind, track_id, box]

        for i, obj in enumerate(self.curentObjects):
            if obj[4]>self.MAX_FRAMES_NOT_SEEM:
                self.curentObjects.pop(i)
            else:
                obj[4]+=1

        if detections.size(0)>0:
            for j, detection in enumerate(detections):
                obj_class, box= int(detection[5].item()), (int(detection[0].item()), int(detection[1].item()), int(detection[2].item()), int(detection[3].item()))

                bestInd= -1
                bestArea= 0

                for i, obj in enumerate(self.curentObjects):
                    if obj_class == obj[1]:
                        curentArea= findIntersection(obj[2], box)

                        if curentArea>bestArea:
                            bestArea= curentArea
                            bestInd= i

                if bestInd== -1 or bestArea< self.THRESHOLD:
                    if detection[4]>self.MIN_CONF:
                        self.lastId+= 1
                        self.curentObjects.append([self.lastId, obj_class, box, (randint(0, 255), randint(0, 255), randint(0, 255)), 0])
                        
                        result.append([obj_class, self.lastId, box])
                else:
                    self.curentObjects[bestInd][2]= box
                    self.curentObjects[bestInd][4]= 0
                    
                    result.append([obj_class, self.curentObjects[bestInd][0], box])

        return result