import argparse
import json
import cv2
import torch
import numpy as np

from os import mkdir, listdir
from os.path import isdir, split, join

from math import atan2, sin, cos, pi, asin
from scipy.optimize import minimize

from signTracking import Tracker
from pointsDetection import PointsDetector

import sys
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

FRAMES_PATH= 'frames' #Файл, в который будут сохраняться отдельные кадры

class SignsLocalization():
    #Параметры камеры полученные после калибровки
    def __init__(self, modelPath= '', dataPath= '', signSizesPath= '', saveFrames= False):
        self.cameraCoefs= [-1.12247280e-03,  2.05702479e-04,  5.77681379e-01,  5.70756088e-06, -8.57117308e-04,  4.42979786e-01]
        self.adjustments= [1.0, 0.0]

        #Инициализируем YOLOv5
        self.DEVICE= 'cpu' #select_device('')
        self.model= DetectMultiBackend(modelPath, device= self.DEVICE, dnn=False, data= dataPath, fp16= False)
        self.SIGN_NAMES= self.model.names
        self.model.eval()

        self.SAVE_FRAMES= saveFrames

        self.tracker= Tracker()
        self.pointsDetector= PointsDetector(len(self.SIGN_NAMES))

        self.signs= [] #[class, [(box1, coord1), (box2, coord2), ...], lastFrame]

        with open(signSizesPath, "r") as read_file:
            self.signSizes= json.load(read_file)
        for i in self.signSizes:
            for j in i:
                j.append(0)

        self.signsPositions= [] #[class, lat, lon]

    #Находим знаки на двумерных изображениях
    def findBoxes(self, frame, model):
        #Подгатавливаем кадр для YOLO
        origH, origW= frame.shape[:2]

        frame= cv2.resize(frame, INPUT_SIZE)
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame).to(self.DEVICE)
        frame = frame.permute((2, 0, 1))

        frame = frame.half() if model.fp16 else frame.float()
        frame /= 255
        if len(frame.shape) == 3:
            frame = frame[None]

        pred = model(frame)
        pred = non_max_suppression(pred, MIN_CONF/2, 0.25, None, False, max_det= 1000)[0]

        if pred.size(0)>0:
            for i, box in enumerate(pred):
                box[0], box[1], box[2], box[3]= box[0].item()/INPUT_SIZE[1]*origW, box[1].item()/INPUT_SIZE[0]*origH, box[2].item()/INPUT_SIZE[1]*origW, box[3].item()/INPUT_SIZE[0]*origH

                if box[0]<1 or box[1]<1 or box[2]>origW-2 or box[3]>origH-2:
                    pred= torch.cat((pred[:i], pred[i+1:]), axis= 0)

        return pred

    #Обрабатывает одно изображение
    def processNextImage(self, frame, position):

        pred= self.findBoxes(frame, self.model)

        origH, origW= frame.shape[:2]
        if pred.size(0)>0:
            for box in pred:
                if box[4].item()> MIN_CONF:
                    x0, y0, x1, y1= round(box[0].item()), round(box[1].item()), round(box[2].item()), round(box[3].item())
                    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2) #Выделяем знак прямоугольником
        
        cv2.imshow('frame', frame)

        # trackRes= self.tracker.addDetections(pred)

        # for track in trackRes:
        #     if track[1]>=len(self.signs):
        #         self.signs.append([track[0], [(track[2], position)], frame])
        #     else:
        #         self.signs[track[1]][1].append((track[2], position))
        #         self.signs[track[1]][2]= frame

    def findSignsPositions(self):
        def linesIntersection(variables, lines):
            x1, y1, z1= variables

            result= 0.0
            for coefs in lines:
                a= cos(coefs[1])*cos(coefs[0])
                b= cos(coefs[1])*sin(coefs[0])
                c= sin(coefs[1]) #-1м???

                x0, y0= coefs[2:4]
                result+= ((b*(x1-x0)-a*(y1-y0))**2+(c*(y1-y0)-b*z1)**2+(c*(x1-x0)-a*z1)**2)/(a**2+b**2+c**2)

            return result

        def findRot(points_2d, points_3d):
            #Параметры камеры полученные после калибровки
            fx= 1.04772907e+03
            fy= 1.04456275e+03
            cx= 5.71316559e+02
            cy= 3.87625164e+02

            CAMERA_MATRIX= np.matrix([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

            revtval, rvecs, tvecs= cv2.solvePnP(np.array(points_3d, dtype="double"), points_2d, CAMERA_MATRIX, np.zeros((4,1)), flags=0)

            R = cv2.Rodrigues(rvecs)[0]
            roll = 180*atan2(-R[2][1], R[2][2])/pi
            pitch = 180*asin(R[2][0])/pi
            yaw = 180*atan2(-R[1][0], R[0][0])/pi
            rot_params= [roll,pitch,yaw]
            return rot_params

        for sign in self.signs:
            if len(sign[1])>2:
                if self.pointsDetector.isWeightsExist[sign[0]]:
                    w, h= sign[1][-1][0][2]- sign[1][-1][0][0], sign[1][-1][0][3]- sign[1][-1][0][1]

                    x0, y0, x1, y1= round(sign[1][-1][0][0]-w*0.05), round(sign[1][-1][0][1]-h*0.05), round(sign[1][-1][0][2]+w*0.05), round(sign[1][-1][0][3]+h*0.05)
                    
                    if x0<0:
                        x0= 0
                    if y0<0:
                        y0= 0
                    if x1>sign[2].shape[1]:
                        x1= sign[2].shape[1]
                    if y1>sign[2].shape[0]:
                        y1= sign[2].shape[0]

                    cropedImage= sign[2][y0:y1, x0:x1]

                    origH, origW= cropedImage.shape[:2]
                    cropedImage= cv2.resize(cropedImage, (self.pointsDetector.INPUT_SIZE, self.pointsDetector.INPUT_SIZE))
                    points= self.pointsDetector.evaluteModel((cropedImage,), sign[0])

                    points_2d= np.array(((x0+points[0].item()/self.pointsDetector.INPUT_SIZE*origW, y0+points[1].item()/self.pointsDetector.INPUT_SIZE*origH), (x0+points[2].item()/self.pointsDetector.INPUT_SIZE*origW, y0+points[3].item()/self.pointsDetector.INPUT_SIZE*origH), (x0+points[4].item()/self.pointsDetector.INPUT_SIZE*origW, y0+points[5].item()/self.pointsDetector.INPUT_SIZE*origH), (x0+points[6].item()/self.pointsDetector.INPUT_SIZE*origW, y0+points[7].item()/self.pointsDetector.INPUT_SIZE*origH)), dtype="double")

                    rot= findRot(points_2d, self.signSizes[sign[0]])[0]

                    lines= []
                    for oneMeasure in sign[1]:
                        xMid, yMid= (oneMeasure[0][0]+oneMeasure[0][2])/2.0, (oneMeasure[0][1]+oneMeasure[0][3])/2.0

                        alpha= xMid*self.cameraCoefs[0]+yMid*self.cameraCoefs[1]+self.cameraCoefs[2]+oneMeasure[1][2]
                        if alpha>pi:
                            alpha-= 2*pi
                        elif alpha<-pi:
                            alpha+= 2*pi

                        lines.append((alpha, xMid*self.cameraCoefs[3]+yMid*self.cameraCoefs[4]+self.cameraCoefs[5], (oneMeasure[1][1]-sign[1][0][1][1])*62.8E3, (oneMeasure[1][0]-sign[1][0][1][0])*111.1E3))

                    translation= minimize(linesIntersection, (0.0, 0.0, 0.0), (lines,)).x

                    deltXGlob, deltYGlob= cos(oneMeasure[1][2])*self.adjustments[0]-sin(oneMeasure[1][2])*self.adjustments[1], sin(oneMeasure[1][2])*self.adjustments[0]+cos(oneMeasure[1][2])*self.adjustments[1]
                    
                    curentSign= {'type': self.SIGN_NAMES[sign[0]],
                                'lat': sign[1][0][1][0]+(translation[1]+deltYGlob)/111.1E3,
                                'lon': sign[1][0][1][1]+(translation[0]+deltXGlob)/62.8E3, 
                                'height': translation[2]+1}

                    self.signsPositions.append(curentSign)

def getLog(logPath):
    logFile= open(logPath, 'r')
    logFileLines= logFile.readlines()

    result= []
    for s in logFileLines:
        splitedS= s.split('\t')

        hour, minute, second= splitedS[0].split(':')
        second, mSecond= second.split('.')

        result.append([((int(hour)*60+int(minute))*60+int(second))*1000+int(mSecond), float(splitedS[1].replace(',', '.')), float(splitedS[2].replace(',', '.'))])

    return result

def calcAngles(log):
    FACTOR= 111.1/62.8

    curentY, prevY, curentX, prevX= log[1][1], log[0][1], log[1][2], log[0][2]

    log[0].append(atan2((curentY-prevY)*FACTOR, curentX-prevX))

    prevX, prevY= curentX, curentY
    for i in range(2, len(log)):
        curentY, curentX= log[i][1], log[i][2]

        if ((curentY-prevY)*62.8E3)**2+((curentX-prevX)*111.1E3)**2<0.2:
            log[i-1].append(log[i-2][3])
        else:
            log[i-1].append(atan2((curentY-prevY)*FACTOR, (curentX-prevX)))

            prevX, prevY= curentX, curentY

if __name__ == "__main__":
    #Задаём парсер аргументов
    parser = argparse.ArgumentParser(description='A tutorial of argparse!')
    parser.add_argument("--YOLO_weights", default= ".\\weights\\signsDetector\\YOLOv5_1.pt", type= str, help="Веса YOLOv5")
    parser.add_argument("--points_weights", default= ".\\weights\\pointsDetector", type= str, help="Папка с весами детектора точек")
    parser.add_argument("--min_conf", default= 0.7, type= float, help= "Минимальная уверенность YOLOv5 в ответе при, которой он считается верным")
    parser.add_argument("--im_size", default= 704, type= int, help= "Размер изображения, на котором происходило обучение сети")
    parser.add_argument("--data", default= 'custom_data.yaml', type= str, help= "Путь к YAML файлу с названиями классов")
    parser.add_argument("--signSizes", default= 'signSizes.json', type= str, help= "Путь к YAML файлу с размерами знаков")
    parser.add_argument("--tracks_path", default= '.\\tracks', type= str, help= "Путь к видео со знаками")
    parser.add_argument("--log_path", default= '.\\040220_104824_track.log', type= str, help="Путь к log-файлу")
    parser.add_argument("--minutes", default= 0.5, type= float, help= "Количество минут, которые необходимо обработать")

    #Вылавливаем аргументы командной строки
    args = parser.parse_args()
    YOLO_WEIGHTS= args.YOLO_weights
    points_WEIGHTS= args.points_weights
    MIN_CONF= args.min_conf
    INPUT_SIZE= (args.im_size, args.im_size)
    DATA_PATH= args.data
    SIGNS_SIZES_PATH= args.signSizes
    TRACKS_PATH= args.tracks_path
    LOG_PATH= args.log_path
    MINUTES= args.minutes

    JSON_OUTPUT_PATH= split(LOG_PATH)[-1].replace('log', 'json') #Файл, в который будет сохраняться результат

    trackNames= [_ for _ in listdir(TRACKS_PATH) if _.endswith('.mp4')]

    if len(trackNames)<1:
        print("Videos not found")
        exit()

    trackCounter= 0
    cap = cv2.VideoCapture(join(TRACKS_PATH, trackNames[trackCounter])) #Создаём поток видео
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    log= getLog(LOG_PATH)
    calcAngles(log)

    start_mSeconds, lat, lon, angle= log[0]
    start_start_mSeconds= start_mSeconds
    curent_mSeconds= start_mSeconds

    localizator= SignsLocalization(modelPath= YOLO_WEIGHTS, dataPath= DATA_PATH, signSizesPath= SIGNS_SIZES_PATH)
    signs= []

    i= 0
    while True:
        if MINUTES>0 and curent_mSeconds-start_start_mSeconds> int(MINUTES*60*1000):
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, curent_mSeconds-start_mSeconds) #Устанавливаем видеопоток на нужную милисекунду

        ret, origFrame = cap.read() #Берём кадр

        if not ret:
            trackCounter+= 1
            if trackCounter>= len(trackNames):
                print("Stream end. Exiting.")
                break

            start_mSeconds= curent_mSeconds
            cap = cv2.VideoCapture(join(TRACKS_PATH, trackNames[trackCounter])) #Создаём поток видео
            if not cap.isOpened():
                print("Cannot open video")
                exit()

            continue

        localizator.processNextImage(origFrame, (lat, lon, angle))

        i+= 1

        curent_mSeconds, lat, lon, angle= log[i]

        if cv2.waitKey(1) == ord('q'): #Завершаем просмотр видеопотока по нажатию q
            break

    localizator.findSignsPositions()
    
    #Заносим словарь питон в JSON файл
    with open(JSON_OUTPUT_PATH, "w") as write_file:
        json.dump(localizator.signsPositions, write_file, sort_keys=False, indent=4, ensure_ascii=False, separators=(',', ': '))