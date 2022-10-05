import cv2
import json
import torch
import argparse
from os import mkdir, listdir
from os.path import isdir, split
from math import atan2, pi, sin, cos
from signTracking import Tracker
from scipy.optimize import fsolve

import sys
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

FRAMES_PATH= '.\\frames' #Файл, в который будут сохраняться отдельные кадры

class SignsLocalization():
    #Параметры камеры полученные после калибровки
    def __init__(self, fx= 1.04772907e+03, fy= 1.04456275e+03, cx= 5.71316559e+02, cy= 3.87625164e+02, modelPath= '', dataPath= '', saveFrames= False):
        self.FX= fx
        self.FY= fy
        self.CX= cx
        self.CY= cy

        #Инициализируем YOLOv5
        self.DEVICE= select_device('')
        self.model= DetectMultiBackend(modelPath, device= self.DEVICE, dnn=False, data= dataPath, fp16= False)
        self.SIGN_NAMES= self.model.names
        self.model.eval()

        self.SAVE_FRAMES= saveFrames

        self.tracker= Tracker()
        self.signs= [] #[class, box1, coord1, box2, coord2]
        self.signsPositions= [] #[class, lat, lon]

        #Находим знаки на двумерных изображениях
    def findBoxes(self, img, model):
        frame= cv2.resize(img, INPUT_SIZE)
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame).to(self.DEVICE)
        frame = frame.permute((2, 0, 1))

        frame = frame.half() if model.fp16 else frame.float()
        frame /= 255
        if len(frame.shape) == 3:
            frame = frame[None]

        pred = model(frame)
        pred = non_max_suppression(pred, MIN_CONF, 0.25, None, False, max_det= 1000)[0]

        return pred

    #Обрабатывает одно изображение
    def processNextImage(self, frame, position):
        #Подгатавливаем кадр для YOLO
        origH, origW= frame.shape[:2]
        pred= self.findBoxes(frame, self.model)

        trackRes= self.tracker.addDetections(pred)

        for  i, box in enumerate(trackRes[0]):
            trackRes[0][i][2]= (round(float(box[2][0]*origW/INPUT_SIZE[0])), round(float(box[2][1]*origH/INPUT_SIZE[1])), round(float(box[2][2]*origW/INPUT_SIZE[0])), round(float(box[2][3]*origH/INPUT_SIZE[1])))
        for  i, box in enumerate(trackRes[1]):
            trackRes[1][i][2]= (round(float(box[2][0]*origW/INPUT_SIZE[0])), round(float(box[2][1]*origH/INPUT_SIZE[1])), round(float(box[2][2]*origW/INPUT_SIZE[0])), round(float(box[2][3]*origH/INPUT_SIZE[1])))
        
        for newTrack in trackRes[0]:
            self.signs.append([newTrack[0], newTrack[2], position, newTrack[2], position])
        for oldTrack in trackRes[1]:
            dist= ((position[0]-self.signs[oldTrack[1]][4][0])*111.1E3)**2+((position[1]-self.signs[oldTrack[1]][4][1])*62.8E3)**2
            if dist> 10.0:
                self.signs[oldTrack[1]][1]= self.signs[oldTrack[1]][3]
                self.signs[oldTrack[1]][2]= self.signs[oldTrack[1]][4]
                self.signs[oldTrack[1]][3]= oldTrack[2]
                self.signs[oldTrack[1]][4]= position

    def findSignsPositions(self):
        def findPointPosition(variables, fx, fy, cx, cy, a, c, beta1, beta2, u1, v1, u2):
            Xl1, Xl2, Yl1, Zl1, Zl2, Xg, Yg, Zg= variables

            f1 = Xg*cos(beta1)+Zg*sin(beta1)-Xl1
            f2= (Xg-a)*cos(beta2)+(Zg-c)*sin(beta2)-Xl2
            f3= Yl1-Yg
            f4= Zg*cos(beta1)-Xg*sin(beta1)-Zl1
            f5= (Zg-c)*cos(beta2)-(Xg-a)*sin(beta2)-Zl2
            f6= Xl1*fx+Zl1*cx-u1*Zl1
            f7= Yl1*fy+Zl1*cy-v1*Zl1
            f8= Xl2*fx+Zl2*cx-u2*Zl2

            return (f1, f2, f3, f4, f5, f6, f7, f8)

        for sign in self.signs:
            if sign[1][0]==sign[3][0] and sign[1][1]==sign[3][1] and sign[1][2]==sign[3][2] and sign[1][3]==sign[3][3]:
                break

            lat1, lon1, lat2, lon2= sign[2][0], sign[2][1], sign[4][0], sign[4][1]

            a, c= (lat1-lat2)*111.1E6, (lon2-lon1)*62.9E6

            beta1, beta2= sign[2][2], sign[4][2]
            u1, v1, u2= sign[1][0], sign[1][1], sign[3][0]

            res= fsolve(findPointPosition, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (self.FX, self.FY, self.CX, self.CY, a, c, beta1, beta2, u1, v1, u2))

            self.signsPositions.append((self.SIGN_NAMES[sign[0]], lat1-res[5]/111.1E6, lon1+res[7]/62.8E6))

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

    prevDeltY, prevDeltX= log[1][1]- log[0][1], log[1][2]- log[0][2]

    log[0].append(atan2(prevDeltY*FACTOR, prevDeltX))

    for i in range(1, len(log)-1):
        curentDeltY, curentDeltX= log[i+1][1]- log[i][1], log[i+1][2]- log[i][2]

        if (curentDeltX*62.8E3)**2+(curentDeltY*111.1E3)**2<0.2:
            log[i].append(log[i-1][3])
        else:
            log[i].append(atan2((prevDeltY+curentDeltY)*FACTOR/2, (prevDeltX+curentDeltX)/2))

            curentDeltX, curentDeltY= prevDeltX, prevDeltY

    log[-1].append(atan2(curentDeltY, curentDeltX))


if __name__ == "__main__":
    #Задаём парсер аргументов
    parser = argparse.ArgumentParser(description='A tutorial of argparse!')
    parser.add_argument("--weights", default= ".\\weights\\YOLOv5.pt", type= str, help="Веса YOLOv5")
    parser.add_argument("--min_conf", default= 0.7, type= float, help="Минимальная уверенность YOLOv5 в ответе при, которой он считается верным")
    parser.add_argument("--im_size", default= 1280, type= int, help="Размер изображения, на котором происходило обучение сети")
    parser.add_argument("--source", default= 'input_video.mp4', type= str, help="Путь к видео, которое необходимо обработать")
    parser.add_argument("--data", default= 'custom_data.yaml', type= str, help="Путь к YAML файлу с названиями классов")
    parser.add_argument("--period", default= 500, type= int, help="Промежутки времени в миллисекундах, через которые необходимо обрабатывать кадры")
    parser.add_argument("--save_frames", default= True, type= bool, help="Если True сохраняет кадры в которых есть знаки в папку frames")
    parser.add_argument("--tracks_path", default= '.\\tracks', type= str, help="Путь к видео со знаками")
    parser.add_argument("--log_path", default= '.\\040220_104824_track.log', type= str, help="Путь к log-файлу")
    parser.add_argument("--minutes", default= 0, type= int, help="Количество минут, которые необходимо обработать")

    #Вылавливаем аргументы командной строки
    args = parser.parse_args()
    MODEL_PATH= args.weights
    MIN_CONF= args.min_conf
    INPUT_SIZE= (args.im_size, args.im_size)
    VIDEO_PATH= args.source
    DATA_PATH= args.data
    PERIOD= args.period
    SAVE_FRAMES= args.save_frames
    TRACKS_PATH= args.tracks_path
    LOG_PATH= args.log_path
    MINUTES= args.minutes

    JSON_OUTPUT_PATH= split(LOG_PATH)[-1].replace('log', 'json') #Файл, в который будет сохраняться результат

    if SAVE_FRAMES and not isdir(FRAMES_PATH):
        mkdir(FRAMES_PATH) #Создаём папку для вырезанных кадров

    trackNames= [_ for _ in listdir(TRACKS_PATH) if _.endswith('.mp4')]

    if len(trackNames)<1:
        print("Videos not found")
        exit()

    trackCounter= 0
    cap = cv2.VideoCapture(TRACKS_PATH+'\\'+trackNames[trackCounter]) #Создаём поток видео
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    log= getLog(LOG_PATH)
    calcAngles(log)

    start_mSeconds, lat, lon, angle= log[0]
    start_start_mSeconds= start_mSeconds
    curent_mSeconds= start_mSeconds

    localizator= SignsLocalization(modelPath= MODEL_PATH, dataPath= DATA_PATH)
    signs= []

    i= 0
    while True:
        if MINUTES>0 and curent_mSeconds-start_start_mSeconds>MINUTES*60*1000:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, curent_mSeconds-start_mSeconds) #Устанавливаем видеопоток на нужную милисекунду

        ret, origFrame = cap.read() #Берём кадр

        if not ret:
            trackCounter+= 1
            if trackCounter>= len(trackNames):
                print("Stream end. Exiting.")
                break

            start_mSeconds= curent_mSeconds
            cap = cv2.VideoCapture(TRACKS_PATH+'\\'+trackNames[trackCounter]) #Создаём поток видео
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
