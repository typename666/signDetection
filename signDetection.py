import cv2
import json
import torch
import argparse
from os import mkdir
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

JSON_OUTPUT_PATH= "signsDetection_result.json" #Файл, в который будет сохраняться результат
FRAMES_PATH= '.\\frames' #Файл, в который будут сохраняться отдельные кадры

if __name__ == "__main__":
    result= {'frames': []} #итоговый словарь со знаками 

    #Задаём парсер аргументов
    parser = argparse.ArgumentParser(description='A tutorial of argparse!')
    parser.add_argument("--weights", default= ".\\weights\\YOLOv5.pt", type= str, help="Веса YOLOv5")
    parser.add_argument("--min_conf", default= 0.7, type= float, help="Минимальная уверенность YOLOv5 в ответе при, которой он считается верным")
    parser.add_argument("--im_size", default= 1024, type= int, help="Размер изображения, на котором происходило обучение сети")
    parser.add_argument("--source", default= 'input_video.mp4', type= str, help="Путь к видео, которое необходимо обработать")
    parser.add_argument("--data", default= 'custom_data.yaml', type= str, help="Путь к YAML файлу с названиями классов")
    parser.add_argument("--save_frames", default= False, type= bool, help="Если True сохраняет кадры в которых есть знаки в папку frames")

    #Вылавливаем аргументы командной строки
    args = parser.parse_args()
    MODEL_PATH= args.weights
    MIN_CONF= args.min_conf
    INPUT_SIZE= (args.im_size, args.im_size)
    VIDEO_PATH= args.source
    DATA_PATH= args.data
    SAVE_FRAMES= args.save_frames

    #Инициализируем YOLOv5
    device= select_device('')
    model= DetectMultiBackend(MODEL_PATH, device= device, dnn=False, data= DATA_PATH, fp16= False)
    names= model.names
    model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH) #Создаём поток видео
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    if SAVE_FRAMES:
        mkdir(FRAMES_PATH) #Создаём папку для вырезанных кадров

    second= 0 #Секунда видео
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC,second * 1000) #Устанавливаем видеопоток на нужную секунду

        ret, origFrame = cap.read() #Берём кадр

        if not ret:
            print("Stream end. Exiting.")
            break

        #Подгатавливаем кадр для YOLO
        origH, origW= origFrame.shape[:2]
        frame= cv2.resize(origFrame, INPUT_SIZE)
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame).to(device)
        frame = frame.permute((2, 0, 1))

        frame = frame.half() if model.fp16 else frame.float()
        frame /= 255
        if len(frame.shape) == 3:
            frame = frame[None]

        pred = model(frame, augment= False, visualize= False) #Прогоняем кадр через YOLO
        pred = non_max_suppression(pred, 0.6, 0.25, None, False, max_det= 1000)[0]

        #Заносим координаты и названия знаков в словарь
        signs= []
        if pred.size()[0]>0:
            for box in pred:

                if box[4]>MIN_CONF:
                    boxX1= round(float(origW*box[0]/INPUT_SIZE[1]))
                    boxY1= round(float(origH*box[1]/INPUT_SIZE[0]))
                    boxX2= round(float(origW*box[2]/INPUT_SIZE[1]))
                    boxY2= round(float(origH*box[3]/INPUT_SIZE[0]))

                    signs.append({'x1': boxX1, 'y1': boxY1, 'x2': boxX2, 'y2': boxY2, 'name': names[int(box[5])]})

            if len(signs)>0:
                result['frames'].append({'second': second, 'signs': signs})

                if SAVE_FRAMES:
                    cv2.imwrite(FRAMES_PATH+'\\frame'+str(second)+'.jpg', origFrame) #Сохраняем кадр из видео, на котором есть знаки, отдельно

        second= second+1

        if cv2.waitKey(1) == ord('q'): #Завершаем просмотр видеопотока по нажатию q
            break

#Заносим словарь питон в JSON файл
with open(JSON_OUTPUT_PATH, "w") as write_file:
    json.dump(result, write_file, sort_keys=False, indent=4, ensure_ascii=False, separators=(',', ': '))