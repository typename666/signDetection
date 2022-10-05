# trafficSignDetection
Программа распознаёт дорожные знаки на видео и заносит их в файл JSON.

# Требования
opencv-python>=4.1.1<br />
torch>=1.7.0<br />
torchvision>=0.8.1

# Запуск
python signLocalizationFromTwoPoints.py --weights .\\weights\\YOLOv5.pt --min_conf 0.7 --im_size 1024 --source input_video.mp4 --data custom_data.yaml --save_frames False

Введите "python signLocalizationFromTwoPoints.py --help" для помощи.

# Добавление новых знаков
Для обучения YOLOv5 на новых знаках необходимо запустить скрипт yolov5/train.py, для того чтобы узнать его параметры необходимо ввести "python train.py --help".<br />
Наиболее важным является параметр --data, который отвечает за имя файла в формате YAML, в котором должны быть указаны названия классов, а также пути к папкам в которых хранятся изображения для обучения и проверки детектора и папки в которых хранятся текстовые файлы с параметрами одного прямоугольника обозначающего выделенный объект в формате "Номер класса x y ширина высота" (пример в папке trainingDataExample).
