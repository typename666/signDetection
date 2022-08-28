# trafficSignDetection
Программа распознаёт дорожные знаки на видео и заносит их в файл JSON.

# Требования
opencv-python>=4.1.1
torch>=1.7.0
torchvision>=0.8.1

# Запуск
python signDetection.py --weights .\\weights\\YOLOv5.pt --min_conf 0.7 --im_size 1024 --source input_video.mp4 --data custom_data.yaml

Введите "python signDetection.py --help" для помощи.
