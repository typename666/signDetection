import torch
import torch.nn as nn
from torch.optim import Adam

from os.path import join, isfile, isdir
from os import listdir

import csv
import cv2

WEIGHTS_PATH= join('weights', 'pointsDetector')

class AlexNet(nn.Module):
    def __init__(self, num_classes= 8):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Linear(3456, 2**4),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(2**4, 2**4),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(2**4, 2**4),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(2**4, 2**3),
            nn.ReLU())
        self.fc4= nn.Sequential(
            nn.Linear(2**3, num_classes))
        
    def forward(self, input1):
        out = self.layer1(input1)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out

class PointsDetector():
  def __init__(self, nClasses):
    self.INPUT_SIZE= 80

    self.device= 'cpu'
    print('Device: '+ self.device)

    self.networks= []
    for i in range(0, nClasses):
      self.networks.append(AlexNet().to(self.device))

    self.isWeightsExist= []
    for i in range(nClasses):
      weightsPath= join(WEIGHTS_PATH, '{}.pt'.format(i))

      self.isWeightsExist.append(isfile(weightsPath))
      if self.isWeightsExist[i]:
          self.networks[i].load_state_dict(torch.load(weightsPath, map_location=torch.device('cpu')))

    self.LR= 1.0e-4
    self.loss= nn.MSELoss()

  def trainNetwork(self, x_train, y_train, epochs, n):
    x_train_img_t= torch.tensor(x_train).to(self.device).permute(0, 3, 1, 2).float()
    y_train= torch.tensor(y_train).to(self.device).float()

    model= self.networks[n]

    weightsPath= join(WEIGHTS_PATH, str(n)+'.pt')
    if isfile():
        model.load_state_dict(torch.load())

    optimizer = Adam(model.parameters(), lr= self.LR)

    bestTrainLoss= 1.0E6

    # Forward pass
    model.train()

    print('Network: {}'.format(n))
    for epoch in range(epochs): 
      outputs = model(x_train_img_t)
      trainLoss = self.loss(outputs, y_train)
            
      # Backward and optimize
      optimizer.zero_grad()
      trainLoss.backward()
      optimizer.step()

      if epoch%100==0:
        if trainLoss< bestTrainLoss:
          bestTrainLoss= trainLoss
          torch.save(model.state_dict(), weightsPath)

        print('Epoch: {}'.format(epoch))
        print('Train loss: {}'.format(trainLoss))

    print('\n\n')

  def evaluteModel(self, x_evalute, n):
    x_evalute_t= torch.tensor(x_evalute).to(self.device).permute(0, 3, 1, 2).float()

    model= self.networks[n]

    model.eval()
    return model(x_evalute_t)

def readTrainData(signsPath= 'signs'):    
    allImages= []
    allPoints= []

    signsDirs= listdir(signsPath)

    for signPath in signsDirs:
      if isdir(join(signsPath, signPath)):
        csvFileName= join(signsPath, signPath, 'points.csv')

        if isfile(csvFileName):
          with open(csvFileName, 'r', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')

            signClass= int(signPath)
            lastImgName= ''
            imagesPath= join(signsPath, signPath)
            for row in spamreader:
              strRow= row[0]

              parsedRow= strRow.split(',')

              curentImgName= parsedRow[3]
                
              x= float(parsedRow[1])
              y= float(parsedRow[2])

              while signClass>=len(allImages):
                allImages.append([])
                allPoints.append([])

              if curentImgName==lastImgName:
                  allPoints[signClass][-1].append(x)
                  allPoints[signClass][-1].append(y)
              else:
                  if len(allPoints[signClass])>0 and len(allPoints[signClass][-1])==3:
                    allPoints[signClass].pop()
                    allImages[signClass].pop()

                  img= cv2.imread(join(imagesPath, curentImgName))

                  if img is None:
                    continue
                    
                  allImages[signClass].append(img)
                  allPoints[signClass].append([x, y])

              lastImgName= curentImgName

    return allImages, allPoints

if __name__ == "__main__":
    x_train, y_train = readTrainData('/content/signs') #Читаем картинки и координаты точек на них

    networks= Networks(len(x_train))

    for i in range(0, len(x_train)):
        if len(x_train[i])>0:
            networks.trainNetwork(x_train[i], y_train[i], 10000, i)