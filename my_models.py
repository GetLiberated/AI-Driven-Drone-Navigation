"""
Models

Collection of models that are used for this project.
"""
import cv2
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class YOLOv5:
    def __init__(self):
        self.model = None

    def init(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
        device = torch.device("cuda")
        self.model.to(device)

    def predict(self, img):
        results = self.model(img)
        results.render()
        return results.pandas().xyxy[0]


class MiDaS:
    def __init__(self):
        self.device = None
        self.model = None
        self.transform = None

    def init(self):
        self.device = torch.device("cuda")
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        # self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        # self.transform = midas_transforms.small_transform

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        return output


class AlexNet:
    def __init__(self, device=torch.device("cuda:0")):
        self.model = None
        self.device = device
        self.images_train = None
        self.images_test = None
        self.labels_train = None
        self.labels_test = None

    def alexnet(self):
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3)
        )
        return model

    def validate(self, model, images_test, labels_test, batch_size=64):
        total = 0
        correct = 0
        for i in range(0, len(images_test), batch_size):
            images = images_test[i:i + batch_size].to(self.device)
            x = model(images)
            value, pred = torch.max(x, 1)
            pred = pred.data.cpu()
            total += x.size(0)
            if (i + batch_size) > len(images_test):
                for j in range(len(images_test) - i):
                    if pred[j] == labels_test[j + i]:
                        correct += 1
            else:
                for j in range(batch_size):
                    if pred[j] == labels_test[j + i]:
                        correct += 1
        return correct * 100. / total

    def train(self, numb_epoch=3, lr=1e-3, batch_size=64):
        if self.images_train is None:
            self.load_dataset()
        images_train = torch.from_numpy(self.images_train)
        labels_train = torch.from_numpy(self.labels_train).to(torch.int64)
        images_test = torch.from_numpy(self.images_test)
        labels_test = torch.from_numpy(self.labels_test).to(torch.int64)

        accuracies = []
        cnn = self.alexnet().to(self.device)
        cec = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
        max_accuracy = 0
        for epoch in range(numb_epoch):
            for i in range(0, len(images_train), batch_size):
                images = images_train[i:i + batch_size].to(self.device)
                labels = labels_train[i:i + batch_size].to(self.device)
                optimizer.zero_grad()
                pred = cnn(images)
                loss = cec(pred, labels)
                loss.backward()
                optimizer.step()
            accuracy = float(self.validate(cnn, images_test, labels_test))
            accuracies.append(accuracy)
            print('Epoch:', epoch + 1, "Accuracy :", accuracy, '%')
            if accuracy > max_accuracy:
                self.model = copy.deepcopy(cnn)
                max_accuracy = accuracy
                print("New Best Model with Accuracy: ", accuracy)
        # print(accuracies)
        print("Saving best model...")
        torch.save(self.model.state_dict(), "alexnet.pth")

    def load(self):
        self.model = self.alexnet().to(self.device)
        self.model.load_state_dict(torch.load("alexnet.pth"))

    def predict(self, x):
        x = cv2.resize(x, dsize=(224, 224))
        # x = (x - np.mean(x)) / np.std(x)
        x = cv2.normalize(x, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        T = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        with torch.no_grad():
            pred = self.model(torch.unsqueeze(T(x), axis=0).float().to(self.device))
            pred = F.softmax(pred, dim=-1).cpu().numpy()
            # pred_idx = np.argmax(pred)
            # print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")
            # return pred_idx
            return pred

    def load_dataset(self):
        depth_images = np.load("depth_images.npy")
        depth_labels = np.load("depth_labels.npy")
        resized_depth_images = np.array([[cv2.resize(depth_images[0], dsize=(224, 224))]])
        for img in depth_images[1:]:
            resized_depth_images = np.append(resized_depth_images, [[cv2.resize(img, dsize=(224, 224))]], axis=0)
        # depth_images = (resized_depth_images - np.mean(resized_depth_images)) / np.std(resized_depth_images) # Standardisation
        depth_images = cv2.normalize(resized_depth_images, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalisation
        # depth_images = resized_depth_images # Raw

        self.images_train = depth_images[:2000]
        self.labels_train = depth_labels[:2000]
        self.images_test = depth_images[2000:]
        self.labels_test = depth_labels[2000:]
