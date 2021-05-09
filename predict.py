import numpy as np
from unet_model import *
import torch
import cv2
import glob
from PIL import Image

if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=UNet(n_channels=1,n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth',map_location=device))
    net.eval()
    tests_path = glob.glob("D:/UNET-try/DRIVE/test"+'/images/*.tif')
    n=int(0)
    for test_path in tests_path:
        n=n+1
        img=cv2.imread(test_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=img.reshape(1,1,img.shape[0],img.shape[1])
        img_tensor=torch.from_numpy(img)
        img_tensor=img_tensor.to(device=device,dtype=torch.float32)
        pred=net(img_tensor)
        pred=np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        cv2.imwrite("D:/UNET-try/DRIVE/test/img/predict{}.png".format(n),pred)
        print(n)


