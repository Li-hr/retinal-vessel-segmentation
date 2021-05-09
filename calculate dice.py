import numpy as np
import cv2
import matplotlib.pyplot as plt
def calcDice(predict, groundtruth):
    predict=np.array(predict)
    groundtruth=np.array(groundtruth)
    if predict.shape == groundtruth.shape:
        print("same size")
        predict=np.where(predict[...,:] < 128, 0, 1)
        groundtruth = np.where(groundtruth[..., :] < 128, 0, 1)

    predict_n = np.ones(predict.shape) - predict
    groundtruth_n = np.ones(predict.shape) - groundtruth
    TP = np.sum(predict * groundtruth)
    FP = np.sum(predict * groundtruth_n)
    TN = np.sum(predict_n * groundtruth_n)
    FN = np.sum(predict_n * groundtruth)
    print("lou: ",'%.4f' %(TP/(TP+FP+FN)))
    dice = 2 * np.sum(predict * groundtruth) / (np.sum(predict) + np.sum(groundtruth))
    print("dice: ",'%.4f' %dice)
    print("TPR:",'%.4f' %(TP/(TP+FN)))
    print("FPR:",'%.4f' %(FP/(FP+TN)))
    print("TNR:",'%.4f' %(TN/(FP+TN)))
    print("FNR:",'%.4f' %(FN/(TP+FN)))
    print("ACC:",'%.4f' %((TP+TN)/(TP+TN+FP+FN)))
for n in range(10,21,1):
    groundtruth=cv2.imread("D:/UNET-try/DRIVE/test/img/manual{}.png".format(n))
    predict=cv2.imread("MFresult{}.tif".format(n))
    calcDice(predict, groundtruth)
# groundtruth=cv2.imread("D:/UNET-try/DRIVE/test/img/manual10.png")
# predict=cv2.imread("D:/UNET-try/DRIVE/test/img/predict10.png")
# calcDice(predict,groundtruth)

# def sigmoid(z):
#     return 1/(1+np.exp(-z))
# #
# nums=np.arange(-10,10,step=1)
# fig,ax=plt.subplots(figsize=(12,8))
#
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.set_xticks([-10, -5, 0, 5, 10])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([0, 0.5, 1])
# ax.plot(nums,sigmoid(nums),'r')
# plt.title('sigmoid')
# plt.show()

# def tanh(x):
#     return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot(111)
#
# x = np.linspace(-10, 10)
# y = tanh(x)
#
#
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.set_xticks([-10, -5, 0, 5, 10])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([-1, -0.5, 0.5, 1])
# plt.plot(x, y, label="Tanh", color="red")
#
# plt.legend()
# plt.show()

# def ReLU(x):
#     return np.maximum(0,x)
# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot(111)
#
# x = np.linspace(-10, 10)
# y = ReLU(x)
#
#
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.set_xticks([-10,10])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([0, 10])
#
# plt.plot(x, y, label="ReLU(x)", color="red")
# plt.legend()
# plt.show()
