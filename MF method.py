import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_equal(img, z_max=255):
    H, W = img.shape
    # S is the total of pixels
    S = H * W * 1.

    out = img.copy()

    sum_h = 0.

    for i in range(0, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)

    return out


#local HE
def localHE(img):
    clahe=cv2.createCLAHE(clipLimit=10,tileGridSize=(10,10))
    dst=clahe.apply(img)
    return dst

#homofilter
def homofilter(I):
    I = np.double(I)
    m, n = I.shape
    rL = 0.5
    rH = 2
    c = 2
    d0 = 20
    I1 = np.log(I + 1)
    FI = np.fft.fft2(I1)
    n1 = np.floor(m / 2)
    n2 = np.floor(n / 2)
    D = np.zeros((m, n))
    H = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = ((i - n1) ** 2 + (j - n2) ** 2)
            H[i, j] = (rH - rL) * (np.exp(c * (-D[i, j] / (d0 ** 2)))) + rL
    I2 = np.fft.ifft2(H * FI)
    I3 = np.real(np.exp(I2) - 1)
    I4 = I3 - np.min(I3)
    I4 = I4 / np.max(I4) * 255
    dstImg = np.uint8(I4)
    return dstImg

#Gabor filter
def Gaborfilter(img):
    for theta in range(1,19):
        image_container = []
        c=theta*10
        retval=cv2.getGaborKernel(ksize=(3,3), sigma=2*np.pi, theta=c, lambd=np.pi/2, gamma=0.5)
        retval /= 1.5 * retval.sum()
        image_gabor=cv2.filter2D(img,-1,retval)

        image_container.append(image_gabor)
    # cv2.imshow("   ",image_gabor)
    # cv2.waitKey(0)
    pass

#Gauss MF
def gaussmatchfilter(sigma=1,YLength=10):
    filters=[]
    width=np.ceil(np.sqrt(6*np.ceil(sigma)+1)**2+YLength**2)
    if np.mod(width,2)==0:
        width=width+1
    width=int(width)
    for theta in np.arange(0,np.pi,np.pi/32):
        matchfilter=np.zeros((width,width),dtype=np.float)
        for x in range(width):
            for y in range(width):
                halfLength=(width-1)/2
                x_=(x-halfLength)*np.cos(theta)+(y-halfLength)*np.sin(theta)
                y_=-(x-halfLength)*np.sin(theta)+(y-halfLength)*np.cos(theta)
                if abs(x_) > (YLength-1)/2:
                    matchfilter[x][y]=0
                elif abs(y_) > (YLength-1)/2:
                    matchfilter[x][y]=0
                else:
                    matchfilter[x][y]=-np.exp(-.5*(x_/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        m=0.0
        for i in range(matchfilter.shape[0]):
            for j in range(matchfilter.shape[1]):
                if matchfilter[i][j]<0:
                        m=m+1
        mean=np.sum(matchfilter)/m
        for i in range(matchfilter.shape[0]):
            for j in range(matchfilter.shape[1]):
                if matchfilter[i][j]<0:
                    matchfilter[i][j]=matchfilter[i][j]-mean
        filters.append(matchfilter)

    return filters


def max_reciever(img,filters):
    out=np.zeros_like(img)
    for kern in filters:
        fig=cv2.filter2D(img,cv2.CV_8U,kern,borderType=cv2.BORDER_REFLECT)
        np.maximum(out,fig,out)
    return out


def pass_mask(mask, img):
    # qwe = reverse_image(img)
    qwe = img.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                qwe[i][j] = 0
    # asd = cv2.filter2D(qwe, cv2.CV_8U, mask)
    return qwe


def Agpass_mask(mask, img):
    # qwe = reverse_image(img)
    qwe = img.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                qwe[i][j] = 255
    # asd = cv2.filter2D(qwe, cv2.CV_8U, mask)
    return qwe

#OSTU segmentation
def histseg(img,level):
    Img=img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < level:
                Img[i][j] = 0
            else:
                Img[i][j] = 255
    return Img

#AUC calculate
def CalcuAUC(img1,img2):
    correct=0
    wrong=0
    vessel_detected=0
    vessel_undetected=0
    black=0
    black_wh=0
    Prediction = 0
    Recall = 0
    AUC=None
    if img1.shape == img2.shape:
        print("same size")
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if img2[i][j] == 0:
                    if img1[i][j] == img2[i][j]:
                        black = black +1
                    else:
                        back_wh = black_wh+1
                elif img2[i][j] == 255:
                    if img1[i][j] == img2[i][j]:
                        vessel_detected = vessel_detected +1
                    else:
                        vessel_undetected=vessel_undetected+1
    else:
        print("Img1 and Img2 are not same size")
    correct=vessel_detected+black
    wrong=vessel_undetected+black_wh
    TP=vessel_detected
    FN=vessel_undetected
    FP=black_wh
    TN=black
    Prediction=TP/(TP+FP)
    Recall=TP/(TP+FN)
    Specificity=TN/(FP+TN)
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    AUC=correct/(correct+wrong)
    print("accuracy is: ",AUC)
    print("Prediction is:",Prediction,"Recall: ",Recall)
    print("True prop",(correct+wrong)/(584*565))
    print("total",)


    return FPR,TPR


def genmask(mask):
    matrix=np.zeros(mask.shape[:2])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                matrix[i][j] == 0
            else:
                matrix[i][j] == 1
    return matrix




if __name__ == '__main__':
    for n in range(14,15):
        vessel = cv2.imread("D:/UNET-try/DRIVE/test/images/{}_test.png".format(n+1))
        mask = cv2.imread("D:/UNET-try/DRIVE/test/mask/{}.png".format(n+1),cv2.IMREAD_GRAYSCALE)
        manual = cv2.imread("D:/UNET-try/DRIVE/test/img/manual{}.png".format(n+1),cv2.IMREAD_GRAYSCALE)

        b, g, r = cv2.split(vessel)

# for k in range(120, 150, 15):
#         g = pass_mask(mask, g)
        blurImg = cv2.GaussianBlur(g, (5, 5), 0)
    #Histogram equalization(whole)
        heImg=cv2.equalizeHist(g)
    #Adaptive HE
        AdpheImg=localHE(blurImg)
    #homo filter
        homopic=homofilter(AdpheImg)
    #match filter
        filters=gaussmatchfilter(sigma=1,YLength=10)
        MFImg_=max_reciever(homopic,filters)
    #OTSU
        ret1,th1=cv2.threshold(MFImg_,0,255,cv2.THRESH_OTSU)

# for rh in range(1,11):
#     c=rh
#     rh=rh/5+2
#     filterpic=homomorphic_filter(g,rh=rh)
#     plt.subplot(2,5,c)
#     plt.imshow(filterpic,cmap='gray')
# plt.show()
    #c = pass_mask(mask, predictImg)
    # cv2.imshow("vessel1",homopic)
    # cv2.imshow("vessel2",g)

        final_img=pass_mask(mask,th1)
        cv2.imwrite('MFresult{}.tif'.format(n),final_img)


















# # 获得混淆矩阵
# def BinaryConfusionMatrix(prediction, groundtruth):
#     """Computes scores:
#     TP = True Positives    真正例
#     FP = False Positives   假正例
#     FN = False Negatives   假负例
#     TN = True Negatives    真负例
#     return: TP, FP, FN, TN"""
#
#     TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
#     FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
#     FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
#     TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
#
#     return TN, FP, FN, TP
#
#
# # 精准率和 或 查准率的计算方法
# def get_precision(prediction, groundtruth):
#     _, FP, _, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     precision = float(TP) / (float(TP + FP) + 1e-6)
#     return precision
#
#
# # 召回率和 或 查全率的计算方法
# def get_recall(prediction, groundtruth):
#     TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     recall = float(TP) / (float(TP + FN) + 1e-6)
#     return recall
#
#
# # 准确率的计算方法
# def get_accuracy(prediction, groundtruth):
#     TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     accuracy = float(TP + TN) / (float(TP + FP + FN + TN) + 1e-6)
#     return accuracy
#
#
# def get_sensitivity(prediction, groundtruth):
#     return get_recall(prediction, groundtruth)
#
#
# def get_specificity(prediction, groundtruth):
#     TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     specificity = float(TN) / (float(TN + FP) + 1e-6)
#     return specificity
#
#
# def get_f1_score(prediction, groundtruth):
#     precision = get_precision(prediction, groundtruth)
#     recall = get_recall(prediction, groundtruth)
#     f1_score = 2 * precision * recall / (precision + recall)
#     return f1_score
#
#
# # Dice相似度系数，计算两个样本的相似度，取值范围为[0, 1], 分割结果最好为1，最坏为0
# def get_dice(prediction, groundtruth):
#     TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     dice = 2 * float(TP) / (float(FP + 2 * TP + FN) + 1e-6)
#     return dice
#
#
# # 交并比 一般都是基于类进行计算, 值为1这一类的iou
# def get_iou1(prediction, groundtruth):
#     TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     iou = float(TP) / (float(FP + TP + FN) + 1e-6)
#     return iou
#
#
# # 交并比 一般都是基于类进行计算, 值为0这一类的iou
# def get_iou0(prediction, groundtruth):
#     TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
#     iou = float(TN) / (float(FP + TN + FN) + 1e-6)
#     return iou
#
#
# # 基于类进行计算的IoU就是将每一类的IoU计算之后累加，再进行平均，得到的就是基于全局的评价
# # 平均交并比
# def get_mean_iou(prediction, groundtruth):
#     iou0 = get_iou0(prediction, groundtruth)
#     iou1 = get_iou1(prediction, groundtruth)
#     mean_iou = (iou1 + iou0) / 2
#     return mean_iou

