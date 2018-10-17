import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import rgb2hex

from sklearn import ensemble
from sklearn import datasets

def load_image(img_path):
    img = Image.open(img_path)
    arr = np.array(img)
    return arr

def GradientBoostingClassifier(img):
    # print("img.shape",img.shape)
    # img.shape (720, 1280, 3)
    # 720*1280

    res_img=img.reshape(-1,3)
    # (408960, 3)
    
    # print("res_img",res_img)
    # [[ 76 102 132]
    #  [ 76 102 132]
    #  [ 76 102 132]
    #  ...
    #  [110 110  51]
    #  [110 110  51]
    #  [110 110  51]]

    # plt.imshow(res_img.reshape(480,852,3))
    # plt.show()

    res_img_g=res_img.mean(axis=1).astype("uint8")
    # print("res_img_g",res_img_g.shape)
    # res_img_g (921600,)

    # print("res_img_g",res_img_g)
    # [103 103 103 ...  90  90  90]

    # labels, y = np.unique(res_img_g, return_inverse=True)
    # print("labels",labels)
    # print("y",y)
    # y [276 276 276 ... 185 185 185]
    # print("y",len(y))
    # y 921600

    y=res_img_g
    # print("y",y.shape)
    # y (921600,)

    # print("res_img_g",res_img_g)
    # [0.49019608 0.49019608 0.49019608 ... 0.87581699 0.88366013 0.88366013]

    # train and test data of X
    X_train, X_test = res_img[:6000], res_img[6000:12000]
    # print("X_train, X_test",X_train.shape, X_test.shape)
    # X_train, X_test (600000, 3) (321600, 3)

    # train and test data of y
    # y_train, y_test = res_img_g[:600000], res_img_g[600000:]
    y_train, y_test = y[:6000], y[6000:12000]
    # print("y_train, y_test",y_train.shape, y_test.shape)
    # y_train, y_test (600000,) (321600,)

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    dd=clf.predict(res_img)
    print("dd",dd)
    # dd [276 276 276 ... 355 355 355]
    print("dd",dd.shape)
    # dd (921600,)

    re_dd=dd.reshape(720,1280)

    # Standardization
    re_dd=(re_dd-np.mean(re_dd))/np.std(re_dd)

    # Normalise [0,1]
    re_dd=(re_dd-np.min(re_dd))/np.ptp(re_dd)

    plt.imshow(re_dd,cmap="gray")
    plt.show()
    
    # https://raw.githubusercontent.com/youngminpark2559/test/master/boostings/gradient_b/pics/0008_ref_bi_20_22_dtf_50_50.png
    # https://raw.githubusercontent.com/youngminpark2559/test/master/boostings/gradient_b/pics/GradientBoostingClassifier_on_ref_img.png


img_path="/mnt/1T-5e7/0008_ref_bi_20_22_dtf_50_50.png"
l_img=load_image(img_path)
GradientBoostingClassifier(l_img)
