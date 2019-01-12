import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_mask(img):

    THRESHOLD = 125

    height,width = img.shape[:2]
    mask = np.zeros((height,width),dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # delta_0 = (img[i][j][0] - img[i][j][1])**2
            # delta_1 = (img[i][j][0] - img[i][j][2])**2
            # delta_2 = (img[i][j][1] - img[i][j][2])**2

            delta_0 = (int(img[i][j][0]) - int(img[i][j][1]))**2
            delta_1 = (int(img[i][j][0]) - int(img[i][j][2]))**2
            delta_2 = (int(img[i][j][1]) - int(img[i][j][2]))**2

            delta = delta_0 + delta_1 + delta_2
            if delta < THRESHOLD:
                mask[i][j] = 1
            # print('({}, {}, {})'.format(img[i][j][0],img[i][j][1],img[i][j][2]))
            # print('({}, {}, {})'.format(delta_0,delta_1,delta_2))

    # print(mask)
    # plt.subplot(221)
    # plt.imshow(img)

    # plt.subplot(222)
    # plt.imshow(mask,cmap='binary')

    # mask_img = cv2.bitwise_and(img,img,mask=mask)
    # plt.subplot(223)
    # plt.imshow(mask_img)

    # plt.show()
    return mask


img = cv2.imread('sample_2.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print(get_mask(img))