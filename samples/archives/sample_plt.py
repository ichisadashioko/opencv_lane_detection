import cv2
import numpy as np
import matplotlib.pyplot as plt


def grab_frame(cap):
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def birdview(img, shrink_ratio):
    height, width = img.shape[:2]
    SKYLINE = 85
    src_pts = np.float32([[0, SKYLINE], [width, SKYLINE],
                          [0, height], [width, height]])

    dst_width = width
    dst_height = height - SKYLINE + 120
    dsize = (dst_width, dst_height)

    dst_pts = np.float32([[0, 0], [dst_width, 0], [
                         dst_width*shrink_ratio, dst_height], [dst_width*(1-shrink_ratio), dst_height]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    dst = cv2.warpPerspective(img, M, dsize)

    return dst


cap = cv2.VideoCapture('sample.avi')

plot_rows = 3
plot_cols = 4
plot_n = plot_rows*plot_cols
# plots = []
axes_plots = []
images_plots = []

for i in range(plot_n):
    ax = plt.subplot(plot_rows, plot_cols, i+1)
    axes_plots.append(ax)

    im = ax.imshow(grab_frame(cap))
    images_plots.append(im)

plt.ion()
plt.pause(5)

while(cap.isOpened()):
    print('cap is opening')
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # cv2.imshow('abs_sobelx lightness',abs_sobelx)

    for i in range(10):

        ratio = i*0.05
        bird_view = birdview(frame, ratio)
        label = '{}%'.format(int(ratio*100))

        axes_plots[i].set_xlabel(label)
        images_plots[i].set_data(bird_view)

        # cv2.imshow(label,bird_view)
    plt.pause(0.1)
    # cv2.imshow('frame',gray)
    # if cv2.waitKey(25) & 0xff == ord('q'):
    #     break

plt.ioff()
plt.show()
cap.release()
# cv2.destroyAllWindows()
