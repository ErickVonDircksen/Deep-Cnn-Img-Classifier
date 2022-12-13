import os
import cv2
import random
import numpy as np

directory = 'data\/Tilapia'


def transform(fileName):
    
    IN_IMAGE = fileName
    OUT_PATH = 'data\outImage'

    if os.path.sep != OUT_PATH[-1]:
        OUT_PATH += os.path.sep

    name = IN_IMAGE.split(os.path.sep)[-1].split('.')[0] + '_'
    IN_IMAGE = cv2.imread(IN_IMAGE)

    SIZE_Y = IN_IMAGE.shape[0]
    SIZE_X = IN_IMAGE.shape[1]

    if os.path.exists(OUT_PATH) is False:
        os.makedirs(OUT_PATH)

    # Blur image and save it to OUT_PATH + 'blurred5x5.png'
    cv2.imwrite(OUT_PATH + name + 'blurred 3x3.png', cv2.blur(IN_IMAGE, (3, 3)))
    cv2.imwrite(OUT_PATH + name + 'blurred 15x15.png', cv2.blur(IN_IMAGE, (15, 15)))
    cv2.imwrite(OUT_PATH + name + 'gaussian blurred 15x15.png', cv2.GaussianBlur(IN_IMAGE, (15, 15), 0))
    
    # Draw a random black box on the image and save it to OUT_PATH + 'blackbox.png'
    def blackbox(img, w, h):
        img = img.copy()
        w = int(w)
        h = int(h)
        x = random.randint(0, img.shape[1] - w)
        y = random.randint(0, img.shape[0] - h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        return img

    cv2.imwrite(OUT_PATH + name + 'blackbox 10%.png', blackbox(IN_IMAGE, SIZE_X*0.1, SIZE_Y*0.1))
    cv2.imwrite(OUT_PATH + name + 'blackbox 25%.png', blackbox(IN_IMAGE, SIZE_X*0.25, SIZE_Y*0.25))


    # Rotate the image and save it to OUT_PATH + 'rotated.png'
    def rotate(img, angle):
        return cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1), (img.shape[1], img.shape[0]))

    cv2.imwrite(OUT_PATH + name + 'rotated 45.png', rotate(IN_IMAGE, 45))
    cv2.imwrite(OUT_PATH + name + 'rotated 90.png', rotate(IN_IMAGE, 90))
    cv2.imwrite(OUT_PATH + name + 'rotated 180.png', rotate(IN_IMAGE, 180))
   

    # Dilate the image and save it to OUT_PATH + 'dilated.png'
    def dilate(img, iterations):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(img, kernel, iterations=iterations)

    cv2.imwrite(OUT_PATH + name + 'dilated 1x.png', dilate(IN_IMAGE, 1))
    cv2.imwrite(OUT_PATH + name + 'dilated 20x.png', dilate(IN_IMAGE, 4))
    cv2.imwrite(OUT_PATH + name + 'dilated 50x.png', dilate(IN_IMAGE, 5))

    # Change hue of the image and save it to OUT_PATH + 'hue.png'
    def hue(img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.add(h, value)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(OUT_PATH + name + 'hue 10.png', hue(IN_IMAGE, 10))
    cv2.imwrite(OUT_PATH + name + 'hue -10.png', hue(IN_IMAGE, -10))
    cv2.imwrite(OUT_PATH + name + 'hue 180.png', hue(IN_IMAGE, 180))
    cv2.imwrite(OUT_PATH + name + 'hue -180.png', hue(IN_IMAGE, -180))
    cv2.imwrite(OUT_PATH + name + 'hue 270.png', hue(IN_IMAGE, 270))
    cv2.imwrite(OUT_PATH + name + 'hue -270.png', hue(IN_IMAGE, -270))

    # Compress image with JPEG and save it
   
    cv2.imwrite(OUT_PATH + name + 'compressed 10.jpg', IN_IMAGE, [cv2.IMWRITE_JPEG_QUALITY, 10])
    cv2.imwrite(OUT_PATH + name + 'compressed 50.jpg', IN_IMAGE, [cv2.IMWRITE_JPEG_QUALITY, 50])
    cv2.imwrite(OUT_PATH + name + 'compressed 75.jpg', IN_IMAGE, [cv2.IMWRITE_JPEG_QUALITY, 75])

    print('Saving to:', OUT_PATH)
    
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    transform(f)
