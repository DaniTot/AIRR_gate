import cv2 as cv
import numpy as np

class Harris:
    def __init__(self):
        self.img = None

        self.block_size = None
        self.ksize = None
        self.harris_param = None
        self.threshold = None

    def load_image(self, img, block_size=2, ksize=3, k=0.04, threshold=1.2):
        self.img = img
        self.block_size = block_size
        self.ksize = ksize
        self.harris_param = k
        self.threshold = threshold

    def corner_detect(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        corner = cv.cornerHarris(gray, self.block_size, self.ksize, self.harris_param)
        # result is dilated for marking the corners, not important
        dst = cv.dilate(corner, None)
        # Threshold for an optimal value, it may vary depending on the image.
        self.img[dst > self.threshold * dst.max()] = [0, 0, 255]
        return self.img

    def trackbar(self, img):
        window_name = "Settings"
        cv.namedWindow(window_name)

        def nothing(x):
            pass

        low_block = 2
        high_block = 10
        cv.createTrackbar("block size", window_name, 1, 10, nothing)

        low_ksize = 3
        high_ksize = 10
        cv.createTrackbar("ksize", window_name, 1, 10, nothing)

        low_k = 4
        high_k = 50
        cv.createTrackbar("k", window_name, 1, 50, nothing)

        low_th = 2
        high_th = 50
        cv.createTrackbar("threshhold", window_name, 1, 50, nothing)

        while True:
            orig = img.copy()
            cv.imshow("in", orig)
            block = cv.getTrackbarPos("block size", window_name)
            ksize = cv.getTrackbarPos("ksize", window_name)
            k = cv.getTrackbarPos("k", window_name)
            th = cv.getTrackbarPos("threshhold", window_name)
            # HarCor.load_image(img, block_size=low_block, ksize=low_ksize, k=low_k/100, threshold=low_th/100)
            self.load_image(orig, block_size=block, ksize=ksize * 2 - 1, k=k / 100, threshold=th / 100)
            print(block, ksize * 2 - 1, k / 100, th / 100)
            out1 = self.corner_detect()

            cv.imshow(window_name, out1)

            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break