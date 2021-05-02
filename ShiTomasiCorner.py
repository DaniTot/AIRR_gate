import cv2 as cv
import numpy as np

class ShiTomas:
    def __init__(self):
        self.img = None
        self.n_corners = 25
        self.quality = 0.01
        self.dist = 10

    def load_image(self, img, n_corners=25, quality_level=0.01, min_dist=10):
        self.img = img
        self.n_corners = n_corners
        self.quality = quality_level
        self.dist = min_dist

    def corner_detect(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray, self.n_corners, self.quality, self.dist)
        if corners is not None:
            corners = np.int0(corners)
        return corners

    def trackbar(self, img):
        window_name = "Settings"
        cv.namedWindow(window_name)

        def nothing(x):
            pass

        low_n = 1
        high_n = 50
        cv.createTrackbar("n corners", window_name, 1, 50, nothing)

        low_qlty = 1
        high_qlty = 100
        cv.createTrackbar("quality lvl", window_name, 1, 100, nothing)

        low_dst = 1
        high_dst = 100
        cv.createTrackbar("min dist", window_name, 1, 100, nothing)

        while True:
            orig = img.copy()
            out = img.copy()
            cv.imshow("in", orig)
            n = cv.getTrackbarPos("n corners", window_name)
            qlty = cv.getTrackbarPos("quality lvl", window_name)
            dst = cv.getTrackbarPos("min dist", window_name)
            self.load_image(orig, n, qlty/100., dst)
            print(n, qlty/100., dst)
            corners = self.corner_detect()
            if corners is not None:
                for i in corners:
                    x, y = i.ravel()
                    cv.circle(out, (x, y), 3, [0, 0, 255], -1)

            cv.imshow(window_name, out)

            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break