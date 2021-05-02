import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class FeatureExtractor:
    def __init__(self, *args):
        self.method_in = []
        self.add_method(*args)

        self.detector_options = {}    # Feature point detectors
        self.descriptor_options = {}  # Feature descriptors
        self.combined_options = {}    # Combined methods

        self.img = None
        self.gray = None
        self.out_img = None
        self.kp = None
        self.des = None

        self.setup()

    def setup(self):
        self.detector_options.update({"Harris": self.do_harris, "ShiTomasi": self.do_shitomasi,
                                      "FAST": self.do_fast, "STAR": self.do_star})                # Feature detectors
        self.descriptor_options.update({"BRIEF": self.do_brief})                                  # Feature descriptors
        self.combined_options.update({"SIFT": self.do_sift, "ORB": self.do_orb})                  # Combined methods

    def load_image(self, img):
        self.img = img
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.out_img = img.copy()
        self.kp = None
        self.des = None

    def set_method(self, *args):
        self.method_in = []
        self.add_method(*args)

    def add_method(self, *args):
        if args:
            for arg in args:
                assert type(arg) is str
                self.method_in.append(arg)

    def detect_compute(self):
        for method in self.method_in:
            if method in {**self.detector_options, **self.descriptor_options, **self.combined_options}.keys():
                if self.kp is None:
                    {**self.detector_options, **self.combined_options}[method]()
                if self.kp is not None and self.des is None:
                    try:
                        self.descriptor_options[method]()
                    except KeyError as err:
                        print(self.kp, self.des)
                        cv.imshow("err", self.gray)
                        cv.waitKey()
                        raise err

    def get_kp(self):
        if self.kp is None:
            self.detect_compute()
        return self.kp

    def get_mp_kp_des(self):
        if self.kp is None or self.des is None:
            self.detect_compute()
        assert self.kp is not None and self.des is not None
        out_lst = []
        for i in range(len(self.kp)):
            point = self.kp[i]
            desc = self.des[i]
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, desc)
            out_lst.append(temp)
        return out_lst

    def get_des(self):
        if self.des is None:
            self.detect_compute()
        return self.des

    def show_keypoints(self):
        self.out_img = cv.drawKeypoints(self.gray, self.kp, None, (255, 0, 0), 4)
        plt.imshow(self.out_img)
        plt.show()

    def do_harris(self, block_size=2, ksize=3, k=0.04, threshold=1.2):
        raise NotImplementedError
        gray = np.float32(self.gray)
        corner = cv.cornerHarris(gray, block_size, ksize, harris_param)
        # result is dilated for marking the corners, not important
        dst = cv.dilate(corner, None)
        # Threshold for an optimal value, it may vary depending on the image.
        self.img[dst > threshold * dst.max()] = [0, 0, 255]

    def do_shitomasi(self, n_corners=25, quality_level=0.01, min_dist=10):
        assert self.kp is None
        corners = cv.goodFeaturesToTrack(self.gray, n_corners, quality_level, min_dist)
        if corners is not None:
            self.kp = np.int0(corners)

    def do_fast(self, nonmax_supression=True):
        assert self.kp is None
        fast = cv.FastFeatureDetector_create()
        if nonmax_supression is False:
            fast.setNonmaxSuppression(0)
        # find and draw the keypoints
        self.kp = fast.detect(self.gray, None)

    def do_star(self):
        assert self.kp is None
        star = cv.xfeatures2d.StarDetector_create()
        self.kp = star.detect(self.gray, None)

    def do_brief(self):
        assert self.des is None
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        self.kp, self.des = brief.compute(self.gray, self.kp)

    def do_sift(self):
        # https://docs.opencv.org/master/d7/d60/classcv_1_1SIFT.html
        assert self.kp is None and self.des is None
        sift = cv.SIFT_create()
        self.kp, self.des = sift.detectAndCompute(self.gray, None)
        if len(self.kp) == 0:
            self.kp = None

    def do_orb(self):
        # https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html
        assert self.kp is None and self.des is None
        orb = cv.ORB_create()
        self.kp = orb.detect(self.gray, None)
        self.kp, self.des = orb.compute(self.gray, self.kp)