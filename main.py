from ImageLoading import ImageLoader
from HarrisCorner import Harris
import cv2 as cv
import numpy as np
import argparse


def main():
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()

    # Harris corner detection. Trackbar is used for tuning
    HarCor = Harris()
    HarCor.trackbar(img)

    # TODO: Method 1.1: Pure SIFT feature matching with the gate

    # TODO: Method 1.2: Pure SIFT feature matching with the white checker pattern at corners

    # TODO: Method 2.1: Colorfilter to gate colors (dark blue, light blue, white); feature matching around those colors

    # TODO: Method 2.2: Colorfilter to gate colors (dark blue, light blue, white);
    #  feature matching with the gate around those colors

    # TODO: Method 2.3: Colorfilter to gate colors (dark blue, light blue, white);
    #  feature matching with the white checker patttern around those colors


if __name__ == "__main__":
    main()
