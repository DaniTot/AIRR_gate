from ImageLoading import ImageLoader
from FeatureExtractor import FeatureExtractor
import cv2 as cv
import numpy as np
import cProfile
import pstats


def roi_checker(rows, cols, ratio_test, min_matches, sample_size, ImLo, FE, des_ref):
    false_neg = 0
    false_pos = 0
    true_pos = 0
    total_tests = 0
    total_cost = 0
    while True:
        # Load image
        img = ImLo.load_image()
        if img is None or total_tests >= sample_size:
            print(total_tests, true_pos, false_pos)
            print("True pos rate:", true_pos / (total_tests*4), "False pos rate:", false_pos / (total_tests*4))
            print("Total cost:", total_cost, "Average cost:", total_cost/total_tests/4)
            break

        rois = []
        stitched = np.zeros(img.shape, np.uint8)
        y_corner_min = img.shape[0]
        y_corner_max = 0
        x_corner_min = img.shape[1]
        x_corner_max = 0
        for row in range(0, rows):
            for col in range(0, cols):
                y_min = int(row * img.shape[0] / rows)
                y_max = int(row * img.shape[0] / rows + img.shape[0] / rows)
                x_min = int(col * img.shape[1] / cols)
                x_max = int(col * img.shape[1] / cols + img.shape[1] / cols)
                roi = img[y_min:y_max, x_min:x_max]

                FE.load_image(roi)
                FE.set_method("SIFT")
                kp = FE.get_kp()
                des = FE.get_des()

                if kp is not None:
                    if len(kp) >= 2:
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

                        search_params = dict(checks=50)  # or pass empty dictionary
                        flann = cv.FlannBasedMatcher(index_params, search_params)
                        matches = flann.knnMatch(des_ref, des, k=2)

                        # Apply ratio test
                        good = []
                        for m, n in matches:
                            if m.distance < ratio_test * n.distance:
                                good.append(m)

                        if len(good) >= min_matches:
                            pts = np.int32([kp[m.trainIdx].pt for m in good])
                            important_roi = False
                            for pt in pts:
                                # Elso x, masodik y, 0,0 bal alul
                                if pt[0] + x_min < x_corner_min:
                                    x_corner_min = pt[0] + x_min
                                    important_roi = True
                                if pt[0] + x_min > x_corner_max:
                                    x_corner_max = pt[0] + x_min
                                    important_roi = True
                                if pt[1] + y_min < y_corner_min:
                                    y_corner_min = pt[1] + y_min
                                    important_roi = True
                                if pt[1] + y_min > y_corner_max:
                                    y_corner_max = pt[1] + y_min
                                    important_roi = True
                            if important_roi:
                                rois.append(((y_min, y_max), (x_min, x_max)))
                                stitched[y_min:y_max, x_min:x_max] = roi

        if len(rois) > 0:
            print(total_tests)
            gt = ImLo.get_ground_truth()
            print("truth", gt)
            print("approx", x_corner_min, y_corner_min, x_corner_min, y_corner_max, x_corner_max, y_corner_max, x_corner_max, y_corner_min)
            cv.imshow("done ", stitched)

            cv.circle(img, (x_corner_min, y_corner_min), radius=4, color=(0, 0, 255), thickness=-1)
            cv.circle(img, (x_corner_min, y_corner_max), radius=4, color=(0, 0, 255), thickness=-1)
            cv.circle(img, (x_corner_max, y_corner_max), radius=4, color=(0, 0, 255), thickness=-1)
            cv.circle(img, (x_corner_max, y_corner_min), radius=4, color=(0, 0, 255), thickness=-1)

            cv.circle(img, (gt[0], gt[1]), radius=4, color=(0, 255, 0), thickness=-1)
            cv.circle(img, (gt[2], gt[3]), radius=4, color=(0, 255, 0), thickness=-1)
            cv.circle(img, (gt[4], gt[5]), radius=4, color=(0, 255, 0), thickness=-1)
            cv.circle(img, (gt[6], gt[7]), radius=4, color=(0, 255, 0), thickness=-1)

            cv.line(img, (gt[0], gt[1]), (x_corner_min, y_corner_min), color=(0, 255, 0), thickness=2)
            cv.line(img, (gt[2], gt[3]), (x_corner_max, y_corner_min), color=(0, 255, 0), thickness=2)
            cv.line(img, (gt[4], gt[5]), (x_corner_max, y_corner_max), color=(0, 255, 0), thickness=2)
            cv.line(img, (gt[6], gt[7]), (x_corner_min, y_corner_max), color=(0, 255, 0), thickness=2)

            cost = np.sqrt((gt[0] - x_corner_min)**2 + (gt[1] - y_corner_min)**2) + \
                   np.sqrt((gt[2] - x_corner_max)**2 + (gt[3] - y_corner_min)**2) + \
                   np.sqrt((gt[4] - x_corner_max)**2 + (gt[5] - y_corner_max)**2) + \
                   np.sqrt((gt[6] - x_corner_min)**2 + (gt[7] - y_corner_max)**2)
            total_cost += cost
            print(cost)
            cv.imshow("rect ", img)
            cv.waitKey()
            truepos = int(input("How many true_pos? "))
            false_pos += int(input("How many false_pos? "))
            false_neg += 4 - truepos
            true_pos += truepos

        total_tests += 1


def cost_checker(rows, cols, ratio_test, min_matches, sample_size, ImLo, FE, des_ref):
    total_tests = 0
    costs = np.array([])
    while True:
        # Load image
        img = ImLo.load_image()
        if img is None or total_tests >= sample_size:
            print("total tests:", total_tests, "sum:", np.sum(costs), "median:", np.median(costs), "std:", np.std(costs))
            break

        print(total_tests)

        rois = []
        y_corner_min = img.shape[0]
        y_corner_max = 0
        x_corner_min = img.shape[1]
        x_corner_max = 0
        for row in range(0, rows):
            for col in range(0, cols):
                y_min = int(row * img.shape[0] / rows)
                y_max = int(row * img.shape[0] / rows + img.shape[0] / rows)
                x_min = int(col * img.shape[1] / cols)
                x_max = int(col * img.shape[1] / cols + img.shape[1] / cols)
                roi = img[y_min:y_max, x_min:x_max]

                FE.load_image(roi)
                FE.set_method("SIFT")
                kp = FE.get_kp()
                des = FE.get_des()

                if kp is not None:
                    if len(kp) >= 2:
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

                        search_params = dict(checks=50)  # or pass empty dictionary
                        flann = cv.FlannBasedMatcher(index_params, search_params)
                        matches = flann.knnMatch(des_ref, des, k=2)

                        # Apply ratio test
                        good = []
                        for m, n in matches:
                            if m.distance < ratio_test * n.distance:
                                good.append(m)

                        if len(good) >= min_matches:
                            pts = np.int32([kp[m.trainIdx].pt for m in good])
                            important_roi = False
                            for pt in pts:
                                # Elso x, masodik y, 0,0 bal alul
                                if pt[0] + x_min < x_corner_min:
                                    x_corner_min = pt[0] + x_min
                                    important_roi = True
                                if pt[0] + x_min > x_corner_max:
                                    x_corner_max = pt[0] + x_min
                                    important_roi = True
                                if pt[1] + y_min < y_corner_min:
                                    y_corner_min = pt[1] + y_min
                                    important_roi = True
                                if pt[1] + y_min > y_corner_max:
                                    y_corner_max = pt[1] + y_min
                                    important_roi = True
                            if important_roi:
                                rois.append(((y_min, y_max), (x_min, x_max)))

        gt = ImLo.get_ground_truth()
        cost = np.sqrt((gt[0] - x_corner_min) ** 2 + (gt[1] - y_corner_min) ** 2) + \
               np.sqrt((gt[2] - x_corner_max) ** 2 + (gt[3] - y_corner_min) ** 2) + \
               np.sqrt((gt[4] - x_corner_max) ** 2 + (gt[5] - y_corner_max) ** 2) + \
               np.sqrt((gt[6] - x_corner_min) ** 2 + (gt[7] - y_corner_max) ** 2)
        costs = np.append(costs, [cost])

        total_tests += 1


def compute_checker(rows, cols, ratio_test, min_matches, sample_size, ImLo, FE, des_ref):
    total_tests = 0

    def main_checker_profile_rec():
        y_corner_min = img.shape[0]
        y_corner_max = 0
        x_corner_min = img.shape[1]
        x_corner_max = 0
        for row in range(0, rows):
            for col in range(0, cols):
                y_min = int(row * img.shape[0] / rows)
                y_max = int(row * img.shape[0] / rows + img.shape[0] / rows)
                x_min = int(col * img.shape[1] / cols)
                x_max = int(col * img.shape[1] / cols + img.shape[1] / cols)
                roi = img[y_min:y_max, x_min:x_max]

                FE.load_image(roi)
                FE.set_method("SIFT")
                kp = FE.get_kp()
                des = FE.get_des()

                if kp is not None:
                    if len(kp) >= 2:
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

                        search_params = dict(checks=50)  # or pass empty dictionary
                        flann = cv.FlannBasedMatcher(index_params, search_params)
                        matches = flann.knnMatch(des_ref, des, k=2)

                        # Apply ratio test
                        good = []
                        for m, n in matches:
                            if m.distance < ratio_test * n.distance:
                                good.append(m)

                        if len(good) >= min_matches:
                            pts = np.int32([kp[m.trainIdx].pt for m in good])
                            for pt in pts:
                                if pt[0] + x_min < x_corner_min:
                                    x_corner_min = pt[0] + x_min
                                if pt[0] + x_min > x_corner_max:
                                    x_corner_max = pt[0] + x_min
                                if pt[1] + y_min < y_corner_min:
                                    y_corner_min = pt[1] + y_min
                                if pt[1] + y_min > y_corner_max:
                                    y_corner_max = pt[1] + y_min

    profiler = cProfile.Profile()
    profiler.enable()
    while True:
        # Load image
        img = ImLo.load_image()
        if img is None or total_tests >= sample_size:
            print("total tests:", total_tests, )
            break

        main_checker_profile_rec()

        total_tests += 1

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def main(method=1, sample_size=50, rows=8, cols=8, ratio_test=0.60, min_matches=2):
    """

    :param method:  1 ->    Manually count the True Positive and False Positive Regions of Interest on each image.
                            Used for measuring the TP and FP rate.
                    2 ->    Gather statistical data on the cost.
                    3 ->    Measure the compute time.
    :param sample_size:     The number of images to be used.
    :param rows:    The image is divided into this many rows for RoI creation. See the method section of the report.
    :param cols:    The image is divided into this many columns for RoI creation. See the method section of the report.
    :param ratio_test:  The ratio (between 0 and 1) used in Lowe's ratio test.
                        Lower ratio only leaves the better quality matches.
    :param min_matches: Minimum number of good matches (i.e. passed Lowe's test) required in a box to be designated as RoI.
    :return:
    """
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load reference
    ref = ImLo.load_reference("reference_checker_large.png")

    FE = FeatureExtractor()

    FE.load_image(ref)
    FE.set_method("SIFT")
    des_ref = FE.get_des()

    if method == 1:
        roi_checker(rows, cols, ratio_test, min_matches, sample_size, ImLo, FE, des_ref)
    elif method == 2:
        cost_checker(rows, cols, ratio_test, min_matches, sample_size, ImLo, FE, des_ref)
    elif method == 3:
        compute_checker(rows, cols, ratio_test, min_matches, sample_size, ImLo, FE, des_ref)


if __name__ == "__main__":
    main(method=3, sample_size=10, rows=8, cols=8, ratio_test=0.60, min_matches=2)
