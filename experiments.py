from ImageLoading import ImageLoader
from HarrisCorner import Harris
from ShiTomasiCorner import ShiTomas
from FeatureExtractor import FeatureExtractor
import cv2 as cv
import numpy as np
import cProfile
import pstats


def main_tracker(corner_type):
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()

    # Harris corner detection. Trackbar is used for tuning
    if corner_type == "Harris":
        HarCor = Harris()
        HarCor.load_image(img)
        HarCor.trackbar(img)

    # Shi-Tomasi corner detection. Trackbar is used for tuning
    if corner_type == "ShiTomasi":
        ShiTomCor = ShiTomas()
        ShiTomCor.load_image(img)
        ShiTomCor.trackbar(img)

    if corner_type == "SIFT":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        out = img.copy()
        out = cv.drawKeypoints(gray, kp, out, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow("SIFT keypoints", out)
        cv.waitKey()

    if corner_type == "SURF":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        surf = cv.xfeatures2d.SURF_create(50000)
        surf.setExtended(True)
        kp, des = surf.detectAndCompute(gray, None)
        out = img.copy()
        out = cv.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
        cv.imshow("SURF keypoints", out)
        cv.waitKey()

    if corner_type == "USURF":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        surf = cv.xfeatures2d.SURF_create(50000)
        surf.setUpright(True)
        surf.setExtended(True)
        kp, des = surf.detectAndCompute(gray, None)
        out = img.copy()
        out = cv.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
        cv.imshow("SURF keypoints", out)
        cv.waitKey()

    if corner_type == "FAST":
        pass

    if corner_type == "BRIEF":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Initiate FAST detector
        star = cv.xfeatures2d.StarDetector_create()
        # star = cv.SIFT_create()
        # Initiate BRIEF extractor
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        # find the keypoints with STAR
        kp = star.detect(gray, None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(gray, kp)
        print(brief.descriptorSize())
        print(des.shape)
        out = img.copy()
        out = cv.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
        cv.imshow("BRIEF keypoints", out)
        cv.waitKey()
        pass

    if corner_type == "ORB":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(gray, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(gray, kp)
        # draw only keypoints location,not size and orientation
        out = cv.drawKeypoints(gray, kp, None, color=(0, 255, 0), flags=0)
        cv.imshow("ORB keypoints", out)
        cv.waitKey()
        pass


def main_bf_orb():
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()
    ref = ImLo.load_reference()

    FEref = FeatureExtractor()
    FEref.load_image(ref)
    FEref.set_method("ORB")
    kp_ref = FEref.get_kp()
    des_ref = FEref.get_des()

    FE = FeatureExtractor()
    FE.load_image(img)
    FE.set_method("ORB")
    kp = FE.get_kp()
    des = FE.get_des()

    BF = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = BF.match(des_ref, des)
    matches = sorted(matches, key=lambda x: x.distance)
    print(matches)
    print(type(matches))

    # Draw first 10 matches.
    img3 = cv.drawMatches(ref, kp_ref, img, kp, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("matches", img3)
    cv.waitKey()

def main_bf_sift():
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()
    ref = ImLo.load_reference()

    FEref = FeatureExtractor()
    FEref.load_image(ref)
    FEref.set_method("SIFT")
    kp_ref = FEref.get_kp()
    des_ref = FEref.get_des()

    FE = FeatureExtractor()
    FE.load_image(img)
    FE.set_method("SIFT")
    kp = FE.get_kp()
    des = FE.get_des()

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_ref, des, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print(len(good))
    # Draw first 10 matches.
    img3 = cv.drawMatchesKnn(ref, kp_ref, img, kp, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("matches", img3)
    cv.waitKey()

def main_flann_sift():
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()
    ref = ImLo.load_reference()

    FEref = FeatureExtractor()
    FEref.load_image(ref)
    FEref.set_method("SIFT")
    kp_ref = FEref.get_kp()
    des_ref = FEref.get_des()

    FE = FeatureExtractor()
    FE.load_image(img)
    FE.set_method("SIFT")
    kp = FE.get_kp()
    des = FE.get_des()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ref, des, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(ref, kp_ref, img, kp, matches, None, **draw_params)

    cv.imshow("matches", img3)
    cv.waitKey()

def main_flann_sift_homo():
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()
    ref = ImLo.load_reference()

    FEref = FeatureExtractor()
    FEref.load_image(ref)
    FEref.set_method("SIFT")
    kp_ref = FEref.get_kp()
    des_ref = FEref.get_des()

    FE = FeatureExtractor()
    FE.load_image(img)
    FE.set_method("SIFT")
    kp = FE.get_kp()
    des = FE.get_des()

    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ref, des, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)
        matchesMask = mask.ravel().tolist()
        h, w, d = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img = cv.polylines(img, [np.int32(dst)], True, [0, 0, 255], 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(ref, kp_ref, img, kp, good, None, **draw_params)
    cv.imshow("matches", img3)
    cv.waitKey()

def AIIR_search():
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()
    ref = ImLo.load_reference("reference_logo.png")

    FEref = FeatureExtractor()
    FEref.load_image(ref)
    FEref.set_method("SIFT")
    kp_ref = FEref.get_kp()
    des_ref = FEref.get_des()

    FE = FeatureExtractor()
    FE.load_image(img)
    FE.set_method("SIFT")
    kp = FE.get_kp()
    des = FE.get_des()

    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ref, des, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    MIN_MATCH_COUNT = 5
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)
        matchesMask = mask.ravel().tolist()
        h, w, d = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img = cv.polylines(img, [np.int32(dst)], True, [0, 0, 255], 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(ref, kp_ref, img, kp, good, None, **draw_params)
    cv.imshow("matches", img3)
    cv.waitKey()

def find_logo_in_roi(rows=8, cols=8):
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")
    # Load image
    img = ImLo.load_image()
    ref = ImLo.load_reference("reference_checker_large.png")

    FE = FeatureExtractor()

    FE.load_image(ref)
    FE.set_method("SIFT")
    kp_ref = FE.get_kp()
    des_ref = FE.get_des()

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

            # cv.imshow("roi", roi)
            # cv.waitKey()

            FE.load_image(roi)
            FE.set_method("SIFT")
            kp = FE.get_kp()
            des = FE.get_des()

            if kp is not None:
                if len(kp) >= 2:
                    print(len(kp))
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

                    search_params = dict(checks=50)  # or pass empty dictionary
                    flann = cv.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des_ref, des, k=2)


                    # Apply ratio test
                    good = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good.append(m)

                    if len(good) >= 4:
                        print("asd", np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2))
                        print(np.float32([kp[m.trainIdx].pt for m in good]))
                        pts = np.int32([kp[m.trainIdx].pt for m in good])
                        for pt in pts:
                            # Elso x, masodik y, 0,0 bal alul
                            if pt[0] + x_min < x_corner_min:
                                x_corner_min = pt[0] + x_min
                            if pt[0] + x_min > x_corner_max:
                                x_corner_max = pt[0] + x_min
                            if pt[1] + y_min < y_corner_min:
                                y_corner_min = pt[1] + y_min
                            if pt[1] + y_min > y_corner_max:
                                y_corner_max = pt[1] + y_min
                        rois.append(((y_min, y_max), (x_min, x_max)))
                        img3 = cv.drawMatches(ref, kp_ref, roi, kp, good, None, matchColor=(0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS )
                        # cv.imshow("matches", img3)
                        # cv.waitKey()
                        stitched[y_min:y_max, x_min:x_max] = roi

    cv.imshow("done", stitched)
    cv.circle(img, (x_corner_min, y_corner_min), radius=4, color=(0, 0, 255), thickness=-1)
    cv.circle(img, (x_corner_min, y_corner_max), radius=4, color=(0, 0, 255), thickness=-1)
    cv.circle(img, (x_corner_max, y_corner_max), radius=4, color=(0, 0, 255), thickness=-1)
    cv.circle(img, (x_corner_max, y_corner_min), radius=4, color=(0, 0, 255), thickness=-1)
    cv.imshow("rect", img)
    print(x_corner_min, y_corner_min, x_corner_max, y_corner_max)
    cv.waitKey()


def main_full_side(rows=8, cols=8, ratio_test=0.65, min_matches=10):
    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load image set
    ImLo.load_set("train.pickle")

    FE = FeatureExtractor()

    # Load reference
    ref = ImLo.load_reference("reference_bot.png")
    FE.load_image(ref)
    FE.set_method("SIFT")
    kp_ref_bot = FE.get_kp()
    des_ref_bot = FE.get_des()

    ref = ImLo.load_reference("reference_left.png")
    FE.load_image(ref)
    FE.set_method("SIFT")
    kp_ref_left = FE.get_kp()
    des_ref_left = FE.get_des()

    total_good_corners = 0
    total_tests = 0
    while True:
        # Load image
        img = ImLo.load_image()
        if img is None or total_tests>=20:
            print(total_tests, total_good_corners)
            print("Detection rate:", total_good_corners / (total_tests*4))
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

                # cv.imshow("roi", roi)
                # cv.waitKey()

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
                        matches_bot = flann.knnMatch(des_ref_bot, des, k=2)
                        matches_left = flann.knnMatch(des_ref_left, des, k=2)


                        # Apply ratio test
                        good = []
                        for m, n in matches_bot:
                            if m.distance < ratio_test * n.distance:
                                good.append(m)
                        for m, n in matches_left:
                            if m.distance < ratio_test * n.distance:
                                good.append(m)
                        print(len(good))

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
            cv.imshow("done", stitched)
            cv.circle(img, (x_corner_min, y_corner_min), radius=4, color=(0, 0, 255), thickness=-1)
            cv.circle(img, (x_corner_min, y_corner_max), radius=4, color=(0, 0, 255), thickness=-1)
            cv.circle(img, (x_corner_max, y_corner_max), radius=4, color=(0, 0, 255), thickness=-1)
            cv.circle(img, (x_corner_max, y_corner_min), radius=4, color=(0, 0, 255), thickness=-1)
            cv.imshow("rect", img)
            cv.waitKey()
            print(x_corner_min, y_corner_min, x_corner_max, y_corner_max)
            total_good_corners += int(input("How many good corners? "))
        total_tests += 1
