from ImageLoading import ImageLoader
from FeatureExtractor import FeatureExtractor
import itertools
import cv2 as cv
import numpy as np
import multiprocessing
import cProfile
import pstats
import copyreg

# https://stackoverflow.com/questions/50337569/pickle-exception-for-cv2-boost-when-using-multiprocessing/50394788#50394788
def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
    return cv.KeyPoint, (
        keypoint.pt[0],
        keypoint.pt[1],
        keypoint.size,
        keypoint.angle,
        keypoint.response,
        keypoint.octave,
        keypoint.class_id,
    )
copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoint)

def single_process(FE, args, img, des_ref):
    boxes = args
    # print("fe", FE)
    # print("args", args)
    for case in boxes:
        row = case[0]
        col = case[1]
        y_min = int(row * img.shape[0] / rows)
        y_max = int(row * img.shape[0] / rows + img.shape[0] / rows)
        x_min = int(col * img.shape[1] / cols)
        x_max = int(col * img.shape[1] / cols + img.shape[1] / cols)
        roi = img[y_min:y_max, x_min:x_max]

        FE.load_image(roi)
        FE.set_method("SIFT")
        # kp_des_lst = FE.get_mp_kp_des()

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
                    for pt_i in range(len(pts)):
                        pts[pt_i][0] += x_min
                        pts[pt_i][1] += y_min
                    return pts

ratio_test = 0.60
min_matches = 5

threads = 4

rows = 8
cols = 8

if __name__ == "__main__":

    # Create randomized train and test sets
    ImLo = ImageLoader()
    ImLo.create_random_set(train_name="train.pickle", test_name="test.pickle")
    # Load reference
    ref = ImLo.load_reference("reference_checker_large.png")

    FE = FeatureExtractor()

    FE.load_image(ref)
    FE.set_method("SIFT")
    des_ref = FE.get_des()

    total_tests = 0

    profiler = cProfile.Profile()
    profiler.enable()
    while True:
        # Load image
        img = ImLo.load_image()
        if img is None or total_tests >= 50:
            print("total tests:", total_tests, )
            break

        y_corner_min = img.shape[0]
        y_corner_max = 0
        x_corner_min = img.shape[1]
        x_corner_max = 0

        block_args = list(itertools.product(range(0, rows), range(0, cols)))

        # print(list(itertools.product([FE], block_args)))
        process_results = []
        queue = multiprocessing.Queue()

        # pool = multiprocessing.Pool(2)
        # process_results = pool.map(single_process, list(itertools.product([FE], block_args)))


        processes = []
        for process_idx in range(threads):
            process = multiprocessing.Process(target=single_process,
                                              args=(FE, block_args[int(len(block_args) / threads):], img, des_ref))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        while queue.empty() is False:
            process_results.append(queue.get())

        for pts in process_results:
            for pt in pts:
                # Elso x, masodik y, 0,0 bal alul

                if pt[0] < x_corner_min:
                    x_corner_min = pt[0]
                if pt[0] > x_corner_max:
                    x_corner_max = pt[0]
                if pt[1] < y_corner_min:
                    y_corner_min = pt[1]
                if pt[1] > y_corner_max:
                    y_corner_max = pt[1]

        total_tests += 1
        print(total_tests)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

