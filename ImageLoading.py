from pathlib import Path
import numpy as np
import pickle
import cv2 as cv


class ImageLoader:
    def __init__(self):
        self.folder_path = Path.cwd() / "WashingtonOBRace"
        assert Path.is_dir(self.folder_path)
        self._total_len = 690

        self.working_set = None
        self.img_count = None

    def create_random_set(self, train_size=None, train_name="train.pickle", test_size=None, test_name="test.pickle"):
        # https://en.wikipedia.org/wiki/Pareto_principle
        if train_size is None:
            train_size = int(self._total_len * 0.8)
            print("Train set size is set to", train_size)
        if test_size is None:
            test_size = int(self._total_len * 0.2)
            print("Test set size is set to", test_size)
        assert train_size+test_size <= self._total_len, f"The sum of train size ({train_size}), " \
                                                        f"and test size ({test_size}) cannot exceed 690."
        all_corners = np.genfromtxt(self.folder_path / "corners.csv", delimiter=",", dtype=None, encoding="utf8")
        np.random.shuffle(all_corners)
        train_set, test_set = all_corners[:train_size], all_corners[-test_size:]

        with open(train_name, "wb") as out:
            pickle.dump(train_set, out)
        with open(test_name, "wb") as out:
            pickle.dump(test_set, out)
        self.working_set = train_set

    def load_set(self, name="train.pickle"):
        with open(name, "rb") as file:
            self.working_set = pickle.load(file)

    def load_image(self):
        if self.img_count is None:
            self.img_count = 0
        elif self.img_count + 1 < len(self.working_set):
            self.img_count += 1
        else:
            print("Set finished")
            return None
        img_path = self.folder_path.joinpath(self.working_set[self.img_count][0])
        assert img_path.is_file()
        img = cv.imread(img_path.__str__())
        return img

    def load_reference(self, name="reference2.png"):
        img_path = self.folder_path.joinpath(name)
        assert img_path.is_file()
        img = cv.imread(img_path.__str__())
        return img

    def get_ground_truth(self):
        csv = np.genfromtxt(self.folder_path.joinpath("corners1.csv"), delimiter=',', dtype=None)

        idx = 0
        gates = []
        gate_found = False
        while True:
            this_name = csv[idx][0]
            if this_name.decode('UTF-8') == self.working_set[self.img_count][0]:
                gate_found = True
                gates.append([csv[idx][1], csv[idx][2], csv[idx][3],
                              csv[idx][4], csv[idx][5], csv[idx][6],
                              csv[idx][7], csv[idx][8], csv[idx][14]])
            else:
                if gate_found:
                    break
            idx += 1
            if idx >= 690:
                break

        max_size = 0
        max_index = None
        for idx in range(len(gates)):
            if gates[idx][-1] > max_size:
                max_size = gates[idx][-1]
                max_index = idx

        return gates[max_index]
        # print(csv[:, 0])
        # idx = np.argmax(csv[] == self.working_set[self.img_count][0])
