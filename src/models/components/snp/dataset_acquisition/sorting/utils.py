import cv2
import numpy as np
import os
import yaml
import pickle
from collections import defaultdict


class StrokesLoader:
    def __init__(self, path):
        self.path = path
        self.strokes = None
        self.layer = None

    def load_strokes(self):
        path = os.path.join(self.path, "strokes_params.npz")
        data = np.load(path)
        print(f"Loading strokes from: {path}")
        color = 0.5 * (data["x_color"][:, :, :3] + data["x_color"][:, :, 3:])
        self.strokes = np.concatenate([data["x_ctt"], color], axis=-1)
        self.layer = data["x_layer"]
        self.num_strokes = self.strokes.shape[1]

        return self.strokes, self.layer

    def _normalize(self, x, width):
        """
        Take from renderer.py
        """
        return (int)(x * (width - 1) + 0.5)

    def add_segmentation_saliency(self, seg_map, canvas_size):

        segm_info = np.zeros((1, self.num_strokes, 1))

        # Assign a class to each stroke
        for i in range(self.num_strokes):
            x0, y0 = self.strokes[0, i, :2]
            x0 = self._normalize(x0, canvas_size)
            y0 = self._normalize(y0, canvas_size)

            # TODO: only for pet dataset
            if seg_map[y0, x0] == 3:
                segm_info[0, i, 0] = 1.0
            else:
                segm_info[0, i, 0] = seg_map[y0, x0]

        return np.concatenate([self.strokes, segm_info], axis=-1)


########################################################################################################################


def load_segmentation(path, cw):
    sm = cv2.imread(path)
    sm = cv2.resize(sm, (cw, cw), interpolation=cv2.INTER_NEAREST)
    sm = sm[:, :, 0]

    return sm


def extract_salient(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    sal = saliencyMap > 0.9 * saliencyMap.mean()

    return sal


########################################################################################################################


def make_dir_tree(base_path):
    os.makedirs(base_path, exist_ok=True)

    types = ["lkh"]
    tmp = ["index"]

    for t in types:
        os.makedirs(os.path.join(base_path, t), exist_ok=True)
        for s in tmp:
            os.makedirs(os.path.join(base_path, t, s), exist_ok=True)
        if t == "lkh":
            os.makedirs(os.path.join(base_path, t, "lkh_files"), exist_ok=True)

    print("Directory Tree created")


def save_pickle(obj, path):
    path = path + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


########################################################################################################################
class LKHConfig:
    def __init__(self, default_config_path, name, num_nodes, output_path):

        # Load tempalte
        self.laod_lkh_files(default_config_path)

        # Specific parameters
        self.name = name
        self.num_nodes = num_nodes
        self.output_path = os.path.join(output_path, name)
        # add parameters
        self.add_params()

    def laod_lkh_files(self, path):
        with open(os.path.join(path, "problem.yaml"), "r") as f:
            self.problem_file = yaml.safe_load(f)
        with open(os.path.join(path, "conf.yaml"), "r") as f:
            self.conf_file = yaml.safe_load(f)

    def add_params(self):
        # Update problem file
        self.problem_file["NAME"] = "{}.sop".format(self.name)
        self.problem_file["DIMENSION"] = self.num_nodes
        self.problem_file["EDGE_WEIGHT_SECTION"] = self.num_nodes

        # Update config file
        self.conf_file["PROBLEM_FILE"] = os.path.join(self.output_path + ".sop")
        self.conf_file["TOUR_FILE"] = os.path.join(self.output_path + "_solution.txt")

    def parse_files(self, cost_matrix):
        # Write Problem file
        with open(os.path.join(self.output_path + ".sop"), "w") as f:
            f.write(f"NAME:{self.problem_file['NAME']}\n")
            f.write(f"TYPE:{self.problem_file['TYPE']}\n")
            f.write(f"COMMENT:{self.problem_file['COMMENT']}\n")
            f.write(f"DIMENSION:{self.problem_file['DIMENSION']}\n")
            f.write(f"EDGE_WEIGHT_TYPE:{self.problem_file['EDGE_WEIGHT_TYPE']}\n")
            f.write(f"EDGE_WEIGHT_FORMAT:{self.problem_file['EDGE_WEIGHT_FORMAT']}\n")
            f.write(f"EDGE_WEIGHT_SECTION\n{self.problem_file['EDGE_WEIGHT_SECTION']}\n")

            # save the weight matrix
            np.savetxt(f, cost_matrix, delimiter="\t", fmt="%d")
            f.write("\nEOF")

        # Write configuration file
        self.conf_file_path = os.path.join(self.output_path + ".par")
        with open(self.conf_file_path, "w") as f:
            if self.conf_file["SPECIAL"]:
                f.write("SPECIAL\n")

            f.write(f"PROBLEM_FILE = {self.conf_file['PROBLEM_FILE']}\n")
            f.write(f"TOUR_FILE = {self.conf_file['TOUR_FILE']}\n")
            f.write(f"TIME_LIMIT = {self.conf_file['TIME_LIMIT']}\n")
            f.write(f"RUNS = {self.conf_file['RUNS']}")


########################################################################################################################


def check_tour(graph, tour):
    # Check
    all_good = True
    out = ""
    for i in range(len(tour)):
        curr = tour[i]
        following = tour[i + 1 :]
        for x in following:
            if curr in graph.adjlist[x]:
                out += f"Node: {x} should be before: {curr}\n"
    if all_good:
        out += "Precedence Constraints are all satisfied\n"

    i = tour[0]
    cost = 0
    for j in tour[1:]:
        cost += graph.compute_metric(i, j)
        i = j

    out += f"Total cost: {cost}"

    print(out)
    return out


########################################################################################################################
def lkh_cost_matrix(graph, start):
    scale = 1000
    C = np.zeros((graph.n_nodes + 1, graph.n_nodes + 1))

    # Populate the matrix
    for i in range(graph.n_nodes):
        for j in range(graph.n_nodes):
            # Keep the convention of the documentation
            if i == j:
                C[j, i] = 0
            else:
                C[j, i] = graph.compute_metric(i, j)

    # scale
    C = C * scale
    C[0, -1] = 32767
    C = np.rint(C).astype("int16")

    # Other requirements by lkh solver
    C[1:, 0] = -1  # First node comes before all the others
    C[-1, :-1] = -1  # Last node comes after all the others

    for i, adj_i in graph.adjlist.items():
        for j in adj_i:
            C[j, i] = -1

    return C
