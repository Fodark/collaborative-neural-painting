from renderer import Renderer
import torch
import pickle as pkl
from dataset_acquisition.sorting.graph import GraphBuilder
import sys
from tqdm import trange

renderer = Renderer(canvas_size=(256, 256), half_precision=False)

pkl_file = "data/iters_500_gird_6_N_500_clip_0.5/Dog/3b54048dca9ce76b.pkl"
# open pkl_file
strokes = pkl.load(open(pkl_file, "rb"))
strokes = torch.from_numpy(strokes)
# print(strokes.shape)
# sys.exit(0)

_, alphas = renderer.render_single_strokes(strokes)
alphas = alphas.squeeze(1)
# alphas to numpy
alphas = alphas.numpy()
# print(alphas.shape)

print("building graph...")
gb = GraphBuilder(alphas, 0)
gb.build_graph()
adj_list = gb.get_adjlist(hidden=True)
# print(adj_list)

# print(adj_list[0])

import numpy as np

# go from adj_list to adj_matrix
adj_matrix = np.zeros((len(adj_list), len(adj_list)), dtype=bool)
for k, elem in adj_list.items():
    for e in elem:
        adj_matrix[k][e] = True

# print(adj_matrix[:20][:20])
order = []
# get indexes of columns with 0 True values

print("sorting...")
import math

# repeat until no more columns are present
for _ in trange(len(adj_matrix)):
    if len(order) == len(adj_matrix):
        break
    # get indexes of columns with 0 True values
    idxs = np.where(adj_matrix.sum(axis=0) == 0)[0]
    idxs = [idx for idx in idxs if idx not in order]
    # print("choices:", idxs)

    best_idx, lower_x, lower_y = None, 2.0, 2.0
    for _idx in idxs:
        x, y, *_ = strokes[_idx]
        if y < lower_y:  # " or y < lower_y:
            best_idx = _idx
            lower_x, lower_y = x, y
        elif math.isclose(y, lower_y) and x < lower_x:
            best_idx = _idx
            lower_x, lower_y = x, y

    # print(idxs)
    idx = best_idx  # idxs[0]
    # print("picked:", idx)
    order.append(idx)
    # idx = idxs[0]  # later based on coordinates
    # add column with 0 True values to order
    # order.append(idx)
    # set to 0 row and column at index idx
    adj_matrix[:, idx] = False
    adj_matrix[idx, :] = False
# print('------')
# print("order:")
# print(order)

# reorder strokes according to order
# strokes = strokes[order]
print(strokes.shape)
strokes = [strokes[idx] for idx in order]
# from list to tensor
strokes = torch.stack(strokes)
print(strokes.shape)
# strokes = torch.tensor()

print("rendering...")

import cv2

for i in trange(1, len(strokes)):
    image = renderer.draw_on_canvas(strokes[:i].unsqueeze(0))
    image = image.squeeze(0).numpy() * 255
    image = image.transpose(1, 2, 0)
    # rgb to bgr
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ind = f"{i}".zfill(4)
    # save image with cv2 in "test_output" folder
    cv2.imwrite(f"test_output/{ind}.png", image)


# image = renderer.draw_on_canvas(strokes.unsqueeze(0))
# # show image
# import matplotlib.pyplot as plt
# image = image.squeeze(0).numpy()
# # # rearrange image
# image = image.transpose(1, 2, 0)
# # # rgb to bgr
# plt.imshow(image)
# plt.show()


# load images from "test_output" folder and make a gif
import imageio
import os

images = []
for filename in sorted(os.listdir("test_output")):
    if filename.endswith(".png"):
        images.append(imageio.imread(f"test_output/{filename}"))
imageio.mimsave("test.gif", images, duration=0.05)
