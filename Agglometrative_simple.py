import scipy
import pandas as pd
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os

from natsort import natsorted
from scipy.linalg import norm
from PIL import Image, ImageDraw
import plotly.express as px
import json
from urllib.request import urlopen






def argmin(x):
    return divmod(np.nanargmin(x), x.shape[1])


def nansum(a, b):
    result = np.where(
    np.isnan(a+b),
    np.where(np.isnan(a), b, a),
    a+b)
    return result

def nanmean(a, b):
    result = np.nanmean(np.stack((a, b)), axis=0)
    return result

def nanprod(a, b):
    result = np.nanprod(np.stack((a, b)), axis=0)
    return result

def dis_mat_sample():
    np.set_printoptions(edgeitems=3, linewidth=150,
                        formatter=dict(float=lambda x: "%.3g" % x))

    # dis_mat = np.array([[0, 3, 5, 3, 0, 0, 3, 3, 4, 4],
    #                     [3, 0, 2, 6, 3, 3, 6, 0, 1, 1],
    #                     [5, 2, 0, 8, 5, 5, 8, 2, 1, 1],
    #                     [3, 6, 8, 0, 3, 3, 0, 6, 7, 7],
    #                     [0, 3, 5, 3, 0, 0, 3, 3, 4, 4],
    #                     [0, 3, 5, 3, 0, 0, 3, 3, 4, 4],
    #                     [3, 6, 8, 0, 3, 3, 0, 6, 7, 7],
    #                     [3, 0, 2, 6, 3, 3, 6, 0, 1, 1],
    #                     [4, 1, 1, 7, 4, 4, 7, 1, 0, 0],
    #                     [4, 1, 1, 7, 4, 4, 7, 1, 0, 0]]).astype(float)
    dis_mat = np.array([[-1, 3, 5, 3, 0, 0, 3, 3, 4, 4],   # -1 means no connection
                        [3, -1, 2, 6, 3, 3, 6, 0, 1, 1],
                        [5, -1, -1, 8, 5, 5, 8, -1, -1, -1],
                        [3, 6, 8, -1, 3, 3, 0, 6, 7, 7],
                        [0, 3, 5, 3, -1, 0, 3, 3, 4, 4],
                        [0, 3, 5, 3, 0, -1, 3, 3, 4, 4],
                        [3, 6, 8, 0, 3, 3, -1, 6, 7, 7],
                        [3, 0, 2, 6, 3, 3, 6, -1, 1, 1],
                        [4, 1, 1, 7, 4, 4, 7, 1, -1, 0],
                        [4, 1, 1, 7, 4, 4, 7, 1, 0, -1]]).astype(float)
    # mask = np.eye(len(dis_mat), len(dis_mat))

    # masked_dis = np.ma.masked_array(dis_mat, mask=mask)
    masked_dis = dis_mat
    return masked_dis


def make_list(obj):
    if isinstance(obj, list):
        return obj
    return [obj]

def make_list(obj):
    if isinstance(obj, list):
        return obj
    return [obj]

def traverse_node(label, node, index_labels):
    if node.is_terminal():
        index_labels.append((node.index, label))
        print("node.index, label:", index_labels[-1][0], label)
    else:
        traverse_node(label, node.left, index_labels)
        traverse_node(label, node.right, index_labels)

def extract_labels(nodes):

    cleaned_nodes = []
    for node in nodes:
        if isinstance(node, Node):
            cleaned_nodes.append(node)
    index_labels = []

    for idx, node in enumerate(cleaned_nodes):
        label = idx
        node_index = []
        traverse_node(label, node, index_labels)
    return index_labels


class Node(object):
    def __init__(self, fea=0, left=None, right=None, children_dist=None, index=None):

        self.__fea = fea
        self.left = left
        self.right = right
        self.children_dist = children_dist

        self.depth = self.__calc_depth()
        self.height = self.__calc_height()
        self.value = fea
        self.index = index

        # use Group Average to calculate distance

    def distance(self, other):
        return abs(self.__fea - other.__fea)

    def is_terminal(self):
        return self.left is None and self.right is None

    def __calc_depth(self):
        if self.is_terminal():
            return 0
        return max(self.left.depth, self.right.depth) + self.children_dist

    def __calc_height(self):
        if self.is_terminal():
            return 1
        return max(self.left.height, self.right.height) + 1

    def merge(self, other, distance):
        return Node((self.__fea + other.__fea) / 2,
                    self, other, children_dist=distance)


def sample_AgglomerativeClustering(dis_mat, conn_mat, n_cluster_list):
    if not isinstance(n_cluster_list, list):
        n_cluster_list = [n_cluster_list]
    n_cluster_list = sorted(n_cluster_list, reverse=True)

    # initial nodes
    nodes = []
    new_nodes = []
    for i in range(len(dis_mat)):
        nodes.append(Node(index=i))
        new_nodes.append(Node(index=i))

    layers = [[] for i in range(len(dis_mat))]

    node_cnt_mat = conn_mat.copy().astype(float)
    node_cnt_mat[node_cnt_mat == 0] = np.nan
    np.fill_diagonal(node_cnt_mat, np.nan)

    class_cnt = len(dis_mat)

    eliminated_eles = []
    labels_all_rounds = []
    reported_cnt = 0
    while class_cnt > 1:
        new_node = merge_minpair(dis_mat, nodes, node_cnt_mat, eliminated_eles)
        class_cnt -= 1
        #         print()

        if class_cnt in n_cluster_list:
            labels_a_round = []
            index_labels = extract_labels(nodes)
            labels_a_round = labels_a_round + index_labels
            reported_cnt += 1
            labels_a_round = sorted(labels_a_round, key=lambda x: x[0])
            labels_all_rounds.append(labels_a_round)
        if reported_cnt == len(n_cluster_list):
            break

    return labels_all_rounds

# def set_eye_nan(a):


def merge_minpair(dis_mat, nodes, node_cnt_mat, eliminated_eles):
    #     print("len(nodes):", len(nodes))
    min_pair = argmin(dis_mat)
    min_distance = np.nanmin(dis_mat)

    eliminated_idx = max(min_pair)

    eliminated_eles.append(eliminated_idx)

    eliminated_ele = dis_mat[eliminated_idx].copy()
    #     print("eliminated_ele:", eliminated_ele)
    new_idx = min(min_pair)

    #     print("scores:", features[eliminated_idx], features[new_idx])
    new_node_cnt_row = nansum(node_cnt_mat[new_idx], node_cnt_mat[eliminated_idx])
    # row1 = nanprod(dis_mat[new_idx], node_cnt_mat[new_idx])
    row1 = dis_mat[new_idx]  * node_cnt_mat[new_idx]
    # row2 = nanprod(dis_mat[eliminated_idx], node_cnt_mat[eliminated_idx])
    row2 = dis_mat[eliminated_idx] * node_cnt_mat[eliminated_idx]
    new_row = nansum(row1, row2)

    new_row = new_row / new_node_cnt_row
    dis_mat[new_idx] = new_row
    dis_mat[:, new_idx] = new_row.T
    node_cnt_mat[new_idx] = new_node_cnt_row
    node_cnt_mat[:, new_idx] = new_node_cnt_row.T
    # conn_mat[new_idx] = nanmean(dis_mat[new_idx], dis_mat[eliminated_idx])

    dis_mat[eliminated_idx] = np.nan
    dis_mat[:, eliminated_idx] = np.nan
    node_cnt_mat[eliminated_idx] = np.nan
    node_cnt_mat[:, eliminated_idx] = np.nan

    np.fill_diagonal(dis_mat, np.nan)
    np.fill_diagonal(node_cnt_mat, np.nan)

    node1 = nodes[eliminated_idx]
    node2 = nodes[new_idx]

    new_node = node1.merge(node2, min_distance)
    nodes[new_idx] = new_node
    nodes[eliminated_idx] = None

    # print(dis_mat)
    # print()

    class_cnt = len(dis_mat) - len(eliminated_eles) + 1
    if class_cnt % 1 == 0:
        # print("min_pair:", min_pair)
        print("current class_cnt, min_pair, eliminated_idx, new_idx, min_distance:", class_cnt, min_pair, eliminated_idx, new_idx, min_distance)
        # print("Current class_cnt: ", class_cnt)
    #         print()

    return new_node


def get_world_matrix():
    dis_mat = world_matrix.copy()
    # dis_mat = 1000 - dis_mat
    mask = np.eye(len(dis_mat), len(dis_mat))

    masked_dis = np.ma.masked_array(dis_mat, mask=mask)
    return masked_dis

def list_to_file(a_list, file_path, header=None):
    with open(file_path, 'w') as f:
        if header:
            f.write(str(header) + '\n')
        for i in a_list:
            line = ",".join([str(x) for x in i]) + '\n'
            f.write(line)

def get_sample_conn_mat():
    conn_mat = np.array([[0,1,1,1,1,1,1,1,1,1],
                         [1,0,1,1,1,1,1,1,1,1],
                         [1,0,0,1,1,1,1,0,0,0],
                         [1,1,1,0,1,1,1,1,1,1],
                         [1,1,1,1,0,1,1,1,1,1],
                         [1,1,1,1,1,0,1,1,1,1],
                         [1,1,1,1,1,1,0,1,1,1],
                         [1,1,1,1,1,1,1,0,1,1],
                         [1,1,1,1,1,1,1,1,0,1],
                         [1,1,1,1,1,1,1,1,1,0]])
    return conn_mat

if __name__ == "__main__":
    country_list = pd.read_csv(
        r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\LA_tract_list.txt',
        header=None, names=['country'])
    gby_list = country_list['country'].tolist()

    saved_file_world = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\LA_matrix.csv.npz'


    world_matrix = scipy.sparse.load_npz(saved_file_world).todense()
    world_matrix = np.array(world_matrix)
    for i in range(len(world_matrix)):
        world_matrix[i, i] = 0

    dis_mat = world_matrix.max() + 1 - world_matrix

    dis_mat[np.where(dis_mat > world_matrix.max())] = np.nan
    np.fill_diagonal(dis_mat, np.nan)

    conn_mat = np.where(world_matrix > 0, 1, 0)
    # print(conn_mat)
    # dis_mat.mask = 1-conn_mat
    n_cluster_list = [5, 10, 20, 40, 60]
    n_cluster_list = sorted(n_cluster_list, reverse=True)
    # n_cluster_list = [2770, 2760]

    # print(1-conn_mat)
    saved_path = os.getcwd()
    dis_mat[dis_mat == -1] = np.nan
    print(dis_mat)
    labels_all_rounds = sample_AgglomerativeClustering(dis_mat, conn_mat, n_cluster_list)
    for i, r in enumerate(labels_all_rounds):
        print(f"Clustering results for {str(n_cluster_list[i])} classes (node index, label):")
        print(r)
        new_name = os.path.join(saved_path, "cluster_" + str(n_cluster_list[i]) + ".csv")
        list_to_file(r, new_name, header="node_index,class")



