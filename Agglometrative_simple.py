import scipy
import pandas as pd
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import numba as nb
from numba import jit
from natsort import natsorted
import os
import scipy

from natsort import natsorted
from scipy.linalg import norm
from PIL import Image, ImageDraw
import plotly.express as px
import json
from urllib.request import urlopen



@jit(nopython=True)
def count_not_all_nan_rows(a):
    cnt = len(a)
    #     print(cnt)
    for row in a:
        #         print(row)
        tags = np.ones(len(a))

        for i in range(len(a)):
            #             print(i)
            #             print(np.isnan(row[i]))
            if np.isnan(row[i]):
                #                 print(tags)
                tags[i] = 0

        if tags.sum() == 0:
            cnt -= 1
    return cnt

def argmin(x):
    return divmod(np.nanargmin(x), x.shape[1])

@jit(nopython=True)
def nansum(a, b):
    result = np.where(
    np.isnan(a+b),
    np.where(np.isnan(a), b, a),
    a+b)
    return result

@jit(nopython=True)
def nanmean(a, b):
    result = np.nanmean(np.stack((a, b)), axis=0)
    return result

@jit(nopython=True)
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
    '''
    Conduct agglomerative clustering. Reference: UPGMA (unweighted pair group method with arithmetic mean), https://en.wikipedia.org/wiki/UPGMA.
    :param dis_mat: precomputed distance matrix
    :param conn_mat: connectivity matrix
    :param n_cluster_list: a list of number of clusters
    :return: a list with cluster results, each element is a list of tuple (node.index, label).
    '''
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

    # class_cnt = len(dis_mat)
    class_cnt = count_not_all_nan_rows(dis_mat)

    eliminated_eles = []
    labels_all_rounds = []
    reported_cnt = 0
    while class_cnt > 1:
        new_node = merge_minpair(dis_mat, nodes, node_cnt_mat, eliminated_eles)
        # class_cnt -= 1
        class_cnt = count_not_all_nan_rows(dis_mat)

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

@jit(nopython=True)
def matrix_oper(dis_mat, node_cnt_mat, eliminated_idx, new_idx):
    #     print("scores:", features[eliminated_idx], features[new_idx])
    new_node_cnt_row = nansum(node_cnt_mat[new_idx], node_cnt_mat[eliminated_idx])
    # row1 = nanprod(dis_mat[new_idx], node_cnt_mat[new_idx])
    row1 = dis_mat[new_idx] * node_cnt_mat[new_idx]
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

# @jit(nopython=True)
def merge_minpair(dis_mat, nodes, node_cnt_mat, eliminated_eles):
    #     print("len(nodes):", len(nodes))
    min_pair = argmin(dis_mat)
    min_distance = np.nanmin(dis_mat)

    eliminated_idx = max(min_pair)

    eliminated_eles.append(eliminated_idx)

    eliminated_ele = dis_mat[eliminated_idx].copy()
    #     print("eliminated_ele:", eliminated_ele)
    new_idx = min(min_pair)

    # #     print("scores:", features[eliminated_idx], features[new_idx])
    # new_node_cnt_row = nansum(node_cnt_mat[new_idx], node_cnt_mat[eliminated_idx])
    # # row1 = nanprod(dis_mat[new_idx], node_cnt_mat[new_idx])
    # row1 = dis_mat[new_idx]  * node_cnt_mat[new_idx]
    # # row2 = nanprod(dis_mat[eliminated_idx], node_cnt_mat[eliminated_idx])
    # row2 = dis_mat[eliminated_idx] * node_cnt_mat[eliminated_idx]
    # new_row = nansum(row1, row2)
    #
    # new_row = new_row / new_node_cnt_row
    # dis_mat[new_idx] = new_row
    # dis_mat[:, new_idx] = new_row.T
    # node_cnt_mat[new_idx] = new_node_cnt_row
    # node_cnt_mat[:, new_idx] = new_node_cnt_row.T
    # # conn_mat[new_idx] = nanmean(dis_mat[new_idx], dis_mat[eliminated_idx])
    #
    # dis_mat[eliminated_idx] = np.nan
    # dis_mat[:, eliminated_idx] = np.nan
    # node_cnt_mat[eliminated_idx] = np.nan
    # node_cnt_mat[:, eliminated_idx] = np.nan
    #
    # np.fill_diagonal(dis_mat, np.nan)
    # np.fill_diagonal(node_cnt_mat, np.nan)

    matrix_oper(dis_mat, node_cnt_mat, eliminated_idx, new_idx)

    node1 = nodes[eliminated_idx]
    node2 = nodes[new_idx]

    new_node = node1.merge(node2, min_distance)
    nodes[new_idx] = new_node
    nodes[eliminated_idx] = None

    # print(dis_mat)
    # print()

    class_cnt0 = len(dis_mat) - len(eliminated_eles) + 1
    class_cnt = count_not_all_nan_rows(dis_mat)
    if class_cnt % 1 == 0:
        # print("min_pair:", min_pair)
        print("current class_cnt, min_pair, eliminated_idx, new_idx, min_distance:", class_cnt0, class_cnt, min_pair, eliminated_idx, new_idx, min_distance)
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
        # for i in a_list:
        line = "\n".join([str(x) for x in a_list])
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


def csv_to_matrix(csv_path):
    df = pd.read_csv(csv_path, dtype={"place_i": str, 'place_j': str})
    df = df[df['place_i'] != df['place_j']]
    tweet_user_min = 30
    df = df[df['place_i_users'] > tweet_user_min]
    df = df[df['place_j_users'] > tweet_user_min]
    gby_o = df.groupby('place_i')['place_i'].count().index.to_list()
    gby_d = df.groupby('place_j')['place_j'].count().index.to_list()
    # gby = gby_o + gby_d
    gby_list = gby_o + gby_d
    gby_set = set(gby_list)
    gby_list = list(gby_set)
    gby_list = natsorted(gby_list)
    path = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    basename = basename[:-4] + "_placename_list.txt"
    saved_path = os.path.join(path, basename)
    list_to_file(gby_list, saved_path)
    print(df)
    # build distance matrix
    dis_mat = np.zeros((len(gby_list[:]), len(gby_list[:])))
    print("dis_mat shape:", dis_mat.shape)

    fips_dic = {}
    for i in range(len(gby_list[:])):
        #     print(i, gby_list[i])
        fips_dic[gby_list[i]] = i


    for idx, row in df.iterrows():
        interval = 100000
        i = fips_dic[row['place_i']]
        j = fips_dic[row['place_j']]
        i = int(i)
        j = int(j)
        dis_mat[i, j] = row['pci']
        dis_mat[j, i] = row['pci']
        if idx % interval == 0:
            print(idx, i, j, dis_mat[i, j], dis_mat[j, i], row['place_i'], row['place_j'])

    print("dis_mat.sum(),  df['pci'].sum:", dis_mat.sum(),  df['pci'].sum())
    saved_file = saved_path.replace("_placename_list.txt", "_matrix.csv")
    s_dis_mat = scipy.sparse.coo_matrix(dis_mat)
    scipy.sparse.save_npz(saved_file, s_dis_mat, compressed=True)
    print("Csv_to_matrix() done:", csv_path)


def LA_clustering():
    placename = pd.read_csv(
        r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_LosAngeles_PCI_2019_placename_list.txt',
        header=None, names=['placename'])
    placename_list = placename['placename'].tolist()

    saved_file_world = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_LosAngeles_PCI_2019_matrix.csv.npz'

    world_matrix = scipy.sparse.load_npz(saved_file_world).todense()
    world_matrix = np.array(world_matrix)
    np.fill_diagonal(world_matrix, 0)

    small_value = 0.00000001

    # dis_mat = world_matrix.max() + 1 - world_matrix
    dis_mat = 1 / (world_matrix / 1000 + small_value)

    dis_mat[np.where(dis_mat == (1 / small_value))] = np.nan

    dis_mat[np.where(dis_mat > world_matrix.max())] = np.nan
    np.fill_diagonal(dis_mat, np.nan)

    conn_mat = np.where(world_matrix > 0, 1, 0)

    n_cluster_list = [5, 10, 15, 20, 30, 40, 50, 60, 100]
    n_cluster_list = sorted(n_cluster_list, reverse=True)

    # saved_path = os.getcwd()
    saved_path = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index'

    dis_mat[dis_mat == -1] = np.nan
    print(dis_mat)
    labels_all_rounds = sample_AgglomerativeClustering(dis_mat, conn_mat, n_cluster_list)
    for i, r in enumerate(labels_all_rounds):
        print(f"Clustering results for {str(n_cluster_list[i])} classes (node index, label):")
        r = [str(placename_list[item[0]]) + "," + str(item[1]) for item in r]
        print(r)
        new_name = os.path.join(saved_path, "LA_cluster_" + str(n_cluster_list[i]) + ".csv")
        list_to_file(r, new_name, header="node_index,class")

#
# def NYC_clustering():
#     country_list = pd.read_csv(
#         r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_NewYorkCity_PCI_2019_placename_list.txt',
#         header=None, names=['fips'])
#     gby_list = country_list['fips'].tolist()
#
#     saved_file_world = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_NewYorkCity_PCI_2019_matrix.csv.npz'
#
#     world_matrix = scipy.sparse.load_npz(saved_file_world).todense()
#     world_matrix = np.array(world_matrix)
#     np.fill_diagonal(world_matrix, 0)
#
#     small_value = 0.00000001
#     dis_mat = 1 / (world_matrix / 1000 + small_value)
#
#     dis_mat[np.where(dis_mat == (1 / small_value))] = np.nan
#     np.fill_diagonal(dis_mat, np.nan)
#
#     conn_mat = np.where(world_matrix > 0, 1, 0)
#
#     n_cluster_list = [5, 10, 15, 20, 30, 40, 50, 60, 100]
#     n_cluster_list = sorted(n_cluster_list, reverse=True)
#
#     # saved_path = os.getcwd()
#     saved_path = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index'
#     # dis_mat[dis_mat == -1] = np.nan
#     print(dis_mat)
#     labels_all_rounds = sample_AgglomerativeClustering(dis_mat, conn_mat, n_cluster_list)
#     for i, r in enumerate(labels_all_rounds):
#         print(f"Clustering results for {str(n_cluster_list[i])} classes (node index, label):")
#         print(r)
#         new_name = os.path.join(saved_path, "cluster_" + str(n_cluster_list[i]) + ".csv")
#         list_to_file(r, new_name, header="node_index,class")

def count_not_nan_rows(a):
    cnt = len(a)
    for row in a:
        if np.isnan(row):
            cnt - 1
    return cnt

def NYC_clustering():
    placename_list = pd.read_csv(
        r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_NewYorkCity_PCI_2019_placename_list.txt',
        header=None, names=['placename'])
    placename_list = placename_list['placename'].tolist()

    saved_file_world = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_NewYorkCity_PCI_2019_matrix.csv.npz'

    world_matrix = scipy.sparse.load_npz(saved_file_world).todense()
    world_matrix = np.array(world_matrix)
    np.fill_diagonal(world_matrix, 0)

    small_value = 0.00000001
    dis_mat = 1 / (world_matrix / 1000 + small_value)

    dis_mat[np.where(dis_mat == (1 / small_value))] = np.nan
    np.fill_diagonal(dis_mat, np.nan)

    conn_mat = np.where(world_matrix > 0, 1, 0)

    n_cluster_list = [5, 10, 15, 20, 30, 40, 50, 60, 100]
    n_cluster_list = sorted(n_cluster_list, reverse=True)

    saved_path = os.getcwd()
    saved_path = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index'
    # dis_mat[dis_mat == -1] = np.nan
    print(dis_mat)
    labels_all_rounds = sample_AgglomerativeClustering(dis_mat, conn_mat, n_cluster_list)
    for i, r in enumerate(labels_all_rounds):
        print(f"Clustering results for {str(n_cluster_list[i])} classes (node index, label):")
        r = [str(placename_list[item[0]]) + "," + str(item[1]) for item in r]
        print(r)
        new_name = os.path.join(saved_path, "NYC_cluster_" + str(n_cluster_list[i]) + ".csv")
        list_to_file(r, new_name, header="node_index,class")
if __name__ == "__main__":
    LA_clustering()
    # csv_to_matrix(r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_NewYorkCity_PCI_2019.csv')
    # csv_to_matrix(r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\place_connectivity\0223\Place-Connectivity-Index\US_CensusTract_LosAngeles_PCI_2019.csv')
    NYC_clustering()



