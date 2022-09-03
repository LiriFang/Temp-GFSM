import os
import numpy as np
import multiprocessing as mp
from itertools import repeat
import scipy.sparse as sp
from collections import defaultdict
from torch_geometric.data import Data, Batch
from utils import sparse_to_dense
from copy import deepcopy
import torch

def encode_onehot(labels, classes_dict=None):

    if not classes_dict:
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def shuffle(list1, list2,list3,list4):
    temp = list(zip(list1, list2,list3,list4))
    np.random.shuffle(temp)
    return zip(*temp)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class DPPINDatasetLoader(object):
    """A dataset of Dynamic PPIN github
    Use Dynamic_graph_temporal_signal.
    ORF-V
    ORF-U
    ORF-DU
    ORF-M
    ORF-DE
    BRF
    LTR
    IER
    NA

    """

    def __init__(self, filedir, lags): # lags as num_of_snapshot
        self._read_data(filedir)
        self._get_edges()
        self._node_idx()
        self._get_targets_and_features()
        self._get_dataset(lags)


    def _read_data(self, filedir):
        # print('loading dataset from ' + filedir)
        self._dataset = np.genfromtxt(os.path.join(filedir,'Dynamic_Network.txt'), delimiter=',')
        # feature_dir = filedir.split('graph')[0]
        self._dataset_target_features = np.genfromtxt(os.path.join(filedir,'Node_Features.txt'), delimiter=',')

    def _get_edges(self):
        self.edges = []
        for i in self._dataset:
            self.edges.append((int(i[0]),int(i[1]),int(i[2]),i[3]))

    def _node_idx(self):
        self.max_idx = int(np.max(self._dataset[:,:2]))
        self.min_ts = int(np.min(self._dataset[:,2]))
        self.max_ts = int(np.max(self._dataset[:,2]))

    def _get_targets_and_features(self):
        #self.features = [row for row in self._dataset_target_features[:,1:self.max_ts+1].T]
        self.features = [row for row in self._dataset_target_features[:,1:].T]
    def _get_dataset(self, lags: int=1):
        # self.lags = lags
        if self.max_ts - self.min_ts >= lags:
            self.lags = int((self.max_ts - self.min_ts) / lags)
        else:
            self.lags = self.max_ts - self.min_ts
        self.dataset = TemporalGraph(self.edges, self.features,  self.lags, self.min_ts, self.max_ts) #self.edge_weights #self.targets,


class TemporalGraph:
    def __init__(self, edge_indices,  features, lags, min_ts, max_ts):
        self.edge_indices = edge_indices # temp adj (src, dst, ts, ew)
        self.features = features

        self.lags = lags
        self.min_ts = min_ts
        self.max_ts = max_ts

    def _check_temporal_consistency(self):
        assert len(self.features) == len(self.targets), "Temporal dimension inconsistency."

    def _get_edge_index(self):
        edge_index = [[],[]]
        edge_weights = []
        ts = []
        # for t in range(self.t_start, self.t_end):
        full_edge = deepcopy(self.edge_indices)
        src_node_l_org = defaultdict(float)
        for (src, dst, timestamp, ew) in full_edge:
            if self.t_start <= timestamp < self.t_start + self.lags:
                edge_index[0].append(src)
                edge_index[1].append(dst)
                ts.append(timestamp)
                edge_weights.append(ew)
                if timestamp > src_node_l_org[src]: src_node_l_org[src] = timestamp
                if timestamp > src_node_l_org[dst]: src_node_l_org[dst] = timestamp
        src_node_l = list(src_node_l_org.keys())
        cut_time_l = list(src_node_l_org.values())
        # assert len(src_node_l) > 0, "src node list is empty."
        assert len(edge_index[0]) == len(edge_index[1]), "snapshot num of edges inconsistency."
        assert len(edge_index[0]) == len(edge_weights), "snapshot num of edges and weights inconsistency"
        return torch.LongTensor(np.array(edge_index)), torch.FloatTensor(np.array(edge_weights)), torch.FloatTensor(np.array(ts)),\
               torch.LongTensor(np.array(src_node_l)), torch.FloatTensor(np.array(cut_time_l))

    def _get_features(self):

        if self.features[self.t_start-self.min_ts:self.t_start + self.lags-self.min_ts] is None:
            return self.features[self.t_start-self.min_ts:self.t_start + self.lags-self.min_ts].T
        else:
            # return torch.FloatTensor(self.features[self.t_start-self.min_ts:self.t_start + self.lags-self.min_ts]).T

            return np.array(self.features[self.t_start - self.min_ts:self.t_start + self.lags - self.min_ts]).T
        # return np.array(self.features).T

    def _get_snapshot(self, edge_index, edge_weight, ts, src_node_l, cut_time_l):
        x = self._get_features()
        # print('x size', x.shape)
        # edge_index,  edge_weight, ts, src_node_l, cut_time_l = self._get_edge_index()
        # x = x[src_node_l.to_list()]
        # print('x size', x.shape)
        adj = sparse_to_dense(edge_index, src_node_l.tolist(), edge_weight)
        snapshot = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_weight,
                        src_node_l=src_node_l,
                        cut_time_l=cut_time_l,
                        # min_ts = self.min_ts,
                        adj = adj) # y=y,
        return snapshot

    def __next__(self):
        if self.t_start + self.lags <= self.max_ts: # + 1:
            edge_index,  edge_weight, ts, src_node_l, cut_time_l = self._get_edge_index()
            if len(src_node_l) <= 0:
                self.t_start = self.t_start + self.lags
                return self.__next__()
            else:
                snapshot = self._get_snapshot(edge_index, edge_weight, ts, src_node_l, cut_time_l)
                self.t_start = self.t_start + self.lags
            return snapshot
        else:
            raise StopIteration

    def __iter__(self):
        self.t_start = self.min_ts
        self.t_end = self.t_start + self.lags
        return self

class ReadDataset(object):
    def __init__(self, args):#args
        self.sample_c_n = args.sample_c_n
        self.lags = args.lags
        self.batch_sz = args.batch_size
        self.directory = args.dir
        self.k_shot = args.k_shot
        self.k_query = args.k_query
        self.n_way = args.n_way # number of classes in each task
        self.labels = args.labels #list of labels
        self.total_sample_g = args.total_sample_g
        # self.total_sample_g_test = args.total_sample_g_test
        # self.total_sample_c = len(self.labels)

    def next_batch(self, chosen_class, meta_train=1):
        batch_graph_spt = [[] for _ in range(self.batch_sz)]
        batch_graph_qry = [[] for _ in range(self.batch_sz)]
        batch_label_spt = [[] for _ in range(self.batch_sz)]
        batch_label_qry = [[] for _ in range(self.batch_sz)]
        batch_label_nonencode_spt = [[] for _ in range(self.batch_sz)]
        batch_label_nonencode_qry = [[] for _ in range(self.batch_sz)]
        # if meta_train == 1:
            # self.total_sample_g = self.total_sample_g_train
        #else:
            #self.total_sample_g = self.total_sample_g_test
        load_data_pool = mp.Pool(10)

        g_id_qry = [[] for _ in range(self.batch_sz)]
        g_id_spt = [[] for _ in range(self.batch_sz)]
        for b in range(self.batch_sz):
            np.random.shuffle(chosen_class)
            support_file_list = []
            query_file_list = []
            prefix = "graph"
            for c in chosen_class:

                if 'ct1_0' in c: # for social data
                    chosen_sample_range = np.arange(2, int(self.total_sample_g[c]), step=2)

                elif 'ct1_1' in c: # for social data
                    chosen_sample_range = np.arange(1, int(self.total_sample_g[c])+1, step=2)

                else: # for DPPIN data
                    chosen_sample_range = np.arange(0, self.total_sample_g[c])
                    prefix = "tg53_"

                g_id = np.random.choice(chosen_sample_range, self.k_shot + self.k_query, replace=False)
                for i, g in enumerate(g_id):
                    graphname = prefix + str(g)
                    filedir = os.path.join(self.directory, c, graphname)
                    if i < self.k_shot:
                        support_file_list.append(filedir)
                        batch_label_spt[b].append(c)
                        g_id_spt[b].append(g)
                    else:
                        query_file_list.append(filedir)
                        batch_label_qry[b].append(c)
                        g_id_qry[b].append(g)
            
            
            batch_graph_spt[b] = load_data_pool.starmap(DPPINDatasetLoader, zip(support_file_list, repeat(self.lags)))
            batch_graph_qry[b] = load_data_pool.starmap(DPPINDatasetLoader, zip(query_file_list, repeat(self.lags)))
            batch_label_nonencode_spt[b] = batch_label_spt[b]
            batch_label_nonencode_qry[b] = batch_label_qry[b]
            batch_graph_spt[b], batch_label_spt[b], g_id_spt[b],batch_label_nonencode_spt[b] = shuffle(batch_graph_spt[b], encode_onehot(batch_label_spt[b]),g_id_spt[b], batch_label_nonencode_spt[b]) # classes_dict=class_dict
            batch_graph_qry[b], batch_label_qry[b],g_id_qry[b], batch_label_nonencode_qry[b] = shuffle(batch_graph_qry[b], encode_onehot(batch_label_qry[b]), g_id_qry[b],batch_label_nonencode_qry[b])
            assert len(batch_graph_qry[b]) == len(batch_label_qry[b]), 'dataset.py batch graph qry and label qry inconsistency'
        print('====================================')
        print('graph label qry', batch_label_nonencode_qry, 'graph id qry', g_id_qry)
        return batch_graph_spt, batch_label_spt, batch_graph_qry, batch_label_qry

