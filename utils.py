import numpy as np
import torch

def sparse_to_dense(edge_idx, src_node_l:list, edge_attr=None):
    #if max_num_nodes is None:
        #max_num_nodes = edge_idx.max().item() + 1
    max_num_nodes = len(src_node_l)
    # print('max_num_nodes', max_num_nodes)
    adj = torch.zeros((max_num_nodes, max_num_nodes))
    if edge_attr is None:
        for row_id, col_id in zip(edge_idx[0], edge_idx[1]):
            row, col = src_node_l.index(row_id), src_node_l.index(col_id)
            adj[row][col] = 1
            adj[col][row] = 1
            adj[row][row] = 1
            adj[col][col] = 1
    elif edge_attr is not None:
        for row_id, col_id, ew in zip(edge_idx[0], edge_idx[1], edge_attr):
            row, col = src_node_l.index(row_id), src_node_l.index(col_id)
            adj[row][col] = ew
            adj[col][row] = ew
            adj[row][row] = 1
            adj[col][col] = 1
    return adj


class TempNeighbors:
    def __init__(self, edge_indices, max_idx, time_reset, features, uniform=False): # idx_map
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(edge_indices, max_idx)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.features = np.array(features)
        self.time_reset = time_reset
        self.off_set_l = off_set_l

        self.uniform = uniform

    def init_off_set(self, edge_indices, max_idx):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, ts, ew in edge_indices:
            adj_list[src].append((dst, ts, ew))
            adj_list[dst].append((src, ts, ew))


        n_idx_l = []
        n_ts_l = []
        e_ew_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_ew_l.extend([x[2] for x in curr])
            n_ts_l.extend([x[1] for x in curr])

            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_ew_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        off_set_l = self.off_set_l
        neighbors_features = []
        src_idx = int(src_idx)
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        # print(type(self.features))
        for t, n_idx in zip(neighbors_ts, neighbors_idx):
            neighbors_features.append(self.features[t-self.time_reset][n_idx])
       
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0: # neighbors_idx_map
            return neighbors_idx, neighbors_ts, np.zeros((0,self.features.shape[0]))  #, neighbors_e_idx # neighbors_idx_map
        neighbors_features = np.stack(neighbors_features, axis=0)
        # print('neighbors features shape', neighbors_features.shape,neighbors_idx.shape, neighbors_ts.shape)
        left = 0
        right = len(neighbors_idx) - 1

        if neighbors_ts[left] == neighbors_ts[right] == cut_time: # neighbors_idx_map
            return neighbors_idx, neighbors_ts, neighbors_features # neighbors_idx_map

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:

            return neighbors_idx[:right], neighbors_ts[:right], neighbors_features[:right] # neighbors_e_idx[:right], # neighbors_idx_map
        else:
            return neighbors_idx[:left], neighbors_ts[:left], neighbors_features[:left] # neighbors_e_idx[:left], # neighbors_idx_map

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=5):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_features_batch = np.zeros((len(src_idx_l), num_neighbors, 1)).astype(np.float32)
        # print(out_ngh_features_batch.shape)
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_ts, ngh_features = self.find_before(src_idx, cut_time+1) #ngh_eidx,
            # print(ngh_features.shape)
            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    try:
                        out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    except TypeError:
                        print(ngh_idx, sampled_idx, 'typeerror')
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_features_batch[i,:] = out_ngh_features_batch[i,:][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]

                    assert (len(ngh_idx) <= num_neighbors)
                    assert (len(ngh_ts) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_features_batch[i, num_neighbors - len(ngh_ts):] = ngh_features
        # print('out_ngh_features_batch shape in utils2021', out_ngh_features_batch.shape)           
        return out_ngh_node_batch, out_ngh_t_batch, out_ngh_features_batch # out_ngh_eidx_batch,

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, z, f = self.get_temporal_neighbor(src_idx_l, cut_time_l+1, num_neighbors)
        node_records = [x]
        feature_records = [f]
        t_records = [z]
        for _ in range(k - 1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1]  # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_t_batch, out_ngh_features_batch = self.get_temporal_neighbor(ngn_node_est,
                                                                                                 ngn_t_est,
                                                                                                 num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors)  # [N, *([num_neighbors] * k)]
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_features_batch = out_ngh_features_batch.reshape(*orig_shape, num_neighbors)
            node_records.append(out_ngh_node_batch)
            t_records.append(out_ngh_t_batch)
            feature_records.append(out_ngh_features_batch)
        return node_records, t_records, feature_records

