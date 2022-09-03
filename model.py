

from layers import *
from utils import sparse_to_dense
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, activation=None):
        super(GCN, self).__init__()
        self._nfeat = nfeat
        self._nhid = nhid
        self._activation = activation
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index, eval=False):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        if eval:
            return x
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            return x

    def functional_forward(self, x, adj, weights, eval=False):

        x = F.relu(self.gc1.functional_forward(x, adj, id=1, weights=weights))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2.functional_forward(x, adj, id=2, weights=weights)
        if eval:
            return x
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3.functional_forward(x, adj, id=3, weights=weights)
            return F.log_softmax(x, dim=1)


class TGAN(torch.nn.Module):
    def __init__(self, n_feat, nhid, num_neighbors,device, attn_mode='prod', use_time='time',
                 num_layers=2, n_head=2, null_idx=0, dropout=0.1):
        super(TGAN, self).__init__()

        self.num_layers = num_layers

        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.num_neighbors = num_neighbors
        self.feat_dim = n_feat
        self.nhid = nhid
        self.linear = nn.Linear(self.feat_dim, self.nhid)
        self.n_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        self.use_time = use_time
        self.device = device

        self.attn_model_list = torch.nn.ModuleList([AttnModel(self.nhid,
                                                              self.nhid, # self.feat_dim,
                                                              attn_mode=attn_mode,
                                                              n_head=n_head,
                                                              drop_out=dropout) for _ in range(num_layers)])

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.nhid)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.nhid)
        else:
            raise ValueError('invalid time option!')


    def forward(self, ngh_finder, src_node_feat, src_idx_l, cut_time_l, update_weights=None, module_name=''):

        if update_weights is None:
            src_embed = self.tem_conv(ngh_finder, src_node_feat, src_idx_l, cut_time_l, self.num_layers, self.num_neighbors)
        else:
            src_embed = self.functional_tem_conv(ngh_finder, src_node_feat, src_idx_l, cut_time_l, self.num_layers, self.num_neighbors, update_weights, module_name)
        return src_embed



    def tem_conv(self, ngh_finder, src_node_feat, src_idx_l, cut_time_l, curr_layers, num_neighbors):
        assert (curr_layers >= 0)
        batch_size = len(src_idx_l)
        cut_time_l = np.array(cut_time_l)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(self.device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))

        if curr_layers == 0:
            return  self.linear(torch.from_numpy(src_node_feat).float().to(self.device))
        else:
            src_node_conv_feat = self.tem_conv(ngh_finder, src_node_feat, src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)
            src_ngh_node_batch, src_ngh_t_batch, src_ngh_feat_batch = ngh_finder.get_temporal_neighbor(src_idx_l,
                                                                                                       cut_time_l,
                                                                                                                 num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)
            src_ngh_node_conv_feat = self.tem_conv(ngh_finder, src_ngh_feat_batch,
                                                   src_ngh_node_batch.flatten(),
                                                   src_ngh_t_batch.flatten(),
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)

            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)


            src_node_conv_feat = src_node_conv_feat.view(-1,src_node_conv_feat.size(-1))

            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   mask)
            return local

    def functional_tem_conv(self, ngh_finder, src_node_feat, src_idx_l, cut_time_l, curr_layers,  num_neighbors, update_weights, module_name_):

        assert (curr_layers >= 0)

        batch_size = len(src_idx_l)

        cut_time_l = np.array(cut_time_l)

        cut_time_l_th = torch.from_numpy(np.array(cut_time_l)).float().to(self.device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)

        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder.functional_forward(torch.zeros_like(cut_time_l_th), update_weights=update_weights, module_name=module_name_+'time_encoder.')

        if curr_layers == 0:
            src_node_feat = torch.from_numpy(src_node_feat).float().to(self.device)
            src_node_feat = F.softmax(torch.matmul(src_node_feat, update_weights[module_name_+'linear.weight'].T) + update_weights[module_name_+'linear.bias'], dim=0)
            return src_node_feat
        else:
            src_node_conv_feat = self.functional_tem_conv(ngh_finder, src_node_feat, src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors, update_weights=update_weights, module_name_ = module_name_)
            src_ngh_node_batch, src_ngh_t_batch, src_ngh_feat_batch = ngh_finder.get_temporal_neighbor(src_idx_l,
                                                                                                       cut_time_l,
                                                                                                       num_neighbors=num_neighbors)  # , src_ngh_eidx_batch


            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            if type(src_ngh_t_batch_delta) != np.ndarray:
                src_ngh_t_batch_th = src_ngh_t_batch_delta.float()
            else:
                src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)

            src_ngh_node_conv_feat = self.functional_tem_conv(ngh_finder, src_ngh_feat_batch, src_ngh_node_batch.flatten(),
                                                   src_ngh_t_batch.flatten(),
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors,
                                                              update_weights=update_weights, module_name_=module_name_)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder.functional_forward(src_ngh_t_batch_th, update_weights, module_name=module_name_+'time_encoder.')
            src_node_conv_feat = src_node_conv_feat.view(-1,src_node_conv_feat.size(-1))
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m.functional_forward(src_node_conv_feat,
                                                      src_node_t_embed,
                                                      src_ngh_feat,
                                                      src_ngh_t_embed,
                                                      mask,
                                                      update_weights,
                                                      module_name=module_name_+'attn_model_list.{}'.format(str(curr_layers-1)))
            return local


class GNN_SnapAuto(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout): #
        super(GNN_SnapAuto, self).__init__()

        self.gc_decode_structure1 = GraphConvolution(nfeat, nhid)

        self.gc_community_prob = GraphConvolution(nfeat, nhid)
        self.gc_community_value = GraphConvolution(nfeat, nhid)

        self.nhid = nhid
        self.nfeat = nfeat
        self.dropout = dropout
        self.args = args

        if args.hop_concat_type == 'fc':
            self.concat_weight = nn.Linear(nhid, nhid)
        elif args.hop_concat_type == 'attention':
            self.concat_weight = nn.Parameter(torch.FloatTensor(nhid, 1), requires_grad=True)
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.nhid)
        self.concat_weight.data.uniform_(-stdv, stdv)


    def forward(self, x, adj):
        gc_z = self.gc_decode_structure1(x, adj)
        decoder_adj = torch.sigmoid(torch.mm(gc_z, gc_z.transpose(0, 1)))
        return x, torch.mean((adj - decoder_adj).pow(2))

    def functional_forward(self, x, adj, update_weights, module_name=''):
        # 'gc_decode_structure1.weight', 'gc_decode_structure1.bias',
        # 'gc_community_prob.weight', 'gc_community_prob.bias',
        # 'gc_community_value.weight', 'gc_community_value.bias',
        # 'gc_structure3.weight', 'gc_structure3.bias',
        # 'concat_weight.weight', 'concat_weight.bias'])
        gc_z = self.gc_decode_structure1.functional_forward(x, adj, update_weights, module_name=module_name+'gc_decode_structure1')
        decoder_adj = torch.sigmoid(torch.mm(gc_z, gc_z.transpose(0, 1)))
        return x, torch.mean((adj - decoder_adj).pow(2))

    def forward_community(self, x, adj):
        gc_z = self.gc_community_value(x, adj)
        gc_s = F.softmax(self.gc_community_prob(x, adj), dim=1)
        x = F.normalize(torch.mm(gc_s.transpose(0, 1), gc_z), dim=0).unsqueeze(0)

        if self.args.hop_concat_type == 'mean':
            return torch.mean(x, dim=0, keepdim=True)
        elif self.args.hop_concat_type == 'attention':
            att_weight = F.softmax(torch.matmul(x, self.concat_weight), dim=1)
            return torch.sum(x*att_weight, dim=1)

    def functional_forward_community(self, x, adj, update_weights, module_name=''):
        # 'gc_community_prob.weight', 'gc_community_prob.bias',
        # 'gc_community_value.weight', 'gc_community_value.bias',
        # 'gc_structure3.weight', 'gc_structure3.bias',
        # 'concat_weight.weight', 'concat_weight.bias'])
        gc_z = self.gc_community_value.functional_forward(x, adj, update_weights, module_name=module_name+'gc_community_value')
        gc_s = F.softmax(self.gc_community_prob.functional_forward(x, adj, update_weights, module_name=module_name+'gc_community_prob'), dim=1)
        x = F.normalize(torch.mm(gc_s.transpose(0, 1), gc_z), dim=0).unsqueeze(0)
        if self.args.hop_concat_type == 'fc':
            return torch.mm(x, update_weights[module_name+'concat_weight.weight'].T) + update_weights[module_name+'concat_weight.bias']
        elif self.args.hop_concat_type == 'mean':
            return torch.mean(x, dim=0, keepdim=True)
        elif self.args.hop_concat_type == 'attention':
            att_weight = F.softmax(torch.matmul(x, update_weights[module_name+'concat_weight']), dim=1) #update_weights[module_name+'concat_weight.weight']
            return torch.sum(x * att_weight, dim=1)



class Classifier(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout=0.1):
        super(Classifier, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.dropout = dropout
        self.lags = args.lags
        self.num_neighbors = args.num_neighbors
        self.snapshot_concat = args.snapshot_concat
        if self.snapshot_concat == 'attention':
            self.attn = nn.Parameter(torch.FloatTensor(self.nhid, 1), requires_grad=True) #self.nhid*self.max_length *self.max_length

        self.dropout = nn.Dropout(self.dropout)

        self.device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

        self.autoencoder = GNN_SnapAuto(args, self.nhid, self.nhid, self.dropout)
        self.tempgnn = TGAN(self.nfeat, self.nhid, self.num_neighbors, device=self.device,num_layers=args.num_layers,use_time=args.use_time)

    def forward(self, episode_spt):
        output_repr, loss_r_episode = [], 0
        num_episode = len(episode_spt)
        for ep in range(len(episode_spt)):  # num of instances in a episode
            loss_r_snapshot = 0
            snapshot_embed = []
            train_ngh_finder = TempNeighbors(episode_spt[ep].edges, episode_spt[ep].max_idx, episode_spt[ep].min_ts, episode_spt[ep].features, uniform=True)

            init_features_list, src_node_list, cut_time_list, adj_list, edge_idx_list, edge_attr_list= [], [], [], [], [],[]

            for t, snapshot in enumerate(episode_spt[ep].dataset):
                min_ts = snapshot.cut_time_l.min()
                time_list = snapshot.cut_time_l.int()-min_ts
                if t == 0:
                    init_features_list = snapshot.x[snapshot.src_node_l.tolist(),time_list.int().tolist()]
                elif t > 0:
                    init_features_list = np.append(init_features_list, snapshot.x[snapshot.src_node_l,time_list.int().tolist()], axis=0)

                src_node_list.append(snapshot.src_node_l)
                cut_time_list.append(snapshot.cut_time_l)
            node_embed_total = self.tempgnn(train_ngh_finder, np.expand_dims(init_features_list,axis=1), torch.cat(src_node_list), torch.cat(cut_time_list))
            count = 0
            for t, snapshot in enumerate(episode_spt[ep].dataset):

                node_embed_resize_torch = node_embed_total[count: count + snapshot.src_node_l.shape[0]]

                count += snapshot.src_node_l.shape[0]

                adj = snapshot.adj.to(self.device)
                node_embed, loss_r = self.autoencoder(node_embed_resize_torch.to(self.device),
                                                      adj)
                snapshot_embed.append(self.autoencoder.forward_community(node_embed, adj))
                loss_r_snapshot += loss_r
                del adj
            temp_repr = torch.stack(snapshot_embed, dim=1) # size

            if self.snapshot_concat == 'attention':
                attn_weights = F.softmax(torch.matmul(temp_repr, self.attn), dim=0)
                temp_repr = torch.sum(temp_repr * attn_weights, dim=1)

            elif self.snapshot_concat == 'sum':
                temp_repr = torch.mean(temp_repr, dim=1)

            output_repr.append(temp_repr)
            loss_r_episode += loss_r_snapshot/ (t + 1)

        return torch.cat(output_repr, dim=0), loss_r_episode/num_episode

    def functional_forward(self, episode_spt, update_weights, module_name_=''):
        output_repr, loss_r_episode,output_snapshot_repr = [], 0, []
        num_episode = len(episode_spt)
        for ep in range(len(episode_spt)):  # num of instances in a episode
            
            loss_r_snapshot = 0
            snapshot_embed = []

            train_ngh_finder = TempNeighbors(episode_spt[ep].edges, episode_spt[ep].max_idx, episode_spt[ep].min_ts,
                                             episode_spt[ep].features, uniform=True)

            init_features_list, src_node_list, cut_time_list = [], [], []

            for t, snapshot in enumerate(episode_spt[ep].dataset):
                
                min_ts = snapshot.cut_time_l.min()
                time_list = snapshot.cut_time_l.int()-min_ts
                if t == 0:
                    # init_features_list = snapshot.x[snapshot.src_node_l]
                    init_features_list = snapshot.x[snapshot.src_node_l.tolist(),time_list.int().tolist()]
                elif t > 0:
                    # init_features_list = np.append(init_features_list, snapshot.x[snapshot.src_node_l], axis=0)
                    init_features_list = np.append(init_features_list, snapshot.x[snapshot.src_node_l,time_list.int().tolist()], axis=0)
                src_node_list.append(snapshot.src_node_l)
                cut_time_list.append(snapshot.cut_time_l)
                
            node_embed_total = self.tempgnn(train_ngh_finder, np.expand_dims(init_features_list,axis=1), torch.cat(src_node_list), torch.cat(cut_time_list), update_weights=update_weights, module_name=module_name_+'tempgnn.')
            count = 0
            for t, snapshot in enumerate(episode_spt[ep].dataset):

                node_embed_resize_torch = node_embed_total[count: count + snapshot.src_node_l.shape[0]]

                count += snapshot.src_node_l.shape[0]

                adj = snapshot.adj.to(self.device)
                node_embed, loss_r = self.autoencoder.functional_forward(node_embed_resize_torch.to(self.device),
                                                      adj, update_weights, module_name=module_name_ + 'autoencoder.')
                snapshot_embed.append(self.autoencoder.functional_forward_community(node_embed, adj, update_weights,
                                                                                    module_name=module_name_ + 'autoencoder.'))
                loss_r_snapshot += loss_r

            temp_repr = torch.stack(snapshot_embed, dim=1)

            output_snapshot_repr.append(temp_repr.detach().cpu().tolist())
            if self.snapshot_concat == 'attention':
                attn_weights = F.softmax(torch.matmul(temp_repr, update_weights[module_name_ + 'attn']) , dim=0) #+ update_weights[module_name_ + 'attn.bias']

                temp_repr = torch.sum(temp_repr * attn_weights, dim=1)
            elif self.snapshot_concat == 'sum':
                temp_repr = torch.mean(temp_repr, dim=1)
            loss_r_snapshot /= t + 1  # autoencoder reconstruction loss
            output_repr.append(temp_repr)
            loss_r_episode += loss_r_snapshot

        return torch.cat(output_repr, dim=0), loss_r_episode/num_episode, output_snapshot_repr#torch.cat(output_snapshot_repr, dim=0)
