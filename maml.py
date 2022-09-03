
from model import *
from    torch import optim
from copy import deepcopy
import torch.nn.functional as F


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def proto_loss_spt(logits, y_t, n_support):

    target_cpu = torch.from_numpy(np.array(y_t)).to('cpu')

    target_cpu = target_cpu.view(-1, target_cpu.size(2))

    input_cpu = logits.to('cpu')

    def supp_idxs(c):
        result = target_cpu.eq(c)

        index = torch.ones(target_cpu.size(0))
        for col in range(target_cpu.size(1)):
            index = index.mul(result[:, col])
        return index.nonzero(as_tuple=True)[0][:n_support]


    classes = torch.unique(target_cpu, dim=0) #.squeeze(0)
    n_classes = len(classes)
    n_query = n_support
    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(supp_idxs, classes))).view(-1)

    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(-1)).float().mean()
    return loss_val, acc_val, prototypes


def proto_loss_qry(logits, y_t, prototypes):

    target_cpu = torch.from_numpy(np.array(y_t)).to('cpu')

    input_cpu = logits.to('cpu')
    target_cpu = target_cpu.view(-1, target_cpu.size(2))
    classes = torch.unique(target_cpu, dim=0)

    n_classes = len(classes)


    n_query = int(logits.shape[0] / n_classes)

    def supp_idxs(c):

        result = target_cpu.eq(c)

        index = torch.ones(target_cpu.size(0))
        for col in range(target_cpu.size(1)):

            index = index.mul(result[:, col])

        return index.nonzero(as_tuple=True)[0]

    query_idxs = torch.stack(list(map(supp_idxs, classes))).view(-1)

    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(-1)).float().mean()
    return loss_val, acc_val

class BilevelProto(nn.Module):
    """
    another ref: https://github.com/geopanag/pandemic_tgnn
    """
    def __init__(self, args):
        super(BilevelProto, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_shot
        self.k_qry = args.k_query
        self.batch_size = args.batch_size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.task_num = args.num_tasks
        self.ae_weight = args.ae_weight
        self.weight_decay = args.weight_decay

        self.net = Classifier(args, args.nfeat, args.nhid, dropout=0.1)
        self.meta_train = args.meta_train
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay=self.weight_decay)


    def forward_BilevelProto(self, datagenerator, chosen_class_list): #x_spt, y_spt, x_qry, y_qry,device
        """
        b: number of tasks
        setsz: the size for each task
         x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        y_spt:   [b, setsz]
        x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
         y_qry:   [b, querysz]
        :return:
        """

        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        self.meta_optim.zero_grad()
        print('batch size meta training', self.batch_size)
        for task in range(len(chosen_class_list)):
            if  torch.cuda.max_memory_allocated() != 0:
                print('memory usage at beginning of task', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())

            print('Training task num ', task, 'chosen classes:', chosen_class_list[task])
            x_spt, y_spt, x_qry, y_qry = datagenerator.next_batch(chosen_class_list[task], self.meta_train)

            output_repr = []
            loss_r_task = []
            # print(len(x_spt))
            # output_repr, loss_r_task = self.net(Batch.from_data_list(x_spt))
            for meta_batch in range(self.batch_size):
                episode_spt = x_spt[meta_batch]

                temp_repr, loss_r = self.net(episode_spt)
                loss_r_task.append(loss_r)
                output_repr.append(temp_repr)
                del loss_r, temp_repr

            loss, acc, prototypes = proto_loss_spt(torch.cat(output_repr, dim=0), y_spt, self.k_spt)
            print('support loss + accuracy', loss, acc)
            losses_s[0] += float(loss.item()) + float(torch.stack(loss_r_task).mean()) * self.ae_weight
            loss_r_task = torch.stack(loss_r_task).mean()
            grad = torch.autograd.grad(loss + loss_r_task* self.ae_weight, self.net.parameters(), retain_graph=True,
                                       allow_unused=True)
            del loss_r_task, loss

            fast_weights = {}
            for p, (name, init) in zip(grad, self.net.named_parameters()):
                if p is not None:
                    w = init - self.update_lr * p
                else:
                     w = init
                fast_weights[name] = w
            del grad


            with torch.no_grad():
                logits_q = []
                loss_r_task = []
                for meta_batch in range(self.batch_size):
                    logits_q_b, loss_r_b = self.net(x_qry[meta_batch])
                    loss_r_task.append(loss_r_b)
                    logits_q.append(logits_q_b)
                    del loss_r_b, logits_q_b
                loss_q, acc_q = proto_loss_qry(torch.cat(logits_q, dim=0), y_qry, prototypes)
                loss_r_task = float(torch.stack(loss_r_task).mean())
                losses_q[0] += float(loss_q) + loss_r_task * self.ae_weight
                corrects[0] = corrects[0] + float(acc_q)

            # print('this is the loss and accuracy after the first update')
            with torch.no_grad():
                logits_q = []
                loss_r_task = []
                for meta_batch in range(self.batch_size):
                    logits_q_b, loss_r_b,_ = self.net.functional_forward(x_qry[meta_batch], fast_weights, module_name_='') #'net.'

                    loss_r_task.append(loss_r_b)
                    logits_q.append(logits_q_b)
                    del loss_r_b, logits_q_b
                loss_q, acc_q = proto_loss_qry(torch.cat(logits_q, dim=0), y_qry, prototypes)
                if self.update_step == 1:
                    loss_r_task =torch.stack(loss_r_task).mean()
                    losses_q[1] += (loss_q +  loss_r_task* self.ae_weight).detach()
                elif self.update_step > 1:
                    loss_r_task = float(torch.stack(loss_r_task).mean())
                    losses_q[1] += float(loss_q) + loss_r_task * self.ae_weight
                corrects[1] = corrects[1] + float(acc_q)

            # print('memory usage after training first update step ', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            for k in range(1, self.update_step):
                print('calculating update step ', k+1)
                # 1. run the i-th task and compute loss for k=1~K-1
                loss_r_task = []
                output_repr = []

                for meta_batch in range(self.batch_size):
                    logits_b, loss_r_b,_ = self.net.functional_forward(x_spt[meta_batch], fast_weights, module_name_='') #'net.'
                    loss_r_task.append(loss_r_b)
                    output_repr.append(logits_b)
                    del loss_r_b, logits_b
                loss, _, prototypes = proto_loss_spt(torch.cat(output_repr, dim=0), y_spt, self.k_spt)
                losses_s[k] += float(loss.item()) + float(torch.stack(loss_r_task).mean()) * self.ae_weight
                # 2. compute grad on theta_pi
                loss_r_task = torch.stack(loss_r_task).mean()
                grad = torch.autograd.grad(loss + loss_r_task * self.ae_weight,
                                           fast_weights.values(), retain_graph=True, allow_unused=True) # retain_graph=True,
                del loss_r_task, loss, output_repr
                # 3. theta_pi = theta_pi - train_lr * grad
                for p, (name, init) in zip(grad, fast_weights.items()):
                    # print(p, name, init)
                    if p is not None:
                        w = init - self.update_lr * p
                    else: #elif p is None:
                        w = init
                    fast_weights[name] = w
                del grad, p, init, w

                logits_q = []
                loss_r_task = []
                for meta_batch in range(self.batch_size):
                    logits_q_b, loss_r_b,_ = self.net.functional_forward(x_qry[meta_batch], fast_weights, module_name_='') #'net.'
                    loss_r_task.append(loss_r_b)
                    logits_q.append(logits_q_b)
                    del loss_r_b, logits_q_b
                loss_q, acc_q = proto_loss_qry(torch.cat(logits_q, dim=0), y_qry, prototypes)
                if k + 1 == self.update_step:
                    loss_r_task = torch.stack(loss_r_task).mean()
                    losses_q[k + 1] += (loss_q + loss_r_task * self.ae_weight).detach()
                elif k + 1 < self.update_step:
                    loss_r_task = float(torch.stack(loss_r_task).mean())
                    losses_q[k + 1] += float(loss_q) + loss_r_task * self.ae_weight

                corrects[k + 1] = corrects[k + 1] + float(acc_q)
            loss_q = loss_q + loss_r_task * self.ae_weight
            # loss_q.requires_grad = True
            loss_q.backward()
            print(np.array(corrects)/(task+1))    
            del fast_weights
            # torch.cuda.empty_cache()
        self.meta_optim.step()
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / self.task_num

        accs = np.array(corrects) / (self.task_num)

        return accs, float(loss_q)

    def finetunning_BilevelProto(self, datagenerator, chosen_class_list): # x_spt, y_spt, x_qry, y_qry, device): #c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,


        corrects = [0 for _ in range(self.update_step_test + 1)]
        losses_q = [0 for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net)
        for task in range(len(chosen_class_list)):
            x_spt, y_spt, x_qry, y_qry = datagenerator.next_batch(chosen_class_list[task], meta_train=self.meta_train)

            print('Training task num ', task, 'chosen classes:', chosen_class_list[task])
            # 1. run the i-th task and compute loss for k=0
            output_repr = []
            loss_r_task = []
            for meta_batch in range(self.batch_size):
                episode_spt = x_spt[meta_batch]
                temp_repr, loss_r = self.net(episode_spt)
                loss_r_task.append(loss_r)
                output_repr.append(temp_repr)
            loss, _, prototypes = proto_loss_spt(torch.cat(output_repr, dim=0), y_spt, self.k_spt)
            loss_r_task = torch.stack(loss_r_task).mean()
            grad = torch.autograd.grad(loss +  loss_r_task* self.ae_weight, net.parameters(), allow_unused=True)
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.named_parameters())))
            fast_weights = {}
            for p, (name, init) in zip(grad, net.named_parameters()):
                if p is not None:
                    w = init - self.update_lr * p
                else: #if p is None:
                    w = init
                fast_weights[name] = w
            del grad
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = []
                loss_r_task = []
                for meta_batch in range(self.batch_size):
                    logits_q_b, loss_r_b = self.net(x_qry[meta_batch]) #, net.parameters()
                    loss_r_task.append(loss_r_b)
                    logits_q.append(logits_q_b)
                loss_q, acc_q = proto_loss_qry(torch.cat(logits_q, dim=0), y_qry, prototypes)
                losses_q[0] += float(loss_q) + float(torch.stack(loss_r_task).mean())
                corrects[0] = corrects[0] + acc_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = []
                loss_r_task = []
                for meta_batch in range(self.batch_size):
                    logits_q_b, loss_r_b, output_snap_q = self.net.functional_forward(x_qry[meta_batch], fast_weights,
                                                                   module_name_='')  # 'net.'
                    loss_r_task.append(loss_r_b)
                    logits_q.append(logits_q_b)

                loss_q, acc_q = proto_loss_qry(torch.cat(logits_q, dim=0), y_qry, prototypes)
                loss_r_task = float(torch.stack(loss_r_task).mean())
                losses_q[1] += float(loss_q) + loss_r_task
                corrects[1] = corrects[1] + acc_q

            for k in range(1, self.update_step_test):
                print('update meta testing step ', str(k))
                # 1. run the i-th task and compute loss for k=1~K-1
                loss_r_task = []
                output_repr = []
                for meta_batch in range(self.batch_size):
                    logits_b, loss_r_b, _  = self.net.functional_forward(x_spt[meta_batch], fast_weights,
                                                                 module_name_='')  # 'net.'
                    loss_r_task.append(loss_r_b)
                    output_repr.append(logits_b)
                loss, _, prototypes = proto_loss_spt(torch.cat(output_repr, dim=0), y_spt, self.k_spt)

                # 2. compute grad on theta_pi
                loss_r_task = torch.stack(loss_r_task).mean()
                grad = torch.autograd.grad(loss +  loss_r_task* self.ae_weight, fast_weights.values(),retain_graph=True,
                                        allow_unused=True) #
                # 3. theta_pi = theta_pi - train_lr * grad

                for p, (name, init) in zip(grad, fast_weights.items()):
                    if p is not None:
                        w = init - self.update_lr * p
                    else: #if p is None:
                        w = init
                    fast_weights[name] = w

                logits_q = []
                loss_r_task = []
                output_snap_repr = []
                for meta_batch in range(self.batch_size):
                    logits_q_b, loss_r_b, output_snap_q = self.net.functional_forward(x_qry[meta_batch], fast_weights,
                                                                   module_name_='')  # 'net.'
                    loss_r_task.append(loss_r_b)
                    logits_q.append(logits_q_b)
                    output_snap_repr.append(output_snap_q)

                loss_q, acc_q = proto_loss_qry(torch.cat(logits_q, dim=0), y_qry, prototypes)
                corrects[k + 1] = corrects[k + 1] + acc_q
                loss_r_task = float(torch.stack(loss_r_task).mean())
                losses_q[k + 1] = float(loss_q) + loss_r_task * self.ae_weight
            print('accuracy: ', np.array(corrects)/(task+1))

        del net
        accs = np.array(corrects)/len(chosen_class_list)
        loss = np.array(losses_q)/len(chosen_class_list)
        print('type of accuracy', accs, type(accs))

        return accs, loss, output_snap_repr#torch.cat(output_snap_repr,dim=0) #torch.cat(output_snap_repr, dim=0)

    def forward(self, datagenerator, chosen_class_list):
        accs, loss = self.forward_BilevelProto(datagenerator, chosen_class_list)
        return accs, loss

    def finetuning(self, datagenerator, chosen_class_list):
        accs, loss, logits = self.finetunning_BilevelProto(datagenerator, chosen_class_list)
        return accs, loss, logits
