# reference: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
# reference: https://github.com/dragen1860/MAML-Pytorch

import argparse
import time
from maml import BilevelProto
import random
from dataset import *
import psutil
import jsonlines
##############
# Parameters #
##############
parser = argparse.ArgumentParser(description='tempTag')

parser.add_argument("--dir", default="D:/D-PPIN-main", type=str)
parser.add_argument("--labels", default='dblp_ct1_1,dblp_ct1_0,facebook_ct1_1,facebook_ct1_0,tumblr_ct1_1,tumblr_ct1_0,highschool_ct1_1,highschool_ct1_0,infectious_ct1_1,infectious_ct1_0,mit_ct1_1,mit_ct1_0', type=lambda s:[item for item in s.split(',')])
parser.add_argument('--logdir', type=str, default='logs-sd-b1-', help='directory for summaries and checkpoints.')
parser.add_argument('--output_file', type=str, default='output-sd-', help='directory for evaluation accuracy')

# data generate hyperparameters
parser.add_argument("--random_seed", default=2021, type=int)
parser.add_argument("--num_tasks", default=15, type=int, help='num of task max 56')
parser.add_argument("--batch_size", default=10, type=int, help='num of episodes of each batch')
parser.add_argument('--n_way', type=int, default=2, help='number of classes of each task')
parser.add_argument('--k_shot', type=int, default=5, help='number of support examples')
parser.add_argument('--k_query', type=int, default=1, help='number of query examples')
parser.add_argument("--sample_c_n", default=6, type=int, help='total class num in meta training') #default 100
parser.add_argument('--total_sample_g', default={'dblp_ct1_1':755,'dblp_ct1_0':755,'facebook_ct1_1':995,'facebook_ct1_0':995,'highschool_ct1_1':179,'highschool_ct1_0':179, 'infectious_ct1_1':199, 'infectious_ct1_0':199,'mit_ct1_1':79,'mit_ct1_0':79,'tumblr_ct1_1':373, 'tumblr_ct1_0':373}, type=dict, help='number of instances of each classes')
parser.add_argument("--lags", type=int, default=5, help='time lags of generating snapshot')

# training hyperparameters
parser.add_argument('--meta_lr', type=float, default=0.001, help='meta-level outer learning rate')
parser.add_argument('--update_lr', type=float, default=0.0001, help='task-level inner update learning rate')
parser.add_argument('--metatrain_iterations', type=int, default=30, help='meta training iterations')
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5) #original 5
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument("--test_load_epoch", default=20, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--meta_train', default=1, type=int, help='meta training process: 1 or meta testing process: 0')

# model hyperparameters
parser.add_argument('--snapshot_concat', type=str, default='attention', help='snapshot embedding concat method')
parser.add_argument('--use_time', type=str, default='time', help='time encoder method')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--ae_weight', type=float, default=0.7, help='the weight of autoencoder loss')
parser.add_argument('--hop_concat_type', type=str, default='attention', help='fc or attention or mean')
# parser.add_argument('--module_type', type=str, default='sigmoid', help='sigmoid')
parser.add_argument('--nhid', type=int, default=32, help='Number of hidden units.') #32
parser.add_argument('--nfeat', type=int, default=1, help='Number of feature dimensions')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for tgan attention aggregation')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--num_neighbors", default=10, type=int)

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.device))
torch.backends.cudnn.benchmark = True

random.seed(args.random_seed)
np.random.seed(args.random_seed)

flag = 1
args.labels = np.array(args.labels)
exp_string = "metalr.{}_".format(args.meta_lr)  + "innerlr.{}_".format(args.update_lr) + "hidden.{}_".format(args.nhid) + "n_way.{}_".format(args.n_way) + "k_shot.{}_".format(args.k_shot) + "k_query.{}_".format(args.k_query)


SAVE_EPOCH = 10


def train(args, model, data_generator=None,
          verbose=True):
    if verbose:
        print('Begin training...')
    max_memory = 0

    loss_list, acc_list = [], []

    output_dir = args.logdir + '/' + exp_string + '/'

    print('loading training data...')
    start_time = time.time()
    choice_range_class = np.arange(0, args.sample_c_n)
    # labels = np.array(['facebook_ct1_'+str(x) for x in args.labels])

    chosen_class_list = []
    for i in range(args.num_tasks):
        class_id = np.random.choice(choice_range_class, args.n_way, replace=False)
        
        chosen_class_list.append(args.labels[class_id])
    chosen_class_list = np.array(chosen_class_list)


    for epoch in range(1, args.metatrain_iterations+1):
        acc, loss_q = model(data_generator, chosen_class_list) #device

        max_memory = max(max_memory, float(psutil.virtual_memory().used/(1024**3)))
        print(epoch, loss_q, acc)
        loss_list.append(loss_q)
        acc_list.append(acc)


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if epoch % SAVE_EPOCH == 0 and epoch != 0:
            print(epoch, loss_q, acc)
            print('time: ', time.time()-start_time)
            torch.save(model.state_dict(), output_dir + 'model_epoch_{}'.format(epoch))

        output = {'epoch num': epoch, 'meta_train_loss': np.array(loss_list).tolist(), 'meta_train_accuracy': np.array(acc_list).tolist()}


        with jsonlines.open(output_dir + args.output_file, mode='a') as writer:
            writer.write(output)
    print(loss_list, acc_list)
    print("avg_loss:{}".format(np.array(loss_list).mean(axis=0)))
    print("avg_acc:{}".format(np.array(acc_list).mean(axis=0)))
    if verbose:
        print('Finished.')



def evaluate(args, model, data_generator=None,
             verbose=True):
    if verbose:
        print('Begin evaluating...')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')
    output_dir = args.logdir + '/' + exp_string + '/'


    print('Loading meta testing samples...')
    choice_range_class = np.arange(args.sample_c_n, len(args.labels))
    chosen_class_list = []
    for i in range(args.num_tasks):
        class_id = np.random.choice(choice_range_class, args.n_way, replace=False)
        chosen_class_list.append(args.labels[class_id])
    chosen_class_list = np.array(chosen_class_list)


    print('start evaluating')
    loss_list, acc_list = [], []
    for epoch in range(args.test_load_epoch):
        acc, loss_q = model.finetuning(data_generator, chosen_class_list)
        loss_list.append(loss_q) #.item()
        acc_list.append(acc) #.tolist()
        print(epoch, acc, loss_q)
        # ipdb.set_trace()


        output = {'epoch num': epoch, 'meta_test_accuracy': np.array(acc_list).tolist()}
        with jsonlines.open(output_dir + args.output_file, mode='a') as writer:
            writer.write(output)

    loss = np.array(loss_list).mean(axis=0)
    acc = np.array(acc_list).mean(axis=0)
    print("testing results: loss is {}, acc is {}".format(loss, acc))


    if verbose:
        print('Finished.')


def main():
    data_generator = ReadDataset(args)

    model = BilevelProto(args).to(device)
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    train(args, model, data_generator)

    print('evaluate')
    args.meta_train = 0
    model_dir = args.logdir + '/' + exp_string + '/' + 'model_epoch_{}'.format(30)

    model.load_state_dict(torch.load(model_dir))

    evaluate(args, model, data_generator)



# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
