import os
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model import MVAE
from utils import *
from aslloss import *
from sklearn.model_selection import KFold
torch.cuda.set_device(0)

def elbo_loss(recon_left, left, recon_right, right, recon_diff,diff,mu, logvar,
              lambda_left=1.0, lambda_right=1.0, lambda_diff = 1.0,annealing_factor=1.0):

    left_bce, right_bce,diff_bce = 0, 0,0  # default params
    if recon_left is not None and left is not None:
        left_bce = F.mse_loss(recon_left, left)

    if recon_right is not None and right is not None:
        right_bce = F.mse_loss(recon_right, right)

    if recon_diff is not None and diff is not None:
        diff_bce = F.mse_loss(recon_diff, diff)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    ELBO = lambda_left * left_bce + lambda_right * right_bce + lambda_diff * diff_bce + annealing_factor * KLD
    # print('image_bce=',image_bce,'text_bce=',text_bce,'KLD=',KLD)
    return ELBO


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-left', type=float, default=1.,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-right', type=float, default=1.,
                        help='multipler for text reconstruction [default: 1]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: True]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()



    sub_data = np.load('data_vox8_sub1_lr.npy')
    label = np.load('label.npy')
    label = np.transpose(label)
    label = torch.from_numpy(label)
    sub_data = np.transpose(sub_data, (2, 0, 1))



    left = sub_data[:, 0:180, :].reshape(2196, -1)
    right= sub_data[:, 180:360, :].reshape(2196, -1)
    n_splits = 10
    k_fold = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    map_kfold = torch.zeros(n_splits+1)
    map_class_kfold = torch.zeros(n_splits + 1, 27)
    overall = torch.zeros(n_splits + 1, 3)
    topk = torch.zeros(n_splits + 1, 3)


    #
    # sub_data = sub_data.reshape(2196, -1)
    # v_f = np.transpose(v_f)
    left = torch.from_numpy(left)
    right = torch.from_numpy(right)
    diff = left-right
    cost = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
    elbo_ratio = 1.5
    classify_ratio = 0.1




    def train_model(epoch,train_loader,model,optimizer,train_x0,train_x1,train_x2,train_y):
        model.train()
        train_loss_meter = AverageMeter()
        train_elbo_loss_meter = AverageMeter()
        train_classify_loss_meter = AverageMeter()
        # NOTE: is_paired is 1 if the example is paired
        N_mini_batches = len(train_loader)
        for batch_idx, (left, right,diff, labels) in enumerate(train_loader):
            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0
            if args.cuda:
                left = left.cuda()
                right = right.cuda()
                diff = diff.cuda()
                labels = labels.cuda()
            left = Variable(left)
            right = Variable(right)
            diff = Variable(diff)
            labels = Variable(labels)
            batch_size = len(left)

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            recon_left_1, recon_right_1, recon_diff_1,mu_1, logvar_1, y1 = model(left, right,diff)
            recon_left_2, recon_right_2, recon_diff_2,mu_2, logvar_2, y2 = model(left)
            recon_left_3, recon_right_3, recon_diff_3,mu_3, logvar_3, y3 = model(right=right)
            recon_left_4, recon_right_4, recon_diff_4, mu_4, logvar_4, y4 = model(diff=diff)

            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_left_1, left, recon_right_1, right, recon_diff_1,diff,mu_1, logvar_1,
                                   lambda_left=args.lambda_left, lambda_right=args.lambda_right,lambda_diff=1,
                                   annealing_factor=annealing_factor)
            left_loss = elbo_loss(recon_left_2, left, recon_right_2, right, recon_diff_2,diff,mu_2, logvar_2,
                                   lambda_left=args.lambda_left, lambda_right=args.lambda_right,lambda_diff=1,
                                   annealing_factor=annealing_factor)
            right_loss = elbo_loss(recon_left_3, left, recon_right_3, right,recon_diff_3,diff, mu_3, logvar_3,
                                  lambda_left=args.lambda_left, lambda_right=args.lambda_right,lambda_diff=1,
                                  annealing_factor=annealing_factor)
            diff_loss = elbo_loss(recon_left_4, left, recon_right_4, right,recon_diff_4,diff, mu_4, logvar_4,
                                  lambda_left=args.lambda_left, lambda_right=args.lambda_right,lambda_diff=1,
                                  annealing_factor=annealing_factor)

            classify_loss = cost(y1, labels) + cost(y2, labels) + cost(y3, labels) + cost(y4,labels)
            train_loss = elbo_ratio * (joint_loss + left_loss + right_loss + diff_loss) + classify_ratio * classify_loss
            train_loss_meter.update(train_loss.data, batch_size)
            train_elbo_loss_meter.update(joint_loss + left_loss + right_loss, batch_size)
            train_classify_loss_meter.update(classify_loss, batch_size)
            # compute gradients and take step
            train_loss.backward()
            optimizer.step()


        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))
        model.eval()
        _, _, _, _,_, y = model(train_x0.cuda(), train_x1.cuda(),train_x2.cuda())
        _, map_train = average_precision(y, train_y)
        print('map_train', map_train)
        return train_loss_meter.avg, train_elbo_loss_meter.avg, train_classify_loss_meter.avg, map_train


    def test_model(epoch,test_loader,model,test_x0,test_x1,test_x2,test_y):
        model.eval()
        test_loss_meter = AverageMeter()
        test_elbo_loss_meter = AverageMeter()
        test_classify_loss_meter = AverageMeter()
        for batch_idx, (left,right,diff ,labels) in enumerate(test_loader):
            if args.cuda:
                left = left.cuda()
                right = right.cuda()
                diff = diff.cuda()
                labels = labels.cuda()
            left = Variable(left, volatile=True)
            right = Variable(right, volatile=True)
            diff = Variable(diff, volatile=True)
            labels = Variable(labels, volatile=True)
            batch_size = len(left)

            recon_left_1, recon_right_1,recon_diff_1, mu_1, logvar_1, y1 = model(left, right,diff)
            recon_left_2, recon_right_2, recon_diff_2,mu_2, logvar_2, y2 = model(left)
            recon_left_3, recon_right_3, recon_diff_3,mu_3, logvar_3, y3 = model(right=right)
            recon_left_4, recon_right_4, recon_diff_4, mu_4, logvar_4, y4 = model(diff = diff)

            joint_loss = elbo_loss(recon_left_1, left, recon_right_1, right, recon_diff_1,diff,mu_1, logvar_1)
            left_loss = elbo_loss(recon_left_2, left, recon_right_2, right, recon_diff_2,diff,mu_2, logvar_2)
            right_loss = elbo_loss(recon_left_3, left, recon_right_3, right,recon_diff_3,diff, mu_3, logvar_3)
            diff_loss = elbo_loss(recon_left_4, left, recon_right_4, right, recon_diff_4, diff, mu_4, logvar_4)

            classify_loss = cost(y1, labels) + cost(y2, labels) + cost(y3, labels) + cost(y4,labels)
            test_loss = elbo_ratio * (joint_loss + left_loss + right_loss + diff_loss) + classify_ratio * classify_loss
            test_loss_meter.update(test_loss.data, batch_size)
            test_elbo_loss_meter.update(joint_loss + left_loss + right_loss, batch_size)
            test_classify_loss_meter.update(classify_loss, batch_size)
        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        _, _, _, _, _,y = model(test_x0.cuda(), test_x1.cuda(),test_x2.cuda())
        _, map_test = average_precision(y, test_y)
        print('map_test', map_test)

        return test_loss_meter.avg, test_elbo_loss_meter.avg, test_classify_loss_meter.avg


    #### eval ####
    for k, (train, test) in enumerate(k_fold.split(left, label)):
        print('this is %d fold' % (k + 1))
        train_xl = left[train]
        train_xr = right[train]
        train_xd = diff[train]
        train_y = label[train]

        test_xl = left[test]
        test_xr = right[test]
        test_xd = diff[test]
        test_y = label[test]

        H = torch.zeros(27, 27)
        for j in range(27):
            N_j = torch.sum(train_y[:, j])
            for kt in range(27):
                p = torch.where(train_y[:, j] == 1)
                N_jk = torch.sum(train_y[p[0], kt])
                H[j, kt] = N_jk / N_j
        # H = H - torch.eye(27)
        H = H.cuda()

        train_dataset = torch.utils.data.TensorDataset(train_xl, train_xr, train_xd,train_y)
        test_dataset = torch.utils.data.TensorDataset(test_xl,test_xr,test_xd,test_y)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        torch.set_default_tensor_type(torch.DoubleTensor)
        model = MVAE(n_inputs=1440,n_outputs=1440,n_latents=args.n_latents,H=H)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
        if args.cuda:
            model.cuda()
        # cost = torch.nn.MultiLabelSoftMarginLoss()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_elbo_loss, train_classify_loss, map_train = train_model(epoch,train_loader=train_loader,model=model,optimizer=optimizer,train_x0=train_xl,train_x1=train_xr,train_x2 = train_xd,train_y = train_y)
            test_loss, test_elbo_loss, test_classify_loss = test_model(epoch,test_loader,model=model,test_x0=test_xl,test_x1=test_xr,test_x2 = test_xd,test_y = test_y)
        ### test whole brain ###
        _, _, _, _,_ ,y = model(test_xl.cuda(), test_xr.cuda(),test_xd.cuda())
        a,map_kfold[k] = average_precision(y, test_y.cuda())
        overall[k] = torch.tensor(evaluation(y, test_y.cuda())).reshape(1, 3)
        topk[k] = torch.tensor(evaluation_topk(y, test_y.cuda(), 5)).reshape(1, 3)
        map_class_kfold[k] = a.reshape(1, 27)

    overall[-1] = torch.mean(overall[:n_splits], dim=0)
    topk[-1] = torch.mean(topk[:n_splits], dim=0)
    map_kfold[-1] = torch.mean(map_kfold[:n_splits], dim=0)
    map_class_kfold[-1] = torch.mean(map_class_kfold[:n_splits], dim=0)

    a = './sub1_result'
    if not os.path.isdir(a):
        os.makedirs(a)
    np.savetxt(a+'/map_kfold.csv', map_kfold)
    np.savetxt(a+'/map_class_kfold.csv', map_class_kfold, delimiter=',')
    np.savetxt(a+'/overall.csv', overall, delimiter=',')
    np.savetxt(a+'/topk.csv', topk, delimiter=',')



