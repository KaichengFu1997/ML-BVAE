import torch

def evaluation(scores_, targets_):
    n, n_class = scores_.shape
    Na, Nc, Np, Ng = torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        Ng[k] = torch.sum(targets == 1) #实际为正
        Np[k] = torch.sum(scores >= 0)  #预测为正
        Nc[k] = torch.sum(targets * (scores >= 0)) #正确预测为正
        Na[k] = (torch.sum((scores < 0) * (targets == 0)) + Nc[k]) / n
    Np[Np == 0] = 1 #防止除0
    OP = torch.sum(Nc) / torch.sum(Np)
    OR = torch.sum(Nc) / torch.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)


    return OP,OR,OF1

def evaluation_topk(scores_, targets_,k):
    n, n_class = scores_.shape
    scores_new = torch.zeros((n, n_class)) - 1
    index = scores_.topk(k, 1, True, True)[1].cpu().numpy()
    for i in range(n):
        for ind in index[i]:
            scores_new[i, ind] = 1 if scores_[i, ind] >= 0 else -1
    scores_ = scores_new.cuda()
    Na, Nc, Np, Ng = torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        Ng[k] = torch.sum(targets == 1) #实际为正
        Np[k] = torch.sum(scores >= 0)  #预测为正
        Nc[k] = torch.sum(targets * (scores >= 0)) #正确预测为正
        Na[k] = (torch.sum((scores < 0) * (targets == 0)) + Nc[k]) / n
    Np[Np == 0] = 1 #防止除0
    OP = torch.sum(Nc) / torch.sum(Np)
    OR = torch.sum(Nc) / torch.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)


    return OP,OR,OF1

def average_precision(scores_, targets_):
    n, n_class = scores_.shape
    ap = torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        sorted, indices = torch.sort(scores, dim=0, descending=True)
        pos_count = 0. #真实为正
        total_count = 0. #所有预测为正
        precision_at_i = 0.
        for i in indices:
            label = targets[i]
            total_count += 1
            if label == 0:
                continue
            else:
                pos_count += 1
                precision_at_i += pos_count / total_count
        ap[k] = precision_at_i / pos_count
    return ap,torch.mean(ap)



