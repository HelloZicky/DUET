from sklearn import metrics
import heapq
import numpy as np
import math
import itertools
import torch


def calculate_user_ndcg_hr(n=5, *buffer):
    user_num = 0
    ndcg_ = 0
    hr_ = 0
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = sorted(actions, key=lambda x: x[1], reverse=True)
        top_items = np.array(actions)[:n, :]
        num_postive = int(sum(np.array(actions)[:, 2]))
        if not 0 < num_postive:
            continue
        dcg = 0
        idcg = 0
        for i, (user_id, score, label) in enumerate(top_items):
            if label == 1:
                dcg += math.log(2) / math.log(i + 2)
            if i < num_postive:
                idcg += math.log(2) / math.log(i + 2)

        ndcg_ += dcg / idcg
        hr_ += 1 if any(item[2] for item in top_items) else 0
        user_num += 1

    ndcg = ndcg_ / user_num
    hr = hr_ / user_num
    return ndcg, hr


def calculate_overall_logloss(*buffer):
    prob, y = buffer
    logloss = float(metrics.log_loss(np.array(y), prob))

    return logloss


def calculate_overall_auc(*buffer):
    prob, y = buffer
    fpr, tpr, thresholds = [], [], []
    auc = float(metrics.roc_auc_score(np.array(y), prob))

    return auc, fpr, tpr, thresholds


def calculate_user_auc(*buffer):
    user_num = 0
    auc_ = 0
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = list(actions)
        prob = np.array(actions)[:, 1]
        y = np.array(actions)[:, 2]
        if not 0 < np.sum(y) < len(y):
            continue
        auc_ += float(metrics.roc_auc_score(np.array(y), prob))
        user_num += 1
    auc = auc_ / user_num
    return auc


def calculate_user_auc_misrec(*buffer):
    user_num = 0
    auc_ = 0
    fig1_list = []
    for user_id, actions in itertools.groupby(buffer[0], key=lambda x: x[0]):
        actions = list(actions)
        prob = np.array(actions)[:, 1]
        prob_trigger = np.array(actions)[:, 2]
        y = np.array(actions)[:, 3]
        mis_rec = np.mean(np.array(actions)[:, 4])
        auc_item = float(metrics.roc_auc_score(np.array(y), prob))
        auc_item_trigger = float(metrics.roc_auc_score(np.array(y), prob_trigger))
        auc_ += auc_item
        fig1_list.append([auc_item, auc_item_trigger, mis_rec])
        user_num += 1
    auc = auc_ / user_num
    return auc, fig1_list


def calculate_overall_acc(*buffer):
    prob, y = buffer
    acc = metrics.accuracy_score(y, prob)

    return acc


def calculate_overall_roc(*buffer):
    prob, y = buffer
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), prob, pos_label=1)

    return fpr, tpr


def calculate_overall_recall(*buffer):
    prob, y = buffer
    recall = metrics.recall_score(y, prob, average="macro")
    return recall


def calculate_overall_precision(*buffer):
    prob, y = buffer
    precision = metrics.precision_score(y, prob, average="macro")
    return precision


if __name__ == '__main__':
    records = [
        [3, 0.151, 0],
        [3, 0.152, 0],
        [3, 0.103, 0],
        [3, 0.174, 1],
        [3, 0.135, 0],
        [3, 0.126, 0],
        [1, 0.151, 1],
        [1, 0.152, 0],
        [1, 0.103, 0],
        [1, 0.174, 0],
        [1, 0.135, 0],
        [2, 0.151, 0],
        [2, 0.152, 0],
        [2, 0.103, 1],
        [2, 0.174, 0],
        [2, 0.135, 0],
        [2, 0.126, 0],
    ]
    # ndcg, hr = calculate_user_ndcg_hr(np.array(records)[:, 0], np.array(records)[:, 1], np.array(records)[:, 2])
    # ndcg, hr = calculate_user_ndcg_hr(np.hstack(np.array(records)[:, 0], np.array(records)[:, 1], np.array(records)[:, 2]))
    # ndcg, hr = calculate_user_ndcg_hr(torch.Tensor(np.array(records)[:, 0]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 1]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 2]).view(-1, 1))
    # user_id_list = torch.Tensor([3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,]).view(-1, 1)
    # score_list = torch.Tensor([0.15, 0.15, 0.10, 0.17, 0.13, 0.12, 0.15, 0.15, 0.10, 0.17, 0.13, 0.15, 0.15, 0.10, 0.17, 0.13, 0.12,]).view(-1, 1)
    # label_list = torch.Tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,]).view(-1, 1)
    # print(torch.Tensor(np.array(records)[:, 0]).view(-1, 1))
    # ndcg, hr = calculate_user_ndcg_hr(torch.cat(torch.Tensor(np.array(records)[:, 0]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 1]).view(-1, 1),
    #                                   torch.Tensor(np.array(records)[:, 2]).view(-1, 1), dim=1))
    # ndcg, hr = calculate_user_ndcg_hr(user_id_list, score_list, label_list)
    ndcg, hr = calculate_user_ndcg_hr(records)
    print(ndcg, hr)
