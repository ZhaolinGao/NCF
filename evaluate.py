import numpy as np
import torch


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users

def metrics(model, user_num, item_num, train_data, test_data, top_k):

    batch_size = 100
    current_user = 0
    predictions = []

    for i in range(user_num//batch_size + 1):
        if current_user+batch_size >= user_num:
            users = np.arange(current_user, user_num)
            size = user_num-current_user
        else:
            users = np.arange(current_user, current_user+batch_size)
            current_user += batch_size
            size = batch_size

        users = np.expand_dims(np.repeat(users, item_num), axis=1)
        items = np.expand_dims(np.arange(item_num), axis=1)
        items = np.repeat(items, size, axis=1).transpose().reshape((-1, 1))

        prediction = model(torch.tensor(users).cuda().long(), torch.tensor(items).cuda().long()).view(size, item_num).detach().cpu().numpy()
        predictions.append(prediction)

    predictions = np.concatenate(predictions, axis=0)

    topk = 50
    predictions[train_data.nonzero()] = np.NINF

    ind = np.argpartition(predictions, -topk)
    ind = ind[:, -topk:]
    arr_ind = predictions[np.arange(len(predictions))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predictions)), ::-1]
    pred_list = ind[np.arange(len(predictions))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20, 50]:
        recall.append(recall_at_k(test_data, pred_list, k))

    all_ndcg = ndcg_func([*test_data.values()], pred_list[list(test_data.keys())])
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20, 50]]

    return recall, ndcg


    # HR, NDCG = [[] for k in top_k], [[] for k in top_k]



    # for user, item, label in test_loader:
    #     user = user.cuda()
    #     item = item.cuda()

    #     predictions = model(user, item)

    #     for k in range(len(top_k)):
    #         _, indices = torch.topk(predictions, top_k[k])
    #         recommends = torch.take(
    #                 item, indices).cpu().numpy().tolist()

    #         gt_item = item[0].item()
    #         HR[k].append(hit(gt_item, recommends))
    #         NDCG[k].append(ndcg(gt_item, recommends))

    # return np.mean(HR, axis=1), np.mean(NDCG, axis=1)
