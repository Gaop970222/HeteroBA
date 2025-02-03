from distutils.command.clean import clean

import torch

def evaluation(model_name, victim_backdoor_model, victim_clean_model, posion_testset_index,\
               clean_testset_index,poison_labels,clean_labels,poison_hg,clean_hg, target_type,testset_index):
    logits_clean = victim_clean_model.evaluate(poison_hg)
    logits_poison = victim_backdoor_model.evaluate(poison_hg)

    pred_clean = logits_clean.argmax(1).cpu()
    pred_poison = logits_poison.argmax(1).cpu()
    clean_test_acc = (pred_clean[clean_testset_index]==clean_labels[clean_testset_index]).float().mean()
    posion_test_acc = (pred_poison[clean_testset_index]==clean_labels[clean_testset_index]).float().mean()
    CAD = clean_test_acc-posion_test_acc
    print("clean_test_acc:",clean_test_acc)
    print("posion_test_acc:",posion_test_acc)
    print("CAD is ", CAD.item())
    wrong_indices = torch.tensor(posion_testset_index, dtype=torch.long)[torch.where(pred_poison[torch.tensor(posion_testset_index, dtype=torch.long)] != poison_labels[torch.tensor(posion_testset_index, dtype=torch.long)])[0]]
    ASR =  (pred_poison[posion_testset_index]==poison_labels[posion_testset_index]).float().mean().item()
    print("ASR is ", ASR)
    accuracy = calculate_accuracy(model_name, clean_hg, target_type, victim_clean_model,clean_labels, posion_testset_index)
    return CAD, ASR, accuracy

def calculate_accuracy(model_name,clean_hg,target_type, victim_clean_model,clean_labels, posion_testset_index):
    """
    计算在正常情况下的准确率。

    参数:
    pred_clean: 预测结果的张量
    clean_labels: 真实标签的张量
    testset_index: 测试集的索引

    返回:
    准确率(float)
    """
    logits =  victim_clean_model.evaluate(clean_hg)
    # if model_name == 'HGT':
    #     logits= victim_clean_model(clean_hg,target_type)
    # elif model_name == 'RGCN':
    #     logits = victim_clean_model(clean_hg, clean_hg.ndata['inp'])
    # elif model_name == 'HAN':
    #     logits = victim_clean_model(clean_hg, clean_hg.ndata['inp'][target_type])
    # else:
    #     raise NotImplementedError
    pred = logits.argmax(1).cpu()
    accuracy = (pred[posion_testset_index] == clean_labels[posion_testset_index]).float().mean().item()
    print("Under Normal Circumstances, accuracy is", accuracy)
    return accuracy