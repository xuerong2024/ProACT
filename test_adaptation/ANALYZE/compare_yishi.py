import pandas as pd
import random
random.seed(49)
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
'''
simple_radio2_200.xlsx: 高年资医师山西测试集
simple_radio1_200_2.xlsx: 中年资医师山西测试集
simple_radio3_200.xlsx: 低年资医师山西测试集

simple_radio1_200.xlsx: 高年资医师贵州测试集
simple_radio2_200_2.xlsx: 中年资医师贵州测试集
simple_radio3_200_2.xlsx: 低年资医师贵州测试集
'''
for experi in range(1,2):
    # 读取CSV文件
    # data_our = pd.read_csv('/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/' + 'shanxi_test_org_dataprob_result.csv')
    # data_come = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/COME/0.0001lr_30batch_10epoch/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_tent = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/TENT/0.0001lr_30batch_10epoch/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_shot = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/SHOT/15batch_0.0001lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_cotta = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/COTTA/8batch_3e-05lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_adacontrast = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/Adacontrast/10batch_3e-05lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_sar = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/SAR/30batch_0.0001lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_rmt = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/RMT/10batch_0.0001lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_roid = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/ROID/10batch_0.0001lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_tea = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/tea/10batch_1e-05lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_dpl = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/DPL_LN/20batch_0.0001lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_efsda = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/EFSDA/10batch_1e-05lr/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_ourtta=pd.read_csv('/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/' + 'shanxi_test_dataprob_result.csv')
    # data_radio1_200=pd.read_excel('/disk3/wjr/dataset/nejm/simple_radio3_200.xlsx')

    data_our = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/' + 'guizhou_test_org_dataprob_result.csv')
    data_come = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/COME/0.0001lr_30batch_10epoch/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_tent = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/TENT/0.0001lr_30batch_10epoch/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_shot = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/SHOT/15batch_0.0001lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_cotta = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/COTTA/8batch_3e-05lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_adacontrast = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/Adacontrast/10batch_3e-05lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_sar = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/SAR/30batch_0.0001lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_rmt = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/RMT/10batch_0.0001lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_roid = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/ROID/10batch_0.0001lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_tea = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/tea/10batch_1e-05lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_dpl = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/DPL_LN/20batch_0.0001lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_efsda = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/EFSDA/10batch_1e-05lr/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_ourtta = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_3e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/' + 'guizhou_test_dataprob_result.csv')
    data_radio1_200 = pd.read_excel('/disk3/wjr/dataset/nejm/simple_radio3_200_2.xlsx')
    labels = data_come['Finding Labels'].values
    # 从 [0, 2000] 范围内随机选择 100 个**不重复**的整数
    # random_numbers = random.sample(range(0, labels.shape[0] - 1), 100)
    # with open('/disk3/wjr/dataset/nejm/shanxi_test_chosen/random100_2.txt', 'r') as f:
    with open('/disk3/wjr/dataset/nejm/guizhou_test_chossen/guizhou_100_2.txt', 'r') as f:
        names_from_txt = f.read().strip().split('\n')
    # to_remove = random.sample(names_from_txt, 17)
    # # 从列表中移除选中的元素
    # for item in to_remove:
    #     names_from_txt.remove(item)
    # for name in names_from_txt:
    #     print(name)
    # 获取特定列的数据内容
    img_name_1 = data_our['Image Index']
    random_numbers = img_name_1[img_name_1.isin(names_from_txt)].index.tolist()

    radio1_pred=[]
    for random_number in random_numbers:
        img_name=data_our['Image Index'][random_number]
        aa=data_radio1_200[data_radio1_200['Image Index'] == img_name].index
        radio1_pred.append(data_radio1_200['Pred Labels'].values[aa][0])
    radio1_pred_array = np.array(radio1_pred)
    labels = data_our['Finding Labels'].values[random_numbers]

    preds_come = data_come['Pred Labels'].values[random_numbers]
    preds_our = data_our['Global_Local Pred Labels'].values[random_numbers]
    # preds_our_tta = data_ourtta['Global Pred Labels'].values[random_numbers]
    preds_our_tta = data_ourtta['Global_Local Pred Labels'].values[random_numbers]
    preds_tent = data_tent['Pred Labels'].values[random_numbers]
    preds_shot = data_shot['Pred Labels'].values[random_numbers]
    preds_cotta = data_cotta['Pred Labels'].values[random_numbers]
    preds_adacontrast = data_adacontrast['Pred Labels'].values[random_numbers]
    preds_sar = data_sar['Pred Labels'].values[random_numbers]
    preds_rmt = data_rmt['Pred Labels'].values[random_numbers]
    preds_roid = data_roid['Pred Labels'].values[random_numbers]
    preds_tea = data_tea['Pred Labels'].values[random_numbers]
    preds_dpl = data_dpl['Pred Labels'].values[random_numbers]
    preds_efsda = data_efsda['Pred Labels'].values[random_numbers]

    aa=len(set(random_numbers))
    print(f"是否有重复: {len(random_numbers) != len(set(random_numbers))}")


    # 将所有模型的预测结果存入一个字典
    models = {
        'radio1': radio1_pred_array,
        'our': preds_our,
        'tent': preds_tent,
        'shot': preds_shot,
        'cotta': preds_cotta,
        'adacontrast': preds_adacontrast,
        'sar': preds_sar,
        'rmt': preds_rmt,
        'roid': preds_roid,
        'tea': preds_tea,
        'dpl': preds_dpl,
        'come':preds_come,
        'efsda':preds_efsda,
        'our_tta':preds_our_tta
    }

    # 存储所有准确率（可选，方便后续排序或绘图）
    accuracies = {}
    kappas = {}
    kappas_radio1 = {}
    # 循环计算每个模型的准确率
    for name, preds in models.items():
        # 计算混淆矩阵的基本元素
        TP = np.sum((preds == "Sick") & (labels == "Sick"))  # 真阳性
        TN = np.sum((preds == "Health") & (labels == "Health"))  # 真阴性
        FP = np.sum((preds == "Sick") & (labels == "Health"))  # 假阳性
        FN = np.sum((preds == "Health") & (labels == "Sick"))  # 假阴性

        # 计算敏感性和特异性
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        acc = (preds == labels).mean()
        accuracies[name] = acc
        kappa = cohen_kappa_score(labels, preds)
        kappas[name]=kappa

        kappa_radio1 = cohen_kappa_score(radio1_pred_array, preds)
        kappas_radio1[name] = kappa_radio1
        print(f'acc_{name}: {acc}, sens_{name}: {sensitivity}, spec_{name}: {specificity}, kappa_{name}_wlabel: {kappa}, kappa_{name}_wradio1: {kappa_radio1}')  # 保留4位小数更专业

