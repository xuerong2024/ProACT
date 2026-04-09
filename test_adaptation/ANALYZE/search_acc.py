import pandas as pd
import random
random.seed(49)
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
for experi in range(1,2):
    # 读取CSV文件
    data_resnet = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/resnet50/wmask_nosubcls_nodropout2/resnet50_lr8e-05/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_convnext = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnext_wmask_nosubcls_nodropout2/convnext_lr5e-05/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_dinov2 = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/dinov2/wmask_nosubcls_nodropout2/dinov2_lr3e-05_testall/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_our = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/analyze/' + 'new_shanxi_test_global_local_dataprob_result.csv')
    data_swin = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/swintiny/wmask_nosubcls_nodropout2/swintiny_lr0.0001/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_pcam = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/pcam/wmask_nosubcls_nodropout2/pcam_lr3e-05/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_mambaout = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/mambaout/wmask_nosubcls_nodropout2/mambaout_lr3e-05_testall/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_vmamba = pd.read_csv(
        '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vmamba/lr_3e-05/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_slgms = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    # data_slgms = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/analyze/' + 'shanxi_test_dataprob_result.csv')
    data_dmalnet=pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/DMALNET/_lr5e-05/models_params/RR_avgpool/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    # data_pneullm = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/llm_lr0.0003_weightsseed_10/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_pneullm = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/shanxi_test_data_result.csv')
    data_vit = pd.read_csv(
        '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vit-base/wmask_nosubcls_nodropout2/vit-base_lr3e-05/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_pka2net = pd.read_csv(
        '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/PKA2_Net/wmask_nosubcls_nodropout2/PKA2_Net_lr5e-05_testall/analyze/' + 'shanxi_test_global_dataprob_result.csv')
    data_medmamba = pd.read_csv(
        '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/medmamba/wmask_nosubcls_nodropout2/medmamba_lr3e-05_testall/analyze/' + 'shanxi_test_global_dataprob_result.csv')

    data_radio1_200=pd.read_excel('/disk3/wjr/dataset/nejm/simple_radio3_200.xlsx')

    # data_resnet = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/resnet50/wmask_nosubcls_nodropout2/resnet50_lr8e-05/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_convnext = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnext_wmask_nosubcls_nodropout2/convnext_lr5e-05/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_dinov2 = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/dinov2/wmask_nosubcls_nodropout2/dinov2_lr3e-05_testall/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_our = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/analyze/' + 'new_guizhou_test_global_local_dataprob_result.csv')
    # data_swin = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/swintiny/wmask_nosubcls_nodropout2/swintiny_lr0.0001/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_pcam = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/pcam/wmask_nosubcls_nodropout2/pcam_lr3e-05/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_mambaout = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/mambaout/wmask_nosubcls_nodropout2/mambaout_lr3e-05_testall/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_vmamba = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vmamba/lr_3e-05/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # # data_slgms = pd.read_csv(
    # #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_slgms = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs_2/s-vmamba_5e-5/analyze/' + 'guizhou_test_dataprob_result.csv')
    #
    # data_dmalnet = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/DMALNET/_lr5e-05/models_params/RR_avgpool/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # # data_pneullm = pd.read_csv(
    # #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/llm_lr0.0003_weightsseed_10/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_pneullm = pd.read_csv(
    #     '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/87_83_llm_lr0.0008_weightsseed_10/guizhou_test_data_result.csv')
    # data_vit = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vit-base/wmask_nosubcls_nodropout2/vit-base_lr3e-05/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_pka2net = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/PKA2_Net/wmask_nosubcls_nodropout2/PKA2_Net_lr5e-05_testall/analyze/' + 'guizhou_test_global_dataprob_result.csv')
    # data_medmamba = pd.read_csv('/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/medmamba/wmask_nosubcls_nodropout2/medmamba_lr3e-05_testall/analyze/' + 'guizhou_test_global_dataprob_result.csv')

    labels = data_resnet['Finding Labels'].values
    # 从 [0, 2000] 范围内随机选择 100 个**不重复**的整数
    # random_numbers = random.sample(range(0, labels.shape[0] - 1), 100)
    with open('/disk3/wjr/dataset/nejm/shanxi_test_chosen/random100_2.txt', 'r') as f:
    # with open('/disk3/wjr/dataset/nejm/guizhou_test_chossen/guizhou_100_2.txt', 'r') as f:
        names_from_txt = f.read().strip().split('\n')
    # to_remove = random.sample(names_from_txt, 17)
    # # 从列表中移除选中的元素
    # for item in to_remove:
    #     names_from_txt.remove(item)
    # for name in names_from_txt:
    #     print(name)
    # 获取特定列的数据内容
    img_name_1 = data_resnet['Image Index']
    random_numbers = img_name_1[img_name_1.isin(names_from_txt)].index.tolist()
    img_name_1=data_resnet['Image Index'][random_numbers]
    img_name_2 = data_convnext['Image Index'][random_numbers]
    img_name_3 = data_dinov2['Image Index'][random_numbers]
    img_name_4 = data_our['Image Index'][random_numbers]
    img_name_5 = data_swin['Image Index'][random_numbers]
    img_name_6 = data_pcam['Image Index'][random_numbers]
    img_name_7 = data_slgms['Image Index'][random_numbers]
    img_name_8 = data_mambaout['Image Index'][random_numbers]
    img_name_9 = data_vmamba['Image Index'][random_numbers]
    img_name_10 = data_dmalnet['Image Index'][random_numbers]
    img_name_11 = data_pneullm['Image Index'][random_numbers]
    img_name_12 = data_vit['Image Index'][random_numbers]
    img_name_13 = data_pka2net['Image Index'][random_numbers]
    img_name_14 = data_medmamba['Image Index'][random_numbers]

    if (img_name_1 != img_name_2).any() or (img_name_1 != img_name_3).any() or (img_name_1 != img_name_4).any() or (
            img_name_1 != img_name_5).any() or (img_name_1 != img_name_6).any() or (img_name_1 != img_name_7).any() or (
            img_name_1 != img_name_8).any() or (img_name_1 != img_name_11).any() or (img_name_1 != img_name_14).any() or (
            img_name_1 != img_name_12).any() or (img_name_1 != img_name_13).any():
        print('False!!!')

    radio1_pred=[]
    for random_number in random_numbers:
        img_name=data_resnet['Image Index'][random_number]
        aa=data_radio1_200[data_radio1_200['Image Index'] == img_name].index
        radio1_pred.append(data_radio1_200['Pred Labels'].values[aa][0])
    radio1_pred_array = np.array(radio1_pred)
    labels = data_resnet['Finding Labels'].values[random_numbers]


    preds_resnet = data_resnet['Pred Labels'].values[random_numbers]
    preds_slgms = data_slgms['Pred Labels'].values[random_numbers]
    preds_convnext = data_convnext['Pred Labels'].values[random_numbers]
    preds_dinov2 = data_dinov2['Pred Labels'].values[random_numbers]
    preds_our = data_our['Pred Labels'].values[random_numbers]
    preds_swin = data_swin['Pred Labels'].values[random_numbers]
    preds_pcam = data_pcam['Pred Labels'].values[random_numbers]
    preds_mambaout = data_mambaout['Pred Labels'].values[random_numbers]
    preds_vmamba = data_vmamba['Pred Labels'].values[random_numbers]
    preds_dmalnet = data_dmalnet['Pred Labels'].values[random_numbers]
    preds_pneullm = data_pneullm['Pred Labels'].values[random_numbers]
    preds_vit = data_vit['Pred Labels'].values[random_numbers]
    preds_pka2net = data_pka2net['Pred Labels'].values[random_numbers]
    preds_medmamba = data_medmamba['Pred Labels'].values[random_numbers]


    aa=len(set(random_numbers))
    print(f"是否有重复: {len(random_numbers) != len(set(random_numbers))}")


    # 将所有模型的预测结果存入一个字典
    models = {
        'radio1': radio1_pred_array,
        'our': preds_our,
        'resnet': preds_resnet,
        'dmalnet': preds_dmalnet,
        'convnext': preds_convnext,
        'pcam': preds_pcam,
        'swin': preds_swin,
        'dinov2': preds_dinov2,
        'vmamba': preds_vmamba,
        'mambaout': preds_mambaout,
        'slgms': preds_slgms,
        'pneullm': preds_pneullm,
        'vit': preds_vit,
        'medmamba':preds_medmamba,
        'pka2net':preds_pka2net
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

