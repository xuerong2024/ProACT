import pandas as pd
for experi in range(1,2):
    # 读取CSV文件
    data_shanxi_calgm = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_org_dataprob_result.csv')
    data_shanxi_calgm_medtta = pd.read_csv('/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_dataprob_result.csv')
    data_guizhou_calgm = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_org_dataprob_result.csv')
    data_guizhou_calgm_medtta = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_dataprob_result.csv')

    data_shanxi_slgms = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/SLGMS/OUR/3e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_org_dataprob_result.csv')
    data_shanxi_slgms_medtta = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/SLGMS/OUR/3e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_dataprob_result.csv')
    data_guizhou_slgms = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/SLGMS/OUR/3e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_org_dataprob_result.csv')
    data_guizhou_slgms_medtta = pd.read_csv(
        '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/SLGMS/OUR/3e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_dataprob_result.csv')

    # # 获取特定列的数据内容
    img_name_4 = data_shanxi_calgm['Image Index']
    img_name_7 = data_shanxi_calgm_medtta['Image Index']

    # if (img_name_4 != img_name_7).any():
    #     print('False19!!!')
    # labels = data_shanxi_calgm['Finding Labels'].values
    # preds_shanxi_slgms = data_shanxi_slgms['Global Pred Labels'].values
    # preds_shanxi_slgms_medtta = data_shanxi_slgms_medtta['Global Pred Labels'].values
    # predss_shanxi_slgms = data_shanxi_slgms['Global Pred logits'].values
    # predss_shanxi_slgms_medtta = data_shanxi_slgms_medtta['Global Pred logits'].values
    # preds_shanxi_calgm = data_shanxi_calgm['Global_Local Pred Labels'].values
    # preds_shanxi_calgm_medtta = data_shanxi_calgm_medtta['Global_Local Pred Labels'].values
    # predss_shanxi_calgm = data_shanxi_calgm['Global_Local Pred logits'].values
    # predss_shanxi_calgm_medtta = data_shanxi_calgm_medtta['Global_Local Pred logits'].values
    # img_name_1 = data_shanxi_calgm['Image Index'].values
    #
    # correct_nums=[]
    # img_names=[]
    # for ii in range(labels.shape[0]):
    #     correct_num = 0
    #     shanxi_calgm_index=0
    #     shanxi_calgm_medtta_index=0
    #     if labels[ii]=='Sick':
    #         if preds_shanxi_slgms[ii] == labels[ii]:
    #             if preds_shanxi_slgms_medtta[ii] == labels[ii]:
    #                 if preds_shanxi_calgm[ii] != labels[ii]:
    #                     if preds_shanxi_calgm_medtta[ii] == labels[ii]:
    #                         # print(img_name_1[ii])
    #                         print(img_name_1[ii], f'calgm org:{predss_shanxi_calgm[ii]}',
    #                                                             f' medtta:{predss_shanxi_calgm_medtta[ii]}', f'slgms org:{predss_shanxi_slgms[ii]}',
    #                                                             f' medtta:{predss_shanxi_slgms_medtta[ii]}')
    #
    #                 # correct_num += 1
    #                 # # print(img_name_1[ii])
    #                 # print(img_name_1[ii], f' org:{predss_shanxi_calgm[ii]}', f' medtta:{predss_shanxi_calgm_medtta[ii]}')



    # 获取特定列的数据内容
    img_name_4 = data_guizhou_calgm['Image Index']
    img_name_7 = data_guizhou_calgm_medtta['Image Index']

    if (img_name_4 != img_name_7).any():
        print('False19!!!')
    labels = data_guizhou_calgm['Finding Labels'].values
    preds_guizhou_slgms = data_guizhou_slgms['Global Pred Labels'].values
    preds_guizhou_slgms_medtta = data_guizhou_slgms_medtta['Global Pred Labels'].values
    predss_guizhou_slgms = data_guizhou_slgms['Global Pred logits'].values
    predss_guizhou_slgms_medtta = data_guizhou_slgms_medtta['Global Pred logits'].values
    img_name_1 = data_guizhou_calgm['Image Index'].values
    preds_guizhou_calgm = data_guizhou_calgm['Global_Local Pred Labels'].values
    preds_guizhou_calgm_medtta = data_guizhou_calgm_medtta['Global_Local Pred Labels'].values
    predss_guizhou_calgm = data_guizhou_calgm['Global_Local Pred logits'].values
    predss_guizhou_calgm_medtta = data_guizhou_calgm_medtta['Global_Local Pred logits'].values

    correct_nums=[]
    img_names=[]
    for ii in range(labels.shape[0]):
        correct_num = 0
        guizhou_calgm_index=0
        guizhou_calgm_medtta_index=0
        if labels[ii]=='Sick':
            if preds_guizhou_calgm[ii] != labels[ii]:
                if preds_guizhou_calgm_medtta[ii] == labels[ii]:
                    if preds_guizhou_slgms[ii] != labels[ii]:
                        if preds_guizhou_slgms_medtta[ii] == labels[ii]:
                            # print(img_name_1[ii])
                            print(img_name_1[ii], f'calgm org:{predss_guizhou_calgm[ii]}',
                                  f' medtta:{predss_guizhou_calgm_medtta[ii]}', f'slgms org:{predss_guizhou_slgms[ii]}',
                                  f' medtta:{predss_guizhou_slgms_medtta[ii]}')

                    # correct_num += 1
                    # # print(img_name_1[ii])
                    # print(img_name_1[ii], f' org:{predss_guizhou_calgm[ii]}', f' medtta:{predss_guizhou_calgm_medtta[ii]}')
