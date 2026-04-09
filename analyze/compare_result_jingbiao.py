import pandas as pd
import re
jingbiao_csvpath = "/disk1/wjr/dataset/shanxi_dataset/zaoshaiorg/jingbiao_result20210113.xls"
csv = pd.read_excel(jingbiao_csvpath)

for experi in range(1):
    experi+=1
    # 读取CSV文件
    data_resnet = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/resnet50/resnet50_bs16_lr3e-05/' + str(experi) + '_test_data_result.csv')
    data_convnext = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/convnext/convnext_bs16_lr3e-05/' + str(experi) + '_test_data_result.csv')
    data_dinov2 = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/dinov2/dinov2_bs16_lr1e-05/' + str(experi) + '_test_data_result.csv')
    data_our = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/9mixedregions_our_7_3/s-vmamba/20seed_teacher_ema0.992_head1024_freezeepoch5_1cls_weight0.5self_loss1cls_local_weight/lr_3e-05/' + str(experi) + '_test_data_result.csv')
    data_swin = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/swintiny/swintiny_bs16_lr1e-05/' + str(experi) + '_test_data_result.csv')
    data_svmamba = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/svmamba/2_svmamba_bs16_lr5e-05/' + str(experi) + '_test_data_result.csv')
    data_vim = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/vim/2_vim_bs16_lr1e-05/' + str(experi) + '_test_data_result.csv')
    data_lgms = pd.read_csv('/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/9mixedregions_our_7_3/s-vmamba-loss_ablation/4vmamba_num_teacher_ema0.992_head1024_freezeepoch5_1cls_weight0.5self_loss1cls_local_weight/lr_3e-05/' + str(experi) + '_test_data_result.csv')
    data_vmamba = pd.read_csv(
        '/disk3/wjr/workspace/sectwo/vmambav2_expeript/shanxi_zaoshai_trainvaltest/contra_experi_7_3/vmamba/2_vmamba_bs16_lr3e-05/' + str(experi) + '_test_data_result.csv')

    # 获取特定列的数据内容
    img_name_1 = data_resnet['Image Index']
    img_name_2 = data_convnext['Image Index']
    img_name_3 = data_dinov2['Image Index']
    img_name_4 = data_our['Image Index']
    img_name_5 = data_swin['Image Index']
    img_name_6 = data_svmamba['Image Index']
    img_name_7 = data_vim['Image Index']
    img_name_8 = data_lgms['Image Index']
    img_name_9 = data_vmamba['Image Index']
    if (img_name_1 != img_name_2).any() or (img_name_1 != img_name_3).any() or (img_name_1 != img_name_4).any() or (
            img_name_1 != img_name_5).any() or (img_name_1 != img_name_6).any() or (img_name_1 != img_name_7).any() or (
            img_name_1 != img_name_8).any() or (
            img_name_1 != img_name_9).any():
        print('False!!!')

    labels = data_resnet['Finding Labels'].values
    preds_resnet = data_resnet['Pred Labels'].values
    preds_convnext = data_convnext['Pred Labels'].values
    preds_dinov2 = data_dinov2['Pred Labels'].values
    preds_our = data_our['Pred Labels'].values
    preds_swin = data_swin['Pred Labels'].values
    preds_pcam = data_svmamba['Pred Labels'].values
    preds_vim = data_vim['Pred Labels'].values
    preds_mambaout = data_lgms['Pred Labels'].values
    preds_vmamba = data_vmamba['Pred Labels'].values
    img_name_1 = data_resnet['Image Index'].values

    for ii in range(labels.shape[0]):
        img_name=img_name_1[ii]
        pattern = r'\d+\.\d+|\d+'  # 匹配整数或浮点数
        numbermatches = re.findall(pattern, img_name)[-1]
        csv_line = csv.loc[(csv["编号"] == int(numbermatches))]
        if not csv_line.empty:
            correct_num = 0
            resnet_index = 0
            convnext_index = 0
            dinov2_index = 0
            swin_index = 0
            pcam_index = 0
            vim_index = 0
            mambaout_index = 0
            vmamba_index = 0
            our_index = 0
            if preds_our[ii] == labels[ii]:
                our_index = 1
                if preds_resnet[ii] == labels[ii]:
                    correct_num += 1
                    resnet_index = 1
                if preds_convnext[ii] == labels[ii]:
                    correct_num += 1
                    convnext_index = 1
                if preds_dinov2[ii] == labels[ii]:
                    correct_num += 1
                    dinov2_index = 1
                if preds_swin[ii] == labels[ii]:
                    correct_num += 1
                    swin_index = 1
                if preds_pcam[ii] == labels[ii]:
                #     correct_num += 1
                    pcam_index = 1
                if preds_vim[ii] == labels[ii]:
                    correct_num += 1
                    vim_index = 1
                if preds_mambaout[ii] == labels[ii]:
                #     correct_num += 1
                    mambaout_index = 1
                if preds_vmamba[ii] == labels[ii]:
                    correct_num += 1
                    vmamba_index = 1

                if correct_num <2 and labels[ii] == 'Sick':
                    print(experi, '--image_name:', img_name_1[ii])
                    print('resnet_index:', resnet_index, 'Score_resnet:', data_resnet['Pred logits'].values[ii], )
                    print('svmamba_index:', pcam_index, 'Score_svmamba:',
                          data_svmamba['Pred logits'].values[ii])
                    print('swin_index:', swin_index, 'Score_swin:',
                          data_swin['Pred logits'].values[ii])
                    print('vim_index:', vim_index, 'Score_vim:',
                          data_vim['Pred logits'].values[ii])
                    print('convnext_index:', convnext_index, 'Score_convnext:',
                          data_convnext['Pred logits'].values[ii])
                    print('dinov2_index:', dinov2_index, 'Score_dinov2:',
                          data_dinov2['Pred logits'].values[ii])
                    print('lgms_index:', mambaout_index, 'Score_lgms:',
                          data_lgms['Pred logits'].values[ii])
                    print('vmamba_index:', vmamba_index, 'Score_vmamba:',
                          data_vmamba['Pred logits'].values[ii])
                    print('our_index', our_index, 'Score_our:',
                          data_our['Pred logits'].values[ii])



# Sick_Scores=data['Pred Sick_Score']
# Health_Scores=data['Pred Health_Score']
#
# # 打印特定列的数据内容
# print(specific_column)
