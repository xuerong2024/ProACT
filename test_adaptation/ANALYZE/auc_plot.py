import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.font_manager
def plot_multiple_roc_curves_our(model_names, model_dirs, valdata, aucs):
    plt.figure(figsize=(8, 6))
    # 设置颜色循环（14种不同颜色）
    # 自定义14种高对比度颜色（涵盖基础色、亮色和深色）
    custom_colors = [
        '#FFA500',  # 橙色
        # '#800080',  # 紫色
        '#FF00FF',  # 粉色
        # '#000080',  # 海军蓝
        '#800000',  # 栗色
        '#008080',  # 橄榄绿
        '#00FF00',  # 鲜绿色
        '#FFC0CB',  # 浅粉色
        '#808000',  # 橄榄色

    ]
    # 设置颜色循环
    plt.rc('axes', prop_cycle=plt.cycler('color', custom_colors))
    # 绘制对角线基准线
    plt.plot([0, 1], [0, 1], 'k--', zorder=1, linewidth=3)  # 黑色虚线，层级最低

    # plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    for ii in range(len(model_names)):
        model_name=model_names[ii]
        model_dir=model_dirs[ii]
        auc=aucs[ii]
        fpr = np.load(model_dir+"_fpr.npy")
        tpr = np.load(model_dir+"_tpr.npy")
        # 绘制曲线并获取 Line2D 对象
        line = plt.plot(fpr[:-1], tpr[:-1], label=f'{model_name} (AUC = {auc})')
        specific_fpr = fpr[-1]
        specific_tpr = tpr[-1]
        # 使用 plt.scatter 标记特定点
        # line_color = line[-1].get_color()
        # plt.scatter(specific_fpr, specific_tpr, marker='*', color=line_color, s=100, zorder=10,
        #             label=f'{model_name}, Sens.: {specific_tpr:.2f} Spec.: {1 - specific_fpr:.2f}')

        # plt.scatter(specific_fpr, specific_tpr, marker='*', color=line_color, s=100,  zorder=10,
        #         label=f'CALGM (Ours)，敏感度：{specific_tpr:.2f}  特异性：{1 - specific_fpr:.2f}')

    if 'shanxi' in valdata:
        # plt.scatter(1 - 0.84, 0.84, marker='v',
        #     color='#FF0000', s=100, zorder=10,
        #     label='Senior Radiologist, Sens.: 0.84 Spec.: 0.84')
        # plt.scatter(1 - 0.76, 0.90, marker='v',
        #             color='#00FFFF', s=100, zorder=10,
        #             label='Mid-level Radiologist, Sens.: 0.90 Spec.: 0.76')
        # plt.scatter(1 - 0.90, 0.68, marker='v',
        #             color='#008000', s=100, zorder=10,
        #             label='Junior Radiologist 3, Sens.: 0.68 Spec.: 0.90')
        # plt.title('100 Internal Set ROC Curve', fontsize=16)

        plt.scatter(1 - 0.84, 0.84, marker='v',
                    color='#FF0000', s=100, zorder=10,
                    label='高年资医师，敏感度：0.84  特异度：0.84')

        plt.scatter(1 - 0.76, 0.90, marker='v',
                    color='#00FFFF', s=100, zorder=10,
                    label='中年资医师，敏感度：0.90  特异度：0.76')

        plt.scatter(1 - 0.90, 0.68, marker='v',
                    color='#008000', s=100, zorder=10,
                    label='低年资医师，敏感度：0.68  特异度：0.90')
    else:
        # plt.scatter(1 - 0.90, 0.78, marker='v',
        #             color='#FF0000', s=100, zorder=10,
        #             label='Senior Radiologist 1, Sens.: 0.78 Spec.: 0.90')
        # plt.scatter(1 - 0.84, 0.66, marker='v',
        #             color='#00FFFF', s=100, zorder=10,
        #             label='Senior Radiologist 1, Sens.: 0.66 Spec.: 0.84')
        # plt.scatter(1 - 0.98, 0.50, marker='v',
        #             color='#008000', s=100, zorder=10,
        #             label='Junior Radiologist 3, Sens.: 0.50 Spec.: 0.98')
        # plt.title('100 External Set ROC Curve', fontsize=16)


        plt.scatter(1 - 0.90, 0.78, marker='v',
                    color='#FF0000', s=100, zorder=10,
                    label='高年资医师，敏感度：0.78  特异度：0.90')
        plt.scatter(1 - 0.84, 0.66, marker='v',
                    color='#00FFFF', s=100, zorder=10,
                    label='中年资医师，敏感度：0.66  特异度：0.84')
        plt.scatter(1 - 0.98, 0.50, marker='v',
                    color='#008000', s=100, zorder=10,
                    label='低年资医师，敏感度：0.50  特异度：0.98')

    # plt.xlabel('False Positive Rate (1 - Specificity)', fontdict={'size': 16})
    # plt.ylabel('True Positive Rate (Sensitivity)', fontdict={'size': 16})
    #
    # title_text = '100例内部测试集ROC曲线' if 'shanxi' in valdata else '100例外部测试集ROC曲线'
    # plt.title(title_text, fontsize=18, family='Noto Serif CJK JP')
    plt.legend(loc='lower right', prop={'size': 11})

    plt.xlabel('假阳性率（1 - 特异度）',
           fontdict={'family': 'Noto Serif CJK JP', 'size': 18})
    plt.ylabel('真阳性率（敏感度）',
           fontdict={'family': 'Noto Serif CJK JP', 'size': 18})
    plt.legend(loc='lower right', prop={'family': 'Noto Serif CJK JP', 'size': 14})

    if 'shanxi' in valdata:
        save_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/' + 'shanxi_auc_compare.pdf'
    else:
        save_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/' + 'guizhou_auc_compare.pdf'
    plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
    plt.show()
    plt.close()
# 绘制多个模型的 ROC 曲线
model_names = ["SLGMS(w/o TTA)", "SLGMS(w TTA)", "CALGM(w/o TTA)", "CALGM(w TTA)"]

shanxitestchosen_aucs=[0.927, 0.928, 0.932,0.959]
guizhoutestchosen_aucs=[0.932,0.968,0.964,0.971]
'''
山西SLGMS(w/o TTA) 0.927: /disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vmamba/lr_3e-05/analyze/100_shanxi_test_chosen'
山西SLGMS(w TTA) 0.928: /disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/SLGMS/OUR/0.0001lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_local
山西CALGM(w/o TTA) 0.932: /disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_org_local_global
山西CALGM(w TTA) 0.959: /disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_global

贵州SLGMS(w/o TTA) 0.932: /disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/analyze/100_guizhou_test_chosen
贵州SLGMS(w/o TTA) 0.968: /disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/SLGMS/OUR/0.0001lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_global
贵州CALGM(w/o TTA) 0.964: /disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_org_local_global
贵州CALGM(w TTA) 0.971: /disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_local_global
'''
model_dirs_shanxi=['/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vmamba/lr_3e-05/analyze/100_shanxi_test_chosen',
            '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/SLGMS/OUR/0.0001lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_local',
            '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_org_local_global',
            '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/shanxi_test_global',
            ]
model_dirs_guizhou=['/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/analyze/100_guizhou_test_chosen',
            '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/SLGMS/OUR/0.0001lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_global',
            '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_org_local_global',
            '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/CHOOSEN_200/CALGM/OUR/15batch_8e-05lr_15lpara_4prototypes_0.6rule_JSD/analyze/guizhou_test_local_global',
            ]
# plot_multiple_roc_curves(model_names, model_dirs, valdata='analyze/shanxi_test_global', aucs=shanxitest_aucs)
# plot_multiple_roc_curves(model_names, model_dirs, valdata='analyze/guizhou_test_global', aucs=guizhoutest_aucs)
plot_multiple_roc_curves_our(model_names, model_dirs_shanxi, valdata='shanxi', aucs=shanxitestchosen_aucs)
plot_multiple_roc_curves_our(model_names, model_dirs_guizhou, valdata='guizhou', aucs=guizhoutestchosen_aucs)

# '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/',
