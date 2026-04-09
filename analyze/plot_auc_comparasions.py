import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
def plot_multiple_roc_curves(model_names, model_dirs, valdata, aucs):
    plt.figure(figsize=(8, 6))
    # 设置颜色循环（14种不同颜色）
    # 自定义14种高对比度颜色（涵盖基础色、亮色和深色）
    custom_colors = [
        '#0000FF',  # 蓝色
        '#008000',  # 绿色
        '#FFA500',  # 橙色
        '#800080',  # 紫色
        '#FFFF00',  # 黄色
        '#00FFFF',  # 青色
        '#FF00FF',  # 粉色
        '#000080',  # 海军蓝
        '#800000',  # 栗色
        '#008080',  # 橄榄绿
        '#FFC0CB',  # 浅粉色
        '#808000',  # 橄榄色
        '#00FF00',  # 鲜绿色
        '#FF0000',  # 红色
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
        fpr = np.load(model_dir+valdata+"_fpr.npy")
        tpr = np.load(model_dir+valdata+"_tpr.npy")
        # 绘制曲线并获取 Line2D 对象
        line = plt.plot(fpr[:-1], tpr[:-1], label=f'{model_name} (AUC = {auc})')

        # 获取曲线的颜色
        # line_color = line[0].get_color()

        # specific_fpr = fpr[-1]
        # specific_tpr = tpr[-1]
        # # 使用 plt.scatter 标记特定点
        # plt.scatter(specific_fpr, specific_tpr, marker='*', color=line_color, s=100, zorder=10)


    plt.xlabel('False Positive Rate (Sensitivity)')
    plt.ylabel('True Positive Rate (1 - Specificity)')
    if 'shanxi' in valdata:
        plt.title('Internal Set Receiver Operating Characteristic (ROC) Curve Comparison')
    else:
        plt.title('External Set Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc='lower right')


    save_path = '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/analyze/auc/' + valdata.split('/')[-1]+'_auc_compare.png'
    plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
    plt.show()
    plt.close()


# 绘制多个模型的 ROC 曲线
model_names = ["ResNet", "ViT","Swin Transformer", "ConvNeXt", "DINOv2", "VMamba","MambaOut","PCAM","DMALNET","PKA2Net","MedMamba","PneumoLLM","SLGMS", "Ours"]
shanxitest_aucs=[0.882,0.767,0.886,0.896,0.908,0.904,0.875,0.877,0.908,0.759,0.744,0.891,0.929,0.963]
guizhoutest_aucs=[0.908,0.839,0.924,0.939,0.830,0.925,0.937,0.948,0.898,0.718,0.802,0.873,0.916,0.967]
model_dirs=['/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/resnet50/wmask_nosubcls_nodropout2/resnet50_lr8e-05/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vit-base/wmask_nosubcls_nodropout2/vit-base_lr3e-05/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/swintiny/wmask_nosubcls_nodropout2/swintiny_lr0.0001/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/convnext_wmask_nosubcls_nodropout2/convnext_lr5e-05/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/dinov2/wmask_nosubcls_nodropout2/dinov2_lr3e-05_testall/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/vmamba/lr_3e-05/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/mambaout/wmask_nosubcls_nodropout2/mambaout_lr3e-05_testall/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/pcam/wmask_nosubcls_nodropout2/pcam_lr3e-05/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/DMALNET/_lr5e-05/models_params/RR_avgpool/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/PKA2_Net/wmask_nosubcls_nodropout2/PKA2_Net_lr5e-05_testall/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/medmamba/wmask_nosubcls_nodropout2/medmamba_lr3e-05_testall/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/pneumollm_336/llm_lr0.0003_weightsseed_10/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/',
            '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/new_contra_20250626/convnext/g512_l256_tree_5cls_2cls_cam_wdatamixup_3/0.5_cam_0.5_camalign_0.3_cloc_lr8e-05/']
plot_multiple_roc_curves(model_names, model_dirs, valdata='analyze/shanxi_test_global', aucs=shanxitest_aucs)
plot_multiple_roc_curves(model_names, model_dirs, valdata='analyze/guizhou_test_global', aucs=guizhoutest_aucs)
# '/disk3/wjr/workspace/sec_nejm/nejm_baseline_expeript_new/allshanxi_sick_health/writer_analyze/512/allstage1_health1_1/selected_dcm_saomiao_900_health308_sick_592/slgms/9mixedsubs/s-vmamba_3e-5/',
