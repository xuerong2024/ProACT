import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.font_manager
def plot_multiple_roc_curves_our():
    plt.figure(figsize=(8, 6))
    # 设置颜色循环（14种不同颜色）
    # 自定义14种高对比度颜色（涵盖基础色、亮色和深色）
    custom_colors = [
        '#0000FF',  # 蓝色
        '#FFA500',  # 橙色
        # '#008000',  # 绿色
        # '#800080',  # 紫色
        # '#FFFF00',  # 黄色
        # '#00FFFF',  # 青色
        # '#FF00FF',  # 粉色
        # '#000080',  # 海军蓝
        # '#800000',  # 栗色
        # '#008080',  # 橄榄绿
        # '#FFC0CB',  # 浅粉色
        # '#808000',  # 橄榄色
        # '#00FF00',  # 鲜绿色
        # '#FF0000',  # 红色

    ]
    # 设置颜色循环
    plt.rc('axes', prop_cycle=plt.cycler('color', custom_colors))
    # reliable_radio=[20,30,40,50,60,70,80,90,100]
    # shanxi_performance=[87.18, 88.19, 89.65, 90.73, 91.54, 91.88, 92.44, 92.35, 91.14]
    # guizhou_performance=[95.92, 95.37, 96.21, 96.28, 96.13, 96.37, 96.6, 95.81, 96.15]
    # plt.plot(reliable_radio, shanxi_performance, 'o-',label=r'$p$-山西测试域')
    # plt.plot(reliable_radio, guizhou_performance, 'o-', label=r'$p$-贵州测试域')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlabel(r'$p$ 取值',
    #            fontdict={'family': 'Noto Serif CJK JP', 'size': 25})
    # plt.ylabel('平均性能',
    #            fontdict={'family': 'Noto Serif CJK JP', 'size': 25})
    # plt.legend(loc='best', prop={'family': 'Noto Serif CJK JP', 'size': 20})
    # save_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/' + f'reliable_sample_radio.pdf'

    # rule_loss_radio = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # shanxi_performance = [90.19, 92.19, 92.37, 92.42, 92.34, 92.39, 92.44, 92.39, 92.41, 92.35, 92.47]
    # guizhou_performance = [95.66, 96.36, 96.5, 96.58, 96.53, 96.57, 96.6, 96.61, 96.59, 96.51, 96.49]
    # plt.plot(rule_loss_radio, shanxi_performance, 'o-', label=r'$\lambda_r$ - 山西测试域')
    # plt.plot(rule_loss_radio, guizhou_performance, 'o-', label=r'$\lambda_r$ - 贵州测试域')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlabel(r'$\lambda_r$ 取值',
    #            fontdict={'family': 'Noto Serif CJK JP', 'size': 25})
    # plt.ylabel('平均性能',
    #            fontdict={'family': 'Noto Serif CJK JP', 'size': 25})
    # plt.legend(loc='best', prop={'family': 'Noto Serif CJK JP', 'size': 20})
    # save_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/' + f'rule_loss_radio.pdf'

    para_radios = [5,10,15,20, 25,30,35, 40, 50, 60, 70, 80, 90, 100]
    shanxi_performance = [92.1, 92.35, 92.44, 92.51, 92.07, 92.15, 91.68, 91.3, 91.27, 91.55, 91.58, 91.49, 91.41, 89.52]
    guizhou_performance = [95.68, 96.63, 96.6, 96.54, 96.74, 95.78, 96.06, 95.62, 96.13, 96.39, 96.6, 95.39, 95.73, 94.59]
    plt.plot(para_radios, shanxi_performance, 'o-', label=r'$p$-山西测试域')
    plt.plot(para_radios, guizhou_performance, 'o-', label=r'$p$-贵州测试域')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 101, 10), fontsize=16)  # 从0到100，步长为10
    plt.xlabel(r'$\rho$ 取值',
               fontdict={'family': 'Noto Serif CJK JP', 'size': 25})
    plt.ylabel('平均性能',
               fontdict={'family': 'Noto Serif CJK JP', 'size': 25})
    plt.legend(loc='best', prop={'family': 'Noto Serif CJK JP', 'size': 20})
    save_path = '/disk3/wjr/workspace/sec_proj4/baseline_expeript/TTA/' + f'para_radios_radio.pdf'

    plt.savefig(save_path, dpi=500, bbox_inches='tight')  # 保存图像，设置分辨率和裁剪边距
    plt.show()
    plt.close()
plot_multiple_roc_curves_our()