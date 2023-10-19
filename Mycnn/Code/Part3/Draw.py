import matplotlib.pyplot as plt
def drawPlot(x, y1, y2, title, x_des, y_des, y1_des, y2_des, save=False, savePath='', name='need_a_name'):
    plt.clf()  # 清除上次内容
    lw = 2  # 线粗
    mks = 3  # 关节点大小
    plt.plot(x, y1, label=y1_des, color='blue', marker='o',
             linestyle='-', linewidth=lw, markersize=mks)
    plt.plot(x, y2, label=y2_des, color='red', marker='o',
             linestyle='--', linewidth=lw, markersize=mks)
    # 添加标题和标签
    plt.title(title)  # 图表标题
    plt.xlabel(x_des)  # X轴标签
    plt.ylabel(y_des)  # Y轴标签
    # 显示图表
    plt.legend()
    plt.grid(True)  # 添加网格线
    if save:
        plt.savefig(savePath+'/'+name)
    else:
        plt.show()
