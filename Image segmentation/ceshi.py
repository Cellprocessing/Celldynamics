from matplotlib import pyplot as plt
import numpy as np
train_loss=[11,10,9,8,7,6,5,4,3,2,1]
test_loss=[10,9,8,7,6,5,4,3,2,1,0.5]
train_loss = train_loss
val_loss = test_loss
x = np.linspace(0, (len(train_loss)-1)*5,  len(train_loss))  # 起始值为1，终止值为10000，数据个数10000的等差数列
print(x)

fig = plt.figure(figsize=(12, 12))
sub = fig.add_subplot(111)

sub.plot(x, train_loss[:len(train_loss)], color='orange', linewidth=2, label='Training Loss')
sub.plot(x, val_loss[:len(train_loss)], color='green',  linewidth=2, label='Validation Loss')

sub.tick_params(labelsize=15)
sub.set_xlim([0, 1+max(x)])   #设置坐标轴范围，0~10000
sub.set_ylim([0.0, 1+max(train_loss)])   #设置坐标轴范围，0.0~1.0
# sub.set_xticks(np.arange(0, 10001,
#                          1000))  # arange()类似linspace，初始值为0，终止值为<10001，步长为1000的等差数列。如果等差数列最后一个数为10001，则10001不包含进来。
# sub.set_xticks(np.arange(0, 10001, 500), minor=True)  # set_xticks()设置x轴坐标刻度，minor=True表示设置副刻度
# sub.set_yticks(np.arange(0, 1.21, 0.1))
# sub.set_yticks(np.arange(0, 1.21, 0.02), minor=True)
# sub.tick_params(axis='both', which='major', direction='inout', length=25, width=7, pad=50,
#                 labelsize=45)  # 设置刻度属性
# sub.tick_params(axis='both', which='minor', direction='in', length=20, width=3, pad=50,
#                 labelsize=45)  # axis='both'选择要进行设置的坐标轴为x和y轴， which='major'选择要进行设置的刻度是主刻度， direction='inout',设置刻度的方向是由里到外，即穿插坐标轴。 length和width设置线长和宽， pad设置坐标轴标签与坐标轴的距离，labelsize设置标签字体大小。
sub.set_xlabel('epochs', fontsize=15)
sub.set_ylabel('accuracy or loss', fontsize=15)
sub.set_title('Training and Validation Loss', fontsize=15)
plt.legend(fontsize=15)  # 添加图例
# fig.suptitle('figure', fontsize=100) #添加figure总标题
fig.savefig('base_all_SetOut.png', dpi=200)