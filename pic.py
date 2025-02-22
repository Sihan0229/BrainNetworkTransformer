import re
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志文件
log_file = '/root/autodl-tmp/BrainNetworkTransformer/log_bnt2.log'

# 定义一个正则表达式，用来提取每行中的关键信息
log_pattern = r"Epoch\[(\d+/\d+)\].*Train Loss: (\d+\.\d+).*Train Accuracy: (\d+\.\d+)%.*Test Loss: (\d+\.\d+).*Test Accuracy: (\d+\.\d+)%.*Val AUC:(\d+\.\d+).*Test AUC:(\d+\.\d+).*Test Sen:(\d+\.\d+).*LR:(\d+\.\d+)"

# 初始化空列表来存储数据
epochs = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []
val_auc = []
test_auc = []
test_sen = []
lr = []

# 逐行读取日志文件并使用正则表达式提取数据
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(log_pattern, line)
        if match:
            epochs.append(match.group(1))  # 存储 epoch
            train_loss.append(float(match.group(2)))  # 存储训练损失
            train_acc.append(float(match.group(3)))  # 存储训练准确率
            test_loss.append(float(match.group(4)))  # 存储测试损失
            test_acc.append(float(match.group(5)))  # 存储测试准确率
            val_auc.append(float(match.group(6)))  # 存储验证AUC
            test_auc.append(float(match.group(7)))  # 存储测试AUC
            test_sen.append(float(match.group(8)))  # 存储测试灵敏度
            lr.append(float(match.group(9)))  # 存储学习率

# 将数据存储为 pandas DataFrame
df = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': train_loss,
    'Train Accuracy': train_acc,
    'Test Loss': test_loss,
    'Test Accuracy': test_acc,
    'Val AUC': val_auc,
    'Test AUC': test_auc,
    'Test Sen': test_sen,
    'Learning Rate': lr
})

# 绘制训练过程图表
plt.figure(figsize=(12, 8))

# 训练损失和测试损失图
plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='blue')
plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()

# 设置横轴只显示5个epoch值
step = len(df['Epoch']) // 5  # 计算间隔
epoch_ticks = df['Epoch'][::step]  # 选择5个点
plt.xticks(epoch_ticks, rotation=45)  # 设置横轴刻度并旋转

# 训练准确率和测试准确率图
plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', color='blue')
plt.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy')
plt.legend()

# 设置横轴只显示5个epoch值
step = len(df['Epoch']) // 5  # 计算间隔
epoch_ticks = df['Epoch'][::step]  # 选择5个点
plt.xticks(epoch_ticks, rotation=45)  # 设置横轴刻度并旋转

# 验证AUC和测试AUC图
plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['Val AUC'], label='Validation AUC', color='green')
plt.plot(df['Epoch'], df['Test AUC'], label='Test AUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Validation and Test AUC')
plt.legend()

# 设置横轴只显示5个epoch值
step = len(df['Epoch']) // 5  # 计算间隔
epoch_ticks = df['Epoch'][::step]  # 选择5个点
plt.xticks(epoch_ticks, rotation=45)  # 设置横轴刻度并旋转

# 测试灵敏度和学习率图
plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['Test Sen'], label='Test Sensitivity', color='purple')
plt.plot(df['Epoch'], df['Learning Rate'], label='Learning Rate', color='black')
plt.xlabel('Epoch')
plt.ylabel('Test Sensitivity / Learning Rate')
plt.title('Test Sensitivity and Learning Rate')
plt.legend()

# 设置横轴只显示5个epoch值
step = len(df['Epoch']) // 5  # 计算间隔
epoch_ticks = df['Epoch'][::step]  # 选择5个点
plt.xticks(epoch_ticks, rotation=45)  # 设置横轴刻度并旋转

# 调整布局
plt.tight_layout()

# 保存为图片
output_file = '/root/autodl-tmp/BrainNetworkTransformer/bnt_training_results50.png'
plt.savefig(output_file)

# 提示保存成功
print(f"Training results saved as {output_file}")
