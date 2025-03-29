import json
import numpy as np


j = {'results': {'arc_easy': {'acc': 0.7201178451178452, 'acc_stderr': 0.00921207752465653, 'acc_norm': 0.5782828282828283, 'acc_norm_stderr': 0.010133255284012314}, 'arc_challenge': {'acc': 0.41723549488054607, 'acc_stderr': 0.01440982551840308, 'acc_norm': 0.42150170648464164, 'acc_norm_stderr': 0.01443019706932602}, 'winogrande': {'acc': 0.6771902131018153, 'acc_stderr': 0.01314049817335794}, 'piqa': {'acc': 0.7856365614798694, 'acc_stderr': 0.009574842136050976, 'acc_norm': 0.7763873775843307, 'acc_norm_stderr': 0.009721489519176282}, 'boolq': {'acc': 0.654434250764526, 'acc_stderr': 0.008317463342191585}, 'hellaswag': {'acc': 0.5761800438159729, 'acc_stderr': 0.004931525961035749, 'acc_norm': 0.7427803226448915, 'acc_norm_stderr': 0.004362081806560236}}}


data = j['results']
# print(data['arc_challenge']['acc'], data['arc_easy']['acc'], data['boolq']['acc'], data['hellaswag']['acc'], data['piqa']['acc'], data['winogrande']['acc'])

# 计算各项指标的准确率
arc_challenge = data['arc_challenge']['acc'] * 100
arc_easy = data['arc_easy']['acc'] * 100
boolq = data['boolq']['acc'] * 100
hellaswag = data['hellaswag']['acc'] * 100
piqa = data['piqa']['acc'] * 100
winogrande = data['winogrande']['acc'] * 100

# 计算平均值
average = (arc_challenge + arc_easy + boolq + hellaswag + piqa + winogrande) / 6

# 打印各项指标和平均值
print(
    f"{arc_challenge:.2f}",
    f"{arc_easy:.2f}",
    f"{boolq:.2f}",
    f"{hellaswag:.2f}",
    f"{piqa:.2f}",
    f"{winogrande:.2f}",
    sep=","
)

print(f"平均值: {average:.2f}")
