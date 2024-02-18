#import sys
#sys.path.append('/kaggle/input/gincorl/pythonproject1/CORL/Supplementary Material/code/CORL')
import numpy as np
import pandas as pd
from recode.corl import CORL
from recode.util import output_adj
from recode.causal_strength import adj_cs
from castle.common.priori_knowledge import orient_by_priori_knowledge,PrioriKnowledge
import torch


#X = pd.read_csv("D:\pythonproject1\CORL\Supplementary Material\Experimental data\\2011-2020_1.csv",header=1)
X = pd.read_csv("D:\pythonproject1\CORL\Supplementary Material\Experimental data\processed-merge_datanew_1.csv")
#X = X.drop("Outcome", axis=1)  # 删除"Outcome"列
# 定义要删除的列名列表
#columns_to_drop = ["Hb", "GLO", "HBeAb","MPV","A/G RATIO","Mono","TTP","Baso","Gender","Outcome"]  # 用你想要删除的列名替换这些
# 删除指定列
columns_to_drop = ["SEQN","BMXWT","Outcome"]
X = X.drop(columns=columns_to_drop, axis=1)
print(X.columns)  # 检查表格每列的名称

# 检查是否有可用的 GPU 设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 使用CORL算法进行因果推断和学习
# rl learn
corl = CORL(iteration=5000, device_type=device, reward_score_type='BIC')
corl.learn(X)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
skeleton_matrix = corl.causal_matrix
#priori = PrioriKnowledge(8)
#priori.add_required_edges([(7, 2)])
#priori.add_forbidden_edges([(2,7)])
#priori.add_required_edges([(28, 26)])
#priori.add_forbidden_edges([(4,8),(23,24),(16,21),(26,28)])
#priori.add_undirected_edges([(5, 3)])
# 调用函数
#riented_matrix = orient_by_priori_knowledge(skeleton_matrix, priori)

print(corl.causal_matrix)
print(adj_cs(corl.causal_matrix, X))
output_adj(corl.causal_matrix)
#print("Original Skeleton Matrix:")
#print(skeleton_matrix)
#print("\nOriented Matrix based on Priori Knowledge:")
#print(oriented_matrix)
#print(adj_cs(oriented_matrix, X))
