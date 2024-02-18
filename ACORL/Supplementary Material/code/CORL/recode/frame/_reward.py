# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..utils.validation import Validation


class GPRMine(object):
    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1, "sigma_f": 1}
        self.optimize = optimize
        self.alpha = 1e-10
        self.m = None

    def fit(self, y, median, p_eu):
        self.train_y = np.asarray(y)
        K = self.kernel(median, p_eu)
        np.fill_diagonal(K, 1)
        self.K_trans = K.copy()
        K[np.diag_indices_from(K)] += self.alpha
        # self.KK = K.copy()

        self.L_ = cholesky(K, lower=True)  # Line 2
        # self.L_ changed, self._K_inv needs to be recomputed
        self._K_inv = None
        self.alpha_ = cho_solve((self.L_, True), self.train_y)  # Line 3
        self.is_fit = True

    def predict(self, return_std=False):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        K_trans = self.K_trans
        y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        if return_std == False:
            return y_mean
        else:
            raise ('To cal std')

    def kernel(self, median, p_eu):
        p_eu_nor = p_eu / median
        K = np.exp(-0.5 * p_eu_nor)
        K = squareform(K)
        return K


class Reward(object):
    """
    Used for calculate reward for ordering-based Causal discovery

    In ordering-based methods, only the variables selected in previous decision
    steps can be the potential parents of the currently selected variable.
    Hence, author design the rewards in the following cases:
    `episodic reward` and `dense reward`.
    Reward类是用于计算基于排序的因果发现方法中的奖励。在这种方法中，只有前面决策步骤中选择的变量才能成为当前选择变量的潜在父节点。因此，作者设计了以下几种情况下的奖励：episodic reward和dense reward。
    1.Reward类的构造函数接受以下参数：
input_data：输入数据，用于计算奖励
reward_mode：奖励模式，可以是'episodic'或'dense'，默认为'episodic'
score_type：评分类型，可以是'BIC'或'BIC_different_var'，默认为'BIC'
regression_type：回归类型，可以是'LR'、'QR'、'GPR'或'GPR_learnable'，默认为'LR'
alpha：GPR回归中的alpha值，默认为1.0
    2.Reward类的主要方法是cal_rewards，用于计算奖励。它接受以下参数：
graphs：图的批次，每个图表示一个决策序列
positions：位置信息，表示每个决策序列当前选择变量的位置
ture_flag：真实标志，表示是否为真实奖励计算，如果为True，则不会使用缓存的结果，默认为False
gamma：折扣因子，默认为0.98
cal_rewards方法返回奖励列表、归一化后的奖励、最大奖励和TD目标（如果奖励类型为'episodic'）。
Reward类还包括其他辅助方法，如calculate_yerr用于计算预测误差、calculate_LR用于线性回归、calculate_QR用于二次回归、calculate_GPR用于高斯过程回归、calculate_GPR_learnable用于可学习的高斯过程回归等。

最后，Reward类还提供了更新和排序得分的方法，以及更新所有分数的方法

    """

    def __init__(self, input_data, reward_mode='episodic',
                 score_type='BIC', regression_type='LR', alpha=1.0):

        self.input_data = input_data
        self.reward_type = reward_mode
        self.alpha = alpha
        self.n_samples = input_data.shape[0]
        self.seq_length = input_data.shape[1]
        self.d = {}  # store results
        self.d_RSS = [{} for _ in range(self.seq_length)]  # store RSS for reuse
        self.bic_penalty = np.log(input_data.shape[0]) / input_data.shape[0]

        Validation.validate_value(score_type,
                                  ('BIC', 'BIC_different_var'))
        Validation.validate_value(regression_type,
                                  ('LR', 'QR', 'GPR', 'GPR_learnable'))

        self.score_type = score_type
        self.regression_type = regression_type

        self.poly = PolynomialFeatures()

        if self.regression_type == 'GPR_learnable':
            self.kernel_learnable = 1.0 * RBF(length_scale=1.0,
                                              length_scale_bounds=(1e-2, 1e2)) \
                                    + WhiteKernel(noise_level=1.0,
                                                  noise_level_bounds=(
                                                      1e-10, 1e+1))
        elif regression_type == 'LR':
            self.ones = np.ones((input_data.shape[0], 1), dtype=np.float32)
            X = np.hstack((self.input_data, self.ones))
            self.X = X
            self.XtX = X.T.dot(X)
        elif regression_type == 'GPR':
            self.gpr = GPRMine()
            m = input_data.shape[0]
            self.gpr.m = m
            dist_matrix = []
            for i in range(m):
                for j in range(i + 1, m):
                    dist_matrix.append((input_data[i] - input_data[j]) ** 2)
            self.dist_matrix = np.array(dist_matrix)

    def cal_rewards(self, graphs, positions=None, ture_flag=False, gamma=0.98):
        rewards_batches = []
        if not ture_flag:
            for graphi, position in zip(graphs, positions):
                reward_ = self.calculate_reward_single_graph(graphi,
                                                             position=position,
                                                             ture_flag=ture_flag)
                rewards_batches.append(reward_)
        else:
            for graphi in graphs:
                reward_ = self.calculate_reward_single_graph(graphi,
                                                             ture_flag=ture_flag)
                rewards_batches.append(reward_)

        max_reward_batch = -float('inf')
        reward_list, normal_batch_reward = [], []
        for nu, (reward_, reward_list_) in enumerate(rewards_batches):
            reward_list.append(reward_list_)
            normalized_reward = -reward_
            normal_batch_reward.append(normalized_reward)
            if normalized_reward > max_reward_batch:
                max_reward_batch = normalized_reward
        normal_batch_reward = np.stack(normal_batch_reward)
        reward_list = - np.stack(reward_list)

        if self.reward_type == 'episodic':
            G = 0
            td_target = []
            for r in np.transpose(reward_list, [1, 0])[::-1]:
                G = r + gamma * G
                td_target.append(G)
        elif self.reward_type == 'dense':
            td_target = None
        else:
            raise ValueError(f"reward_type must be one of ['episodic', "
                             f"'dense'], but got ``{self.reward_type}``.")

        return reward_list, normal_batch_reward, max_reward_batch, td_target

    def calculate_yerr(self, X_train, y_train, XtX=None, Xty=None):
        if self.regression_type == 'LR':
            return self.calculate_LR(X_train, y_train, XtX, Xty)
        elif self.regression_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.regression_type == 'GPR':
            return self.calculate_GPR(y_train, XtX)
        elif self.regression_type == 'GPR_learnable':
            return self.calculate_GPR_learnable(X_train, y_train)
        else:
            raise TypeError(f"The parameter `regression_type` must be one of "
                            f"[`LR`, `QR`, `GPR`, `GPR_learnable`], "
                            f"but got ``{self.regression_type}``.")

    def calculate_LR(self, X_train, y_train, XtX, Xty):
        """Linear regression"""

        theta = np.linalg.solve(XtX, Xty)
        y_pre = X_train.dot(theta)
        y_err = y_pre - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        """quadratic regression"""

        X_train = self.poly.fit_transform(X_train)[:, 1:]
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        return self.calculate_LR(X_train, y_train, XtX, Xty)

    def calculate_GPR(self, y_train, XtX):
        p_eu = XtX  # our K1 don't sqrt
        med_w = np.median(p_eu)
        self.gpr.fit(y_train, med_w, p_eu)
        pre = self.gpr.predict()
        return y_train - pre

    def calculate_GPR_learnable(self, X_train, y_train):
        gpr = GPR(kernel=self.kernel_learnable, alpha=0.0).fit(X_train, y_train)
        return y_train.reshape(-1, 1) - gpr.predict(X_train).reshape(-1, 1)

    def calculate_reward_single_graph(self, graph_batch, position=None,
                                      ture_flag=False):

        graph_to_int2 = list(np.int32(position))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if not ture_flag:
            if graph_batch_to_tuple in self.d:
                graph_score = self.d[graph_batch_to_tuple]
                reward = graph_score[0]
                return reward, np.array(graph_score[1])

        RSS_ls = []
        for i in range(self.seq_length):
            RSSi = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)

        RSS_ls = np.array(RSS_ls)
        if self.regression_type == 'GPR' or self.regression_type == 'GPR_learnable':
            reward_list = RSS_ls[position] / self.n_samples
        else:
            reward_list = RSS_ls[position] / self.n_samples

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls) / self.n_samples + 1e-8) + np.sum(graph_batch)*self.bic_penalty/self.seq_length
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls) / self.n_samples + 1e-8))
            # + np.sum(graph_batch)*self.bic_penalt
        else:
            raise TypeError(f"The parameter `score_type` must be one of "
                            f"[`BIC`,`BIC_different_var`], "
                            f"but got ``{self.score_type}``.")
        if not ture_flag:
            self.d[graph_batch_to_tuple] = (BIC, reward_list)

        return BIC, np.array(reward_list)

    def cal_RSSi(self, i, graph_batch):
        col = graph_batch[i]
        str_col = str(col)
        if str_col in self.d_RSS[i]:
            RSSi = self.d_RSS[i][str_col]
            return RSSi
        if np.sum(col) < 0.1:
            y_err = self.input_data[:, i]
            y_err = y_err - np.mean(y_err)
        else:
            cols_TrueFalse = col > 0.5
            if self.regression_type == 'LR':
                cols_TrueFalse = np.append(cols_TrueFalse, True)
                X_train = self.X[:, cols_TrueFalse]
                y_train = self.X[:, i]
                XtX = self.XtX[:, cols_TrueFalse][cols_TrueFalse, :]
                Xty = self.XtX[:, i][cols_TrueFalse]
                y_err = self.calculate_yerr(X_train, y_train, XtX, Xty)
            elif self.regression_type == 'GPR':
                X_train = self.input_data[:, cols_TrueFalse]
                y_train = self.input_data[:, i]
                p_eu = pdist(X_train, 'sqeuclidean')
                train_y = np.asarray(y_train)
                p_eu_nor = p_eu / np.median(p_eu)
                K = np.exp(-0.5 * p_eu_nor)
                K = squareform(K)
                np.fill_diagonal(K, 1)
                K_trans = K.copy()
                K[np.diag_indices_from(K)] += self.alpha  # 1e-10
                L_ = cholesky(K, lower=True)  # Line 2
                alpha_ = cho_solve((L_, True), train_y)  # Line 3
                y_mean = K_trans.dot(alpha_)  # Line 4 (y_mean = f_star)
                y_err = y_train - y_mean
            elif self.regression_type == 'GPR_learnable':
                X_train = self.input_data[:, cols_TrueFalse]
                y_train = self.input_data[:, i]
                y_err = self.calculate_yerr(X_train, y_train, X_train, y_train)
            else:
                raise TypeError(f"The parameter `regression_type` must be one of "
                                f"[`LR`, `GPR`, `GPR_learnable`], "
                                f"but got ``{self.regression_type}``.")
        RSSi = np.sum(np.square(y_err))
        self.d_RSS[i][str_col] = RSSi

        return RSSi

    def penalized_score(self, score_cyc, lambda1=1, lambda2=1):
        score, cyc = score_cyc
        return score + lambda1 * float(cyc > 1e-5) + lambda2 * cyc

    def update_scores(self, score_cycs):
        ls = []
        for score_cyc in score_cycs:
            ls.append(score_cyc)
        return ls

    def update_all_scores(self):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_l in score_cycs:
            ls.append((graph_int, (score_l[0], score_l[-1])))
        return sorted(ls, key=lambda x: x[1][0])

