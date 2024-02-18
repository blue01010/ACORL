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


import torch
import torch.nn as nn

from ._base_network import PointerDecoder


class LSTMDecoder(PointerDecoder):
    """LSTM + Pointer Network"""

    def __init__(self, input_dim, hidden_dim, device=None) -> None:
        # input of Decoder is output of Encoder, e.g. embed_dim
        super(LSTMDecoder, self).__init__(input_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.lstm_cell = nn.LSTMCell(input_size=hidden_dim,
                                     hidden_size=hidden_dim,
                                     device=self.device)

    def forward(self, x) -> tuple:
        """"""
        self.batch_size = x.shape[0]
        self.seq_length = x.shape[1]
        self.encoder_output = x  # 保存起来有用

        #计算输入的均值，作为初始状态
        #s_i = torch.mean(x, 1)
        # 计算输入的加权均值，作为初始状态
        #weights = torch.tensor([0.076,0.259, 0.084, 0.069, 0.087, 0.166, 0.124, 0.135]) #768
        #weights = torch.tensor([0.073538965771951, 0.26472777747729126, 0.08512898261670966, 0.0690812408947065, 0.08783742214434963,
         #0.15883197129194607, 0.12797482448510306, 0.1328788153179429]) #2000
        #weights = torch.tensor([0.04765188483143822, 0.002259965881622904, 0.00374595076298854, 0.009117018929937902, 0.013812225165747349,
             #0.010399383603227244, 0.004049828417652452, 0.006800693367745239, 0.013992522436645713, 0.00476175825208173,
             #0.15163891358715492, 0.02002893588842882, 0.7117409188753291]) #1w
        #weights = torch.tensor([0.048, 0.002, 0.004, 0.009, 0.014, 0.1, 0.004, 0.007,0.014, 0.005,0.152,0.020, 0.711])

        #weights = torch.tensor([0.028, 0.0011, 0.009, 0.0011, 0.008, 0.007, 0.017, 0.008,0.008, 0.007,
         #0.010,0.008,0.010,0.010,0.008,0.009,0.008,0.009,0.008, 0.007, 0.006,0.007,0.007,0.005,0.004,0.004,0.764,0.002])
        #weights = torch.tensor([0.022821576763485476, 0.010373443983402491, 0.008298755186721992, 0.010373443983402491, 0.010373443983402491,
                    #0.0051867219917012455, 0.015560165975103735, 0.007261410788381743, 0.008298755186721992, 0.007261410788381743,
                    #0.011410788381742738, 0.008298755186721992, 0.007261410788381743, 0.006224066390041494, 0.010373443983402491,
                    #0.006224066390041494, 0.007261410788381743, 0.00933609958506224, 0.007261410788381743, 0.0051867219917012455,
                    #0.006224066390041494, 0.0051867219917012455, 0.007261410788381743, 0.0051867219917012455, 0.006224066390041494,
                    #0.0051867219917012455, 0.0051867219917012455, 0.0051867219917012455, 0.0051867219917012455, 0.0051867219917012455,
                    #0.7593360995850622])
        #weights = torch.tensor([0.04533129953989036, 0.0034246110242935886, 0.008959188681829537, 0.01579448148708333,
                               # 0.013666173001194994, 0.024691208341148633, 0.009800100265964552, 0.007391148443252275,
                               # 0.007574711933817087, 0.013418283002318778, 0.006780761909953891, 0.009408907061133683,
                               # 0.00040521878964675087, 0.011000033201419769, 0.009538937575005657,
                               # 0.006771261920656845, 0.006574208987340383, 0.010336907681210634, 0.0009820562119272353,
                               # 0.0022085430922014473, 0.005118679808822244, 0.0017616620391887202,
                               # 0.009598053398952238, 0.008181592888538185, 0.010076701457228283, 0.015505742144370761,
                               # 0.009104547372095716, 0.06370193012138292, 0.03138597566467626, 0.6315070729534552]) #tc
        weights = torch.tensor( [0.03948813707419695, 0.0038075046018570375, 0.010233973851781604, 0.01879747356040285,
                                 0.026152883303869803, 0.01113266225816518, 0.009463250586155256, 0.008919017859452998,
                                 0.017723176756426054, 0.007306709623015475, 0.011601350089540878, 0.00017132129466283796,
                                 0.010911458094132822, 0.009717517530612691, 0.01055855177422131, 0.00623965251226391,
                                 0.01024430769739099, 0.0019270288757870452, 0.0030295487136624277, 0.005775255669624407,
                                 0.0020027120065931638, 0.009229152869518264, 0.009067114244239501, 0.010815758220790311,
                                 0.016735558003653932, 0.00916925715206985, 0.06455409309631484, 0.03500834009232201,
                                 0.6202172325872757]) #29特征标准化
        #weights = torch.tensor( [0.002461122763798919, 0.029003653300955452, 0.00940627825664583, 0.007938813663099255,
                                # 0.004436876826524175, 0.020937693613101098, 0.01914689660395538, 0.0220048873805941,
                                 #0.017572093001635886, 0.01685572083492114, 0.018091513617630242, 0.010783726343228375,
                                 #0.008817608724423623, 0.018926846960239323, 0.018129350667856252, 0.016570210720223934,
                                 #0.022007846994425928, 0.017410517522650444, 0.03624307985802068, 0.03711323984196473,
                                 #0.019269291448820308, 0.020112900077284526, 0.03514329687196138, 0.03239885547112192,
                                 #0.019442143021169554, 0.23501372944723325, 0.2422330184782494, 0.033019218307243844,
                                 #0.00284883144565684, 0.003999666359324783, 0.002661071576039446])  # 筛选31特征标准化



        weights = weights.unsqueeze(0).unsqueeze(2)
        s_i = torch.sum(x * weights, dim=1)
        hi_ci = (torch.zeros((self.batch_size, self.hidden_dim), device=s_i.device),
                 torch.zeros((self.batch_size, self.hidden_dim), device=s_i.device))
        h_list = []
        c_list = []
        s_list = []
        action_list = []
        prob_list = []
        for step in range(self.seq_length):
            h_list.append(hi_ci[0])
            c_list.append(hi_ci[1])
            s_list.append(s_i)

            s_i, hi_ci, pos, prob = self.step_decode(input=s_i, state=hi_ci)

            action_list.append(pos)
            prob_list.append(prob)

        h_list = torch.stack(h_list, dim=1).squeeze()  # [Batch,seq_length,hidden]
        c_list = torch.stack(c_list, dim=1).squeeze()  # [Batch,seq_length,hidden]
        s_list = torch.stack(s_list, dim=1).squeeze()  # [Batch,seq_length,hidden]

        # Stack visited indices
        actions = torch.stack(action_list, dim=1)  # [Batch,seq_length]
        mask_scores = torch.stack(prob_list, dim=1)  # [Batch,seq_length,seq_length]
        self.mask = torch.zeros(1, device=self.device)

        return actions, mask_scores, s_list, h_list, c_list


class MLPDecoder(PointerDecoder):
    """Multi Layer Perceptions + Pointer Network"""

    def __init__(self, input_dim, hidden_dim, device=None) -> None:
        super(MLPDecoder, self).__init__(input_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                         device=device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.mlp = self.feedforward_mlp

    def forward(self, x) -> tuple:

        self.batch_size = x.shape[0]
        self.seq_length = x.shape[1]
        self.encoder_output = x

        s_i = torch.mean(x, 1)

        s_list = []
        action_list = []
        prob_list = []
        for step in range(self.seq_length):
            s_list.append(s_i)
            s_i, _, pos, prob = self.step_decode(input=s_i, state=None)

            action_list.append(pos)
            prob_list.append(prob)
        s_list = torch.stack(s_list, dim=1).squeeze()  # [Batch,seq_length,hidden]

        # Stack visited indices
        actions = torch.stack(action_list, dim=1)  # [Batch,seq_length]
        mask_scores = torch.stack(prob_list, dim=1)  # [Batch,seq_length,seq_length]
        self.mask = torch.zeros(1, device=self.device)

        return actions, mask_scores, s_list, s_list, s_list
