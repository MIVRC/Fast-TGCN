import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable


# def Adj_matrix_gen(face):
#     B, N = face.shape[0], face.shape[1]
#     adj = (face.repeat(1, 1, N).view(B, N*N, 3) == face.repeat(1, N, 1))
#     adj = adj[:, :, 0] + adj[:, :, 1] + adj[:, :, 2]
#     adj = adj.view(B, N, N)
#     adj = torch.where(adj == True, 1., 0.)
#
#     return adj

def Adj_matrix_gen(face):
    B, N = face.shape[0], face.shape[1]
    adj_1_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 0])
    adj_1_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 1])
    adj_1_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 2])
    adj_2_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 0])
    adj_2_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 1])
    adj_2_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 2])
    adj_3_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 0])
    adj_3_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 1])
    adj_3_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 2])
    adj = adj_1_1 + adj_1_2 + adj_1_3 + adj_2_1 + adj_2_2 + adj_2_3 + adj_3_1 + adj_3_2 + adj_3_3
    adj = adj.view(B, N, N)
    adj = torch.where(adj >= 1, 1., 0.)

    return adj


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class Attention(nn.Module):
    def __init__(self, channel, head_num):
        super(Attention, self).__init__()
        self.wQ = nn.Conv1d(channel, channel, kernel_size=1)
        self.wK = nn.Conv1d(channel, channel, kernel_size=1)
        self.wV = nn.Conv1d(channel, channel, kernel_size=1)
        self.h_n = head_num

        self.conv = nn.Conv1d(channel, channel, kernel_size=1)
        # self.conv2 = nn.Conv1d(channel*2, channel, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, N = x.shape
        Q, K, V = self.wQ(x).view(B, -1, N), self.wK(x).view(B, -1, N), self.wV(x).view(B, -1, N)

        att = Q * self.softmax(K) + V
        # att = self.softmax(att)
        # print(att.shape)

        # x = (att @ V).contiguous().view(B, -1, N)
        #
        # output = self.act(self.conv(x)) + x
        # center = x_ - x
        # neighbor = self.conv1(x_)
        # att = torch.cat([center, neighbor], dim=1)
        # att = self.softmax(self.conv2(att))
        # output = x_ * att

        return att

class AFF(nn.Module):

    def __init__(self, channels=256, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # xa = x + y
        xl = self.local_att(x)
        xg = self.global_att(y)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + y * (1 - wei)
        return xo

class graph(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size):
        super(graph, self).__init__()
        if kernel_size==1:
            self.conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=1),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(outchannel, outchannel, kernel_size=1))
        if kernel_size==3:
            self.conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(outchannel, outchannel, kernel_size=1))
        if kernel_size==5:
            self.conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=5, padding=2),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(outchannel, outchannel, kernel_size=1))
    def forward(self, x, adj):
        x = self.conv(x) @ adj

        return x



class GCN(nn.Module):
    def __init__(self, inchannel, outchannel, dim, depth, kernel_size):
        super(GCN, self).__init__()
        self.gcn = nn.ModuleList([
            graph(dim, dim, kernel_size)
            for i in range(depth)])
        self.head = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.tail = nn.Sequential(nn.Conv1d(outchannel, outchannel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, adj):
        x = self.head(x)
        shortcut = x
        for g in self.gcn:
            x = g(x, adj)
        x = self.tail(x) + shortcut

        return x

class Baseline(nn.Module):
    def __init__(self, in_channels=12, output_channels=8):
        super(Baseline, self).__init__()
        # self.k = k
        print("Baseline")
        ''' coordinate stream '''
        self.bn1_c = nn.BatchNorm1d(64)
        self.bn2_c = nn.BatchNorm1d(256)
        self.bn3_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(64, 64, kernel_size=1, bias=False))


        self.conv1_n = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(64, 64, kernel_size=1, bias=False))



        self.FTM_c1 = STNkd(k=12)
        self.FTM_n1 = STNkd(k=12)

        '''feature-wise attention'''

        self.fa_1 = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2))

        # self.fa_2 = nn.Sequential(nn.Conv1d(512 * 3, 512, kernel_size=1, bias=False),
        #                         # nn.BatchNorm1d(1024),
        #                         nn.LeakyReLU(0.2))

        ''' feature fusion '''
        self.pred1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Linear(256, 128, bias=False),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Linear(128, output_channels, bias=False))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)

        # self.fusion = nn.Conv1d(128, 256, kernel_size=1)

        self.aff_coor_1 = AFF(128, 0.5)
        self.aff_nor_1 = AFF(128, 0.5)
        self.aff_coor_2 = AFF(256, 0.5)
        self.aff_nor_2 = AFF(256, 0.5)
        self.aff_coor_3 = AFF(512, 0.5)
        self.aff_nor_3 = AFF(256, 0.5)
        self.aff_coor_4 = AFF(512, 0.5)
        self.aff_nor_4 = AFF(512, 0.5)

        self.aff_1 = AFF(128, 1)
        self.aff_2 = AFF(256, 1)
        self.aff_3 = AFF(512, 1)
        self.aff_4 = AFF(512, 1)

        # self.aff_fu_1 = AFF(512, 1)
        # self.aff_fu_2 = AFF(512, 1)
        # self.aff_fu_3 = AFF(512, 1)

        self.gcn_coor_1_1 = GCN(64, 128, 128, 2, 1)
        self.gcn_coor_1_2 = GCN(128, 128, 128, 2, 1)
        self.gcn_coor_1_3 = GCN(128, 128, 128, 2, 1)
        self.gcn_nor_1_1 = GCN(64, 128, 128, 2, 1)

        self.gcn_coor_2_1 = GCN(128, 256, 256, 1, 1)
        self.gcn_coor_2_2 = GCN(128, 256, 256, 1, 1)
        # self.gcn_coor_2_3 = GCN(256, 256, 256, 2, 1)
        self.gcn_nor_2_1 = GCN(128, 256, 256, 1, 1)
        # self.gcn_nor_2_2 = GCN(128, 256, 256, 8, 1)

        self.gcn_coor_3_1 = GCN(256, 512, 512, 2, 1)
        self.gcn_coor_3_2 = GCN(256, 512, 512, 2, 1)
        self.gcn_nor_3_1 = GCN(256, 512, 512, 2, 1)
        # self.gcn_nor_3_2 = GCN(256, 512, 512, 8, 1)

        self.gcn_coor_4_1 = GCN(512, 512, 512, 2, 1)
        self.gcn_coor_4_2 = GCN(512, 512, 512, 2, 1)
        self.gcn_nor_4_1 = GCN(512, 512, 512, 2, 1)
        # self.gcn_nor_4_2 = GCN(256, 512, 512, 8, 1)

        self.fu_1 = nn.Sequential(nn.Conv1d(128+256, 512, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(512, 512, kernel_size=1, bias=False))

        self.fu_2 = nn.Sequential(nn.Conv1d(512+512, 512, kernel_size=1, bias=False),
                                  self.bn3_c,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv1d(512, 512, kernel_size=1, bias=False))

    def forward(self, x, index_face):
        adj = Adj_matrix_gen(index_face)
        # adj = adj @ adj @ adj
        adj = adj @ adj
        coor = x[:, :12, :]
        # fea = coor
        nor = x[:, 12:, :]

        coor = self.conv1_c(coor)
        nor = self.conv1_n(nor)

        coor1 = self.gcn_coor_1_1(coor, adj)
        coor1 = self.gcn_coor_1_2(coor1, adj)
        coor1 = self.gcn_coor_1_3(coor1, adj)
        nor1 = self.gcn_nor_1_1(nor, adj)
        coor_nor1 = self.aff_1(coor1, nor1)

        coor2_1 = self.gcn_coor_2_1(coor1, adj)
        coor2_2 = self.gcn_coor_2_2(coor1, adj)
        coor2 = self.aff_coor_2(coor2_1, coor2_2)
        nor2 = self.gcn_nor_2_1(nor1, adj)
        coor_nor2 = self.aff_2(coor2, nor2)

        coor3_1 = self.gcn_coor_3_1(coor2, adj)
        coor3_2 = self.gcn_coor_3_2(coor2, adj)
        coor3 = self.aff_coor_3(coor3_1, coor3_2)
        nor3 = self.gcn_nor_3_1(nor2, adj)
        coor_nor3 = self.aff_3(coor3, nor3)

        coor4_1 = self.gcn_coor_4_1(coor3, adj)
        coor4_2 = self.gcn_coor_4_2(coor3, adj)
        coor4 = self.aff_coor_4(coor4_1, coor4_2)
        nor4 = self.gcn_nor_4_1(nor3, adj)
        coor_nor4 = self.aff_4(coor4, nor4)

        x1 = torch.cat((coor_nor1, coor_nor2), dim=1)
        x1 = self.fu_1(x1)
        x2 = torch.cat((coor_nor3, coor_nor4), dim=1)
        x2 = self.fu_2(x2)
        x = torch.cat((x1, x2), dim=1)
        # x = coor_nor3
        x = self.fa_1(x).transpose(-1, -2)

        # x = torch.cat((x1, x2), dim=2)


        x = self.pred1(x)
        x = self.pred2(x)
        x = self.pred3(x)

        score = self.pred4(x)
        score = F.log_softmax(score, dim=2)
        return score
