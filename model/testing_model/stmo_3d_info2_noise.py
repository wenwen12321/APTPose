import torch
import torch.nn as nn
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        #self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w1 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        #self.w2 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class FCBlock(nn.Module):

    def __init__(self, channel_in, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.channel_in = channel_in
        self.stage_num = 3
        self.p_dropout = 0.1
        #self.fc_1 = nn.Linear(self.channel_in, self.linear_size)
        self.fc_1 = nn.Conv1d(self.channel_in, self.linear_size, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        #self.fc_2 = nn.Linear(self.linear_size, channel_out)
        self.fc_2 = nn.Conv1d(self.linear_size, channel_out, kernel_size=1)

        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)

        return x

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        sem_layers = args.sem_layers
        stride_num = args.stride_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        # self.encoder = FCBlock(2*self.num_joints_in, channel, 2*channel, 1)
        #######################################################################################
        self.encoder1 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)
        self.encoder2 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)
        self.encoder3 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)
        self.encoder4 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)
        #######################################################################################

        self.Transformer = Transformer(layers, channel, d_hid, length=length)
        self.Transformer_reduce = Transformer_reduce(len(stride_num), channel, d_hid, \
            length=length, stride_num=stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

        # 【Geometric knowledge】(adding by 1/6)
        ################################################################
        self.fcn_vec = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*(self.num_joints_out-1), kernel_size=1)
        )

        self.fcn_vec1 = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*(self.num_joints_out-1), kernel_size=1)
        )
        ################################################################

    def forward(self, x):
        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape # (b, f ,j, c) = (160, 9, 17, 2)
        b,f,j,c = x.shape

        ##【Adding noise】(adding by 2/12)
        ################################################################
        ## Gaussian noise
        noise_mean = 0
        noise_std = 0.05 # {0.01, 0.03, 0.05}

        noise = (torch.randn(b, f, j, c) * noise_std + noise_mean).cuda()

        x_noisy = x + noise
        x= x_noisy
        ################################################################

        x = x.view(x.shape[0], x.shape[1], -1) # (160, 9, 34)
        x = x.permute(0, 2, 1).contiguous()  # (160, 34, 9)

        # x = self.encoder(x) # (160, 256, 9)
        ## Fix stageII bug
        ################################################
        x1 = self.encoder1(x) # (160, 256, 9)
        x2 = self.encoder2(x) # (160, 256, 9)
        x3 = self.encoder3(x) # (160, 256, 9)
        x4 = self.encoder4(x) # (160, 256, 9)

        x = x4 * x3 * x2 * x1
        ################################################

        x = x.permute(0, 2, 1).contiguous() # (160, 9, 256)
        x = self.Transformer(x) # (160, 9, 256)

        x_VTE = x # (160, 9, 256)
        x_VTE = x_VTE.permute(0, 2, 1).contiguous() # (160, 256, 9)
        # 【Geometric knowledge】(adding by 1/6)
        ################################################################
        boneVec_VTE = self.fcn_vec1(x_VTE) # (b, (j-1), f) = (160, 48, 9)
        boneVec_VTE = boneVec_VTE.view(x_shape[0], self.num_joints_out-1, -1, x_VTE.shape[2]) # 160, 16, 3, 9
        boneVec_VTE = boneVec_VTE.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (160, 3, 9, 16, 1)
        ################################################################
        x_VTE = self.fcn_1(x_VTE) # (160, 51, 9)

        x_VTE = x_VTE.view(x_shape[0], self.num_joints_out, -1, x_VTE.shape[2])
        x_VTE = x_VTE.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (160, 3, 9, 17, 1)

        ## stride Transformer
        x = self.Transformer_reduce(x) # (160, 1, 256)
        x = x.permute(0, 2, 1).contiguous() # (160, 256, 1)
        # 【Geometric knowledge】(adding by 1/6)
        ################################################################
        boneVec = self.fcn_vec(x) # (b, (j-1), f) = (160, 48, 1)
        boneVec = boneVec.view(x_shape[0], self.num_joints_out-1, -1, x.shape[2])
        boneVec = boneVec.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (160, 3, 1, 16, 1)
        ################################################################
        x = self.fcn(x) # (160, 51, 1)

        x = x.view(x_shape[0], self.num_joints_out, -1, x.shape[2])
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (160, 3, 1, 17, 1)
        
        # 【Geometric knowledge】(adding by 1/6)
        ################################################################
        return x, x_VTE, boneVec, boneVec_VTE
        ################################################################

        # return x, x_VTE




