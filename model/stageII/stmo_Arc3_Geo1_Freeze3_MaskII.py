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

## 【Freeze weight】(adding by 1/24) - stageII Masking (modified by 1/29)
########################################################################################################
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        sem_layers = args.sem_layers
        stride_num = args.stride_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

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

        ## 【Freeze weight】(adding by 1/23) - fix bug (multi-encoder)
        ## 【Freeze weight】(adding by 1/24) - stageII Masking 
        ######################################################################
        # (1) spatial mask
        self.spatial_mask_num = args.spatial_mask_num
        self.spatial_mask_token = nn.Parameter(torch.randn(1,1,2))
        self.encoder1 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)

        # (2) bone-length mask
        self.bone_mask_num = 1 # (一次 mask 一個 bone-legth => 2個joint)
        self.bone_mask_token = nn.Parameter(torch.randn(1, 1, 2))
        self.encoder2 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)

        # (3) Limb_mask1 - arm/leg
        self.limb_mask1_num = 3
        self.limb_mask1_token = nn.Parameter(torch.randn(1, 1, 2))
        self.encoder3 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)

        # (4) Limb_mask2 - left/right
        self.limb_mask2_num = 6
        self.limb_mask2_token = nn.Parameter(torch.randn(1, 1, 2))
        self.encoder4 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)
        ######################################################################

    #【Freeze weight】(adding by 1/24) - stageII Masking
    ##########################################################
    def forward(self, x_in, mask, spatial_mask, bone_mask, limb_mask1, limb_mask2):
    ##########################################################    
        x_in = x_in[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()
        x_shape = x_in.shape # (b, f ,j, c) = (160, 9, 17, 2)
        b,f,_,_ = x_in.shape

        ## 【Hierarchical masking - all】(adding by 12/6)(modiefied by 1/29)
        ## 【Freeze weight】(adding by 1/23) - fix bug (multi-encoder)
        #########################################################################################################
        
        ## Hierarchical masking machanism
        ################################################
        # (1) spatial mask out
        x1 = x_in.clone()
        x1[:, spatial_mask] = self.spatial_mask_token.expand(b, self.spatial_mask_num*f, 2) # 有 mask 的地方用 nn.parameter 補起來

        # (2) bone-length mask out
        x2 = x_in.clone()
        for i in range(bone_mask.shape[0]): # frame_num
            count_true = list(bone_mask[i]).count(True)
            # t = x3[0, i].shape
            # print(t)
            x2[:, i, bone_mask[i]] = self.bone_mask_token.expand(b, count_true, 2) # 根據每個 frame 有幾個 joint 被 mask 去 expand nn.Parameters (因為可能 mask 到同一點，所以 mask joint number 可能不同)
        # x2[:,bone_mask] = self.bone_mask_token.expand(b,self.bone_mask_num*2*f,2)

        # (3) Limb1 - arm/leg mask out
        x3 = x_in.clone()
        x3[:,limb_mask1] = self.limb_mask1_token.expand(b,self.limb_mask1_num*f,2)

        # (4) Limb2 - left/right mask out
        x4 = x_in.clone()
        x4[:,limb_mask2] = self.limb_mask2_token.expand(b,self.limb_mask2_num*f,2)
        ################################################


        ## SEM (MLP) - Arc3. (4) Limb2-left/right -> (1)joint
        # (4-1) Limb2 - left/right SEM4
        x4 = x4.view(b, f, -1) # (b, f, (jc))
        x4 = x4.permute(0,2,1).contiguous() # (b, (jc), f)

        x4 = self.encoder4(x4) # (b, e, f)
        x4 = x4.permute(0,2,1).contiguous() # (b, f, e)
        
        # (3-1) Limb1 - arm/leg SEM3
        x3 = x3.view(b, f, -1) # (b, f, (jc))
        x3 = x3.permute(0,2,1).contiguous() # (b, (jc), f)

        x3 = self.encoder3(x3) # (b, e, f)
        x3 = x3.permute(0,2,1).contiguous() # (b, f, e)
        
        # (2-1) bone-length SEM2
        x2 = x2.view(b, f, -1) # (b, f, (jc))
        x2 = x2.permute(0,2,1).contiguous() # (b, (jc), f)

        x2 = self.encoder2(x2) # (b, e, f)
        x2 = x2.permute(0,2,1).contiguous() # (b, f, e)
        
        # (1-1) spatial SEM1
        x1 = x1.view(b, f, -1) # (b, f, (jc))
        x1 = x1.permute(0,2,1).contiguous() # (b, (jc), f)

        x1 = self.encoder1(x1) # (b, e, f)
        x1 = x1.permute(0,2,1).contiguous() # (b, f, e)

        ## Fusion Feature
        x = x4 * x3 * x2 * x1

        ## 【Masking Architecture3】(adding by 1/6) - limb2joint
        ###############################################################
        # feat_l1_l2 = x1 * x2 # fusion (limb1) (limb2) features
        # feat_l1_l2_b = feat_l1_l2 * x3 # fusion (l1 and l2 features) and (bone) features
        # x = feat_l1_l2_b * x # Hierarchical spatial features
        ###############################################################
        #########################################################################################################
        
        ## TEM (Transformer)
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

########################################################################################################




