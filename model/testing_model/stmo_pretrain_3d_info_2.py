import torch
import torch.nn as nn
from model.block.vanilla_transformer_encoder_pretrain import Transformer, Transformer_dec
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
import numpy as np

## 【Hierarchical Features Fusion Module】(adding by 1/5)
#############################################################################
# Non-local helper, used for Attentive Fusion
class NL(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=256, kernel_size=1):
        super(NL, self).__init__()

        self.hidden_dim = hidden_dim

        # self.input_conv = nn.Conv1d(4*hidden_dim, 2*hidden_dim, 1, bias=False)
        # self.input_conv1 = nn.Conv1d(2*hidden_dim, hidden_dim, 1, bias=False)

        self.conv_q = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        self.conv_k = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        self.conv_v = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1, bias=False)
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        # x.shape (b, 4e, f) = (160, 1024, 9)
        # x = self.input_conv(x) # (b, 2e, f) = (160, 512, 9)
        # x = self.input_conv1(x) # (b, e, f) = (160, 256, 9)

        query = self.conv_q(x).unsqueeze(1) # (160, 1, 256, 9)
        key = self.conv_k(x).unsqueeze(0) # (1, 160, 256, 9)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5) # (160, 160, 9)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x) # (160, 256, 9)
        virt_feats = (att.unsqueeze(2) * value).sum(1) # (160, 256, 9)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)
        
        x = x + virt_feats
        x = self.conv_out(x)
        return x
#############################################################################

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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

class Model_MAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        sem_layers = args.sem_layers
        stride_num = args.stride_num
        # self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.length = length
        dec_dim_shrink = 2

        # self.encoder = FCBlock(2*self.num_joints_in, channel, 2*channel, 1)

        self.Transformer = Transformer(layers, channel, d_hid, length=length)
        self.Transformer_dec = Transformer_dec(layers-1, channel//dec_dim_shrink, d_hid//dec_dim_shrink, length=length)

        self.encoder_to_decoder = nn.Linear(channel, channel//dec_dim_shrink, bias=False)
        self.encoder_LN = LayerNorm(channel)
        
        self.fcn_dec = nn.Sequential(
            nn.BatchNorm1d(channel//dec_dim_shrink, momentum=0.1),
            nn.Conv1d(channel//dec_dim_shrink, 2*self.num_joints_out, kernel_size=1)
        )

        # self.fcn_1 = nn.Sequential(
        #     nn.BatchNorm1d(channel, momentum=0.1),
        #     nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        # )

        self.dec_pos_embedding = nn.Parameter(torch.randn(1, length, channel//dec_dim_shrink))
        self.mask_token = nn.Parameter(torch.randn(1, 1, channel//dec_dim_shrink))

        # self.spatial_mask_token = nn.Parameter(torch.randn(1, 1, 2))


        # 【Stage II Masking】(adding by 1/10)】
        #########################################################################################################
        # (1) spatial mask
        self.spatial_mask_num = args.spatial_mask_num
        self.spatial_mask_token = nn.Parameter(torch.randn(1,1,2))
        self.encoder1 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)

        # (2) bone-length mask
        self.bone_mask_num = 1 # (一次 mask 一個 bone-legth => 2個joint)
        self.bone_mask_token = nn.Parameter(torch.randn(1, 1, 2))
        self.encoder2 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)

        # (3) Limb_mask1 - arm/leg
        self.limb_mask1_num = 3 * args.limb_mask_num
        self.limb_mask1_token = nn.Parameter(torch.randn(1, 1, 2))
        self.encoder3 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)

        # (4) Limb_mask2 - left/right
        self.limb_mask2_num = 6 * args.hbody_mask_num
        self.limb_mask2_token = nn.Parameter(torch.randn(1, 1, 2))
        self.encoder4 = FCBlock(2*self.num_joints_in, channel, 2*channel, sem_layers)
        #########################################################################################################

        #【3D_info_1】(adding by 1/26)
        ###############################################################################################
        # """
        # Reprojection matrix
        #     - Arc   : MLP (FCBlock, Residual block)
        #     - hidden: 2*channel(2*256)
        #     - input : 3D pose (3*17)
        #     - output: 2D pose (2*17)
        # """
        # self.reprojection_matrix = FCBlock(3*self.num_joints_in, 2*self.num_joints_out, 2*channel, 1) #channel_in, channel_out, linear_size, block_num
        self.fcn_dec_3d = nn.Sequential(
            nn.BatchNorm1d(channel//dec_dim_shrink, momentum=0.1),
            nn.Conv1d(channel//dec_dim_shrink, 3*self.num_joints_out, kernel_size=1)
        )
        ###############################################################################################


    ## 【Hierarchical masking - all】(adding by 12/6)
    #########################################################################################################
    def forward(self, x_in, mask, spatial_mask, bone_mask, limb_mask1, limb_mask2): # 只有mask的shape=9, 其他mask=(9,17)
    #########################################################################################################
    # def forward(self, x_in, mask, spatial_mask, limb_mask):
        # x_in = (b, c, f, j, 1) # 160,2,9,17,1
        x_in = x_in[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous() # b,f,j,c = 160,9,17,2
        b,f,j,c = x_in.shape # (b, f, j, c)

        # 【Stage II Masking】(adding by 1/10)
        #########################################################################################################
        ## Hierarchical masking machanism
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
        x = x4 * x3 * x2 * x1 # (160,9,256)

        ## 【Masking Architecture3】(adding by 1/6) - limb2joint
        ###############################################################
        # feat_l1_l2 = x1 * x2 # fusion (limb1) (limb2) features
        # feat_l1_l2_b = feat_l1_l2 * x3 # fusion (l1 and l2 features) and (bone) features
        # x = feat_l1_l2_b * x # Hierarchical spatial features
        ###############################################################
        #########################################################################################################

        ## TEM (Transformer)
        feas = self.Transformer(x, mask_MAE=mask) # (b, vis_f, e) = (8,2,256)

        feas = self.encoder_LN(feas)
        feas = self.encoder_to_decoder(feas) # (b, vis_f, e/dec_shrink) = (8,2,128)

        B, N, C = feas.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.dec_pos_embedding.expand(B, -1, -1).clone() # (b,f,e/dec_shrink) = (8, 9, 128) # expend 的 -1 表示該維 dim 不變
        pos_emd_vis = expand_pos_embed[:, ~mask].reshape(B, -1, C) # (8, 2, 128)
        pos_emd_mask = expand_pos_embed[:, mask].reshape(B, -1, C) # (8, 7, 128)
        x_full = torch.cat([feas + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # (b, vis+mask_f, e/2) (8,9,128)

        # print('\nfeas', feas.shape) # (b, vis_f, e) = (8, 2, 128)
        # print('\npos_emd_vis', pos_emd_vis.shape) # (b, vis_f, e) = (8, 2, 128)
        # print('\nmask_token', self.mask_token.shape) # (1, 1, 128)
        # print('\npos_emd_mask', pos_emd_mask.shape) # (b, mask_f, e) = (8, 7, 128)
        # print('\nmask_token + pos_emd_mask', (self.mask_token + pos_emd_mask).shape) # (b, mask_f, e) = (8, 7, 128)

        x_out = self.Transformer_dec(x_full, pos_emd_mask.shape[1]) # (b, f, e/2) = (8, 9, 128)

        x_out = x_out.permute(0, 2, 1).contiguous() # (b, e/2, f) = (8, 128, 9)

        # original
        ####################################################################
        # x_out = self.fcn_dec(x_out) # (b, (jc),f) = (8, 34, 9)

        # x_out = x_out.view(b, self.num_joints_out, 2, -1) # (b, j, c, f) = (8, 17, 2, 9)
        # x_out = x_out.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (b,c,f,j,1)=(8, 2, 9, 17, 1)
        
        # return x_out
        ####################################################################

        #【3D_info_1】(adding by 1/26)
        ####################################################################
        # x_out_3d = self.fcn_dec_3d(x_out) # (b, (jc),f) = (8, 51, 9)
        # x_out_2d = self.reprojection_matrix(x_out_3d) # (b, (jc), f) = (8, 34, 9)

        # x_out_3d = x_out_3d.view(b, self.num_joints_out, 3, -1) # (b, j, c, f) = (8, 17, 3, 9)
        # x_out_3d = x_out_3d.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (b,c,f,j,1)=(8, 3, 9, 17, 1)

        # x_out_2d = x_out_2d.view(b, self.num_joints_out, 2, -1) # (b, j, c, f) = (8, 17, 2, 9)
        # x_out_2d = x_out_2d.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (b,c,f,j,1)=(8, 2, 9, 17, 1)
        
        # return x_out_2d, x_out_3d # 2d=(160,2,9,17,1), 3d=(160,3,9,17,1)
        ####################################################################
        
        #【3D_info_2】(adding by 2/10)
        ####################################################################
        """
        不使用 Reprojection module，直接 output 3D & 2D pose，(都是 supervision)
        """
        ## 3D pose
        x_out_3d = self.fcn_dec_3d(x_out) # (b, (jc),f) = (8, 51, 9)
        
        x_out_3d = x_out_3d.view(b, self.num_joints_out, 3, -1) # (b, j, c, f) = (8, 17, 3, 9)
        x_out_3d = x_out_3d.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (b,c,f,j,1)=(8, 3, 9, 17, 1)

        ## 2D pose
        x_out_2d = self.fcn_dec(x_out) # (b, (jc),f) = (8, 34, 9)

        x_out_2d = x_out_2d.view(b, self.num_joints_out, 2, -1) # (b, j, c, f) = (8, 17, 2, 9)
        x_out_2d = x_out_2d.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) # (b,c,f,j,1)=(8, 2, 9, 17, 1)
        
        return x_out_2d, x_out_3d # 2d=(160,2,9,17,1), 3d=(160,3,9,17,1)
        
        ####################################################################
        




