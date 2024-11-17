import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_3dhp_mae import Fusion
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine

## Stage I Pre-training model selection!!!!!!!! #####
from model.stageI.stmo_pretrain_Arc3_Geo1 import Model_MAE

##### Stage II Fine-tuning model selection!!!!!!!! #####
# from model.stageII.stmo_Arc3_Geo1_Freeze3_MaskII import Model ## stageII masking version
from model.testing_model.stmo_3d_info_1 import Model

from thop import clever_format
from thop.profile import profile
import scipy.io as scio

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']
    model_MAE = model['MAE']

    if split == 'train':
        model_trans.train()
        model_refine.train()
        model_MAE.train()
        # 【Freeze weight】(adding by 1/23)
        ###########################################################
        if opt.freeze: 
            model_trans.module.encoder1.bn_1.eval()
            model_trans.module.encoder1.layers[0].batch_norm1.eval()
            model_trans.module.encoder1.layers[0].batch_norm2.eval()

            model_trans.module.encoder2.bn_1.eval()
            model_trans.module.encoder2.layers[0].batch_norm1.eval()
            model_trans.module.encoder2.layers[0].batch_norm2.eval()

            model_trans.module.encoder3.bn_1.eval()
            model_trans.module.encoder3.layers[0].batch_norm1.eval()
            model_trans.module.encoder3.layers[0].batch_norm2.eval()
            
            model_trans.module.encoder4.bn_1.eval()
            model_trans.module.encoder4.layers[0].batch_norm1.eval()
            model_trans.module.encoder4.layers[0].batch_norm2.eval()
        ###########################################################
    else:
        model_trans.eval()
        model_refine.eval()
        model_MAE.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()

    # action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
    # action_error_sum_MAE = define_error_list(actions)

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, 0)):

        if opt.MAE:
            #batch_cam, input_2D, seq, subject, scale, bb_box, cam_ind = data
            if split == "train":
                batch_cam, input_2D, seq, subject, scale, bb_box, cam_ind = data
            else:
                batch_cam, input_2D, seq, scale, bb_box = data
            [input_2D, batch_cam, scale, bb_box] = get_varialbe(split,[input_2D, batch_cam, scale, bb_box])

            N = input_2D.size(0)
            f = opt.frames

            # Temporal masking part
            mask_num = int(f*opt.temporal_mask_rate)
            mask = np.hstack([
                np.zeros(f - mask_num),
                np.ones(mask_num),
            ]).flatten()

            np.random.seed()
            np.random.shuffle(mask)

            mask = torch.from_numpy(mask).to(torch.bool).cuda()

            ## 【Hierarchical masking - all】(adding by 12/6)
            #########################################################################################################
            ## (1) Spatial masking part
            spatial_mask = np.zeros((f, 17), dtype=bool)
            for k in range(f):
                ran = random.sample(range(0, 16), opt.spatial_mask_num)
                spatial_mask[k, ran] = True

            ## (2) bone-length
            bone_length = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
            bone_mask = np.zeros((f, 17), dtype=bool)
            # bone_mask_num = 1 # hyperparameter, which can be fine-tuning (e.g. 1, 2, 3)
            bone_mask_num = opt.bone_mask_num # 1

            for k in range(f):
                ran = random.sample(range(0, 16), bone_mask_num)
                # print(ran)
                for i in ran:
                    b = bone_length[i]
                    # print(i, bone_length[i][0], bone_length[i][1])
                    bone_mask[k, b] = True

            ## (3) Limb1 - arm/leg
            limb_mask_num = opt.limb_mask_num # 1
            limb_mask1 = np.zeros((f, 17), dtype=bool)
            for k in range(f):
                ran = random.sample(range(0, 4), limb_mask_num)
                if ran == [0] : 
                    limb = joints_left[:3] # arm_left
                elif ran == [1]:
                    limb = joints_left[3:] # leg_left
                elif ran == [2]:
                    limb = joints_right[:3] # arm_right
                elif ran == [3]:
                    limb = joints_right[3:] # leg_right

                limb_mask1[k, limb] = True

            ## (4) Limb2 - left/right (Half-body)
            hbody_mask_num = opt.hbody_mask_num # 1
            limb_mask2 = np.zeros((f, 17), dtype=bool)
            for k in range(f):
                ran = random.sample(range(0, 2), hbody_mask_num)
                if ran == [0] : 
                    limb_side = joints_left
                else:
                    limb_side = joints_right

                limb_mask2[k, limb_side] = True
            #########################################################################################################   

            if opt.test_augmentation and split == 'test':
                # input_2D, output_2D = input_augmentation_MAE(input_2D, model_MAE, joints_left, joints_right, mask, spatial_mask)
                ## 【Hierarchical masking - all】(adding by 12/6)
                #########################################################################################################
                input_2D, output_2D = input_augmentation_MAE(input_2D, model_MAE, joints_left, joints_right, mask, spatial_mask, bone_mask, limb_mask1, limb_mask2)
                ######################################################################################################### 
            else:
                input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(
                    torch.cuda.FloatTensor) # input_2D (B,F,J,2) -> (B,2,F,J,1)
                
                # output_2D = model_MAE(input_2D, mask, spatial_mask)
                ## 【Hierarchical masking - all】(adding by 12/6)
                #########################################################################################################
                output_2D = model_MAE(input_2D, mask, spatial_mask, bone_mask, limb_mask1, limb_mask2)
                #########################################################################################################

            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)
            output_2D = output_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)

            loss = mpjpe_cal(output_2D, torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1))


        else:
            #batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data

            if split == "train":
                batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
            else:
                batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

            [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,
                                                                       [input_2D, gt_3D, batch_cam, scale, bb_box])

            N = input_2D.size(0)

            ## 準備 3D pose GT
            out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels) # (b, f, j, c) = (160, 9, 17, 3)
            out_target[:, :, 14] = 0 # 把 root joint 設成 (x, y, z) = (0, 0, 0)
            gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

            # 如果 out_target frame number > 1，把 center frame 當作 out_target_single 
            if out_target.size(1) > 1:
                out_target_single = out_target[:, opt.pad].unsqueeze(1)
                gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
            else:
                out_target_single = out_target
                gt_3D_single = gt_3D

            ## 準備 3D bone vector GT
            # 【Geometric knowledge】(adding by 1/6)
            ##################################################################
            out_target_pose = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels) # (b, f, j, c) = (160, 9, 17, 3)
            # out_target_boneVec[:, :, 0] = 0 # 把 root joint 設成 (x, y, z) = (0, 0, 0)
            out_target_boneVec = get_BoneVecbypose3d(out_target_pose) # (B, F, bone-length, 3) = (160, 9, 16, 3)
            # gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

            if out_target.size(1) > 1:
                out_target_boneVec_single = out_target_boneVec[:, opt.pad].unsqueeze(1) # (160, 1, 16, 3)
                # gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
            else:
                out_target_boneVec_single = out_target_boneVec
                # gt_3D_single = gt_3D
            ##################################################################
            
            if opt.test_augmentation and split =='test':
                # input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, model_trans, joints_left, joints_right)
                # 【Geometric knowledge】(adding by 1/6)
                ##################################################################
                input_2D, output_3D, output_3D_VTE, output_boneVec, output_boneVec_VTE = input_augmentation(input_2D, model_trans, joints_left, joints_right)
                ##################################################################
            else:
                input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)
                
                # output_3D, output_3D_VTE = model_trans(input_2D)
                # 【Geometric knowledge】(adding by 1/6)
                ##################################################################
                output_3D, output_3D_VTE, output_boneVec, output_boneVec_VTE = model_trans(input_2D)
                ##################################################################

            ## 處理 3D pose output 維度轉換 & 乘上 scale
            output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels) # (b,f,j,c)=(160,9,17,3)
            output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels) # (b,f,j,c)=(160,1,17,3)

            # print(f"scale:\n{scale.shape}\n{scale}") # shape=batch_size=(160) # full 1
            output_3D_VTE = output_3D_VTE * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D_VTE.size(1),opt.out_joints, opt.out_channels)
            output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.out_joints, opt.out_channels)
            output_3D_single = output_3D

            ## 處理 3D bone vector output 維度轉換 & 乘上 scale
            # 【Geometric knowledge】(adding by 1/6)
            ##################################################################
            output_boneVec_VTE = output_boneVec_VTE.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints-1, opt.out_channels)
            output_boneVec = output_boneVec.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints-1, opt.out_channels)

            output_boneVec_VTE = output_boneVec_VTE * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_boneVec_VTE.size(1),opt.out_joints-1, opt.out_channels)
            output_boneVec = output_boneVec * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_boneVec.size(1),opt.out_joints-1, opt.out_channels)
            output_boneVec_single = output_boneVec
            ##################################################################

            if split == 'train':
                pred_out = output_3D_VTE
                ## 【Geometric knowledge】(adding by 1/6)
                ##################################################################
                pred_out_bone = output_boneVec_VTE
                ##################################################################

            elif split == 'test':
                pred_out = output_3D_single
                ## 【Geometric knowledge】(adding by 1/6)
                ##################################################################
                pred_out_bone = output_boneVec_single
                ##################################################################

            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)

            if opt.refine:
                pred_uv = input_2D
                uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
                xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
                xyz[:, :, 0, :] = 0
                post_out = model_refine(output_3D_single, xyz)
                loss = mpjpe_cal(post_out, out_target_single)
            else:
                # loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(output_3D_single, out_target_single)
                
                ## 【Geometric knowledge】(adding by 1/6)
                ##################################################################
                ## weight (hyperparameter)
                wp_m = opt.multi_pose_loss    #1   # weight of multiple frame's pose
                wp_s = opt.single_pose_loss   #1   # weight of single frame's pose
                
                wb_m = opt.multi_vector_loss  #0.5 # weight of multiple frame's bone vector
                wb_s = opt.single_vector_loss #0.5 # weight of single frame's bone vector 
                
                w_v = opt.velocity_loss       #0   # weight of velocity
                
                loss = wp_m*mpjpe_cal(pred_out, out_target) + wp_s*mpjpe_cal(output_3D_single, out_target_single) + \
                       wb_m*mpjpe_cal(pred_out_bone, out_target_boneVec) + wb_s*mpjpe_cal(output_boneVec_single, out_target_boneVec_single) + \
                       w_v*mean_velocity_error(pred_out, out_target)
                ##################################################################

        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N) # 把 loss 轉成 numpy 形式丟到 cpu (才能紀錄下來)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()  # compute gradient in the computational graph
            optimizer.step() # update parameters in the model

            if not opt.MAE:

                if opt.refine:
                    post_out[:,:,14,:] = 0
                    joint_error = mpjpe_cal(post_out, out_target_single).item()
                else:
                    pred_out[:,:,14,:] = 0
                    joint_error = mpjpe_cal(pred_out, out_target).item()

                error_sum.update(joint_error*N, N)

        elif split == 'test':
            if opt.MAE:
                # action_error_sum_MAE = test_calculation(output_2D, torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1), action, action_error_sum_MAE, opt.dataset,
                #                                     subject,MAE=opt.MAE)
                joint_error_test = mpjpe_cal(torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1), output_2D).item()
            else:
                pred_out[:, :, 14, :] = 0
                #action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, subject)
                joint_error_test = mpjpe_cal(pred_out, out_target).item()
                out = pred_out
                # if opt.refine:
                #     post_out[:, :, 14, :] = 0
                #     action_error_sum_post_out = test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset, subject)

            if opt.train == 0:
            # if opt.train == 0 or opt.show_eval == 1:
                for seq_cnt in range(len(seq)):
                    seq_name = seq[seq_cnt]
                    if seq_name in data_inference:
                        data_inference[seq_name] = np.concatenate(
                            (data_inference[seq_name], out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                    else:
                        data_inference[seq_name] = out[seq_cnt].permute(2, 1, 0).cpu().numpy()

            error_sum_test.update(joint_error_test * N, N)

    if split == 'train':
        if opt.MAE:
            return loss_all['loss'].avg*1000
        else:
            return loss_all['loss'].avg, error_sum.avg
    elif split == 'test':
        if opt.MAE:
            #p1, p2 = print_error(opt.dataset, action_error_sum_MAE, opt.train)
            return error_sum_test.avg*1000
        if opt.refine:
            p1, p2 = print_error(opt.dataset, action_error_sum_post_out, opt.train)
        else:
            #p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
            if opt.train == 0:
            # if opt.train == 0 or opt.show_eval == 1:
                for seq_name in data_inference.keys():
                    data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
                mat_path = os.path.join(opt.checkpoint, 'inference_data.mat')
                scio.savemat(mat_path, data_inference)

        return error_sum_test.avg

# def input_augmentation_MAE(input_2D, model_trans, joints_left, joints_right, mask, spatial_mask=None):
## 【Hierarchical masking - all】
#########################################################################################################
def input_augmentation_MAE(input_2D, model_trans, joints_left, joints_right, mask, spatial_mask=None, bone_mask=None, limb_mask1=None, limb_mask2=None):
#########################################################################################################  
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    # output_2D_flip = model_trans(input_2D_flip, mask, spatial_mask)
    ## 【Hierarchical masking - all】
    #########################################################################################################
    output_2D_flip = model_trans(input_2D_flip, mask, spatial_mask, bone_mask, limb_mask1, limb_mask2)
    #########################################################################################################

    output_2D_flip[:, 0] *= -1

    output_2D_flip[:, :, :, joints_left + joints_right] = output_2D_flip[:, :, :, joints_right + joints_left]

    # output_2D_non_flip = model_trans(input_2D_non_flip, mask, spatial_mask)
    ## 【Hierarchical masking - all】
    #########################################################################################################
    output_2D_non_flip = model_trans(input_2D_non_flip, mask, spatial_mask, bone_mask, limb_mask1, limb_mask2)
    #########################################################################################################

    output_2D = (output_2D_non_flip + output_2D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_2D

def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)   
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) 

    # output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip)
    ## 【Geometric knowledge】(adding by 1/6)
    ##################################################################
    output_3D_flip, output_3D_flip_VTE, output_3D_flip_boneVec, output_3D_flip_boneVec_VTE = model_trans(input_2D_flip)
    ##################################################################

    output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1

    output_3D_flip_VTE[:, :, :, joints_left + joints_right] = output_3D_flip_VTE[:, :, :, joints_right + joints_left]
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    # output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip)
    ## 【Geometric knowledge】(adding by 1/6)
    ##################################################################
    output_3D_non_flip, output_3D_non_flip_VTE, output_3D_non_flip_boneVec, output_3D_non_flip_boneVec_VTE = model_trans(input_2D_non_flip)
    ##################################################################

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    ## 【Geometric knowledge】(adding by 1/6)
    ##################################################################
    output_3D_boneVec_VTE = (output_3D_non_flip_boneVec_VTE + output_3D_flip_boneVec_VTE) / 2
    output_3D_boneVec = (output_3D_non_flip_boneVec + output_3D_flip_boneVec) / 2
    ##################################################################

    input_2D = input_2D_non_flip

    # return input_2D, output_3D, output_3D_VTE
    ## 【Geometric knowledge】(adding by 1/6)
    ##################################################################
    return input_2D, output_3D, output_3D_VTE, output_3D_boneVec, output_3D_boneVec_VTE
    ##################################################################

## 【Freeze weight】(adding by 1/23)
############################################
def freeze_model(model, to_freeze_dict, keep_step=None):
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            # print(name)
            param.requires_grad = False
        else:
            pass
    
    return model
############################################
if __name__ == '__main__':
    # 設種子
    opt.manualSeed = 1

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 新增 checkpoint train.log (e.g. model_9_pretrain/train.log)
    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    # load dataset (h36m 3d data 跟 define_actions)
    root_path = opt.root_path
    # dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    #dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        #train_data = Fusion(opt=opt, train=True, root_path=root_path)
        train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        #test_data = Fusion(opt=opt, train=False,root_path =root_path)
        test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = 17

    model = {}
    model['trans'] = nn.DataParallel(Model(opt)).cuda()
    model['refine'] = nn.DataParallel(refine(opt)).cuda()
    model['MAE'] = nn.DataParallel(Model_MAE(opt)).cuda()

    ## params, mac
    ######################
    model_test=Model(opt)
    dsize = (2, 2, 81, 17, 1)
    inputs = torch.randn(dsize)
    total_ops, total_params = profile(model_test, (inputs,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    print('MACs:', macs)
    print('Paras:', params)
    ######################

    model_params = 0
    for parameter in model['trans'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)


    # if opt.MAE_test_reload==1:
    #     model_dict = model['MAE'].state_dict()

    #     MAE_test_path = opt.previous_dir

    #     pre_dict_MAE = torch.load(MAE_test_path)
    #     for name, key in model_dict.items():
    #         model_dict[name] = pre_dict_MAE[name]
    #     model['MAE'].load_state_dict(model_dict)

    if opt.MAE_reload == 1:
        model_dict = model['trans'].state_dict() # get the key information of all parameters

        MAE_path = opt.previous_dir

        pre_dict = torch.load(MAE_path)

        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        
        # model['trans'].load_state_dict(model_dict)
        ## 【Freeze weight】(adding by 1/23)
        ####################################################################
        model['trans'].load_state_dict(model_dict, strict=False) #  "strict=False" to ignore non-matching keys.(whether missing or more keys)
        if opt.freeze:
            model['trans'] = freeze_model(model=model['trans'], to_freeze_dict=state_dict)
        ####################################################################


    model_dict = model['trans'].state_dict()
    if opt.reload == 1:

        no_refine_path = opt.previous_dir

        pre_dict = torch.load(no_refine_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]

            # print(model_dict.keys())
            # n = name[7:]
            # model_dict[n] = pre_dict[n]

        model['trans'].load_state_dict(model_dict)
        # print(model['trans'].module.encoder1.fc_1.weight)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload == 1:

        refine_path = opt.previous_refine_name

        pre_dict_refine = torch.load(refine_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    
    # optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    ## 【Freeze weight】(adding by 1/23)
    ####################################################################
    if opt.MAE_reload == 1 and opt.freeze:
        optimizer_all = optim.Adam(filter(lambda p: p.requires_grad, all_param), lr=opt.lr, amsgrad=True)
    else:
        optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    ####################################################################

    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            if not opt.MAE:
                loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
            else:
                loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
        if opt.test == 1:
            if not opt.MAE:
                p1 = val(opt, actions, test_dataloader, model)
            else:
                p1 = val(opt, actions, test_dataloader, model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                if opt.MAE:
                    opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold,
                                                   model['MAE'], 'MAE')

                else:
                    opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], 'no_refine')

                    if opt.refine:
                        opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                              data_threshold, model['refine'], 'refine')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
            # if opt.train == 0 or opt.show_eval == 1:
                print('p1: %.2f' % (p1))

                ## save p1 value as .txt file name
                ############################################
                p1_path = os.path.join(opt.checkpoint, 'p1: %.2f.txt' % (p1))
                np.savetxt(p1_path, np.arange(0.0,5.0,1.0)) # 0,1,2,3,4
                ############################################
                
                break
            else:
                if opt.MAE:
                    logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f' % (
                    epoch, lr, loss, p1))
                    print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f' % (epoch, lr, loss, p1))
                else:
                    logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))
                    print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f' % (epoch, lr, loss, mpjpe, p1))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay
