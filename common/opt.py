import argparse
import os
import math
import time
import torch

#############################################################################################################        
##                                          Hierarchical Masking Mode                                          ##
############################################################################################################# 

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--layers', default=4, type=int)
        self.parser.add_argument('--channel', default=256, type=int)
        self.parser.add_argument('--d_hid', default=512, type=int)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str) # {'cpn_ft_h36m_dbb', 'hr', 'gt'}
        
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        # self.parser.add_argument('--root_path', type=str, default='../../Data/3DHPE/P-STMO_dataset/')# 4090
        self.parser.add_argument('--root_path', type=str, default='/home/final/HD/Data/3DHPE/P-STMO_dataset/')# 4090
        # self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', type=int, default=0)
        self.parser.add_argument('--test', type=int, default=1)
        self.parser.add_argument('--nepoch', type=int, default=80)
        self.parser.add_argument('-b','--batchSize', type=int, default=160)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_refine', type=float, default=1e-5)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=80)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('-f','--frames', type=int, default=243)
        self.parser.add_argument('--pad', type=int, default=121)
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--reload', type=int, default=0)
        self.parser.add_argument('--refine_reload', type=int, default=0)
       
        ## 【Adding get_cosine_schedule_with_warmup】(adding by 12/13)
        # self.parser.add_argument('-c','--checkpoint', type=str, default='scheduler_lr0.001_v3')
        self.parser.add_argument('--warmupepoch', type=int, default=3)

        ## 【9 frames testing】
        # self.parser.add_argument('-c','--checkpoint', type=str, default='9frame/1_11_stageIIMasking')
        
        ## 【243 frames testing】
        self.parser.add_argument('-c','--checkpoint', type=str, default='243frame/2_10_info2')

        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--previous_refine_name', type=str, default='')
        self.parser.add_argument('--manualSeed', type=int, default=1)

        self.parser.add_argument('--MAE', action='store_true')
        self.parser.add_argument('-tmr','--temporal_mask_rate', type=float, default=0)
        self.parser.add_argument('-smn', '--spatial_mask_num', type=int, default=0)
        self.parser.add_argument('-tds', '--t_downsample', type=int, default=1)

        self.parser.add_argument('--MAE_reload', type=int, default=0)
        self.parser.add_argument('-r', '--resume', action='store_true')

        ## 【Freeze weight】(adding by 1/23)
        self.parser.add_argument('--freeze', type=int, default=0, help='--freeze 1 => freeze weight')
        
        ## hyperparameter of APTPose
        ############################################################################
        # Mask ratios of HMPM 
        self.parser.add_argument('-bmn', '--bone_mask_num', type=int, default=1, help='bone-length mask number')
        self.parser.add_argument('-lmn', '--limb_mask_num', type=int, default=1, help='Limb mask number')
        self.parser.add_argument('-hmn', '--hbody_mask_num', type=int, default=1, help='Half-body mask number')
        
        # stage I - Pre-trainning
        # reprojection layers & loss weight
        self.parser.add_argument('--reproj_layers', type=int, default=1)
        self.parser.add_argument('-w_2d', '--pose2d_loss', type=float, default=1, help='loss weight of 2d pose')
        self.parser.add_argument('-w_3d', '--pose3d_loss', type=float, default=0.3, help='loss weight of 3d pose')
        self.parser.add_argument('-w_v', '--velocity_loss', type=float, default=0, help='loss weight of velocity')
        
        # stage I - Fine-tunning
        # loss weight
        self.parser.add_argument('-wp_m', '--multi_pose_loss', type=float, default=1, help='loss weight of multiple frame\'s pose')
        self.parser.add_argument('-wp_s', '--single_pose_loss', type=float, default=1, help='loss weight of single frame\'s pose')
        self.parser.add_argument('-wb_m', '--multi_vector_loss', type=float, default=0.5, help='loss weight of multiple frame\'s bone vector')
        self.parser.add_argument('-wb_s', '--single_vector_loss', type=float, default=0.5, help='loss weight of single frame\'s bone vector')

        # SEM layers (default=1)
        self.parser.add_argument('--sem_layers', type=int, default=1)
        
        # self.parser.add_argument('--show_eval', type=int, default=1, help='show evaluate result in training stage')
        
        ## geometric loss in {pre-train / fine-tune / both}
        self.parser.add_argument('--geo_loss', type=int, default='0', help='1 = with geometric loss in pre-train / 0 = without geometric loss in pre-train}')
        ############################################################################
        
        ## 【COCO dataset】
        ############################################################################
        self.parser.add_argument('--subjects_train', type=str, default='S1,S5,S6,S7,S8') # 'S1,S5,S6,S7,S8,COCO'
        self.parser.add_argument('--subjects_test', type=str, default='S9,S11')
        ############################################################################
        

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        self.opt.pad = (self.opt.frames-1) // 2

        stride_num = {
                '9': [1, 3, 3],
                '27':  [3, 3, 3],
                '351': [3, 9, 13],
                '81': [3, 3, 3, 3],
                '243': [3, 3, 3, 3, 3],
            }

        if str(self.opt.frames) in stride_num:
            self.opt.stride_num = stride_num[str(self.opt.frames)]
        else:
            self.opt.stride_num = None
            print('no stride_num')
            exit()

        # self.opt.subjects_train = 'S1,S5,S6,S7,S8' # move to parser
        # self.opt.subjects_test = 'S9,S11'

        #if self.opt.train:
        logtime = time.strftime('%m%d_%H%M_%S_')

        ckp_suffix = ''
        if self.opt.refine:
            ckp_suffix='_refine'
        elif self.opt.MAE:
            ckp_suffix = '_pretrain'
        else:
            ckp_suffix = '_STMO'
        self.opt.checkpoint = 'checkpoint/'+self.opt.checkpoint + '_%d'%(self.opt.pad*2+1) + \
            '%s'%ckp_suffix

        if not os.path.exists(self.opt.checkpoint):
            os.makedirs(self.opt.checkpoint)

        if self.opt.train:
            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))

            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
       
        return self.opt


#############################################################################################################

#############################################################################################################        
##                                          Debug mode (pretraining)                                                   ##
############################################################################################################# 

# class opts():
#     def __init__(self):
#         self.parser = argparse.ArgumentParser()

#     def init(self):
#         self.parser.add_argument('--layers', default=3, type=int) # layers of transformer
#         self.parser.add_argument('--channel', default=256, type=int)
#         self.parser.add_argument('--d_hid', default=512, type=int)
        
#         self.parser.add_argument('--dataset', type=str, default='h36m')
#         # self.parser.add_argument('--dataset', type=str, default='coco')
        
#         self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
#         # self.parser.add_argument('-k', '--keypoints', default='gt', type=str)
#         # self.parser.add_argument('-k', '--keypoints', default='hr', type=str)
        
#         self.parser.add_argument('--data_augmentation', type=bool, default=True)
#         self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
#         self.parser.add_argument('--test_augmentation', type=bool, default=True)
#         self.parser.add_argument('--crop_uv', type=int, default=0)
#         self.parser.add_argument('--root_path', type=str, default='../../Data/3DHPE/P-STMO_dataset/')
#         # self.parser.add_argument('-a', '--actions', default='*', type=str)
#         self.parser.add_argument('-a', '--actions', default='Directions', type=str)
#         self.parser.add_argument('--downsample', default=1, type=int)
#         self.parser.add_argument('--subset', default=1, type=float)
#         self.parser.add_argument('-s', '--stride', default=1, type=int)
#         self.parser.add_argument('--gpu', default='0', type=str, help='')
#         self.parser.add_argument('--train', type=int, default=1)
#         self.parser.add_argument('--test', type=int, default=1)
#         self.parser.add_argument('--nepoch', type=int, default=20)
#         self.parser.add_argument('-b','--batchSize', type=int, default=160)
#         self.parser.add_argument('--lr', type=float, default=1e-3)
#         self.parser.add_argument('--lr_refine', type=float, default=1e-5)
#         self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
#         self.parser.add_argument('--large_decay_epoch', type=int, default=80)
#         self.parser.add_argument('--workers', type=int, default=8)
#         self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
#         self.parser.add_argument('-f','--frames', type=int, default=9)
#         self.parser.add_argument('--pad', type=int, default=121)
#         self.parser.add_argument('--refine', action='store_true')
#         self.parser.add_argument('--reload', type=int, default=0)
#         self.parser.add_argument('--refine_reload', type=int, default=0)
#         self.parser.add_argument('-c','--checkpoint', type=str, default='Debug/2_24_3dhp')
#         self.parser.add_argument('--previous_dir', type=str, default='')
#         self.parser.add_argument('--n_joints', type=int, default=17)
#         self.parser.add_argument('--out_joints', type=int, default=17)
#         self.parser.add_argument('--out_all', type=int, default=1)
#         self.parser.add_argument('--in_channels', type=int, default=2)
#         self.parser.add_argument('--out_channels', type=int, default=3)
#         self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
#         self.parser.add_argument('-previous_name', type=str, default='')
#         self.parser.add_argument('--previous_refine_name', type=str, default='')
#         self.parser.add_argument('--manualSeed', type=int, default=1)

#         self.parser.add_argument('--MAE', action='store_true', default=1)
#         self.parser.add_argument('-tmr','--temporal_mask_rate', type=float, default=0.8)
#         self.parser.add_argument('-smn', '--spatial_mask_num', type=int, default=2)
#         self.parser.add_argument('-tds', '--t_downsample', type=int, default=1)

#         self.parser.add_argument('--MAE_reload', type=int, default=0)
#         self.parser.add_argument('-r', '--resume', action='store_true')


#         self.parser.add_argument('--warmupepoch', type=int, default=3)
#         self.parser.add_argument('--freeze', type=int, default=0)

#        ## hyperparameter of ERPoseFormer
#         ############################################################################
#         self.parser.add_argument('-bmn', '--bone_mask_num', type=int, default=1, help='bone-length mask number')
#         self.parser.add_argument('-lmn', '--limb_mask_num', type=int, default=1, help='Limb mask number')
#         self.parser.add_argument('-hmn', '--hbody_mask_num', type=int, default=1, help='Half-body mask number')

#         self.parser.add_argument('--reproj_layers', type=int, default=1)
#         self.parser.add_argument('-w_2d', '--pose2d_loss', type=float, default=1, help='loss weight of 2d pose')
#         self.parser.add_argument('-w_3d', '--pose3d_loss', type=float, default=0.3, help='loss weight of 3d pose')
        
#         self.parser.add_argument('-w_v', '--velocity_loss', type=float, default=0, help='loss weight of velocity')
        
#         self.parser.add_argument('-wp_m', '--multi_pose_loss', type=float, default=1, help='loss weight of multiple frame\'s pose')
#         self.parser.add_argument('-wp_s', '--single_pose_loss', type=float, default=1, help='loss weight of single frame\'s pose')
#         self.parser.add_argument('-wb_m', '--multi_vector_loss', type=float, default=0.5, help='loss weight of multiple frame\'s bone vector')
#         self.parser.add_argument('-wb_s', '--single_vector_loss', type=float, default=0.5, help='loss weight of single frame\'s bone vector')
        
#         self.parser.add_argument('--sem_layers', type=int, default=1)
        
#         self.parser.add_argument('--show_eval', type=int, default=1, help='show evaluate result in training stage')
#         ############################################################################
        
#         ## 【COCO dataset】
#         ############################################################################
#         self.parser.add_argument('--subjects_train', type=str, default='S1') # 'S1,S5,S6,S7,S8,COCO'
#         self.parser.add_argument('--subjects_test', type=str, default='S9')
#         ############################################################################
#         # self.parser.add_argument('--subjects_train', type=str, default='S1') # 'S1,S5,S6,S7,S8,COCO'
#         # self.parser.add_argument('--subjects_test', type=str, default='S9')

#     def parse(self):
#         self.init()
#         self.opt = self.parser.parse_args()

#         self.opt.pad = (self.opt.frames-1) // 2

#         stride_num = {
#                 '9': [1, 3, 3],
#                 '27':  [3, 3, 3],
#                 '351': [3, 9, 13],
#                 '81': [3, 3, 3, 3],
#                 '243': [3, 3, 3, 3, 3],
#             }

#         if str(self.opt.frames) in stride_num:
#             self.opt.stride_num = stride_num[str(self.opt.frames)]
#         else:
#             self.opt.stride_num = None
#             print('no stride_num')
#             exit()

#         # self.opt.subjects_train = 'S1,S5,S6,S7,S8'
#         # self.opt.subjects_test = 'S9,S11'
#         # self.opt.subjects_train = 'S1' # move to parser
#         # self.opt.subjects_test = 'S11'

#         #if self.opt.train:
#         logtime = time.strftime('%m%d_%H%M_%S_')

#         ckp_suffix = ''
#         if self.opt.refine:
#             ckp_suffix='_refine'
#         elif self.opt.MAE:
#             ckp_suffix = '_pretrain'
#         else:
#             ckp_suffix = '_STMO'
#         self.opt.checkpoint = 'checkpoint/'+self.opt.checkpoint + '_%d'%(self.opt.pad*2+1) + \
#             '%s'%ckp_suffix

#         if not os.path.exists(self.opt.checkpoint):
#             os.makedirs(self.opt.checkpoint)

#         if self.opt.train:
#             args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
#                     if not name.startswith('_'))

#             file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
#             with open(file_name, 'wt') as opt_file:
#                 opt_file.write('==> Args:\n')
#                 for k, v in sorted(args.items()):
#                     opt_file.write('  %s: %s\n' % (str(k), str(v)))
#                 opt_file.write('==> Args:\n')
       
#         return self.opt


#############################################################################################################        
##                                          Debug mode (fine-tuning)                                                   ##
############################################################################################################# 

# class opts():
#     def __init__(self):
#         self.parser = argparse.ArgumentParser()

#     def init(self):
#         self.parser.add_argument('--layers', default=3, type=int)
#         self.parser.add_argument('--channel', default=256, type=int)
#         self.parser.add_argument('--d_hid', default=512, type=int)
#         self.parser.add_argument('--dataset', type=str, default='h36m')
        
#         # self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
#         self.parser.add_argument('-k', '--keypoints', default='gt', type=str)
#         # self.parser.add_argument('-k', '--keypoints', default='hr', type=str)
        
#         self.parser.add_argument('--data_augmentation', type=bool, default=True)
#         self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
#         self.parser.add_argument('--test_augmentation', type=bool, default=True)
#         self.parser.add_argument('--crop_uv', type=int, default=0)
#         self.parser.add_argument('--root_path', type=str, default='../../Data/3DHPE/P-STMO_dataset/')
#         # self.parser.add_argument('-a', '--actions', default='*', type=str)
#         self.parser.add_argument('-a', '--actions', default='Directions', type=str)
#         self.parser.add_argument('--downsample', default=1, type=int)
#         self.parser.add_argument('--subset', default=1, type=float)
#         self.parser.add_argument('-s', '--stride', default=1, type=int)
#         self.parser.add_argument('--gpu', default='0', type=str, help='')
#         self.parser.add_argument('--train', type=int, default=1)
#         self.parser.add_argument('--test', type=int, default=1)
#         # self.parser.add_argument('--nepoch', type=int, default=60)
#         self.parser.add_argument('--nepoch', type=int, default=20)
    
#         # self.parser.add_argument('-b','--batchSize', type=int, default=160)
#         self.parser.add_argument('-b','--batchSize', type=int, default=160)
#         self.parser.add_argument('--lr', type=float, default=1e-3)
#         self.parser.add_argument('--lr_refine', type=float, default=1e-5)
#         self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
#         self.parser.add_argument('--large_decay_epoch', type=int, default=80)
#         self.parser.add_argument('--workers', type=int, default=8)
#         self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
#         # self.parser.add_argument('-f','--frames', type=int, default=243)
#         self.parser.add_argument('-f','--frames', type=int, default=9)
#         self.parser.add_argument('--pad', type=int, default=121)
#         self.parser.add_argument('--refine', action='store_true')
        
#         self.parser.add_argument('--reload', type=int, default=0) 
        
#         ## 【freeze3 - step2】
#         #########################################################
#         # self.parser.add_argument('--reload', type=int, default=1) ##### freeze3 step2, (load step 1 weight then unfreeze to train) 
#         # self.parser.add_argument('--previous_dir', type=str, default='checkpoint/Debug/1_29_Arc3_Geo1_freeze_9_STMO/no_refine_5_11170.pth')
#         #########################################################
        
#         # 【Freeze weight】(adding by 1/23)
#         self.parser.add_argument('--freeze', type=int, default=0) # freeze=1
        
          
#         self.parser.add_argument('--refine_reload', type=int, default=0)
#         self.parser.add_argument('-c','--checkpoint', type=str, default='Debug/2_20_3dhp')
#         self.parser.add_argument('--previous_dir', type=str, default='checkpoint/Debug/2_20_3dhp_9_pretrain/MAE_1_4064.pth')
#         self.parser.add_argument('--n_joints', type=int, default=17)
#         self.parser.add_argument('--out_joints', type=int, default=17)
#         self.parser.add_argument('--out_all', type=int, default=1)
#         self.parser.add_argument('--in_channels', type=int, default=2)
#         self.parser.add_argument('--out_channels', type=int, default=3)
#         self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
#         self.parser.add_argument('-previous_name', type=str, default='')
#         self.parser.add_argument('--previous_refine_name', type=str, default='')
#         self.parser.add_argument('--manualSeed', type=int, default=1)

#         self.parser.add_argument('--MAE', action='store_true', default=0)
#         self.parser.add_argument('-tmr','--temporal_mask_rate', type=float, default=0)
#         self.parser.add_argument('-smn', '--spatial_mask_num', type=int, default=0)
#         self.parser.add_argument('-tds', '--t_downsample', type=int, default=1)

#         self.parser.add_argument('--MAE_reload', type=int, default=1)
#         self.parser.add_argument('-r', '--resume', action='store_true')


#         self.parser.add_argument('--warmupepoch', type=int, default=3)

#        ## hyperparameter of ERPoseFormer
#         ############################################################################
#         self.parser.add_argument('-bmn', '--bone_mask_num', type=int, default=1, help='bone-length mask number')
#         self.parser.add_argument('-lmn', '--limb_mask_num', type=int, default=1, help='Limb mask number')
#         self.parser.add_argument('-hmn', '--hbody_mask_num', type=int, default=1, help='Half-body mask number')
        
#         self.parser.add_argument('--reproj_layers', type=int, default=1)
#         self.parser.add_argument('-w_2d', '--pose2d_loss', type=float, default=1, help='loss weight of 2d pose')
#         self.parser.add_argument('-w_3d', '--pose3d_loss', type=float, default=0.3, help='loss weight of 3d pose')
        
#         self.parser.add_argument('-w_v', '--velocity_loss', type=float, default=0, help='loss weight of velocity')
        
#         self.parser.add_argument('-wp_m', '--multi_pose_loss', type=float, default=1, help='loss weight of multiple frame\'s pose')
#         self.parser.add_argument('-wp_s', '--single_pose_loss', type=float, default=1, help='loss weight of single frame\'s pose')
#         self.parser.add_argument('-wb_m', '--multi_vector_loss', type=float, default=0.5, help='loss weight of multiple frame\'s bone vector')
#         self.parser.add_argument('-wb_s', '--single_vector_loss', type=float, default=0.5, help='loss weight of single frame\'s bone vector')
        
#         self.parser.add_argument('--sem_layers', type=int, default=1)
        
#         self.parser.add_argument('--show_eval', type=int, default=1, help='show evaluate result in training stage')
#         ############################################################################
        
#         ## 【COCO dataset】
#         ############################################################################
#         self.parser.add_argument('--subjects_train', type=str, default='S1') # 'S1,S5,S6,S7,S8,COCO'
#         self.parser.add_argument('--subjects_test', type=str, default='S9')
#         ############################################################################



#     def parse(self):
#         self.init()
#         self.opt = self.parser.parse_args()

#         self.opt.pad = (self.opt.frames-1) // 2

#         stride_num = {
#                 '9': [1, 3, 3],
#                 '27':  [3, 3, 3],
#                 '351': [3, 9, 13],
#                 '81': [3, 3, 3, 3],
#                 '243': [3, 3, 3, 3, 3],
#             }

#         if str(self.opt.frames) in stride_num:
#             self.opt.stride_num = stride_num[str(self.opt.frames)]
#         else:
#             self.opt.stride_num = None
#             print('no stride_num')
#             exit()

#         # self.opt.subjects_train = 'S1,S5,S6,S7,S8'
#         # self.opt.subjects_test = 'S9,S11'
#         # self.opt.subjects_train = 'S1' # move to parser
#         # self.opt.subjects_test = 'S11'

#         #if self.opt.train:
#         logtime = time.strftime('%m%d_%H%M_%S_')

#         ckp_suffix = ''
#         if self.opt.refine:
#             ckp_suffix='_refine'
#         elif self.opt.MAE:
#             ckp_suffix = '_pretrain'
#         else:
#             ckp_suffix = '_STMO'
#         self.opt.checkpoint = 'checkpoint/'+self.opt.checkpoint + '_%d'%(self.opt.pad*2+1) + \
#             '%s'%ckp_suffix

#         if not os.path.exists(self.opt.checkpoint):
#             os.makedirs(self.opt.checkpoint)

#         if self.opt.train:
#             args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
#                     if not name.startswith('_'))

#             file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
#             with open(file_name, 'wt') as opt_file:
#                 opt_file.write('==> Args:\n')
#                 for k, v in sorted(args.items()):
#                     opt_file.write('  %s: %s\n' % (str(k), str(v)))
#                 opt_file.write('==> Args:\n')
       
#         return self.opt

#############################################################################################################        
##                                          Debug mode (Evaluation)                                                   ##
############################################################################################################# 

# class opts():
#     def __init__(self):
#         self.parser = argparse.ArgumentParser()

#     def init(self):
#         self.parser.add_argument('--layers', default=8, type=int)
#         self.parser.add_argument('--channel', default=256, type=int)
#         self.parser.add_argument('--d_hid', default=512, type=int)
#         self.parser.add_argument('--dataset', type=str, default='h36m')
        
#         self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
#         # self.parser.add_argument('-k', '--keypoints', default='gt', type=str)
#         # self.parser.add_argument('-k', '--keypoints', default='hr', type=str)
        
#         self.parser.add_argument('--data_augmentation', type=bool, default=True)
#         self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
#         self.parser.add_argument('--test_augmentation', type=bool, default=True)
#         self.parser.add_argument('--crop_uv', type=int, default=0)
#         self.parser.add_argument('--root_path', type=str, default='../../Data/3DHPE/P-STMO_dataset/')
#         self.parser.add_argument('-a', '--actions', default='*', type=str)
#         # self.parser.add_argument('-a', '--actions', default='Directions', type=str)
#         self.parser.add_argument('--downsample', default=1, type=int)
#         self.parser.add_argument('--subset', default=1, type=float)
#         self.parser.add_argument('-s', '--stride', default=1, type=int)
#         self.parser.add_argument('--gpu', default='0', type=str, help='')
#         self.parser.add_argument('--train', type=int, default=0)
#         self.parser.add_argument('--test', type=int, default=1)
#         # self.parser.add_argument('--nepoch', type=int, default=60)
#         self.parser.add_argument('--nepoch', type=int, default=20)
    
#         self.parser.add_argument('-b','--batchSize', type=int, default=160)
#         self.parser.add_argument('--lr', type=float, default=1e-3)
#         self.parser.add_argument('--lr_refine', type=float, default=1e-5)
#         self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
#         self.parser.add_argument('--large_decay_epoch', type=int, default=80)
#         self.parser.add_argument('--workers', type=int, default=8)
#         self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
#         self.parser.add_argument('-f','--frames', type=int, default=243)
#         self.parser.add_argument('--pad', type=int, default=121)
#         self.parser.add_argument('--refine', action='store_true')
        
#         self.parser.add_argument('--reload', type=int, default=1) 
        
#         ## 【freeze3 - step2】
#         #########################################################
#         # self.parser.add_argument('--reload', type=int, default=1) ##### freeze3 step2, (load step 1 weight then unfreeze to train) 
#         # self.parser.add_argument('--previous_dir', type=str, default='checkpoint/Debug/1_29_Arc3_Geo1_freeze_9_STMO/no_refine_5_11170.pth')
#         #########################################################
        
#         # 【Freeze weight】(adding by 1/23)
#         self.parser.add_argument('--freeze', type=int, default=0) # freeze=1
        
          
#         self.parser.add_argument('--refine_reload', type=int, default=0)
#         self.parser.add_argument('-c','--checkpoint', type=str, default='')
#         # self.parser.add_argument('--previous_dir', type=str, default='checkpoint/243frame/no_refine_34_4296.pth')
#         # self.parser.add_argument('--previous_dir', type=str, default='checkpoint/Supp/v3_layer6_30_4335.pth')
#         self.parser.add_argument('--previous_dir', type=str, default='checkpoint/Supp/v3_layer8_30_4338.pth')
        
#         self.parser.add_argument('--n_joints', type=int, default=17)
#         self.parser.add_argument('--out_joints', type=int, default=17)
#         self.parser.add_argument('--out_all', type=int, default=1)
#         self.parser.add_argument('--in_channels', type=int, default=2)
#         self.parser.add_argument('--out_channels', type=int, default=3)
#         self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
#         self.parser.add_argument('-previous_name', type=str, default='')
#         self.parser.add_argument('--previous_refine_name', type=str, default='')
#         self.parser.add_argument('--manualSeed', type=int, default=1)

#         self.parser.add_argument('--MAE', action='store_true', default=0)
#         self.parser.add_argument('-tmr','--temporal_mask_rate', type=float, default=0)
#         self.parser.add_argument('-smn', '--spatial_mask_num', type=int, default=0)
#         self.parser.add_argument('-tds', '--t_downsample', type=int, default=1)

#         self.parser.add_argument('--MAE_reload', type=int, default=0)
#         self.parser.add_argument('-r', '--resume', action='store_true')


#         self.parser.add_argument('--warmupepoch', type=int, default=3)

#        ## hyperparameter of ERPoseFormer
#         ############################################################################
#         self.parser.add_argument('-bmn', '--bone_mask_num', type=int, default=1, help='bone-length mask number')
#         self.parser.add_argument('-lmn', '--limb_mask_num', type=int, default=1, help='Limb mask number')
#         self.parser.add_argument('-hmn', '--hbody_mask_num', type=int, default=1, help='Half-body mask number')
        
#         self.parser.add_argument('--reproj_layers', type=int, default=1)
#         self.parser.add_argument('-w_2d', '--pose2d_loss', type=float, default=1, help='loss weight of 2d pose')
#         self.parser.add_argument('-w_3d', '--pose3d_loss', type=float, default=0.3, help='loss weight of 3d pose')
        
#         self.parser.add_argument('-w_v', '--velocity_loss', type=float, default=0, help='loss weight of velocity')
        
#         self.parser.add_argument('-wp_m', '--multi_pose_loss', type=float, default=1, help='loss weight of multiple frame\'s pose')
#         self.parser.add_argument('-wp_s', '--single_pose_loss', type=float, default=1, help='loss weight of single frame\'s pose')
#         self.parser.add_argument('-wb_m', '--multi_vector_loss', type=float, default=0.5, help='loss weight of multiple frame\'s bone vector')
#         self.parser.add_argument('-wb_s', '--single_vector_loss', type=float, default=0.5, help='loss weight of single frame\'s bone vector')
        
#         self.parser.add_argument('--sem_layers', type=int, default=1)
        
#         self.parser.add_argument('--show_eval', type=int, default=1, help='show evaluate result in training stage')
#         ############################################################################
        
#         ## 【COCO dataset】
#         ############################################################################
#         self.parser.add_argument('--subjects_train', type=str, default='S1') # 'S1,S5,S6,S7,S8,COCO'
#         # self.parser.add_argument('--subjects_test', type=str, default='S9')
#         self.parser.add_argument('--subjects_test', type=str, default='S9,S11')
#         ############################################################################



#     def parse(self):
#         self.init()
#         self.opt = self.parser.parse_args()

#         self.opt.pad = (self.opt.frames-1) // 2

#         stride_num = {
#                 '9': [1, 3, 3],
#                 '27':  [3, 3, 3],
#                 '351': [3, 9, 13],
#                 '81': [3, 3, 3, 3],
#                 '243': [3, 3, 3, 3, 3],
#             }

#         if str(self.opt.frames) in stride_num:
#             self.opt.stride_num = stride_num[str(self.opt.frames)]
#         else:
#             self.opt.stride_num = None
#             print('no stride_num')
#             exit()

#         # self.opt.subjects_train = 'S1,S5,S6,S7,S8'
#         # self.opt.subjects_test = 'S9,S11'
#         # self.opt.subjects_train = 'S1' # move to parser
#         # self.opt.subjects_test = 'S11'

#         #if self.opt.train:
#         logtime = time.strftime('%m%d_%H%M_%S_')

#         ckp_suffix = ''
#         if self.opt.refine:
#             ckp_suffix='_refine'
#         elif self.opt.MAE:
#             ckp_suffix = '_pretrain'
#         else:
#             ckp_suffix = '_STMO'
#         self.opt.checkpoint = 'checkpoint/'+self.opt.checkpoint + '_%d'%(self.opt.pad*2+1) + \
#             '%s'%ckp_suffix

#         if not os.path.exists(self.opt.checkpoint):
#             os.makedirs(self.opt.checkpoint)

#         if self.opt.train:
#             args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
#                     if not name.startswith('_'))

#             file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
#             with open(file_name, 'wt') as opt_file:
#                 opt_file.write('==> Args:\n')
#                 for k, v in sorted(args.items()):
#                     opt_file.write('  %s: %s\n' % (str(k), str(v)))
#                 opt_file.write('==> Args:\n')
       
#         return self.opt