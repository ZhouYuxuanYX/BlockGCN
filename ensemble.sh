##!/bin/bash
#python ensemble.py --dataset ntu/xsub --joint-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic \
#--bone-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic  \
#--joint-motion-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic   \
#--bone-motion-dir work_dir/ntu60/csub/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic



# 96.4
#python ensemble.py --dataset ntu/xview --joint-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic \
#--bone-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic  \
#--joint-motion-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic   \
#--bone-motion-dir work_dir/ntu60/cview/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic

python ensemble.py --dataset NW-UCLA --joint-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic \
--bone-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic  \
--joint-motion-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic   \
--bone-motion-dir work_dir/ucla/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic

#python ensemble.py --dataset ntu120/xset --joint-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic \
#--bone-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic  \
#--joint-motion-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic   \
#--bone-motion-dir work_dir/ntu120/cset/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic

#python ensemble.py --dataset ntu120/xsub --joint-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_new_8heads_deterministic \
#--bone-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_bone_new_8heads_deterministic  \
#--joint-motion-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_vel_new_8heads_deterministic   \
#--bone-motion-dir work_dir/ntu120/csub/BlockGCN_decay_110_120_140_epochs_bone_vel_new_8heads_deterministic

