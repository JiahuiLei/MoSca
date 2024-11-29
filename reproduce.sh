# get command line input args: GPU ID and scene name1 scene name 2.
GPU_ID=${1:-0}
TOTAL_NUM_GPUS=${2:-1}
which python

############################################################################################
# * DYCHECK
############################################################################################
scene_names=(
    spin teddy wheel apple block paper-windmill space-out
)
num_scenes=${#scene_names[@]}
scenes_per_gpu=$(((num_scenes + TOTAL_NUM_GPUS - 1) / TOTAL_NUM_GPUS))
scenes_for_gpu=()
for ((i = GPU_ID; i < num_scenes; i += TOTAL_NUM_GPUS)); do
    scenes_for_gpu+=("${scene_names[i]}")
done
echo "GPU $GPU_ID: ${scenes_for_gpu[@]}"

for scene in "${scenes_for_gpu[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --ws ./data/iphone/$scene --cfg ./profile/iphone/iphone_prep.yaml
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --ws ./data/iphone/$scene --cfg ./profile/iphone/iphone_fit.yaml
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --ws ./data/iphone/$scene --cfg ./profile/iphone/iphone_fit_colfree.yaml
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --ws ./data/iphone/$scene --cfg ./profile/iphone/iphone_fit_focalonly.yaml
done
############################################################################################

############################################################################################
# * NVIDIA
############################################################################################
scene_names=(Jumping Skating Truck Umbrella Balloon1 Balloon2 Playground)
num_scenes=${#scene_names[@]}
scenes_per_gpu=$(((num_scenes + TOTAL_NUM_GPUS - 1) / TOTAL_NUM_GPUS))
scenes_for_gpu=()
for ((i = GPU_ID; i < num_scenes; i += TOTAL_NUM_GPUS)); do
    scenes_for_gpu+=("${scene_names[i]}")
done
echo "GPU $GPU_ID: ${scenes_for_gpu[@]}"

for scene in "${scenes_for_gpu[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --ws ./data/nvidia/$scene --cfg ./profile/nvidia/nvidia_prep.yaml
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --ws ./data/nvidia/$scene --cfg ./profile/nvidia/nvidia_fit.yaml
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --ws ./data/nvidia/$scene --cfg ./profile/nvidia/nvidia_fit_colfree.yaml
done
############################################################################################

############################################################################################
# * TUM
############################################################################################
scene_names=(
    rgbd_dataset_freiburg3_sitting_halfsphere
    rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_sitting_static
    rgbd_dataset_freiburg3_sitting_xyz
    rgbd_dataset_freiburg3_walking_halfsphere
    rgbd_dataset_freiburg3_walking_rpy
    rgbd_dataset_freiburg3_walking_static
    rgbd_dataset_freiburg3_walking_xyz
)
num_scenes=${#scene_names[@]}
scenes_per_gpu=$(((num_scenes + TOTAL_NUM_GPUS - 1) / TOTAL_NUM_GPUS))
scenes_for_gpu=()
for ((i = GPU_ID; i < num_scenes; i += TOTAL_NUM_GPUS)); do
    scenes_for_gpu+=("${scene_names[i]}")
done
echo "GPU $GPU_ID: ${scenes_for_gpu[@]}"

for scene in "${scenes_for_gpu[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --ws ./data/tum/$scene --cfg ./profile/tum/tum_prep.yaml --skip_dynamic_resample
    CUDA_VISIBLE_DEVICES=$GPU_ID python lite_moca_reconstruct.py --ws ./data/tum/$scene --cfg ./profile/tum/tum_fit.yaml
done
############################################################################################

############################################################################################
# * SINTEL
############################################################################################
scene_names=(alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3)
num_scenes=${#scene_names[@]}
scenes_per_gpu=$(((num_scenes + TOTAL_NUM_GPUS - 1) / TOTAL_NUM_GPUS))
scenes_for_gpu=()
for ((i = GPU_ID; i < num_scenes; i += TOTAL_NUM_GPUS)); do
    scenes_for_gpu+=("${scene_names[i]}")
done
echo "GPU $GPU_ID: ${scenes_for_gpu[@]}"

for scene in "${scenes_for_gpu[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --ws ./data/sintel/$scene --cfg ./profile/sintel/sintel_prep.yaml --skip_dynamic_resample
    CUDA_VISIBLE_DEVICES=$GPU_ID python lite_moca_reconstruct.py --ws ./data/sintel/$scene --cfg ./profile/sintel/sintel_fit.yaml
done
############################################################################################
