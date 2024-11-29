GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck

CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/shiba --tap_mode=bootstapir --boundary_enhance_th=-1.0
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/shiba

CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/breakdance-flare --dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/breakdance-flare

CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/train --dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/train
