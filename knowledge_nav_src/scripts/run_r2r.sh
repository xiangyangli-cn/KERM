DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k

outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 4
      --lr 1e-5
      --iters 5000
      --log_every 100
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
#python3 -m torch.distributed.launch \
#    --nproc_per_node=${ngpus} main_nav.py $flag  \
#      --tokenizer bert \
#      --resume_file ../datasets/R2R/trained_models/best_val_unseen \
      #--eval_first

# test
CUDA_VISIBLE_DEVICES='3'  python3 main_nav.py $flag  \
      --tokenizer bert --world_size 0 \
      --resume_file ../datasets/R2R/exprs_map/finetune/dagger-vitbase-seed.0-init.aug.45k/ckpts/freeze_best \
      --test --submit