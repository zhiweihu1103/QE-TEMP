export DATA_PATH=../data/FB15k-237-betae
export SAVE_PATH=../logs/FB15k-237/logice_temp
export LOG_PATH=../logs/FB15k-237/logice_temp.out
export MODEL=temp
export FAITHFUL=no_faithful

export MAX_STEPS=450000
export VALID_STEPS=10000
export SAVE_STEPS=10000
export ENT_TYPE_NEIGHBOR=32
export REL_TYPE_NEIGHBOR=64

CUDA_VISIBLE_DEVICES=0 nohup python -u ../main.py --cuda --do_train --do_valid --do_test \
  --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 400 -g 0.375 \
  -lr 0.0001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS \
  --cpu_num 1 --geo logic -logicm "(luk,0,1,0,1600,2)" --test_batch_size 16 --tasks "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up" --print_on_screen \
  --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR \
  > $LOG_PATH 2>&1 &