#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/fastspeech2_libritts/train_fastspeech2.py \
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2_libritts/train_fastspeech2.py \
  --train-dir /dltraining/datasets/dump_libritts/train/ \
  --dev-dir /dltraining/datasets/dump_libritts/valid/ \
  --outdir /dltraining/outdir/ \
  --config ./examples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
  --use-norm 1 \
  --f0-stat /dltraining/datasets/dump_libritts/stats_f0.npy \
  --energy-stat /dltraining/datasets/dump_libritts/stats_energy.npy \
  --mixed_precision 1 \
  --dataset_mapping /dltraining/datasets/dump_libritts/libritts_mapper.json \
  --dataset_config preprocess/libritts_preprocess.yaml \
  --dataset_stats /dltraining/datasets/dump_libritts/stats.npy \
  --resume /dltraining/outdir/checkpoints/ckpt-70000
