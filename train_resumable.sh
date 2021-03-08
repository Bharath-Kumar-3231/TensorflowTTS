DATASET_DIR=/dltraining/datasets
OUTDIR=/dltraining/outdir
CKPT_DIR=$OUTDIR/checkpoints

gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d',')
echo "gpus = $gpus"

taskid=${1}
echo "task id is $taskid"

latestCkptPath=$(ls -t $CKPT_DIR/ckpt-*index | head -1)
ckptExists=false
latestCkpt="NA"
if [[ $latestCkptPath == *"index" ]]; 
then
        latestCkpt="$( echo "$latestCkptPath" | sed -e 's#.index$##' )"
        ckptExists=true
fi
echo "latest checkpoint is $latestCkptPath"

if [[ $ckptExists == true ]]; 
then
CUDA_VISIBLE_DEVICES=$gpus python examples/fastspeech2_libritts/train_fastspeech2.py \
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
    --resume "$latestCkpt"
else
rm -rf /dltraining/outdir/
CUDA_VISIBLE_DEVICES=$gpus python examples/fastspeech2_libritts/train_fastspeech2.py \
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
fi

latestModelPath=$(ls -t $CKPT_DIR/model-*h5 | head -1)
modelExists=false
if [[ $latestModelPath == *"h5" ]]; 
then
  modelExists=true
fi
echo "latest model is $latestModelPath"

if [[ $modelExists == true ]]; 
then
  aws s3 cp "$OUTDIR/config.yml" "s3://murf-models-dev/trained/$taskid/config.yml"
  aws s3 cp "$latestModelPath" "s3://murf-models-dev/trained/$taskid/model.h5"
fi


