#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/fastspeech2_libritts/train_fastspeech2.py \
$DEFAULT_TASK_ID="TASK-1"
DATASET_DIR=/dltraining/datasets
OUTDIR=/dltraining/outdir
CKPT_DIR=$OUTDIR/checkpoints

taskid=${1}

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
    --resume "$latestCkpt"
else
aws cp s3://murf-models-dev/pretrained/fs2-192-80k.h5 "$DATASET_DIR/pretrained_fs2_192-80k.h5"
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
    --pretrained "$DATASET_DIR/pretrained_fs2_192-80k.h5"
fi

latestModelPath=$(ls -t $CKPT_DIR/model-*h5 | head -1)
modelExists=false
if [[ $latestmodelPath == *"h5" ]]; 
then
  modelExists=true
fi
echo "latest model is $latestModelPath"

if [[ $modelExists == true ]]; 
then
  aws cp "$OUTDIR/config.yaml" "s3://murf-models-dev/trained/$taskid/config.yaml"
  aws cp "$latestModelPath" "s3://murf-models-dev/trained/$taskid/model.h5"
fi


