taskid=${1}
echo "task id is $taskid"

CORE_DATASET=/dltraining/datasets/libritts
DATASET_DIR=/dltraining/datasets_$taskid
OUTDIR=/dltraining/outdir_$taskid
CKPT_DIR=$OUTDIR/checkpoints
libritts=$DATASET_DIR/libritts
dump=$DATASET_DIR/dump_libritts
yaml=examples/multiband_melgan/conf/multiband_melgan_clone.yaml
fs2_yaml=examples/fastspeech2_libritts/conf/fastspeech2libritts_clone.yaml

mkdir "$DATASET_DIR"
mkdir "$libritts"

gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d',')
echo "gpus = $gpus"

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
  echo "RESUMING from checkpoint $latestCkpt"
  CUDA_VISIBLE_DEVICES=$gpus python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir $dump/train/ \
  --dev-dir $dump/valid/ \
  --outdir $OUTDIR/ \
  --use-norm 1 \
  --config $yaml \
  --resume "$latestCkpt"
else
  python setupCloneDataset.py --task_id=$taskid --libri_path=$libritts --dataset_path=$CORE_DATASET --for_vocoder='true'
  numSpeakers=$(ls $libritts|wc -l)
  echo "$numSpeakers found in libritts"
  
  rm -rf mfa
  rm -rf /home/ubuntu/Documents
  rm -rf $dump
  
  ./examples/mfa_extraction/scripts/prepare_mfa.sh

  python examples/mfa_extraction/run_mfa.py \
  --corpus_directory $libritts \
  --output_directory ./mfa/parsed \
  --jobs 8

  python examples/mfa_extraction/txt_grid_parser.py \
  --yaml_path $fs2_yaml \
  --dataset_path $libritts \
  --text_grid_path ./mfa/parsed \
  --output_durations_path $libritts/durations \
  --sample_rate 24000

  tensorflow-tts-preprocess --rootdir $libritts \
  --outdir $dump \
  --config preprocess/libritts_preprocess.yaml \
  --dataset libritts

  tensorflow-tts-normalize --rootdir $dump \
  --outdir $dump \
  --config preprocess/libritts_preprocess.yaml \
  --dataset libritts

  python examples/mfa_extraction/fix_mismatch.py \
  --base_path $dump \
  --trimmed_dur_path $libritts/trimmed-durations \
  --dur_path $libritts/durations
  
  pretrainedFile=/dltraining/datasets/generator-720000.h5
  
  if [ ! -f $pretrainedFile ]; then
      echo "Downloading pretrained vocoder from s3"
      aws s3 cp s3://murf-models-dev/pretrained/generator-720000.h5 $pretrainedFile
  fi

  echo "Using PRETRAINED from model $pretrainedFile"
  CUDA_VISIBLE_DEVICES=$gpus python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir $dump/train/ \
  --dev-dir $dump/valid/ \
  --outdir $OUTDIR/ \
  --use-norm 1 \
  --config $yaml \
  --pretrained $pretrainedFile
fi

latestModelPath=$(ls -t $CKPT_DIR/generator-*h5 | head -1)
modelExists=false
if [[ $latestmodelPath == *"h5" ]]; 
then
  modelExists=true
fi
echo "latest model is $latestModelPath"

if [[ $modelExists == true ]]; 
then
  aws s3 cp "$OUTDIR/config.yml" "s3://murf-models-dev/trained/$taskid/vocoder-config.yml"
  aws s3 cp "$latestModelPath" "s3://murf-models-dev/trained/$taskid/generator.h5"
fi
