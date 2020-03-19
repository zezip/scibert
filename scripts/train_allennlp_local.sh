# Run allennlp training locally, serializes to /path/to/needs_citation/models/scibert
# Run from scibert root as 'bash scripts/train_allennlp_local.sh <GPU #>'
#
# edit these variables before running script
DT=$(date '+%d-%m-%Y_%H-%M-%S')
PARENT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$PARENT_DIR")")")")"
SCIBERT_INPUTS="$PROJECT_ROOT"/"processed"/"data"/"scibert"
OUTPUT_DIR="$PROJECT_ROOT"/"models"/"scibert-""$DT"

DATASET='needs_citation'
TASK='text_classification'
with_finetuning='' #'_finetune'  # or '' for not fine tuning

export TRAIN_PATH=$SCIBERT_INPUTS/$DATASET/train.txt
export DEV_PATH=$SCIBERT_INPUTS/$DATASET/dev.txt
export TEST_PATH=$SCIBERT_INPUTS/$DATASET/test.txt

TRAIN_COUNT=$(cat $TRAIN_PATH | wc -l)
DEV_COUNT=$(cat $DEV_PATH | wc -l)
TEST_COUNT=$(cat $TEST_PATH | wc -l)
dataset_size=$(($TRAIN_COUNT + $DEV_COUNT + $TEST_COUNT))

# TODO: make certain these files are actually there
export BERT_VOCAB=$SCIBERT_INPUTS/vocab.txt
export BERT_WEIGHTS=$SCIBERT_INPUTS/weights.tar.gz

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false
export CUDA_DEVICE="$@"

export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=75
export LEARNING_RATE=0.001

if [ -d "$OUTPUT_DIR" ]; then rm -rf $OUTPUT_DIR; fi
python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s $OUTPUT_DIR