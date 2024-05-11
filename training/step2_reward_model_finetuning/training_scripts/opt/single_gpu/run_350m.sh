#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT='./outputs/retrieval' 
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed  main.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --enable_tensorboard --data_split 0,1,0 --data_output_path /tmp/marco_data_files/\
   --data_path /aiarena/gpfs/data/MSMARCO --cache_path ./tmp/marco_data\
   --tensorboard_path $OUTPUT --per_device_train_batch_size 8 \
   --deepspeed --output_dir $OUTPUT \
&> $OUTPUT/training.log

# deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
#    --num_padding_at_beginning 1 --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
#    --enable_tensorboard \
#    --tensorboard_path $OUTPUT \
#    --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
