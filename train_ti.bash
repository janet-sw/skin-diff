#!/bin/bash

SEED=0 # split seed; you can use different seeds by using skleans train_test_split with the Fitzpatrick17k's metadata 
MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
DATA_DIR="/data/finalfitz17k" # path to the images
DATA_SPLIT="/ssd/janet/skin-diff/data_splits/train_light_dark_seed_to_dark_seed=0.csv" # path to the data split (e.g., training data with dark-skinned type)
OUTPUT_DIR_BASE="output/ti/light_only_to_dark_demo/" # path to the output model


DISEASES=("basal_cell_carcinoma" "folliculitis" "nematode_infection" "neutrophilic_dermatoses" \
          "prurigo_nodularis" "psoriasis" "squamous_cell_carcinoma") # subset of conditions considered in the paper

for disease in "${DISEASES[@]}"; do
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/SEED=${SEED}/${disease:0:3}"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi
    
    token="<${disease:0:3}-class>" # better be something unique, use your imagination
    
    accelerate launch textual_inversion.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --train_data_dir="$DATA_DIR" \
        --fitz_split_csv="$DATA_SPLIT" \
        --learnable_property="object" \
        --placeholder_token="$token" \
        --initializer_token="skin" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --mixed_precision="fp16" \
        --max_train_steps=500 \
        --learning_rate=5.0e-04 \
        --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --push_to_hub \
        --output_dir="$OUTPUT_DIR"
done
