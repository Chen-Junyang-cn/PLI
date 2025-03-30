# Overview

This is the **official repository** of the [**ICME 2025 paper**](https://arxiv.org/abs/2311.07622) "*Pretrain like Your Inference: Masked Tuning Improves Zero-Shot Composed Image Retrieval*"

## Abstract
Zero-shot composed image retrieval (ZS-CIR), which takes a textual modification and a reference image as a query to retrieve a target image without triplet labeling, has gained more and more attention. Current ZS-CIR research relies mainly on the generalization ability of pre-trained vision language models (VLMs). However, the pre-trained VLMs and CIR tasks have substantial discrepancies, where the VLMs focus on learning the similarities but CIR aims to learn the modifications of the image guided by text. In this paper, we introduce a novel unlabeled and pre-trained masked tuning approach, which reduces the gap between the pre-trained VLMs and the downstream CIR task. First, to reduce the gap, we reformulate the contrastive learning of the VLMs as the CIR task, where we randomly mask input image patches to generate $\langle$masked image, text, image$\rangle$ triplet from an image-text pair. Then, we propose a simple but novel pre-trained masked tuning method, which uses the text and the masked image to learn the modifications of the original image. With such a simple design, the proposed masked tuning can learn to better capture fine-grained text-guided modifications. Extensive experimental results demonstrate the significant superiority of our approach over the baseline models on four ZS-CIR datasets, including FashionIQ, CIRR, CIRCO, and GeneCIS.

# Usage

## Dataset
<details>
  <summary>The downloaded dataset should look like this (click to expand)</summary>
  
  ```
  data
  └─── cirr
      ├─── captions
      │        cap.VER.test1.json
      │        cap.VER.train.json
      │        cap.VER.val.json
      ├─── captions_ext
      │        cap.ext.VER.test1.json
      │        cap.ext.VER.train.json
      │        cap.ext.VER.val.json
      ├─── image_splits
      │        split.VER.test1.json
      │        split.VER.train.json
      │        split.VER.val.json
      ├─── img_raw  
      │    ├── train
      │    │    ├── 0 # sub-level folder structure inherited from NLVR2 (carries no special meaning in CIRR)
      │    │    │    <IMG0_ID>.png
      │    │    │    <IMG0_ID>.png
      │    │    │         ...
      │    │    ├── 1
      │    │    │    <IMG0_ID>.png
      │    │    │    <IMG0_ID>.png
      │    │    │         ...
      │    │    ├── 2
      │    │    │    <IMG0_ID>.png
      │    │    │    <IMG0_ID>.png
      │    │    └──       ...
      │    ├── dev         
      │    │      <IMG0_ID>.png
      │    │      <IMG1_ID>.png
      │    │           ...
      │    └── test1       
      │           <IMG0_ID>.png
      │           <IMG1_ID>.png
      │                ...
      ├─── img_feat_res152 
      │        <Same subfolder structure as above>
      └─── img_feat_frcnn         
               <Same subfolder structure as above>
  ```
</details>

## Commands
We provide some arguments to choose the model.

<details>
  <summary>Baseline Model</summary>


- Baseline CLIP, specifying clip text and image model:
    ```bash
    CUDA_VISIBLE_DEVICES=3 python3 main.py --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=clip_baseline_model4zscir --device_idx=$CUDA_VISIBLE_DEVICES
    ```
  Evaluate in Genecis: --pretrain_dataset="MSCOCO" is unused, checkpoint_path="None" means no pretrain
  ```bash
  CUDA_VISIBLE_DEVICES=7 python3 genecis_eval.py --dataset=all --experiment_description=test_genecis_CLIP_B_Text_plus_Image --checkpoint_path="None" --img_weight=1 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --compositor='None' --pretrain_dataset="MSCOCO" --tuning_module='freeze' --batch_size=16 --epoch=10 --lr=1e-6 --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=6 python3 genecis_eval.py --dataset=all --checkpoint_path="None" --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_genecis_CLIP_L_Image_Only --compositor='None' --pretrain_dataset="MSCOCO" --tuning_module='freeze' --batch_size=16 --epoch=10 --lr=1e-6 --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  Evaluate in CIRCO
    ```bash
    CUDA_VISIBLE_DEVICES=3 python3 circo_test_submission.py --img_weight=1 --config_path="config/fashionIQ_CLIP_config.json" --checkpoint_path="None" --pretrain_dataset="ImageNet" --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=clip_L_model4zscir_circo --device_idx=$CUDA_VISIBLE_DEVICES --dataset=circo
    ```
- Baseline BLIP, change blip_model_name
    ```bash
    CUDA_VISIBLE_DEVICES=6 python3 main.py --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=blip_baseline_w_img_add_text --device_idx=$CUDA_VISIBLE_DEVICES
    ```
    CIRR test submission: BLIP baseline
    ```bash
    CUDA_VISIBLE_DEVICES=2 python3 cirr_test_submission.py --dataset=cirr --pretrain_dataset="ImageNet" --checkpoint_path='None' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=cirr_blip_baseline_w_img_add_text --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=6 python3 cirr_val_submission.py --img_weight=1 --dataset=cirr --pretrain_dataset="ImageNet" --checkpoint_path='None' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=cirr_blip_B_val --device_idx=$CUDA_VISIBLE_DEVICES
    ```
    Evaluate in CIRCO
    ```bash
    CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --img_weight=1 --dataset=circo --checkpoint_path="None" --pretrain_dataset="ImageNet" --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=circo_blip_baseline_w_img_add_text --device_idx=$CUDA_VISIBLE_DEVICES
    ```
</details>
  
### Training Models

**We provide all commands to reproduce the results in our paper.**

- **BLIP** model, pretrained on **ImageNet**:
    export CUDA_VISIBLE_DEVICES=3
    ```bash
      nohup `python3 main.py  --compositor="projector"  --batch_size=32 --epoch=50 --lr=1e-5 --pretrain_dataset="FashionIQcaptions" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=pretrain_blip_baseline_w_imagenet --device_idx=3` &  
    ```
    evaluate in fashionIQ
    ```bash
    CUDA_VISIBLE_DEVICES=1 python3 main.py --experiment_description=mask_pretrain_blip_baseline_w_ImageNet_10K --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json"  --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=3 python3 main.py --mask_ratio=0.5  --experiment_description=FixCFBug_mask_pretrain_blip_ImageNet_LossEncNoNorm_EvaNormBeforeAdd_Eva0.5*ImgFea --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

    CUDA_VISIBLE_DEVICES=5 python3 main.py --mask_ratio=0.5 --img_weight=0.5  --experiment_description=FixCFBug_mask_pretrain_blip_ImageNet_LossEncNoNorm_EvaNormBeforeAdd --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    ```
    CIRR test submission: pretrain on ImageNet
    ```bash
    CUDA_VISIBLE_DEVICES=2 python3 cirr_test_submission.py --dataset=cirr --pretrain_dataset="ImageNet" --checkpoint_path='experiments/FixCFBug_cirr_mask_pretrain_blip_ImageNet_2023-10-29_0/best.pth' --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=test_cirr_mask_pretrain_blip_ImageNet_mr5 --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.5 --img_weight=0.5 --dataset=cirr  --experiment_description=FixCFBug_cirr_mask_pretrain_blip_ImageNet --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=6 python3 cirr_val_submission.py --mask_ratio=0.5 --img_weight=0.5 --dataset=cirr --pretrain_dataset="ImageNet" --checkpoint_path='experiments/FixCFBug_cirr_mask_pretrain_blip_ImageNet_2023-10-29_0/best.pth' --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=val_cirr_mask_pretrain_blip_ImageNet_mr5 --device_idx=$CUDA_VISIBLE_DEVICES
    ```
    Evaluate in CIRCO
    ```bash
    CUDA_VISIBLE_DEVICES=2 python3 main.py --mask_ratio=0.5 --img_weight=0.5 --dataset=circo --warmup_iters=3 --decay_step=3 --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=mask_pretrain_blip_baseline_w_ImageNet_circo --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=2 python3 circo_test_submission.py --mask_ratio=0.5 --img_weight=0.5 --dataset=circo --checkpoint_path='experiments/FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner_2024-03-02_1/best.pth' --compositor='None' --tuning_module='both' --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=test_sub_mask_pretrain_blip_w_ImageNet --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=2 python3 circo_val_submission.py --mask_ratio=0.5 --img_weight=0.5 --dataset=circo --checkpoint_path='experiments/mask_pretrain_blip_baseline_w_ImageNet_2023-10-28_1/best.pth' --compositor='None' --tuning_module='both' --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --experiment_description=val_sub_mask_pretrain_blip_w_ImageNet_MR50 --device_idx=$CUDA_VISIBLE_DEVICES
    ```
    Evaluate in Genecis
    ```bash
    CUDA_VISIBLE_DEVICES=4 python3 genecis_eval.py --checkpoint_path="None" --dataset=all --mask_ratio=0.5 --img_weight=1 --experiment_description=test_genecis_FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    ```
  
- **CLIP** model, mask pretrained on **ImageNet**:
  FashionIQ: --mask_ratio=0.25
  ```bash
  CUDA_VISIBLE_DEVICES=3,4,5,6 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --batch_size=64 --experiment_description=FixCFBug_mask_pretrain_clip_L_ImageNet --compositor='None' --tuning_module='both' --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=5 python3 main.py --mask_ratio=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixComposedFeaturesBug_mask_pretrain_clip_ImageNet_Nonorm_NormBeforeAdd --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=4 python3 main.py --mask_ratio=0.5 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCFBug_mask_pretrain_clip_ImageNet_LossEncNoNorm_EvaNormBeforeAdd_Eva0.5*ImgFea --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCFBug_mask_pretrain_clip_ImageNet_EvaNormBeforeAdd --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  **Tagetpad**
  Train:
  ```bash
  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --dataset="fashionIQ" --random_seed=42 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_L_ImageLlava_NoCombiner_Targetpad --compositor='None' --tuning_module='both' --batch_size=64 --epoch=5 --lr=1e-6 --pretrain_dataset="imageva" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator"  --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --dataset="fashionIQ" --random_seed=4242 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCFBug_mask_pretrain_clip_ImageLlava_NoCombiner_Targetpad --compositor='None' --tuning_module='both' --batch_size=128 --epoch=5 --lr=1e-6 --pretrain_dataset="imageva" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator"  --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  Evaluate:
  ```bash
  CUDA_VISIBLE_DEVICES=2 python3 cirr_test_submission.py --mask_ratio=0.75 --img_weight=0.25 --dataset=cirr --pretrain_dataset="imageva" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_ImageLlava_NoCombiner_Targetpad_2024-07-23_0/epoch_1.pth' --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=cirr_test_submission_clip_baseline_w_ImageNet --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=0 python3 cirr_test_submission.py --mask_ratio=0.75 --img_weight=0.25 --dataset=cirr --pretrain_dataset="imageva" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_L_ImageLlava_NoCombiner_Targetpad_2024-07-23_0/epoch_2.pth' --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=cirr_test_submission_clip_baseline_w_ImageNet --device_idx=$CUDA_VISIBLE_DEVICES


  CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --mask_ratio=0.75 --img_weight=0.25 --dataset=circo --pretrain_dataset="llava" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_ImageLlava_NoCombiner_Targetpad_2024-07-23_0/epoch_0.pth' --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=circo_test_submission_clip__w_ImageNet --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --mask_ratio=0.75 --img_weight=0.25 --dataset=circo --pretrain_dataset="imageva" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_L_ImageLlava_NoCombiner_Targetpad_2024-07-23_0/epoch_2.pth' --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=circo_test_submission_clip_L_w_Llava --device_idx=$CUDA_VISIBLE_DEVICES


  CUDA_VISIBLE_DEVICES=1 python3 genecis_eval.py --mask_ratio=0.75 --img_weight=0.25 --dataset=all --pretrain_dataset="llava" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_Llava_NoCombiner_2024-07-21_0/epoch_2.pth' --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=genecis_test_submission_clip__w_llava_target --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=2 python3 genecis_eval.py --mask_ratio=0.75 --img_weight=0.25 --dataset=all --pretrain_dataset="llava" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_L_ImageLlava_NoCombiner_Targetpad_2024-07-23_0/epoch_2.pth' --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=genecis_test_submission_clip_L_w_llava_target --device_idx=$CUDA_VISIBLE_DEVICES
  ```


  CIRR:
  ```bash
  CUDA_VISIBLE_DEVICES=6 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --dataset=cirr --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCF_cirrBug_mask_pretrain_clip_B_w_ImageNet --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=1 python3 cirr_test_submission.py --mask_ratio=0.75 --img_weight=0.25 --dataset=cirr --pretrain_dataset="ImageNet" --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_ImageNet_EvaNormBeforeAdd_2024-04-02_1/epoch_2.pth' --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --config_path="config/fashionIQ_CLIP_config.json" --experiment_description=cirr_test_submission_clip_baseline_w_ImageNet --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=1 python3 cirr_val_submission.py --mask_ratio=0.75 --img_weight=0.25 --checkpoint_path='experiments/FixCFBug_mask_pretrain_clip_ImageNet_EvaNormBeforeAdd_2024-04-02_1/epoch_1.pth' --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --batch_size=64 --dataset=cirr --experiment_description=FixCFBug_cirr_mask_pretrain_clip_B_val --compositor='None' --tuning_module='both' --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES  

  CUDA_VISIBLE_DEVICES=6 python3 main.py --mask_ratio=0.5 --img_weight=0.5 --experiment_description=FixCF_cirrBug_mask_pretrain_clip_ImageNet_LossEncNoNorm_EvaNormBeforeAdd_Eva0.5*ImgFea --dataset=cirr --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32"  --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=3,4,5,6 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --batch_size=64 --dataset=cirr --experiment_description=FixCFBug_cirr_mask_pretrain_clip_L_ImageNet --compositor='None' --tuning_module='both' --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES  
  CUDA_VISIBLE_DEVICES=3,4,5,6 python3 cirr_test_submission.py --experiment_description=FixCFBug_test_cirr_mask_pretrain_clip_L_ImageNet --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_Flickr30K_NoCombiner_L_2024-04-02_1/epoch_1.pth" --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --batch_size=64 --dataset=cirr  --compositor='None' --tuning_module='both' --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES  

  CUDA_VISIBLE_DEVICES=1 python3 cirr_test_submission.py --mask_ratio=0.5 --img_weight=0.5 --experiment_description=FixCF_cirrBug_test_submission_mask_pretrain_clip_w_ImageNet_IW0.5_MR0.5 --dataset=cirr --pretrain_dataset="ImageNet" --checkpoint_path='experiments/FixCF_cirrBug_mask_pretrain_clip_ImageNet_LossEncNoNorm_EvaNormBeforeAdd_Eva0.5*ImgFea_2023-10-25_1/best.pth' --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --config_path="config/fashionIQ_CLIP_config.json"  --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  Evaluate in CIRCO:
  ```bash
  CUDA_VISIBLE_DEVICES=2 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --dataset=circo --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=circo_mask_pretrain_clip_ImageNet_NoNorm --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --mask_ratio=0.75 --img_weight=0.25 --decay_step=3 --checkpoint_path='experiments/circo_mask_pretrain_clip_ImageNet_NoNorm_2023-10-28_0/best.pth' --dataset=circo --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=test_sub_circo_mask_pretrain_clip_ImageNet_NoNorm --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_ImageNet_EvaNormBeforeAdd_2024-04-02_1/epoch_1.pth" --mask_ratio=0.75 --img_weight=0.25 --dataset=circo --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=test_circo_mask_pretrain_clip_ImageNet_NoNorm_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --weight_decay=5e-2 --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=3,4,5,6 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --batch_size=64 --dataset=circo --experiment_description=FixCFBug_circo_mask_pretrain_clip_L_ImageNet_mr75_iw25 --compositor='None' --tuning_module='both' --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES  
  CUDA_VISIBLE_DEVICES=3,4,5,6 python3 circo_test_submission.py --experiment_description=FixCFBug_test_circo_mask_pretrain_clip_L_ImageNet --checkpoint_path="experiments/FixCFBug_circo_mask_pretrain_clip_L_ImageNet_mr75_iw25_2023-10-29_0/best.pth" --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --batch_size=64 --dataset=circo  --compositor='None' --tuning_module='both' --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  
- Combiner Network, freeze image and text encoder (without img_weight)
  ```bash
    CUDA_VISIBLE_DEVICES=4 python3 main.py --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCFBug_mask_pretrain_clip_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=2e-5 --weight_decay=1e-2 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset=cirr --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=cirr_FixCFBug_mask_pretrain_clip_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=2e-5 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset=circo --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=circo_FixCFBug_mask_pretrain_clip_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=2e-5 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    
    CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_L_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=5 --lr=1e-5 --weight_decay=1e-2 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=2 python3 main.py --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_ImageNet_Combiner_Norm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=5 --lr=2e-5 --weight_decay=1e-2 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  
    CUDA_VISIBLE_DEVICES=6 python3 main.py --mask_ratio=0.75 --img_weight=1  --experiment_description=FixCFBug_mask_pretrain_BLIP_ImageNet_Combiner --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=6 --lr=1e-6 --weight_decay=2e-5 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  ```
  test
  ```bash
    CUDA_VISIBLE_DEVICES=0 python3 cirr_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_L_Flickr30K_Combiner_NoNorm_2024-04-02_3/epoch_0.pth" --dataset=cirr --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=test_cirr_FixCFBug_mask_pretrain_clip_ImageNet_EvaNormBeforeAdd --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_L_Flickr30K_Combiner_NoNorm_2024-04-02_3/epoch_2.pth" --dataset=circo --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_circo_FixCFBug_mask_pretrain_clip_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    
    CUDA_VISIBLE_DEVICES=5 python3 genecis_eval.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_L_Flickr30K_Combiner_NoNorm_2024-04-02_3/epoch_0.pth" --dataset=all --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_genecis_FixCFBug_mask_pretrain_clip_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=16 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  
    CUDA_VISIBLE_DEVICES=0 python3 cirr_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_L_Flickr30K_Combiner_NoNorm_2024-04-02_3/epoch_1.pth" --dataset=cirr --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_cirr_FixCFBug_mask_pretrain_clip_L_ImageNet_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="ImageNet" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  ```

- CLIP model, mask pretrained on MSCOCO, without Combiner
  FashionIQ:
  ```bash
  CUDA_VISIBLE_DEVICES=3 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCFBug_mask_pretrain_clip_Flickr30K_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0,5,6,7 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_Flickr30K_NoCombiner_L --compositor='None' --tuning_module='both' --batch_size=64 --epoch=5 --lr=1e-6 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=6 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --image_transforms="target" --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=1,2,4,7 python3 main.py --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner_L --compositor='None' --tuning_module='both' --batch_size=64 --epoch=5 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  BLIP:
  ```bash
        CUDA_VISIBLE_DEVICES=5 python3 main.py --mask_ratio=0.5 --img_weight=0.5  --experiment_description=FixCFBug_mask_pretrain_BLIP_Flickr30k_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
      CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.5 --img_weight=0.5  --experiment_description=FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=5e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  **BLIP + Combiner**:
  ```bash
    CUDA_VISIBLE_DEVICES=4 python3 main.py --mask_ratio=0.5 --img_weight=1  --experiment_description=FixCFBug_mask_pretrain_BLIP_MSCOCO_Combiner --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=6 --lr=1e-6 --weight_decay=1e-2 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    ```
  ** CLIP + PHI **:
    img_weight unused
    ```bash
    CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14"  --experiment_description=FixCFBug_mask_pretrain_PHI_MSCOCO --compositor='phi' --tuning_module='freeze' --batch_size=128 --epoch=10 --lr=1e-4 --weight_decay=1e-2 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    ```
  **CLIP + Combiner**:
  ```bash
        CUDA_VISIBLE_DEVICES=1 python3 main.py --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_L_Flickr30K_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=6 --lr=1e-5 --weight_decay=1e-2 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

      CUDA_VISIBLE_DEVICES=2 python3 main.py --mask_ratio=0.75 --img_weight=1 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=FixCFBug_mask_pretrain_clip_L_MSCOCO_Combiner_NoNorm --compositor='Combiner' --tuning_module='freeze' --batch_size=64 --epoch=6 --lr=1e-5 --weight_decay=1e-2 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  CIRR:
  ```bash
    CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset=cirr --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=cirr_mask_pretrain_clip_Flickr30k_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset=cirr --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=cirr_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=1,2,4,7 python3 main.py --dataset=cirr --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=cirr_mask_pretrain_clip_MSCOCO_NoCombiner_L --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  BLIP:
  ```bash
      CUDA_VISIBLE_DEVICES=7 python3 main.py --dataset=cirr --mask_ratio=0.5 --img_weight=0.5  --experiment_description=FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    ```
  
  CIRCO
  ```bash
    CUDA_VISIBLE_DEVICES=1 python3 main.py  --mask_ratio=0.75 --img_weight=0.25 --dataset=circo --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=circo_mask_pretrain_clip_Flickr30k_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=3 python3 main.py  --mask_ratio=0.5 --img_weight=0.5 --dataset=circo --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=circo_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  ```
  BLIP:
  ```bash
        CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset=circo --mask_ratio=0.5 --img_weight=0.5  --experiment_description=FixCFBug_mask_pretrain_BLIP_Flickr30k_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="Flickr30k" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
      CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset=circo --mask_ratio=0.75 --img_weight=0.25  --experiment_description=FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    ```
  CLIP model test:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python3 cirr_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_Flickr30K_NoCombiner_2024-04-02_1/epoch_1.pth" --dataset=cirr --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=test_cirr_FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0 python3 cirr_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner_L_2024-03-09_1/epoch_2.pth" --dataset=cirr --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_cirr_FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=16 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=1 python3 circo_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_Flickr30K_NoCombiner_2024-04-02_1/epoch_1.pth" --mask_ratio=0.75 --img_weight=0.25 --dataset=circo --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=test_circo_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --weight_decay=5e-2 --device_idx=$CUDA_VISIBLE_DEVICES
      CUDA_VISIBLE_DEVICES=3 python3 circo_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner_L_2024-03-09_1/epoch_2.pth" --dataset=circo --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_cirr_FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=16 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=5 python3 genecis_eval.py --dataset=all --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_Flickr30K_NoCombiner_2024-04-02_1/epoch_0.pth"  --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-B/32" --clip_text_model="ViT-B/32" --experiment_description=test_genecis_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='freeze' --batch_size=64 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=5 python3 genecis_eval.py --dataset=all --checkpoint_path="experiments/FixCFBug_mask_pretrain_clip_MSCOCO_NoCombiner_L_2024-03-09_1/epoch_2.pth"  --mask_ratio=0.75 --img_weight=0.25 --clip_image_model="ViT-L/14" --clip_text_model="ViT-L/14" --experiment_description=test_genecis_mask_pretrain_clip_MSCOCO_NoCombiner --compositor='None' --tuning_module='freeze' --batch_size=16 --epoch=10 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='clip_image_encoder' --text_encoders='clip_text_encoder' --config_path="config/fashionIQ_CLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=3 python3 evaluate.py --dataset=change_object
  ```
  BLIP model test:
  ```bash
  CUDA_VISIBLE_DEVICES=2 python3 cirr_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner_2024-03-02_0/best.pth" --dataset=cirr --mask_ratio=0.5 --img_weight=0.5 --experiment_description=test_cirr_FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  CUDA_VISIBLE_DEVICES=2 python3 circo_test_submission.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner_2024-03-17_2/epoch_2.pth" --dataset=circo --mask_ratio=0.5 --img_weight=0.5 --experiment_description=test_circo_FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES
  
  CUDA_VISIBLE_DEVICES=5 python3 genecis_eval.py --checkpoint_path="experiments/FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner_2024-03-11_0/epoch_0.pth" --dataset=all --mask_ratio=0.5 --img_weight=0.5 --experiment_description=test_genecis_FixCFBug_mask_pretrain_BLIP_MSCOCO_NoCombiner --compositor='None' --tuning_module='both' --batch_size=64 --epoch=6 --lr=1e-6 --pretrain_dataset="MSCOCO" --trainer="pretrainer" --evaluator_code="pretrainer_evaluator" --image_encoders='blip_image_encoder' --text_encoders='blip_text_encoder' --blip_model_name="blip_feature_extractor" --blip_model_type="base" --config_path="config/fashionIQ_BLIP_config.json" --device_idx=$CUDA_VISIBLE_DEVICES

  ```