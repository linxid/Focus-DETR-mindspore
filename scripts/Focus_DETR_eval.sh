coco_path=$1
checkpoint=$2
python main.py \
  --output_dir logs/Focus_DETR/R50-MS5-eval \
	-c config/Focus_DETR/Focus_DETR_5scale.py --coco_path $coco_path  \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
