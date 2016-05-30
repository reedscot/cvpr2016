# cvpr2016
Learning Deep Representations of Fine-grained Visual Descriptions

How to train a model:
1. Download the [birds](https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE)
 and [flowers](https://drive.google.com/open?id=0B0ywwgffWnLLZHluQm5SRkdWTDQ) data.
2. Modify the training script (e.g. train_cub_hybrid.sh) to point to your data directory.
3. Run the training script: ./train_cub_hybrid.sh

How to evaluate:
1. Train a model (see above).
2. Modify the eval bash script (e.g. eval_cub_cls.sh) to point to your saved checkpoint.
3. Run the eval script: ./eval_cub_cls.sh

