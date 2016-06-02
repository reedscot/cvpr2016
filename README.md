###<a href="http://arxiv.org/abs/1605.05395">Learning Deep Representations of Fine-grained Visual Descriptions</a>
#####Scott Reed, Zeynep Akata, Bernt Schiele, Honglak Lee

<img src="images/description_embedding.jpg" width="500px" height="250px"/>

How to train a char-CNN-RNN model:

1. Download the [birds](https://drive.google.com/open?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE)
 and [flowers](https://drive.google.com/open?id=0B0ywwgffWnLLcms2WWJQRFNSWXM) data.
2. Modify the training script (e.g. `train_cub_hybrid.sh` for birds) to point to your data directory.
3. Run the training script: `./train_cub_hybrid.sh`

How to evaluate:

1. Train a model (see above).
2. Modify the eval bash script (e.g. `eval_cub_cls.sh` for birds) to point to your saved checkpoint.
3. Run the eval script: `./eval_cub_cls.sh`

