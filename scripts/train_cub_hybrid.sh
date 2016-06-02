
th train_sje_hybrid.lua \
  -data_dir data/cub \
  -image_dir images \
  -ids_file trainvalids.txt \
  -learning_rate 0.0007 \
  -symmetric 1 \
  -max_epochs 200 \
  -savefile sje_cub_c10_hybrid \
  -num_caption 10 \
  -gpuid 3 \
  -print_every 10

