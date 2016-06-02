
# char cnn rnn
echo "DS-SJE char-cnn-rnn classification"
mkdir -p results
th classify_sje_tcnn.lua \
  -data_dir data/cub \
  -num_caption 10 \
  -ttype char \
  -model cv/lm_sje_cub_c10_hybrid_0.00070_1_10_trainvalids.txt.t7 \
  | tee results/cub_cls_char.txt

