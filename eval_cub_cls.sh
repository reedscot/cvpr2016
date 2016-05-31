
# char cnn rnn
echo "DS-SJE char-cnn-rnn"
th classify_sje_tcnn.lua \
  -data_dir data \
  -num_caption 10 \
  -ttype char \
  -model cv/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7 \
  | tee results/cub_cls_nc4_gru18_avg_c512_char.txt

