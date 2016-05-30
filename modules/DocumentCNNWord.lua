
local DocumentCNNWord = {}
function DocumentCNNWord.cnn(vocab_size, emb_dim, dropout)
  dropout = dropout or 0.0

  local net = nn.Sequential()
  -- 30 x alphasize
  net:add(nn.TemporalConvolution(vocab_size, 256, 3))
  net:add(nn.Threshold())
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 10 x 256
  net:add(nn.TemporalConvolution(256, 256, 3))
  net:add(nn.Threshold())
  -- 8 x 256
  net:add(nn.TemporalConvolution(256, 256, 3))
  net:add(nn.Threshold())
  -- 6 x 256
  net:add(nn.Reshape(1280))
  net:add(nn.Linear(1280, 1024))
  net:add(nn.Dropout(dropout))
  net:add(nn.Linear(1024, emb_dim))
  return net
end

return DocumentCNNWord

