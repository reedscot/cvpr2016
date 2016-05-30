
local HybridCNN = {}
function HybridCNN.cnn(vocab_size, emb_dim, dropout, length)
  dropout = dropout or 0.0
  length = length or 8

  local inputs = {}
  table.insert(inputs, nn.Identity()())

  local net = nn.Sequential()
  -- 30 x alphasize
  net:add(nn.TemporalConvolution(vocab_size, 512, 3))
  net:add(nn.Threshold())
  -- 28 x alphasize
  net:add(nn.TemporalConvolution(512, 512, 2))
  net:add(nn.Threshold())
  -- 27 x alphasize
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 9 x alphasize
  net:add(nn.TemporalConvolution(512, 512, 2))
  net:add(nn.Threshold())
  -- 8 x 256
  local h1 = nn.SplitTable(2)(net(inputs[1]))
  local r2 = FixedRNN.rnn(length, 1, 512)(h1)
  out = nn.Linear(512, emb_dim)(nn.Dropout(dropout)(r2))
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return HybridCNN

