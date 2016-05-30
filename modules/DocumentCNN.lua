
local DocumentCNN = {}
function DocumentCNN.cnn(alphasize)
  local net = nn.Sequential()
  -- 1014 x alphasize
  net:add(nn.TemporalConvolution(alphasize, 256, 7))
  net:add(nn.Threshold())
  net:add(nn.Dropout(0.5))
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 336 x 256
  net:add(nn.TemporalConvolution(256, 128, 7))
  net:add(nn.Threshold())
  net:add(nn.Dropout(0.5))
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 110 x 128
  net:add(nn.TemporalConvolution(128, 64, 3))
  net:add(nn.Threshold())
  net:add(nn.Dropout(0.5))
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 36 x 64
  net:add(nn.Reshape(2304))
  -- 2304
  net:add(nn.Linear(2304, 256))
  net:add(nn.Dropout(0.7))
  net:add(nn.Linear(256, 256))
  return net
end

return DocumentCNN

