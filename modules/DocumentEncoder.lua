
local DocumentEncoder = {}
function DocumentEncoder.enc(alphasize, emb_dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- img
  local x = inputs[1]

  -- 1014 x alphasize
  local conv1 = nn.TemporalConvolution(alphasize, 256, 7)(x)
  local thresh1 = nn.Threshold()(conv1)
  local max1 = nn.TemporalMaxPooling(3, 3)(thresh1)
  -- 336 x 256
  local conv2 = nn.TemporalConvolution(256, 256, 7)(max1)
  local thresh2 = nn.Threshold()(conv2)
  local max2 = nn.TemporalMaxPooling(3, 3)(thresh2)
  -- 110 x 256
  local conv3 = nn.TemporalConvolution(256, 256, 3)(max2)
  local thresh3 = nn.Threshold()(conv3)
  -- 108 x 256
  local conv4 = nn.TemporalConvolution(256, 256, 3)(thresh3)
  local thresh4 = nn.Threshold()(conv4)
  -- 106 x 256
  local conv5 = nn.TemporalConvolution(256, 256, 3)(thresh4)
  local thresh5 = nn.Threshold()(conv5)
  -- 104 x 256
  local max3 = nn.TemporalMaxPooling(3, 3)(thresh5)
  -- 34 x 256
  local rs = nn.Reshape(8704)(max3)
  -- 8704
  local fc1 = nn.Linear(8704, 1024)(rs)
  local drop = nn.Dropout(0.5)(fc1)
  local fc2 = nn.Linear(1024, emb_dim)(drop)

  -- To classifier
  local proj = nn.Linear(1024, 14)(drop)
  local pred = nn.LogSoftMax()(proj)

  outputs = {}
  table.insert(outputs, pred)
  table.insert(outputs, fc2)
  return nn.gModule(inputs, outputs)
end

return DocumentEncoder

