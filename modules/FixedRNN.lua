
local FixedRNN = {}
function FixedRNN.rnn(nstep, avg, emb_dim)
  emb_dim = emb_dim or 256
  local inputs = {}
  for n = 1,nstep do
    table.insert(inputs, nn.Identity()())
  end

  local i2h = {}
  local h2h = {}
  local relu = {}
  local sum
  for i,v in ipairs(inputs) do
    i2h[i] = nn.Sequential()
    i2h[i]:add(nn.Linear(emb_dim,emb_dim))
    if i > 1 then
      i2h[i]:share(i2h[1],'weight', 'bias', 'gradWeight', 'gradBias')

      h2h[i-1] = nn.Sequential()
      h2h[i-1]:add(nn.Linear(emb_dim,emb_dim))
      if i > 2 then
        h2h[i-1]:share(h2h[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
      end

      relu[i] = nn.ReLU()(nn.CAddTable()({i2h[i](inputs[i]),
                                          h2h[i-1](relu[i-1])}))
    else
      relu[i] = nn.ReLU()(i2h[i](inputs[i]))
    end
    if i == 1 then
        sum = relu[1]
    else
        sum = nn.CAddTable()({sum, relu[i]})
    end
  end 
  local hid = nn.MulConstant(1./#inputs)(sum)

  local outputs = {}
  table.insert(outputs, hid)
  return nn.gModule(inputs, outputs)
end

return FixedRNN

