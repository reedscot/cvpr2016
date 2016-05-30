
local BidirectionalGRU = {}
function BidirectionalGRU.rnn(nstep, avg, emb_dim)
  if avg == nil then
    avg = 0
  end
  if emb_dim == nil then
    emb_dim = 256
  end
  local inputs = {}
  for n = 1,nstep do
    table.insert(inputs, nn.Identity()())
  end

  -- gates for update
  local i2h_update_fwd = {}
  local h2h_update_fwd = {}
  local i2h_update_rev = {}
  local h2h_update_rev = {}
  -- gates for reset
  local i2h_reset_fwd = {}
  local h2h_reset_fwd = {}
  local i2h_reset_rev = {}
  local h2h_reset_rev = {}
  -- candidate hidden state
  local i2h_fwd = {}
  local h2h_fwd = {}
  local i2h_rev = {}
  local h2h_rev = {}
  -- actual hidden state
  local hids_fwd = {}
  local hids_rev = {}

  -- forward GRU
  for i,v in ipairs(inputs) do
    i2h_update_fwd[i] = nn.Sequential()
    i2h_update_fwd[i]:add(nn.Linear(emb_dim,emb_dim))
    i2h_reset_fwd[i] = nn.Sequential()
    i2h_reset_fwd[i]:add(nn.Linear(emb_dim,emb_dim))
    i2h_fwd[i] = nn.Sequential()
    i2h_fwd[i]:add(nn.Linear(emb_dim,emb_dim))

    if i > 1 then
      i2h_update_fwd[i]:share(i2h_update_fwd[1],'weight', 'bias', 'gradWeight', 'gradBias')
      i2h_reset_fwd[i]:share(i2h_reset_fwd[1],'weight', 'bias', 'gradWeight', 'gradBias')
      i2h_fwd[i]:share(i2h_fwd[1], 'weight', 'bias', 'gradWeight', 'gradBias')

      h2h_update_fwd[i-1] = nn.Sequential()
      h2h_update_fwd[i-1]:add(nn.Linear(emb_dim,emb_dim))
      h2h_reset_fwd[i-1] = nn.Sequential()
      h2h_reset_fwd[i-1]:add(nn.Linear(emb_dim,emb_dim))
      h2h_fwd[i-1] = nn.Sequential()
      h2h_fwd[i-1]:add(nn.Linear(emb_dim,emb_dim))

      if i > 2 then
        h2h_update_fwd[i-1]:share(h2h_update_fwd[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
        h2h_reset_fwd[i-1]:share(h2h_reset_fwd[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
        h2h_fwd[i-1]:share(h2h_fwd[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
      end

      -- compute update and reset gates.
      local update = nn.Sigmoid()(nn.CAddTable()(
        {i2h_update_fwd[i](inputs[i]), h2h_update_fwd[i-1](hids_fwd[i-1])}))
      local reset = nn.Sigmoid()(nn.CAddTable()(
        {i2h_reset_fwd[i](inputs[i]), h2h_reset_fwd[i-1](hids_fwd[i-1])}))

      -- compute candidate hidden state.
      local gated_hidden = nn.CMulTable()({reset, hids_fwd[i-1]})
      local p1 = i2h_fwd[i](inputs[i])
      local p2 = h2h_fwd[i-1](gated_hidden)
      local hidden_cand = nn.Tanh()(nn.CAddTable()({p1, p2}))

      -- use gates to interpolate hidden state.
      local zh = nn.CMulTable()({update, hidden_cand})
      local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update)), hids_fwd[i-1]})
      hids_fwd[i] = nn.CAddTable()({zh, zhm1})
    else
      hids_fwd[i] = nn.Tanh()(i2h_fwd[i](inputs[i]))
    end
  end 

  for i,v in ipairs(inputs) do
    local ix = #inputs - i + 1
    i2h_update_rev[i] = nn.Sequential()
    i2h_update_rev[i]:add(nn.Linear(emb_dim,emb_dim))
    i2h_reset_rev[i] = nn.Sequential()
    i2h_reset_rev[i]:add(nn.Linear(emb_dim,emb_dim))
    i2h_rev[i] = nn.Sequential()
    i2h_rev[i]:add(nn.Linear(emb_dim,emb_dim))

    if i > 1 then
      i2h_update_rev[i]:share(i2h_update_rev[1],'weight', 'bias', 'gradWeight', 'gradBias')
      i2h_reset_rev[i]:share(i2h_reset_rev[1],'weight', 'bias', 'gradWeight', 'gradBias')
      i2h_rev[i]:share(i2h_rev[1], 'weight', 'bias', 'gradWeight', 'gradBias')

      h2h_update_rev[i-1] = nn.Sequential()
      h2h_update_rev[i-1]:add(nn.Linear(emb_dim,emb_dim))
      h2h_reset_rev[i-1] = nn.Sequential()
      h2h_reset_rev[i-1]:add(nn.Linear(emb_dim,emb_dim))
      h2h_rev[i-1] = nn.Sequential()
      h2h_rev[i-1]:add(nn.Linear(emb_dim,emb_dim))

      if i > 2 then
        h2h_update_rev[i-1]:share(h2h_update_rev[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
        h2h_reset_rev[i-1]:share(h2h_reset_rev[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
        h2h_rev[i-1]:share(h2h_rev[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
      end

      -- compute update and reset gates.
      local update = nn.Sigmoid()(nn.CAddTable()(
        {i2h_update_rev[i](inputs[ix]), h2h_update_rev[i-1](hids_rev[i-1])}))
      local reset = nn.Sigmoid()(nn.CAddTable()(
        {i2h_reset_rev[i](inputs[ix]), h2h_reset_rev[i-1](hids_rev[i-1])}))

      -- compute candidate hidden state.
      local gated_hidden = nn.CMulTable()({reset, hids_rev[i-1]})
      local p1 = i2h_rev[i](inputs[ix])
      local p2 = h2h_rev[i-1](gated_hidden)
      local hidden_cand = nn.Tanh()(nn.CAddTable()({p1, p2}))

      -- use gates to interpolate hidden state.
      local zh = nn.CMulTable()({update, hidden_cand})
      local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update)), hids_rev[i-1]})
      hids_rev[i] = nn.CAddTable()({zh, zhm1})
    else
      hids_rev[i] = nn.Tanh()(i2h_rev[i](inputs[ix]))
    end
  end

  local hid
  local projL = {}
  local projR = {}
  local combiner = {}
  for i = 1,nstep do
    projL[i] = nn.Sequential()
    projL[i]:add(nn.Linear(emb_dim,emb_dim))
    projR[i] = nn.Sequential()
    projR[i]:add(nn.Linear(emb_dim,emb_dim))
    combiner[i] = nn.Sequential()
    combiner[i]:add(nn.Linear(emb_dim,emb_dim))

    if i > 1 then
      projL[i]:share(projL[i-1], 'weight', 'bias', 'gradWeight', 'gradBias')
      projR[i]:share(projR[i-1], 'weight', 'bias', 'gradWeight', 'gradBias')
      combiner[i]:share(combiner[i-1], 'weight', 'bias', 'gradWeight', 'gradBias')
    end

    -- combine fwd and reverse directions.
    local hidL = projL[i](hids_fwd[i])
    local hidR = projR[i](hids_rev[i])
    local cur_hid = combiner[i](nn.ReLU()(nn.CAddTable()({hidL, hidR})))

    if i == 1 then
      hid = cur_hid
    else
      hid = nn.CAddTable()({hid, cur_hid})
    end
  end
  hid = nn.MulConstant(1./nstep)(hid)

  local outputs = {}
  table.insert(outputs, hid)
  return nn.gModule(inputs, outputs)
end

return BidirectionalGRU

