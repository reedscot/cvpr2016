
local AttributeEncoder = {}
function AttributeEncoder.enc(input_size, hid_size, dropout)
  dropout = dropout or 0
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Attribute
  local x = inputs[1]
  local outputs = {}
  h = nn.Linear(input_size, hid_size)(x)
  table.insert(outputs, h)
  return nn.gModule(inputs, outputs)
end

return AttributeEncoder

