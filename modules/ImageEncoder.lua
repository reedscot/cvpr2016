
local ImageEncoder = {}
function ImageEncoder.enc(input_size, hid_size, noop)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Image
  local x = inputs[1]
  local outputs = {}

  if noop == 1 then
      table.insert(outputs, x)
  else
      h = nn.Linear(input_size, hid_size)(x)
      table.insert(outputs, h)
  end
  return nn.gModule(inputs, outputs)
end

return ImageEncoder

