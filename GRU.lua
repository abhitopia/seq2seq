require 'nn'
require 'nngraph'
local function GRUStep(input_size, hidden_size)
  local xt = nn.Identity()()
  local htm1 = nn.Identity()()
  local xr = nn.Linear(input_size, hidden_size)(xt)

  local xu = nn.Linear(input_size, hidden_size)(xt)
  local hr = nn.Linear(hidden_size, hidden_size)(htm1)
  local hu = nn.Linear(hidden_size, hidden_size)(htm1)

  local r = nn.Sigmoid()(nn.CAddTable()({xr, hr}))
  local u = nn.Sigmoid()(nn.CAddTable()({xu,hu}))

  local rh = nn.CMulTable()({htm1, r})
  local from_h = nn.Linear(hidden_size, hidden_size)(rh)
  local from_x = nn.Linear(input_size, hidden_size)(xt)

  local htilde = nn.Tanh()(nn.CAddTable()({from_h, from_x}))

  local zh = nn.CMulTable()({u, htilde})
  local zhm1 = nn.CMulTable()({nn.AddConstant(1, false)(nn.MulConstant(-1, false)(u)), htm1})
  local out = nn.CAddTable()({zh, zhm1})

  return nn.gModule({xt,htm1},{out})
end

local GRU, parent = torch.class('nn.GRU','nn.Recurrent')
function GRU:__init(input_size, hidden_size, sequence_len, reverse, h0, learn_h0)
  local h0 = h0 or torch.zeros(hidden_size)
  local function GRUStepCall()
    return GRUStep(input_size, hidden_size)
  end
  parent.__init(self, GRUStepCall(), h0, sequence_len, reverse, learn_h0)
end


