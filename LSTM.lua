require 'nn'
require 'nngraph'

-- Equations taken from Graves' paper: http://arxiv.org/pdf/1308.0850v5.pdf
-- but not constraining the weights from the cell to gate vectors to be diagonal
-- notation: 
-- xt: input at time t
-- stm1: state of the LSTM at time t-1, which is the concatenation of the htm1 and ctm1
-- htm1: actual output of the LSTM at time t-1
-- ctm1: memory cell of the LSTM at time t-1
local function LSTMStep(input_size, hidden_size, batched)
  xt = nn.Identity()()
  stm1 = nn.Identity()()
  htm1 = nn.BatchNarrow(1,1,hidden_size/2,1)(stm1)
  ctm1 = nn.BatchNarrow(1,(hidden_size/2)+1,hidden_size/2,1)(stm1)


  -- compute the last state (hidden+cell)'s contribution to i,f in one go
  if_stm1_contrib = nn.Linear(hidden_size, hidden_size/2*2)(stm1)
  -- compute the input's contribution to i,f,o,c in one go
  ifoc_xt_contrib = nn.Linear(input_size, hidden_size*2)(xt)

  -- compute i,f
  if_xt_contrib = nn.BatchNarrow(1,1,hidden_size, 1)(ifoc_xt_contrib)
  i_f = nn.Sigmoid()(nn.CAddTable()({if_xt_contrib, if_stm1_contrib}))
  i = nn.BatchNarrow(1,1,hidden_size/2,1)(i_f)
  f = nn.BatchNarrow(1,hidden_size/2 + 1, hidden_size/2, 1)(i_f)

  -- compute c
  c_xt_contrib = nn.BatchNarrow(1,hidden_size/2*3+1, hidden_size/2, 1)(ifoc_xt_contrib)
  c_htm1_contrib = nn.Linear(hidden_size/2, hidden_size/2)(htm1)
  c = nn.CAddTable()({nn.CMulTable()({f, ctm1}), nn.CMulTable()({i,nn.Tanh()(nn.CAddTable()({c_xt_contrib, c_htm1_contrib}))})})

  -- compute o
  o_htm1_contrib = nn.Linear(hidden_size/2, hidden_size/2)(htm1)
  o_c_contrib = nn.Linear(hidden_size/2, hidden_size/2)(c)
  o_xt_contrib = nn.BatchNarrow(1,hidden_size+1,hidden_size/2, 1)(ifoc_xt_contrib)
  o = nn.Sigmoid()(nn.CAddTable()({o_htm1_contrib, o_c_contrib, o_xt_contrib}))

  -- compute h
  h = nn.CMulTable()({o, nn.Tanh()(c)})
  -- join the two to create s
  s = nn.JoinTable(1,1)({h,c})
  return nn.gModule({xt,stm1},{s})
end

local LSTM, parent = torch.class('nn.LSTM','nn.Recurrent')
function LSTM:__init(input_size, hidden_size,sequence_len, reverse,h0, learn_h0)
  h0 = h0 or torch.zeros(hidden_size)
  local function LSTMStepCall()
    return LSTMStep(input_size, hidden_size)
  end
  parent.__init(self, LSTMStepCall(), h0, sequence_len, reverse, learn_h0)
end
