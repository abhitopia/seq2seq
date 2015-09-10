require 'nn'

local Recurrent, parent = torch.class('nn.Recurrent','nn.Module')

-- step_module: nn-compliant module that accepts (xt, htm1) and returns ht
-- h0: initial hidden state
-- max_sequence_length: maximum length of sequence you'll see. if a longer sequence is seen, this will be increased, but that isn't efficient
-- learn_h0: whether or not to learn the h0 embedding (not tested)
-- reverse: whether or not to traverse backwards (but the output tensor is still in the correct order)
function Recurrent:__init(step_module, h0, max_sequence_length, reverse,learn_h0)
  parent.__init(self)
  self.masterStepModule = step_module
  self.max_sequence_length = max_sequence_length
  self.reverse = reverse or false
  self.learn_h0 = learn_h0 or false
  self.bias=h0
  self.gradBias = torch.Tensor():resizeAs(self.bias)
  self.modules = {}
end

function Recurrent:type(type)
  parent.type(self, type)
  self.masterStepModule:type(type)
  if #self.modules > 0 then
      print("Warning: called type() but clones already existed, so this will break sharing. Generating new clones.")
      self:resetClones()
  end
end

function Recurrent:reset()
  self.masterStepModule:reset()
  if #self.modules > 0 then
      print ("Warning: called reset() but clones already existed. Not sure if this breaks anything, so right now we'll generate all new clones. This might not be necessary.")
      self:resetClones()
  end 
end

function Recurrent:resetClones()
    self.modules = {}
    self:makeClones()
    collectgarbage()
end

-- TODO: can just fill self.modules to max_sequence_length instead of this
function Recurrent:makeClones()
  local ntimes = self.max_sequence_length - #self.modules
  local clones = self.masterStepModule:sharedCloneManyTimes(ntimes, 'params','gradParams')
  for i=1,#clones do
    table.insert(self.modules, clones[i])
  end
  assert(#self.modules == self.max_sequence_length)
end

function Recurrent:updateOutput_(h0, input)
  -- allocate enough size for the output
  -- torch does this intelligently (i think)
  local osz = input:size()
  osz[input:dim()] = self.bias:size(1)
  self.output:resize(osz)

  local tdim = input:dim() - 1 -- works for 2d,3d

  -- make sure we have enough clones
  if #self.modules < input:size(tdim) then
    --print(input)
    self.max_sequence_length = math.max(self.max_sequence_length, input:size(tdim))
    print("Warning: We found an input of size "..input:size(tdim).." but the maximum sequence length given was "..self.max_sequence_length..". Making "..self.max_sequence_length-#self.modules.." new clones...")
    self:makeClones()
  end

  -- ** Actual forward prop ** 
  local htm1 = h0
  -- NOTE: can't use expand because of GPU bug on zero-strided tensors. When this is fixed, should fix this too for efficiency
  -- this deals with batching
  -- to copy:
  for t=1,input:size(tdim) do
    if self.reverse then t=input:size(tdim)-(t-1) end
    local ht = self.modules[t]:updateOutput({input:select(tdim, t), htm1})
    self.output:select(tdim, t):copy(ht)
    htm1 = ht
  end
  return self.output

end

-- un-expands the tensor. expandpattern is 0 where the dimension was expended, and 1 where the dimension was there before.
-- currently returns e.g (1,1,hid_size) matrix but it seems that copy works.
local function dexpand(tensor, expandpattern)
  assert(tensor:dim() == expandpattern:size(1))
  local curr =tensor
  for i=1,expandpattern:size(1) do
    if expandpattern[i]==0 then curr = curr:sum(i) end
  end
  return curr
end


function Recurrent:updateGradInput_(h0, input, gradOutput)
  self.gradInputTensor:resize(input:size())
  -- there is some lack of clarity here; h0 passed in will always already be expanded, but the gradient of the original h0
  -- should not be expanded, so that the :copy(dexpand(...)) works. maybe refactor some stuf..
  self.gradh0:resize(h0:size())

  local tdim = input:dim() - 1
  for t=input:size(tdim), 1, -1 do
    if self.reverse then t=input:size(tdim)-(t-1) end

    local htm1
    local next_t
    if self.reverse then next_t = t+1 else next_t = t-1 end
    if (next_t > input:size(tdim) or next_t < 1) then htm1=h0 else htm1 = self.output:select(tdim, next_t) end

    -- ** actual backprop **
    local dinputt, dhtm1 = unpack(self.modules[t]:updateGradInput({input:select(tdim, t), htm1}, gradOutput:select(tdim, t)))

    if not (next_t > input:size(tdim) or next_t < 1) then 
        -- normal case
        gradOutput:select(tdim,next_t):add(dhtm1)
    else
        -- end of backprop, go to initial hidden state
        -- in reality, this might be better put in accGradParameters, but that seems sort of pointless, since I'm not sure what that seperation actually does (since backward() just calls both anyways)
        self.gradh0:copy(dhtm1)
    end
    self.gradInputTensor:select(tdim, t):copy(dinputt)
  end

  return self.gradh0, self.gradInputTensor
end
function istable(x)
  return type(x) == 'table' and not torch.typename(x)
end

function Recurrent:updateOutput(input)
  if istable(input) then
    self:setupTableInput()
    return self:updateOutput_(input[1], input[2])
  else
    self:setupTensorInput()
    local h0
    if input:dim() == 3 then h0 = torch.repeatTensor(self.bias, input:size(1), 1) else h0=self.bias end
    return self:updateOutput_(h0, input)
  end
end

function Recurrent:setupTableInput()
  if not self.set then
    self.gradInputTensor = self.gradInput.new()
    self.gradh0 = self.gradInput.new()
    self.gradInput = {self.gradh0, self.gradInputTensor}
    self.set=true
    self.tab='table'
  else
    if not istable(self.gradInput) then
      error("didnt recieve table input but expected to")
    end
  end
end

function Recurrent:setupTensorInput()
  if not self.set then
    self.gradInputTensor=self.gradInput
    -- this is a bit of a hack
    self.gradh0=self.bias.new()
    self.set=true
    self.tab='tensor'
  else
    if istable(self.gradInput) then
      error("grad input was a table at some point")
    end
  end
end

function Recurrent:updateGradInput(input, gradOutput)
  if type(input)=='table' then
    local one, two = self:updateGradInput_(input[1], input[2], gradOutput)
    return self.gradInput
  else
    if input:dim() == 3 then h0 = torch.repeatTensor(self.bias, input:size(1), 1) else h0=self.bias end
    local gradh0, gradInput = self:updateGradInput_(h0, input, gradOutput)
    expandpattern=torch.ByteTensor(gradh0:dim()):zero();expandpattern[gradh0:dim()]=1
    self.gradBias:copy(dexpand(gradh0,expandpattern))
    return self.gradInput
  end
end


function Recurrent:accGradParameters_(h0, input, gradOutput, scale)
  local tdim = input:dim() - 1
  for t=input:size(tdim),1,-1 do
    if self.reverse then t = input:size(tdim) - (t-1) end

    local htm1
    local next_t
    if self.reverse then next_t = t+1 else next_t = t-1 end
    if (next_t > input:size(tdim) or next_t < 1) then htm1=h0 else htm1 = self.output:select(tdim, next_t) end

    self.modules[t]:accGradParameters({input:select(tdim, t), htm1}, gradOutput:select(tdim, t), scale)
  end
end

function Recurrent:accGradParameters(input, gradOutput, scale)
if type(input)=='table' then
    self:accGradParameters_(input[1], input[2], gradOutput, scale)
  else
    self:accGradParameters_(self.bias, input, gradOutput, scale)
    -- this is the same as gradInput
  end
end



function Recurrent:apply(func)
    func(self.masterStepModule)
    for i=1,#self.modules do
        func(self.modules[i])
    end
end

function Recurrent:training()
    function modTrain(mod)
        mod:training()
    end
    self:apply(modTrain)
end

function Recurrent:evaluate()
    function modEval(mod)
        mod:evaluate()
    end
    self:apply(modEval)
end

function Recurrent:parameters()
  p, gp = self.masterStepModule:parameters()
  return p, gp
end

function Recurrent:share(other, ...)
    args = {...}
    function modShare(mod)
        mod:share(other.masterStepModule, unpack(args))
    end
    self:apply(modShare)
end

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



-- Equations taken from Graves' paper: http://arxiv.org/pdf/1308.0850v5.pdf, but not constraining the weights from the cell to gate vectors to be diagonal
-- This formulation is often referred to as "LSTM with peephole connections", since the previous state ctm1 can influence the gates.
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

