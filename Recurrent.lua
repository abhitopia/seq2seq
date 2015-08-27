require 'nn'

local Recurrent, parent = torch.class('nn.Recurrent','nn.Module')

-- step_module: nn-compliant module that accepts (xt, htm1) and returns ht
-- h0: initial hidden state
-- max_sequence_length: maximum length of sequence you'll see. if a longer sequence is seen, this will be increased, but that isn't efficient
-- learn_h0: whether or not to learn the h0 embedding (not tested)
-- reverse: whether or not to traverse backwards (but the output tensor is still in the correct order)
function Recurrent:__init(step_module, h0, max_sequence_length, reverse, learn_h0)
  parent.__init(self)

  self.masterStepModule = step_module
  self.bias = h0
  self.gradBias = torch.Tensor(h0:size())
  self.max_sequence_length = max_sequence_length
  self.learn_h0 = learn_h0 or false
  self.reverse = reverse or false
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

function Recurrent:updateOutput(input)
  -- allocate enough size for the output
  -- torch does this intelligently (i think)
  local osz = input:size()
  osz[input:dim()] = self.bias:size(1)
  self.output:resize(osz)

  local tdim = input:dim() - 1 -- works for 2d,3d

  -- make sure we have enough clones
  if #self.modules < input:size(tdim) then
    self.max_sequence_length = math.max(self.max_sequence_length, input:size(tdim))
    print("Warning: We found an input of size "..input:size(tdim).." but the maximum sequence length given was "..self.max_sequence_length..". Making "..self.max_sequence_length-#self.modules.." new clones...")
    self:makeClones()
  end

  -- ** Actual forward prop ** 
  local htm1
  -- NOTE: can't use expand because of GPU bug on zero-strided tensors. When this is fixed, should fix this too for efficiency
  -- this deals with batching
  if input:dim() == 3 then htm1 = torch.repeatTensor(self.bias, input:size(1), 1) else htm1 = self.bias end
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


function Recurrent:updateGradInput(input, gradOutput)
  self.gradInput:resize(input:size())

  local tdim = input:dim() - 1
  for t=input:size(tdim), 1, -1 do
    if self.reverse then t=input:size(tdim)-(t-1) end

    local htm1
    local next_t
    if self.reverse then next_t = t+1 else next_t = t-1 end
    if (next_t > input:size(tdim) or next_t < 1) then htm1=self.bias else htm1 = self.output:select(tdim, next_t) end

    -- ** actual backprop **
    local dinputt, dhtm1 = unpack(self.modules[t]:updateGradInput({input:select(tdim, t), htm1}, gradOutput:select(tdim, t)))

    if not (next_t > input:size(tdim) or next_t < 1) then 
        -- normal case
        gradOutput:select(tdim,next_t):add(dhtm1)
    else 
        -- end of backprop, go to initial hidden state
        -- in reality, this might be better put in accGradParameters, but that seems sort of pointless, since I'm not sure what that seperation actually does (since backward() just calls both anyways)
        expandpattern=torch.ByteTensor(dhtm1:dim()):zero();expandpattern[dhtm1:dim()]=1
        self.gradBias:copy(dexpand(dhtm1, expandpattern))
    end
    self.gradInput:select(tdim, t):copy(dinputt)
  end

  return self.gradInput
end


function Recurrent:accGradParameters(input, gradOutput, scale)
  local tdim = input:dim() - 1
  for t=input:size(tdim),1,-1 do
    if self.reverse then t = input:size(tdim) - (t-1) end

    local htm1
    local next_t
    if self.reverse then next_t = t+1 else next_t = t-1 end
    if (next_t > input:size(tdim) or next_t < 1) then htm1=self.bias else htm1 = self.output:select(tdim, next_t) end

    self.modules[t]:accGradParameters({input:select(tdim, t), htm1}, gradOutput:select(tdim, t), scale)
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
