require 'nn'
require 'Recurrent'

local BeamSearch = torch.class('BeamSearch')

function BeamSearch:__init(step, lookup, softmax, sos, eos)
  self.stepModule = step
  self.lookupTable = lookup
  self.softmaxLinear = softmax
  self.SOS = sos
  self.EOS = eos
end

-- the input now only expects an initial state, nothing else, so an input_size vector.
function BeamSearch:forward(input)
  print("warning: only beam search of beamlength 1 is currently implemented.")
  print("warning: only works on single inputs currently")
  local outputs = {}

  local htm1 = input
  local inputt = self:firstInput()
  local curr=1

  while true do
    local ht = self.stepModule:updateOutput({inputt, htm1})
    local outputt = self:sampleOutput(ht)
    table.insert(outputs, outputt)
    print(outputt)
    if self:endCondition(outputt) then break end
    inputt=self:nextInput(outputt)
    htm1=ht
    curr=curr+1
  end

  return torch.Tensor(outputs)


end

-- this should perform a softmax classification and take the argmax
function BeamSearch:sampleOutput(ht)
  local pred = self.softmaxLinear:forward(ht)
  local max, idx = pred:max(1)
  return idx[1]
end

-- this should check if outputt == <EOS>
function BeamSearch:endCondition(outputt)
  return outputt==self.EOS
end

-- this should return the word embedding of the output
function BeamSearch:nextInput(outputt)
  return self.lookupTable:forward(torch.ones(1)*outputt)[1]
end

function BeamSearch:firstInput()
  print(self.SOS)
  return self.lookupTable:forward(torch.ones(1)*self.SOS)[1]
end
