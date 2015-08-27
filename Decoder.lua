require 'nn'
require 'Recurrent'
-- LMDecoder is a recurrent model at train time, but a beam-searcher at test time.

local LMDecoder, parent = torch.class('nn.LMDecoder','nn.Recurrent')

-- Beam search
-- the input now only expects an initial state, nothing else, so an input_size vector.
function LMDecoder:beamSearch(input)
  print("warning: only beam search of beamlength 1 is currently implemented.")
  local outputs = {}

  -- initialize h0
  local hsz = input:size()
  hsz[input:dim()] = self.hidden_size
  local htm1 = input.new():resize(hsz):zero()
  local inputt = input

  local curr=1

  while true do
    local ht = self.modules[curr]:updateOutput({inputt, htm1})
    local outputt = self:sampleOutput(ht)
    table.insert(outputs, outputt)
    if self:endCondition(outputt) then break end
    inputt=self:nextInput(outputt)
    htm1=ht
  end
  return outputs
end

-- this should perform a softmax classification and take the argmax
function LMDecoder:sampleOutput(ht)
  error("not implemented, either overwrite the function or set the decoder's sampleOutput function at instantiation time")
end

-- this should check if outputt == <EOS>
function LMDecoder:endCondition(outputt)
  error("not implemented, either overwrite the function or set the decoder's endCondition function at instantiation time")
end

-- this should return the word embedding of the output
function LMDecoder:nextInput(outputt)
  error("not implemented, either overwrite the function or set the decoder's nextInput function at instantiation time")
end

--TODO: does this work
function LMDecoder:updateOutput(input)
  print("Make sure this works!")
  if self.train then return nn.Recurrent.updateOutput(self,input) else return self:beamSearch(input) end
end

function LMDecoder:training()
    function modTrain(mod)
        mod:training()
    end
    self.train=true
    self:apply(modTrain)
end

function LMDecoder:evaluate()
    function modEval(mod)
        mod:evaluate()
    end
    self.train=false
    self:apply(modEval)
end
