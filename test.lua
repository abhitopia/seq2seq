require 'nnrecurrent'
--require 'oldGRU'

gradcheck = paths.dofile('modulegradcheck.lua')
local function RNNTest()
  mod = nn.RNN(5,10,30)
  --mod:reset()
  --
  crit = nn.AbsCriterion()
  input = torch.rand(15,5)
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)

  input = torch.rand(5,15,5)
  target = torch.rand(5,15,10)

  gradcheck.moduledictgradcheck(mod,crit,input,target)
  print("REVERSE")
  mod = nn.RNN(5,10,30, true)
  mod:reset()
  crit = nn.AbsCriterion()
  input = torch.rand(15,5)
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)

  input = torch.rand(5,15,5)
  target = torch.rand(5,15,10)

  gradcheck.moduledictgradcheck(mod,crit,input,target)

end
local function GRUTest()
  mod = nn.GRU(5,10,30)
  --p, gp = mod:getParameters()
  --p:uniform(-.1,.1)
  --mod:reset()
  crit = nn.AbsCriterion()
  input = torch.rand(15,5)
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)

  input = torch.rand(5,15,5)
  target = torch.rand(5,15,10)

  gradcheck.moduledictgradcheck(mod,crit,input,target)

  --print("CUDA")
--  require 'cutorch'
--  require 'cunn'
--  mod = nn.GRU(5,10,30)
--  mod:cuda()
--  p, gp = mod:getParameters()
--  p:uniform(-.1,.1)
--  mod:manualReset()
--  crit = nn.AbsCriterion():cuda()
--  input = torch.rand(15,5):cuda()
--  target = torch.rand(15,10):cuda()
--  gradcheck.moduledictgradcheck(mod,crit,input,target)
--
--  print("REVERSE")
--  mod = nn.GRU(5,10,30, true)
--  mod:reset()
--  crit = nn.AbsCriterion()
--  input = torch.rand(15,5)
--  target = torch.rand(15,10)
--  gradcheck.moduledictgradcheck(mod,crit,input,target)
--
--  input = torch.rand(5,15,5)
--  target = torch.rand(5,15,10)
--
--  gradcheck.moduledictgradcheck(mod,crit,input,target)
end

local function LSTMTest()
  mod = nn.LSTM(5,10,30)
  mod:reset()
  crit = nn.AbsCriterion()
  input = torch.rand(15,5)
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)


  input = torch.rand(5,15,5)
  target = torch.rand(5,15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)
end
local function RNNInputStateTest()
  mod = nn.RNNInputState(5,10,30)
  --mod:reset()
  --
  crit = nn.AbsCriterion()
  input = {torch.rand(15,5), torch.rand(10)}
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)

  input = {torch.rand(5,15,5), torch.rand(1,10)}
  target = torch.rand(5,15,10)

  gradcheck.moduledictgradcheck(mod,crit,input,target)
  print("REVERSE")
  mod = nn.RNNInputState(5,10,30, true)
  mod:reset()
  crit = nn.AbsCriterion()
  input = {torch.rand(15,5), torch.rand(10)}
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)

  input = {torch.rand(5,15,5), torch.rand(5,10)}
  target = torch.rand(5,15,10)

  gradcheck.moduledictgradcheck(mod,crit,input,target)

end
local function GRUInputStateTest()
  mod = nn.GRUInputState(5,10,30)
  --p, gp = mod:getParameters()
  --p:uniform(-.1,.1)
  --mod:reset()
  crit = nn.AbsCriterion()
  input = {torch.rand(15,5), torch.rand(10)}
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)

  input = {torch.rand(5,15,5), torch.rand(5,10)}
  target = torch.rand(5,15,10)

  gradcheck.moduledictgradcheck(mod,crit,input,target)

  --print("CUDA")
--  require 'cutorch'
--  require 'cunn'
--  mod = nn.GRU(5,10,30)
--  mod:cuda()
--  p, gp = mod:getParameters()
--  p:uniform(-.1,.1)
--  mod:manualReset()
--  crit = nn.AbsCriterion():cuda()
--  input = torch.rand(15,5):cuda()
--  target = torch.rand(15,10):cuda()
--  gradcheck.moduledictgradcheck(mod,crit,input,target)
--
--  print("REVERSE")
--  mod = nn.GRU(5,10,30, true)
--  mod:reset()
--  crit = nn.AbsCriterion()
--  input = torch.rand(15,5)
--  target = torch.rand(15,10)
--  gradcheck.moduledictgradcheck(mod,crit,input,target)
--
--  input = torch.rand(5,15,5)
--  target = torch.rand(5,15,10)
--
--  gradcheck.moduledictgradcheck(mod,crit,input,target)
end

local function LSTMInputStateTest()
  mod = nn.LSTMInputState(5,10,30)
  mod:reset()
  crit = nn.AbsCriterion()
  input = torch.rand(15,5)
  target = torch.rand(15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)


  input = torch.rand(5,15,5)
  target = torch.rand(5,15,10)
  gradcheck.moduledictgradcheck(mod,crit,input,target)
end

print("ALL TESTS ARE CPUS")
print("RNN test")
RNNTest()
print("GRU test")
GRUTest()
print("LSTM test")
LSTMTest()
--LSTMTest()
mod = nn.GRU(5,10,30)

p, gp = mod:getParameters()
gp:zero()
--for i=1,#gp do gp[i]:zero() end
input = torch.rand(15,5)
target = torch.rand(15,10)
crit = nn.AbsCriterion()

out = mod:forward(input)
c = crit:forward(out, target)
dout = crit:backward(out, target)
dinput = mod:backward(input, dout)
oldgp = torch.Tensor(gp:size()):copy(gp)
--oldgp = {}
--for i=1,#gp do
--  oldgp_i = torch.Tensor(gp[i]:size()):copy(gp[i])
--  oldgp[i] = oldgp_i
--end

require 'cutorch'
require 'cunn'

cmod = mod:clone()
print("CALLING CUDA")
cmod:cuda()
p, gp = cmod:getParameters()
cmod.modules = {}
gp:zero()
--for i=1, #gp do gp[i]:zero() end
--cmod:manualReset()
cinput = input:cuda()
ctarget = target:cuda()
ccrit = nn.AbsCriterion()
ccrit:cuda()

cout = cmod:forward(cinput)
cc = ccrit:forward(cout, ctarget)
dcout = ccrit:backward(cout, ctarget)
dcinput = cmod:backward(cinput, dcout)

