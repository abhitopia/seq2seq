require 'Extra'
require 'nn'
gc = require 'modulegradcheck'

m = nn.Unbatch()
input = torch.rand(4,3,10)
target = torch.rand(12,10)
criterion = nn.AbsCriterion()
gc.moduleDictCheckGrad(m, criterion, input, target, true, false)
