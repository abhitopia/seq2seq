--[[

This file implements plotting a checkpoint file.

Ankit Kumar
kitofans@gmail.com
]]--

require 'torch'
require 'nn'
require 'gnuplot'
require 'nngraph'
require 'Seq2SeqDataset'
models = require 'models/'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train sequence to sequence learning')
cmd:text()
cmd:text('Options')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-iteration', -1, 'iteration to load. -1 means most recent iteration.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
-- overwrite -1 iteration with the most recent iteration
if opt.iteration == -1 then
  local f = torch.DiskFile(string.format("%s/iterlast.t7", opt.checkpoint_dir), 'r')
  opt.iteration = f:readInt()
end
-- Modified from https://github.com/karpathy/char-rnn
-- this function taken from http://stackoverflow.com/questions/15706270/sort-a-table-in-lua
local function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end


local function parse(loss)
  local iters = {}
  local losses = {}
  for k,v in spairs(loss) do
    table.insert(iters,k)
    table.insert(losses,v)
  end
  return iters, losses
end





checkpoint = torch.load(string.format("%s/iter%d.t7", opt.checkpoint_dir, opt.iteration))
dev_iters,dev_losses = parse(checkpoint.dev_losses)
train_iters,train_losses = parse(checkpoint.train_losses)

gnuplot.plot({'train',torch.Tensor(train_iters), torch.Tensor(train_losses), '-'}, {'dev', torch.Tensor(dev_iters), torch.Tensor(dev_losses), '-'})
--

