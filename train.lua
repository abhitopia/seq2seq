--[[

This file implements training.

Code is based on the train script in https://github.com/karpathy/char-rnn

Ankit Kumar
kitofans@gmail.com
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Recurrent'
require 'GRU'
require 'LSTM'
require 'lfs'
require 'Seq2SeqDataset'
models = require 'models'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train sequence to sequence learning')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-train_data_dir','data/SentimentPTBTrees/binary/train','train data directory, should contain source.txt and target.txt')
cmd:option('-test_data_dir', 'data/SentimentPTBTrees/binary/test', 'test data directory, should contain source.txt and target.txt')
cmd:option('-dev_data_dir', 'data/SentimentPTBTrees/binary/dev', 'dev data directory, should contain source.txt and target.txt')
cmd:option('-truncate_source_vocab_to', 20000, 'max vocab size of the source text')
cmd:option('-truncate_target_vocab_to', 20000, 'max vocab size of the target text')
-- model params
cmd:option('-model', 'basic', 'model to use')
cmd:option('-embedding_size', 100, 'size of word embeddings')
cmd:option('-hidden_size', 200, 'hidden dimension of recurrent nets')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',05760506,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-iterations', 10000, 'how many iterations to go through')
cmd:option('-eval_dev_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-num_dev_batches', 10, 'how many batches to test the validation cost on')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','binary_sentiment_basic','filename to autosave the checkpont to. Will be inside checkpoint_dir/. think of this as the name of the experiment')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
-- misc
cmd:option('-checkgrad', false, 'if true, will not actually train the network, but instead, will just check the gradient.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader classes
print(opt.batch_size)
local train_loader = Seq2SeqDataset(opt.train_data_dir, opt.batch_size, opt.truncate_source_vocab_to, opt.truncate_target_vocab_to)
local packedvocab = {source_i2v=train_loader.source_i2v, source_v2i=train_loader.source_v2i, target_i2v=train_loader.target_i2v, target_v2i=train_loader.target_v2i}
local dev_loader = Seq2SeqDataset(opt.dev_data_dir, 1, 0, 0, packedvocab)

-- build model config
local model_config = {}
model_config.source_vocab_size=#train_loader.source_i2v
model_config.target_vocab_size=#train_loader.target_i2v
model_config.source_max_sequence_length=train_loader.source_max_sequence_length
model_config.target_max_sequence_length=train_loader.target_max_sequence_length
model_config.hidden_size=opt.hidden_size
model_config.embedding_size=opt.embedding_size
model_config.EOS = train_loader.target_v2i['__EOS__']
print('source vocab size: ' .. model_config.source_vocab_size)
print('target vocab size: ' .. model_config.target_vocab_size)

local model = models[opt.model](model_config)
model:training()
local criterion = nn.CrossEntropyCriterion()  -- TODO maybe add clever criterions like CTC

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- ship the model to the GPU, or make everything floats, if desired
if opt.gpuid >= 0 then
  model:cuda()
  criterion:cuda()
elseif opt.real == 'float' then -- (floats don't lose much accuracy, but increase speed.)
  model:float()
  criterion:float()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model:getParameters()

-- initialization
if do_random_init then
params:uniform(-0.08, 0.08) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- do fwd/bwd and return loss, grad_params
if opt.checkgrad then opt.grad_clip=100000000000 end -- hacky but needed to pass gradchecks
function ftrain(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local source, target = train_loader:next_batch()
    local token_dim = target:dim()
    -- this saves memory at the expense of time, when compared to just storing all these tensors in the loader.
    -- really, if the batch dimenson was trailing, or if the arrays were stored column major, this would be simpler,
    -- as the desired memory would be contiguous anyways, and you'd get the best of both worlds.
    local target_no_sos = target:narrow(token_dim, 2, target:size(token_dim)-1):contiguous()
    target_no_sos:resize(target_no_sos:size(1)*target_no_sos:size(2))
    local target_no_eos = target:narrow(token_dim, 1, target:size(token_dim)-1):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        source = source:float():cuda()
        target_no_sos = target_no_sos:float():cuda()
        target_no_eos = target_no_eos:float():cuda()
    elseif opt.real == 'float' then
        source = source:float()
        target_no_sos = target_no_sos:float()
        target_no_eos = target_no_eos:float()
    end
    ------------------- forward/backward pass -------------------
    local out = model:forward({source, target_no_eos})
    local cost = criterion:forward(out, target_no_sos)
    local dout = criterion:backward(out, target_no_sos)
    local din = model:backward({source, target_no_eos}, dout)
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return cost, grad_params
end

function fdev(x, num_batches)
    if x ~= params then
        params:copy(x)
    end
    local total_cost=0
    for i=1,num_batches do
      ------------------ get minibatch -------------------
      local source, target = dev_loader:next_batch()
      local token_dim = target:dim()
      -- this saves memory at the expense of time, when compared to just storing all these tensors in the loader.
      -- really, if the batch dimenson was trailing, or if the arrays were stored column major, this would be simpler,
      -- as the desired memory would be contiguous anyways, and you'd get the best of both worlds.
      local target_no_sos = target:narrow(token_dim, 2, target:size(token_dim)-1):contiguous()
      target_no_sos:resize(target_no_sos:size(1)*target_no_sos:size(2))
      local target_no_eos = target:narrow(token_dim, 1, target:size(token_dim)-1):contiguous()
      if opt.gpuid >= 0 then -- ship the input arrays to GPU
          -- have to convert to float because integers can't be cuda()'d
          source = source:float():cuda()
          target_no_sos = target_no_sos:float():cuda()
          target_no_eos = target_no_eos:float():cuda()
      elseif opt.real == 'float' then
          source = source:float()
          target_no_sos = target_no_sos:float()
          target_no_eos = target_no_eos:float()
      end
      ------------------- forward/backward pass -------------------
      local out = model:forward({source, target_no_eos})
      local cost = criterion:forward(out, target_no_sos)
      total_cost = total_cost + cost
    end
    return total_cost/num_batches
end


if opt.checkgrad then
  local checkgrad = require 'modulegradcheck'
  local source, target = train_loader:next_batch()
  local token_dim = target:dim()
  -- this saves memory at the expense of time, when compared to just storing all these tensors in the loader.
  -- really, if the batch dimenson was trailing, or if the arrays were stored column major, this would be simpler,
  -- as the desired memory would be contiguous anyways, and you'd get the best of both worlds.
  local target_no_sos = target:narrow(token_dim, 2, target:size(token_dim)-1):contiguous()
  target_no_sos:resize(target_no_sos:size(1)*target_no_sos:size(2))
  local target_no_eos = target:narrow(token_dim, 1, target:size(token_dim)-1):contiguous()
  checkgrad.moduleDictCheckGrad(model, criterion, {source, target_no_eos}, target_no_sos, true, true)
  local source,target=train_loader:next_batch()
  function fakenextbatch()
    return source,target
  end
  train_loader.next_batch = fakenextbatch
  diff, _, _ = optim.checkgrad(feval, params, 1e-8)
  print ("Gradient check done. Difference was: "..diff..". Exiting...")
  os.exit()
end


-- start optimization here
train_losses = {}
dev_losses = {}
-- TODO
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.iterations
local loss0 = nil
for i = 1, iterations do
    --local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(ftrain, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    --if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
    --    if epoch >= opt.learning_rate_decay_after then
    --        local decay_factor = opt.learning_rate_decay
    --        optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
    --        print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    --    end
    --end

    -- every now and then or on last iteration
    if i % opt.eval_dev_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local dev_loss = fdev(params, opt.num_dev_batches) -- 2 = validation
        dev_losses[i] = dev_loss
        print(string.format("%d/%d, dev_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, dev_loss, grad_params:norm() / params:norm(), time))
        local savefile = string.format('%s/iter%d.t7', opt.checkpoint_dir, i)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.model = model
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.dev_losses = dev_losses
        checkpoint.iteration = i
        checkpoint.packedvocab = packedvocab
        torch.save(savefile, checkpoint)
        -- convenience file (for testing later)
        local f = torch.DiskFile(string.format('%s/iterlast.t7', opt.checkpoint_dir), 'w')
        f:writeInt(i)
        f:close()
    end

    if i % opt.print_every == 0 then
       print(string.format("%d/%d, train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


