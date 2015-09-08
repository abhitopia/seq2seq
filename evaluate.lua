--[[

This file implements interaction with a checkpoint file.

Ankit Kumar
kitofans@gmail.com
]]--

require 'torch'
require 'nn'
require 'xlua'
require 'nngraph'
require 'Recurrent'
require 'GRU'
require 'LSTM'
require 'Seq2SeqDataset'
require 'BeamSearch'
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
cmd:option('-iterations', 100000, 'how many iterations to go through')
cmd:option('-eval_dev_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-num_dev_batches', 10, 'how many batches to test the validation cost on')

cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-iteration', -1, 'iteration to load. -1 means most recent iteration.')

cmd:option('-savefile','binary_sentiment_basic','filename to autosave the checkpont to. Will be inside checkpoint_dir/. think of this as the name of the experiment')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
-- misc
cmd:option('-checkgrad', false, 'if true, will not actually train the network, but instead, will just check the gradient.')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
-- set model to evaluation mode
torch.manualSeed(opt.seed)
-- overwrite -1 iteration with the most recent iteration
if opt.iteration == -1 then
  local f = torch.DiskFile(string.format("%s/iterlast.t7", opt.checkpoint_dir), 'r')
  opt.iteration = f:readInt()
end


local checkpoint = torch.load(string.format("%s/iter%d.t7", opt.checkpoint_dir, opt.iteration))
local beam_searcher = BeamSearch(checkpoint.decoder.masterStepModule, checkpoint.target_embeddings, checkpoint.pred_linear, checkpoint.packedvocab.target_v2i['__SOS__'], checkpoint.packedvocab.target_v2i['__EOS__'])
local encoder = checkpoint.encoder
local dev_loader = Seq2SeqDataset(opt.dev_data_dir, 1, 0, 0, checkpoint.packedvocab)
local test_loader = Seq2SeqDataset(opt.test_data_dir, 1, 0, 0, checkpoint.packedvocab)
checkpoint.model:evaluate() --TODO: deal with this, make sure this script works.

function test(loader)
  local total = loader:size()
  local corr = 0
  for i=1,total do
    xlua.progress(i,total)
    local source, target = loader:next_batch()
    local token_dim = target:dim()
    local target_no_sos = target:narrow(token_dim, 2, target:size(token_dim)-1):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        source = source:float():cuda()
        target_no_sos = target_no_sos:float():cuda()
    elseif opt.real == 'float' then
        source = source:float()
        target_no_sos = target_no_sos:float()
    end
    local encoding = encoder:forward(source)
    local output = beam_searcher:forward(encoding)
    if output:eq(target_no_sos):sum() == output:nElement() then
      corr = corr + 1
    end
  end

  print ("total: "..total..", correct: "..corr..", accuracy: "..corr/total)
end

print("Testing on dev set...")
test(dev_loader)
print("Testing on test set...")
test(test_loader)
