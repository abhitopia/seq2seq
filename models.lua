require 'nn'
require 'nngraph'
require 'Recurrent'
require 'Extra'
require 'GRU'
require 'LSTM'
models = {}

-- A basic encoder-decoder architecture
function models.basic(model_config)
  -- read params
  local source_vocab_size=model_config.source_vocab_size
  local target_vocab_size=model_config.target_vocab_size
  local source_max_sequence_length=model_config.source_max_sequence_length
  local target_max_sequence_length=model_config.target_max_sequence_length
  print(source_max_sequence_length)
  print(target_max_sequence_length)
  local embedding_size=model_config.embedding_size
  local hidden_size=model_config.hidden_size

  --[[ Encoder: 2-layer GRU ]]
  local encoder = nn.Sequential()
  encoder:add(nn.LookupTable(source_vocab_size, embedding_size))
  encoder:add(nn.GRU(embedding_size,hidden_size,source_max_sequence_length))
  encoder:add(nn.GRU(hidden_size,hidden_size,source_max_sequence_length))
  encoder:add(nn.Laststate(1,2))
  
  --[[ Decoder: 1-layer GRU ]]
  local encoding = nn.Identity()()
  local target_shifted_one = nn.Identity()()
  local embeddings = nn.LookupTable(target_vocab_size,embedding_size)
  local embedded = embeddings(target_shifted_one)
  local decoder_ = nn.GRULMDecoder(embedding_size,hidden_size,target_max_sequence_length)
  local decoding = decoder_({encoding, embedded})
  local decoder = nn.gModule({encoding, target_shifted_one}, {decoding})
  
  --[[ Putting them together ]]
  local source = nn.Identity()() -- source words
  local target_shifted_one_ = nn.Identity()() -- target words shifted over by one : the previous word that conditions the LM decoder
  local encoding_ = encoder(source)
  local decoding_ = decoder({encoding_, target_shifted_one_})
  -- nn linear and the criterions don't accept 3d input, so simply resize it to look like the batchsize is in fact (num_seqs * seq_length), this is the same math
  local unbatched = nn.Unbatch()(decoding_)
  local predLinear = nn.Linear(hidden_size, target_vocab_size)
  local preds = predLinear(unbatched)
  local model = nn.gModule({source, target_shifted_one_}, {preds})

  decoder_:setLookupTable(embeddings)
  decoder_:setSoftmaxLinear(predLinear)
  decoder_:setEOS(model_config.EOS)
  return model
end

return models
