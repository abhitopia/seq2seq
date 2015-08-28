-- A basic encoder-decoder architecture
require 'nn'
require 'nngraph'

source_vocab_size=3
max_source_sequence_length=4
target_vocab_size=3
max_target_sequence_length=4


--[[ Encoder: 2-layer GRU ]]
encoder = nn.Sequential()
print(encoder.modules)
encoder:add(nn.LookupTable(source_vocab_size, 100))
encoder:add(nn.GRU(100,200,max_source_sequence_length))
encoder:add(nn.GRU(200,200,max_source_sequence_length))
encoder:add(nn.Laststate(1,2))

--[[ Decoder: 1-layer GRU ]]
encoding = nn.Identity()()
target_shifted_one = nn.Identity()()
embedded = nn.LookupTable(target_vocab_size,100)(target_shifted_one)
decoding = nn.GRU(100,200,max_target_sequence_length)({encoding, embedded})
decoder = nn.gModule({encoding, target_shifted_one}, {decoding})

--[[ Putting them together ]]
source = nn.Identity()() -- source words
target_shifted_one_ = nn.Identity()() -- target words shifted over by one : the previous word that conditions the LM decoder
encoding_ = encoder(source)
decoding_ = decoder({encoding_, target_shifted_one_})
model = nn.gModule({source, target_shifted_one_}, {decoding_})
