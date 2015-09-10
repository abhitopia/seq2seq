-- [[ implementation of the model from sutskever et al:
-- http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
--
-- Ankit Kumar
-- kitofans@gmail.com
-- ]]

-- deep recurrent network: 
-- require same number of encoder/decoder layers
-- see e.g http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
function models.deep_recurrent(model_config)
    -- params
    local source_vocab_size = model_config.source_vocab_size
    local target_vocab_size = model_config.target_vocab_size
    local source_max_sequence_length = model_config.source_max_sequence_length
    local target_max_sequence_length = model_config.target_max_sequence_length

    -- encoder params
    local encoder_embedding_size = model_config.encoder_embedding_size
    local encoder_hidden_sizes = model_config.encoder_hidden_sizes
    -- decoder params
    local decoder_embedding_size = model_config.decoder_embedding_size
    local decoder_hidden_sizes = model_config.decoder_hidden_sizes

    assert(#encoder_hidden_sizes==#decoder_hidden_sizes, string.format("To use a deep recurrent model you must have an equal number of encoder/decoder layers; got %d encoder layers, %d decoder layers", #encoder_hidden_sizes, #decoder_hidden_sizes))
    local num_layers = #encoder_hidden_sizes

    -- instantiate modules that will be used
    -- [[ Encoder ]] 
    local source_embeddings = nn.LookupTable(source_vocab_size, encoder_embedding_size)

    local encoder_lstm_layers = {}
    local last_states = {}
    encoder_hidden_sizes[0] = encoder_embedding_size
    for i=1,num_layers do
        table.insert(encoder_lstm_layers, recurrent_module(encoder_hidden_sizes[i-1], encoder_hidden_sizes[i], source_max_sequence_length))
        table.insert(last_states, nn.Laststate(1,2))
    end

    -- [[ Decoder: 4-layer LSTM ]]
    local target_embeddings = nn.LookupTable(target_vocab_size, 1000)

    local decoder_lstm_layers = {}
    decoder_hidden_sizes[0] = decoder_embedding_size
    for i=1,#num_layers do
        table.insert(decoder_lstm_layers, recurrent_module(decoder_hidden_sizes[i-1], decoder_hidden_sizes[i], target_max_sequence_length))
    end

    local unbatch_decoding = nn.Unbatch()
    local prediction_linear = nn.Linear(1000, target_vocab_size)

    -- [[ Build model ]] 

    -- [[ Build encoder ]]
    local source = nn.Identity()()
    local source_embedded = source_embeddings(source)
    local encoder_layer_outs = {}
    local encodings = {}
    local curr = source_embedded
    for i=1,num_layers do
        curr = encoder_lstm_layers[i](curr)
        table.insert(encoder_layer_outs, curr)
        table.insert(encodings, last_states[i](curr))
    end

    local encoder = nn.gModule({source},{encodings})

    -- [[ Build decoding ]]
    local target_shifted_one = nn.Identity()() -- <SOS> word1 word2 ... rather than word1 word2 ...
    local decoder_layer_one_out = decoder_lstm_layer_one({layer_one_encoding, target_shifted_one})
    local decoder_layer_two_out = decoder_lstm_layer_two({layer_two_encoding, decoder_layer_one_out})
    local decoder_layer_three_out = decoder_lstm_layer_three({layer_three_encoding, decoder_layer_two_out})
    local decoder_layer_four_out = decoder_lstm_layer_four({layer_four_encoding, decoder_layer_three_out})
    local unbatched = unbatch_decoding(decoder_layer_four_out)
    local predictions = prediction_linear(unbatched)
    local model = nn.gModule({source, target_shifted_one}, {predictions})

    -- For beam search we want an explicit encoder/decoder.
    -- we can't just reuse the same nodes as above because they are tied to the model gmodule,
    -- but we can just re-do the work again using the same modules. annoying, but i don't know
    -- of a way around it, and I'm pretty sure it's fine from a memory standpoint.
    local _source = nn.Identity()()
    local _source_embedded = source_embeddings(_source)

    local _encoder_layer_one_out = encoder_lstm_layer_one(_source_embedded)
    local _encoder_layer_two_out = encoder_lstm_layer_two(_layer_one_out)
    local _encoder_layer_three_out = encoder_lstm_layer_three(_layer_two_out)
    local _encoder_layer_four_out = encoder_lstm_layer_four(_layer_three_out)

    local _layer_one_encoding = encoder_last_state_one(_encoder_layer_one_out)
    local _layer_two_encoding = encoder_last_state_two(_encoder_layer_two_out)
    local _layer_three_encoding = encoder_last_state_three(_encoder_layer_three_out)
    local _layer_four_encoding = encoder_last_state_four(_encoder_layer_four_out)
    local encoder = nn.gModule({_source}, {_layer_one_encoding, _layer_two_encoding, _layer_three_encoding, _layer_four_encoding})


    local _target_shifted_one = nn.Identity()()
    local __layer_one_encoding = nn.Identity()()
    local __layer_two_encoding = nn.Identity()()
    local __layer_three_encoding = nn.Identity()()
    local __layer_four_encoding = nn.Identity()()

    local _decoder_layer_one_out = decoder_lstm_layer_one({__layer_one_encoding, target_shifted_one})
    local _decoder_layer_two_out = decoder_lstm_layer_two({__layer_two_encoding, decoder_layer_one_out})
    local _decoder_layer_three_out = decoder_lstm_layer_three({__layer_three_encoding, decoder_layer_two_out})
    local _decoder_layer_four_out = decoder_lstm_layer_four({__layer_four_encoding, decoder_layer_three_out})
    local _unbatched = unbatch_decoding(_decoder_layer_four_out)
    local _predictions = prediction_linear(_unbatched)
    local decoder = nn.gModule({

    local decoder = nn.gModule({placeholder_layer_one_encoding, placehold_layer_two_encoding, placeholder_layer_three_encoding, 





    local embedding_one = nn.Identity()()
    local embedding_two = nn.Identity()()
    local embedding_three = nn.Identity()()
    local embedding_four = nn.Identity()()


    



