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

local Seq2SeqDataset = {}
Seq2SeqDataset.__index = Seq2SeqDataset

function Seq2SeqDataset.create(data_dir, batch_size, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, Seq2SeqDataset)

    local source_file = path.join(data_dir, 'source.txt')
    local target_file = path.join(data_dir, 'target.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local counts_file = path.join(data_dir, 'counts.t7')
    local source_lengths_file = path.join(data_dir, 'source_lengths.t7')
    local target_lengths_file = path.join(data_dir, 'target_lengths.t7')

    local source_tensor_file = path.join(data_dir, 'TODOADDPREFIXsource_data.t7')
    local target_tensor_file = path.join(data_dir, 'TODOADDPREFIXtarget_data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) and path.exists(counts_file) and path.exists(source_lengths_file) and path.exists(target_lengths_file) path.exists(target_tensor_file) and path.exists(source_tensor_file)) then
        -- prepro files do not exist, generate them
        print("Couldn't find a required file. Running preprocessing...")
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local source_attr = lfs.attributes(source_file)
        local target_attr = lfs.attributes(target_file)
        local vocab_attr = lfs.attributes(vocab_file) -- dont need to check other than vocab bc all made at same time
        local source_tensor_attr = lfs.attributes(source_tensor_file)
        local target_tensor_attr = lfs.attributes(target_tensor_file)
        if source_attr.modification > vocab_attr.modification or source_attr.modification > source_tensor_attr.modification or source_attr.modification > target_tensor_attr.modification then
            print('Looks like source.txt has been changined. Re-running preprocessing...')
            run_prepro = true
        end
        if target_attr.modification > vocab_attr.modification or target_attr.modification > source_tensor_attr.modification or target_attr.modification > target_tensor_attr.modification then
            print('Looks like target.txt has been changed. Re-running preprocessing...')
            run_prepro = true
        end
    end

    if run_prepro then
        -- preprocess into vocab file, vocab counts file, source lengths file, and target lengths file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        Seq2SeqDataset.preprocess_text(source_file, target, vocab_file, counts_file, source_lengths_file, target_lengths_file)
    end

    print('loading preprocessed files...')
    self.counts = torch.load(counts_file)
    self.source_lengths = torch.load(source_lengths_file)
    self.target_lengths = torch.load(target_lengths_file)

    -- we want to do two things: first, truncate vocab to only the top N
    -- and second, batch it intelligently so that we waste the least amount of time in computation.
    
    -- truncate vocab
    -- TODO; remove vocab from preproc
    -- TODO: add split token to preproc for char-seq2seq possibility?
    -- TODO: add truncate_vocab_to param
    self.vocab_to_index = {}
    print("YOU SHOULD MAKE SURE THAT THIS SPAIRS FUNCTION WORKS YOU JUST COPY PASTED IT")
    for k,v in spairs(self.counts, function(t,a,b) return t[a]>t[b] end) do
        table.insert(vocab_to_index, k)
        if #vocab_to_index >= truncate_vocab_to then break end
    end

    -- batch intelligently
    -- each batch will be (batchsize, max_sentence_length),
    -- TODO: imhere
        



    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function Seq2SeqDataset:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function Seq2SeqDataset:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end

-- *** STATIC method ***
function Seq2SeqSeq2SeqDataset.text_to_vocab(in_sourcetextfile, in_targettextfile, out_vocabfile, out_countsfile, out_sourcelengthsfile, out_targetlengthsfile)
    local timer = torch.Timer()
    print('loading text file...')
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all tokens to a set
    -- also record line lengths for batching purposes later
    local counts = {}
    local source_lengths = {}
    for rawdata in io.lines(in_sourcetextfile) do
        local length = 0
        for token in rawdata:gmatch('%S+') do -- note: assumes that splitting purely by spaces is correct. Do your own pre-processing accordingly.
            if not unordered[token] then unordered[token] = 1 else unordered[token] = unordered[token] + 1 end
            length = length + 1
        end
        table.insert(source_lengths, length)
    end

    local target_lengths = {}
    for rawdata in io.lines(in_targettextfile) do
        local length = 0
        for token in rawdata:gmatch('%S+') do
            if not unordered[token] then unordered[token] = 1 else unordered[token] = unordered[token] + 1 end
            length = length + 1
        end
        table.insert(target_lengths, length)
    end
    
    assert(#source_lengths == #target_lengths, "source.txt and target.txt had a different number of lines.")

    -- sort into a table (i.e. keys become 1..N)
    local index_to_vocab = {}
    for token in pairs(counts) do index_to_vocab[#index_to_vocab + 1] = token end
    -- invert `ordered` to create the char->int mapping
    local vocab_to_index = {}
    for i, token in ipairs(index_to_vocab) do
        vocab_to_index[char] = i
    end
   -- -- construct a tensor with all the data
   -- print('putting data into tensors...')
   -- local source_data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
   -- for rawdata in io.lines(in_sourcetextfile) do

   -- f = io.open(in_textfile, "r")
   -- local currlen = 0
   -- rawdata = f:read(cache_len)
   -- repeat
   --     for i=1, #rawdata do
   --         data[currlen+i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
   --     end
   --     currlen = currlen + #rawdata
   --     rawdata = f:read(cache_len)
   -- until not rawdata
   -- f:close()

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving '.. out_countsfile)
    torch.save(out_countsfile, counts)
    print('saving ' .. out_sourcelengthsfile)
    torch.save(out_tensorfile, source_lengths)
    print('saving ' .. out_targetlengthsfile)
    torch.save(out_tensorfile, target_lengths)

end

return Seq2SeqDataset

