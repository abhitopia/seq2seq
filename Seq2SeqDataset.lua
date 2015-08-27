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

-- *** STATIC methods ***
local function i2v_to_v2i(i2v)
  local v2i = {}
  for i,v in ipairs(i2v) do
    v2i[v] = i
  end
  return v2i
end

local function text_to_vocab(in_textfile, out_countsfile, out_lengthsfile)
    local timer = torch.Timer()
    print('loading text file...')
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all tokens to a set
    -- also record line lengths for batching purposes later
    local counts = {}
    local lengths = {}
    for rawdata in io.lines(in_textfile) do
        local length = 0
        for token in rawdata:gmatch('%S+') do -- note: assumes that splitting purely by spaces is correct. Do your own pre-processing accordingly.
            if not counts[token] then counts[token] = 1 else counts[token] = counts[token] + 1 end
            length = length + 1
        end
        table.insert(lengths, length)
    end

     
    -- save output preprocessed files
    print('saving '.. out_countsfile)
    torch.save(out_countsfile, counts)
    print('saving ' .. out_lengthsfile)
    torch.save(out_lengthsfile, lengths)
end


local Seq2SeqDataset = torch.class('S2SDataset')

function Seq2SeqDataset:__init(data_dir, batch_size, truncate_source_vocab_to, truncate_target_vocab_to)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}
    local data_dir = data_dir or "test_datadir"
    local batchsize = batchsize or 2
    local truncate_source_vocab_to = truncate_source_vocab_to or 10000
    local truncate_target_vocab_to = truncate_target_vocab_to or 10000

    local source_file = path.join(data_dir, 'source.txt')
    local target_file = path.join(data_dir, 'target.txt')
    local source_counts_file = path.join(data_dir, 'source_vocab_counts.t7')
    local target_counts_file = path.join(data_dir, 'target_vocab_counts.t7')
    local source_lengths_file = path.join(data_dir, 'source_lengths.t7')
    local target_lengths_file = path.join(data_dir, 'target_lengths.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(source_counts_file) and path.exists(target_counts_file) and path.exists(source_lengths_file) and path.exists(target_lengths_file)) then
        -- prepro files do not exist, generate them
        print("Couldn't find a required file. Running preprocessing...")
        run_prepro = true
    else
        run_prepro=false
        --[[ The commented out stuff below this is some intense checking, maybe add that later before release]]
        -- [[ TODO ]] 
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        --local source_attr = lfs.attributes(source_file)
        --local target_attr = lfs.attributes(target_file)
        --local vocab_attr = lfs.attributes(vocab_file) -- dont need to check other than vocab bc all made at same time
        --local source_tensor_attr = lfs.attributes(source_tensor_file)
        --local target_tensor_attr = lfs.attributes(target_tensor_file)
        --if source_attr.modification > vocab_attr.modification or source_attr.modification > source_tensor_attr.modification or source_attr.modification > target_tensor_attr.modification then
        --    print('Looks like source.txt has been changined. Re-running preprocessing...')
        --    run_prepro = true
        --end
        --if target_attr.modification > vocab_attr.modification or target_attr.modification > source_tensor_attr.modification or target_attr.modification > target_tensor_attr.modification then
        --    print('Looks like target.txt has been changed. Re-running preprocessing...')
        --    run_prepro = true
        --end
    end

    if run_prepro then
        -- preprocess into vocab file, vocab counts file, source lengths file, and target lengths file
        print('one-time setup: preprocessing input text files ' .. source_file .. ' and '.. target_file..'...')
        text_to_vocab(source_file, source_counts_file, source_lengths_file)
        text_to_vocab(target_file, target_counts_file, target_lengths_file)
    end

    print('loading preprocessed files...')
    local source_counts = torch.load(source_counts_file)
    local target_counts = torch.load(target_counts_file)
    local source_lengths = torch.load(source_lengths_file)
    local target_lengths = torch.load(target_lengths_file)
    assert(#source_lengths==#target_lengths, "source and target had different number of sentences")

    -- we want to do two things: first, truncate vocab to only the top N
    -- and second, batch it intelligently so that we waste the least amount of time in computation.
    
    -- truncate vocab
    -- TODO; remove vocab from preproc!
    -- TODO: add split token to preproc for char-seq2seq possibility?
    self.source_i2v = {}
    print("YOU SHOULD MAKE SURE THAT THIS SPAIRS FUNCTION WORKS YOU JUST COPY PASTED IT")
    for k,v in spairs(source_counts, function(t,a,b) return t[a]>t[b] end) do
        table.insert(self.source_i2v, k)
        if #self.source_i2v >= truncate_source_vocab_to then break end
    end

    self.target_i2v = {}
    print("YOU SHOULD MAKE SURE THAT THIS SPAIRS FUNCTION WORKS YOU JUST COPY PASTED IT")
    for k,v in spairs(target_counts, function(t,a,b) return t[a]>t[b] end) do
        table.insert(self.target_i2v, k)
        if #self.target_i2v >= truncate_target_vocab_to then break end
    end


    self.source_v2i = i2v_to_v2i(self.source_i2v)
    self.target_v2i = i2v_to_v2i(self.target_i2v)

    -- add pad and unk
    table.insert(self.source_i2v, '__UNK__')
    self.source_v2i['__UNK__'] = #self.source_i2v
    table.insert(self.source_i2v, '__PAD__')
    self.source_v2i['__PAD__'] = #self.source_i2v

    table.insert(self.target_i2v, '__UNK__')
    self.target_v2i['__UNK__'] = #self.target_i2v
    table.insert(self.target_i2v, '__PAD__')
    self.target_v2i['__PAD__'] = #self.target_i2v


    local sum_lengths = {}
    for i=1,#source_lengths do
      table.insert(sum_lengths,source_lengths[i]+target_lengths[i])
    end

    local batch_assignments = {}
    local batch_offsets={}
    local batch_source_maxlens = {}
    local batch_target_maxlens = {}
    local curr_batch = 0
    local in_curr_batch= batchsize
    for k,v in spairs(sum_lengths, function(t,a,b) return t[a]>t[b] end) do
      if in_curr_batch == batchsize then
        curr_batch=curr_batch+1
        in_curr_batch=0
        batch_source_maxlens[curr_batch] = 0
        batch_target_maxlens[curr_batch] = 0
      end
      if source_lengths[k] > batch_source_maxlens[curr_batch] then batch_source_maxlens[curr_batch] = source_lengths[k] end
      if target_lengths[k] > batch_target_maxlens[curr_batch] then batch_target_maxlens[curr_batch] = target_lengths[k] end

      batch_assignments[k]=curr_batch
      in_curr_batch=in_curr_batch+1
      batch_offsets[k] = in_curr_batch
    end

    self.source_batches = {}
    self.target_batches = {}

    -- you probably shouldn't use a vocab size over 32000 anyways, so a short tensor is fine. anyways we'll check.
    assert(#self.source_i2v < 32767, "your source vocabulary is too big for a ShortTensor storage, but you probably don't want this anyways.")
    assert(#self.target_i2v < 32767, "your target vocabulary is too big for a ShortTensor storage, but you probably don't want this anyways.")
    local spad = self.source_v2i['__PAD__']
    local tpad = self.target_v2i['__PAD__']
    for i=1,curr_batch do
      table.insert(self.source_batches, torch.ShortTensor(batchsize, batch_source_maxlens[i]):fill(spad)) -- 
      table.insert(self.target_batches, torch.ShortTensor(batchsize, batch_target_maxlens[i]):fill(tpad))
    end

    -- now read through source and target again and fill up the tensors
    -- TODO: I'm here. loop through and fill tensors.
    print('reading texts...')
    local curr_line = 1
    for rawdata in io.lines(source_file) do
        local curr_idx = 1
        local batch = batch_assignments[curr_line]
        local batch_offset = batch_offsets[curr_line]
        for token in rawdata:gmatch('%S+') do -- note: assumes that splitting purely by spaces is correct. Do your own pre-processing accordingly.
            local token_idx = self.source_v2i[token] or self.source_v2i['__UNK__']
            self.source_batches[batch][batch_offset][curr_idx] = token_idx
            curr_idx = curr_idx + 1
        end
        curr_line = curr_line + 1
    end

    local curr_line = 1
    for rawdata in io.lines(target_file) do
        local curr_idx = 1
        local batch = batch_assignments[curr_line]
        local batch_offset = batch_offsets[curr_line]
        for token in rawdata:gmatch('%S+') do -- note: assumes that splitting purely by spaces is correct. Do your own pre-processing accordingly.
            local token_idx = self.target_v2i[token] or self.target_v2i['__UNK__']
            self.target_batches[batch][batch_offset][curr_idx] = token_idx
            curr_idx = curr_idx + 1
        end
        curr_line = curr_line + 1
    end

    assert(#self.source_batches == #self.target_batches)
    print("data loading done.")
    -- TODO: deleted here a lot of split stuff. maybe deal with that later.
    
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


