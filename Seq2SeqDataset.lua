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


local Seq2SeqDataset = torch.class('Seq2SeqDataset')

function Seq2SeqDataset:__init(data_dir, batchsize, truncate_source_vocab_to, truncate_target_vocab_to, loadvocab)
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
    local loadvocab = loadvocab or false

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(source_counts_file) and path.exists(target_counts_file) and path.exists(source_lengths_file) and path.exists(target_lengths_file)) then
        -- prepro files do not exist, generate them
        print("Couldn't find a required file. Running preprocessing...")
        run_prepro = true
    else
        run_prepro=false
    end

    if run_prepro then
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
    if not loadvocab then 
      self.source_i2v = {}
      print("YOU SHOULD MAKE SURE THAT THIS SPAIRS FUNCTION WORKS YOU JUST COPY PASTED IT")
      for k,v in spairs(source_counts, function(t,a,b) return t[a]>t[b] end) do
          table.insert(self.source_i2v, k)
          if #self.source_i2v >= truncate_source_vocab_to then break end
      end

      self.target_i2v = {}
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
      -- target also needs start-of-sequence and end-of-sequence tokens
      table.insert(self.target_i2v, '__SOS__')
      self.target_v2i['__SOS__'] = #self.target_i2v
      table.insert(self.target_i2v, '__EOS__')
      self.target_v2i['__EOS__'] = #self.target_i2v
    else
      -- dev/test set, uses train vocab.
      self.source_v2i=loadvocab.source_v2i
      self.source_i2v=loadvocab.source_i2v
      self.target_v2i=loadvocab.target_v2i
      self.target_i2v=loadvocab.target_i2v
    end



    local sum_lengths = {}
    for i=1,#source_lengths do
      table.insert(sum_lengths,source_lengths[i]+target_lengths[i])
    end

    local batch_assignments = {}
    local batch_offsets={}
    local batch_source_maxlens = {[0]=0} -- bit of a hack for the max_seq_len stuff
    local batch_target_maxlens = {[0]=0} -- same here
    self.source_max_sequence_length = 0
    self.target_max_sequence_length = 0
    local curr_batch = 0
    local in_curr_batch= batchsize
    for k,v in spairs(sum_lengths, function(t,a,b) return t[a]>t[b] end) do
      if in_curr_batch == batchsize then
        if batch_source_maxlens[curr_batch] > self.source_max_sequence_length then self.source_max_sequence_length=batch_source_maxlens[curr_batch] end
        if batch_target_maxlens[curr_batch] > self.target_max_sequence_length then self.target_max_sequence_length=batch_target_maxlens[curr_batch] end
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
      table.insert(self.source_batches, torch.ShortTensor(batchsize, batch_source_maxlens[i]):fill(spad)) 
      table.insert(self.target_batches, torch.ShortTensor(batchsize, batch_target_maxlens[i]+2):fill(tpad)) -- +2 is for __SOS__, __EOS__
    end

    -- now read through source and target again and fill up the tensors
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
        local curr_idx = 2 -- start at two to leave room for __SOS__
        local batch = batch_assignments[curr_line]
        local batch_offset = batch_offsets[curr_line]
        for token in rawdata:gmatch('%S+') do -- note: assumes that splitting purely by spaces is correct. Do your own pre-processing accordingly.
            local token_idx = self.target_v2i[token] or self.target_v2i['__UNK__']
            self.target_batches[batch][batch_offset][curr_idx] = token_idx
            curr_idx = curr_idx + 1
        end
        -- add __SOS__, __EOS__
        self.target_batches[batch][batch_offset][1] = self.target_v2i['__SOS__']
        self.target_batches[batch][batch_offset][curr_idx] = self.target_v2i['__EOS__']
        curr_line = curr_line + 1
    end

    assert(#self.source_batches == #self.target_batches)
    print("data loading done.")
    -- TODO: deleted here a lot of split stuff. maybe deal with that later.
    
    self:reset_batches()
    collectgarbage()
    return self
end

function Seq2SeqDataset:reset_batches()
  -- shuffle batches
  self.curr_batch_order = torch.randperm(#self.source_batches)
  -- reset idx
  self.curr_batch_index=1
end

function Seq2SeqDataset:next_batch()
  if self.curr_batch_index > #self.source_batches then
    self:reset_batches()
  end
  local real_index = self.curr_batch_order[self.curr_batch_index]
  self.curr_batch_index = self.curr_batch_index+1
  return self.source_batches[real_index], self.target_batches[real_index]
end

function Seq2SeqDataset:size()
  return #self.source_batches
end

function Seq2SeqDataset:tensor_to_string(tensor, ds)
  assert(ds=='source' or ds=='target')
  if tensor:dim() == 1 then
    local str = ''
    print(tensor[1])
    str = str..self[ds..'_i2v'][tensor[1]]
    for i=2,tensor:size(1) do
      str = str..' '..self[ds..'_i2v'][tensor[i]]
    end
    return str
  elseif tensor:dim() == 2 then
    local strs = {}
    for i=1,tensor:size(1) do
      local slice=tensor[i]
      table.insert(strs, self:tensor_to_string(slice, ds))
    end
    return strs
  end
end

function Seq2SeqDataset:string_to_tensor(inputstr, ds)
  assert(ds=='source' or ds=='target')
  local tokens = {}
  for token in inputstr:gmatch('%S+') do
    table.insert(tokens, token)
  end
  local tensor = torch.Tensor(#tokens)
  for i=1, #tokens do
    tensor[i] = self[ds..'_v2i'][tokens[i]]
  end
  return tensor
end
  
