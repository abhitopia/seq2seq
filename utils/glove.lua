-- read in glove vectors

-- glove_vectors_file: .txt file with glove vectors
-- weights: weights to write glove vectors to
-- v2i: mapping word -> row index for these weights
-- note: weights should be initialized before, this function will not touch rows that do not correspond to glove vectors.
function utils.read_glove(glove_vectors_file, weights, v2i)
  local c = 0
  for line in io.lines(glove_vectors_file) do
    local split_idx = line:find(' ')
    local word = line:sub(1,split_idx-1)
    -- copy into weights if exists
    if v2i[word] then
      c = c + 1
      local tensor = torch.Tensor(line:sub(split_idx+1):split(' '))
      weights[v2i[word]]:copy(tensor)
    end
  end
  print(string.format("Successfully loaded %d glove vectors", c))
  -- just safe to do this
  collectgarbage()
end
