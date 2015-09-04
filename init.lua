require 'nn'
--[[
The following is taken largely from dpnn (https://github.com/nicholas-leonard/dpnn/blob/master/Module.lua).


NOTE: it has NOT been tested on the case where it needs to pick up tensors that are shared; i.e,
 arbitrary tensors that are shared with ones of the name e.g 'weight'.
--]]
local Module = torch.getmetatable('nn.Module')

function Module:sharedCloneManyTimes(ntimes,...)
  print(ntimes)
  print(...)
  -- ... can be some combination of 'params' and 'gradParams', i just like that syntax a bit
  local dotargs = {...}
  local argvalset = {}
  for k,v in ipairs(dotargs) do argvalset[v]=true end

  local valset = {}
  if argvalset['params'] then valset['bias']=true;valset['weight']=true end
  if argvalset['gradParams'] then valset['gradBias']=true;valset['gradWeight']=true end

  --local shareParams = argvalset['params'] or false
  --local shareGradParams = argvalset['gradParams'] or false

  local params, pointers, recursepath ={}, {}, {}
  local function recursiveStore(obj, first_pass)
    if torch.isTensor(obj) then
      error("should never get here, don't call recursivestore on a tensor")
    end

    local paths = {}
    if type(obj) == 'table' then
      for k,v in pairs(obj) do
        if valset[k] and first_pass and torch.isTensor(v) and (v~='__tofill') and (v~='__tofill_was_shared') then 
          table.insert(params, v)
          obj[k]= #params

          if v:storage() then
            pointers[torch.pointer(v:storage():data())] = true
          end
          table.insert(paths, k)

        elseif torch.isTensor(v) and v:storage() then
          if pointers[torch.pointer(v:storage():data())] then
            params[k] = v
            obj[k] = '__tofill_was_shared'
            table.insert(paths, k)
          end
        elseif ((not torch.isTensor(v)) and (type(v) == 'table')) then 
          local path = recursiveStore(v, first_pass)
          local isgoodpath = false
          for ok,ov in pairs(path) do isgoodpath=true;break end
          if isgoodpath then
            local mpath = {}
            mpath[k] = path
            table.insert(paths, mpath)
          end
        end
      end
    else
      error('recursivestore was called on something that wasnt a table')
    end
    return paths
  end

  local recursepaths = recursiveStore(self, true)
  local num = 0

  local curr_nparams = #params

  while true do
    thesepaths = recursiveStore(self, false)
    for i=1,#thesepaths do
      table.insert(recursepaths, thesepaths[i])
    end
    if #params == curr_nparams then break end
    curr_nparams = #params
  end

  function recursiveFill(paths, obj, clone, do_obj)
    for i=1,#paths do
      if type(paths[i]) == 'table' then
        for k,v in pairs(paths[i]) do
          recursiveFill(paths[i][k],obj[k], clone[k], do_obj)
        end
      else
        -- now we're at the bottom
        local k = paths[i]
        if clone[k] == '__tofill_was_shared' then
          if do_obj then obj[k] = params[k] end
          clone[k] = params[k].new():set(params[k])
        elseif type(clone[k]) == 'number' then
          if do_obj then obj[k] = params[clone[k]] end
          clone[k] = params[clone[k]].new():set(params[clone[k]])
        end
      end
    end
  end


  local clones = {}
  -- clone everything but parameters and/or gradients
  for i=1,ntimes do
    local clone = self:clone()
    do_obj = i == ntimes
    recursiveFill(recursepaths, self, clone, do_obj)

    --assert(#params +1 == currfill, "#params was "..#params.." and currfil was "..currfill)
    table.insert(clones, clone)
  end
  return clones
end
torch.include('nnrecurrent', 'Recurrent.lua')
torch.include('nnrecurrent', 'Extra.lua') -- batchnarrow
torch.include('nnrecurrent', 'RNN.lua')
torch.include('nnrecurrent', 'GRU.lua')
torch.include('nnrecurrent', 'LSTM.lua')

