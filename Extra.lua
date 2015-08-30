require 'nn'
require 'nngraph'
--[[
The following is taken largely from dpnn (https://github.com/nicholas-leonard/dpnn/blob/master/Module.lua).


NOTE: it has NOT been tested on the case where it needs to pick up tensors that are shared; i.e,
 arbitrary tensors that are shared with ones of the name e.g 'weight'.
--]]
local Module = torch.getmetatable('nn.Module')

function Module:sharedCloneManyTimes(ntimes,...)
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

local Linear = torch.getmetatable('nn.Linear')

function Linear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      self.output:resize(nframe, self.bias:size(1))
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   elseif input:dim() == 3 then
      local nframe = input:size(1) * input:size(2)
      self.output:resize(input:size(1), input:size(2), self.bias:size(1))
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:view(nframe,-1):addmm(0, self.output:view(nframe, -1), 1, input:view(nframe, -1), self.weight:t())
      self.output:view(nframe,-1):addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      elseif input:dim() == 3 then
         local nframe = input:size(1)*input:size(2)
         self.gradInput:view(nframe, -1):addmm(0, 1, gradOutput:view(nframe, -1), self.weight)
      end

      return self.gradInput
   end
end
function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   elseif input:dim() == 3 then
      local nframe = input:size(1)*input:size(2)
      self.gradWeight:addmm(scale, gradOutput:view(nframe, -1):t(), input:view(nframe, -1))
      self.gradBias:addmv(scale, gradOutput:view(nframe, -1):t(), self.addBuffer)
   end
end
-- we do not need to accumulate parameters when sharing
local BatchNarrow, parent = torch.class('nn.BatchNarrow','nn.Module')

function BatchNarrow:__init(dimension,offset,length, expected)
   parent.__init(self)
   self.dimension=dimension
   self.index=offset
   self.length=length or 1
   self.expected = expected
   if not dimension or not offset then
      error('nn.BatchNarrow(dimension, offset, length)')
   end
end

function BatchNarrow:updateOutput(input)
   if input:dim() == self.expected + 1 then self.useDimension = self.dimension + 1 else self.useDimension = self.dimension end
   if self.length ==-1 then self.useLength = input:size(self.useDimension) +1 - self.index else self.useLength = self.length end
   local output=input:narrow(self.useDimension,self.index,self.useLength)
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function BatchNarrow:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)  
   self.gradInput:zero()
   self.gradInput:narrow(self.useDimension,self.index,self.useLength):copy(gradOutput)
   return self.gradInput
end 



local BidirectionalLaststate, parent = torch.class('nn.BidirectionalLaststate', 'nn.Module')

function BidirectionalLaststate:__init(axis, expected)
  parent.__init(self)
  self.axis=axis
  self.expected = expected
end


function BidirectionalLaststate:updateOutput(input)
  if input:dim() == self.expected + 1 then useaxis = self.axis+1 else useaxis = self.axis end
  hs = input:size(input:dim())
  assert(hs%2 == 0, "bidir laststate didnt get an even size hidden state")
  lastdim = input:dim()-1
  firstoutput = input:select(useaxis,input:size(useaxis)):narrow(lastdim, 1, hs/2)
  secondoutput = input:select(useaxis,1):narrow(lastdim, (hs/2) + 1, hs/2)
  self.output:resize(input:select(useaxis,1):size())
  self.output:narrow(lastdim, 1,hs/2):copy(firstoutput)
  self.output:narrow(lastdim, (hs/2)+1, hs/2):copy(secondoutput)
  return self.output
end

function BidirectionalLaststate:updateGradInput(input, gradOutput)
  if input:dim() == self.expected + 1 then useaxis = self.axis+1 else useaxis = self.axis end
  self.gradInput = torch.zeros(input:size())
  lastdim = input:dim() - 1
  hs = input:size(input:dim())
  self.gradInput:select(useaxis, self.gradInput:size(useaxis)):narrow(lastdim, 1, hs/2):copy(gradOutput:narrow(lastdim, 1, hs/2))
  self.gradInput:select(useaxis, 1):narrow(lastdim, (hs/2) + 1, hs/2):copy(gradOutput:narrow(lastdim, (hs/2)+1, hs/2))
  return self.gradInput
end

local Unbatch, parent = torch.class('nn.Unbatch', 'nn.Module')

function Unbatch:__init(axis, expected)
  parent.__init(self)
end

function Unbatch:updateOutput(input)
  assert(input:dim() == 3) --removelater 
  -- copy and resize
  self.output:resizeAs(input):copy(input)
  self.output:resize(input:size(1)*input:size(2), input:size(3))
  return self.output
end

function Unbatch:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:resize(input:size())
  return self.gradInput
end



local Laststate, parent = torch.class('nn.Laststate', 'nn.Module')

function Laststate:__init(axis, expected)
  parent.__init(self)
  self.axis=axis
  self.expected = expected
end

function Laststate:updateOutput(input)
  if input:dim() == self.expected + 1 then useaxis = self.axis+1 else useaxis = self.axis end
  self.output = input:select(useaxis, input:size(useaxis))
  return self.output
end

function Laststate:updateGradInput(input, gradOutput)
  if input:dim() == self.expected + 1 then useaxis = self.axis+1 else useaxis = self.axis end
  self.gradInput:resizeAs(input):zero()
  self.gradInput:select(useaxis, self.gradInput:size(useaxis)):copy(gradOutput)
  return self.gradInput
end



