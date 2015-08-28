require 'nn'
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
  self.gradInput = torch.zeros(input:size())
  self.gradInput:select(useaxis, self.gradInput:size(useaxis)):copy(gradOutput)
  return self.gradInput
end



