require 'torch'
require 'optim'
gradcheck = {}
-- This function is very close to optim's checkgrad, except it acts on x, which is a typical dict of (potentailly non-flat) parameters.
function gradcheck.dictCheckgrad(opfunc, x, eps)
  -- true gradient
  local _, dC = opfunc(x)

  local eps = eps or 1e-7
  local dC_est = {}

  for i=1,#dC do
    local param = x[i]
    local flatparam = param:view(param:nElement())
    local dflatparam_est = dC[i].new(flatparam:size())

    for i=1, flatparam:size(1) do
      flatparam[i] = flatparam[i] + eps
      local C1 = opfunc(x)
      flatparam[i] = flatparam[i] - 2*eps
      local C2 = opfunc(x)
      flatparam[i] = flatparam[i] + eps
      dflatparam_est[i] = (C1 - C2) / (2 * eps)
    end
    table.insert(dC_est, dflatparam_est:resize(param:size()))
  end

  -- error
  local diff = 0
  local diffDict = {}
  for i,v in pairs(dC) do
    local thisdiff = (torch.norm(dC[i] - dC_est[i]) / torch.norm(dC[i] + dC_est[i]))
    table.insert(diffDict, thisdiff)
    diff = diff + thisdiff
  end

  return diff, dC, dC_est, diffDict
end

function gradcheck.moduleDictCheckGrad(mod, criterion, input, target, do_input, do_params)
  local params, grad_params = mod:parameters()

  function feval(x)
    local params, grad_params = mod:parameters()
    for i=1,#params do
      if params[i]~=x[i] then print("asdasdasd");params[i]:copy(x[i]) end
    end
    for i=1,#grad_params do
      grad_params[i]:zero()
    end
    local output = mod:forward(input)
    local cost = criterion:forward(output, target)
    local d = criterion:backward(output, target)
    local gradinput=mod:backward(input,d)
    return cost, grad_params
  end

  if do_params == nil then do_params = true end
  if do_params then
    timer = torch.Timer()
    diff, dc, dc_Est, thisdd = gradcheck.dictCheckgrad(feval, params, 1e-5)
    print ("parameter gradcheck done")
    print(thisdd)
  end

  -- hacky input check grad
  if type(input) ~= 'table' then input = {input} end
  function ifeval(x)
    local tab=false
    if (type(x) == 'table' and #x == 1) then input = x[1];tab=true else input = x end
    output = mod:forward(input)
    cost = criterion:forward(output, target)
    d = criterion:backward(output, target)
    gradx = mod:backward(input, d)
    if tab then gradx = {gradx} end
    return cost, gradx
  end

  if do_input == nil then do_input = true end
  if do_input then
    diff, dc, dc_Est, dd =gradcheck.dictCheckgrad(ifeval, input, 1e-7)
    print ("input gradcheck done")
    print(dd)
  end
  return p_dc, p_dcest
end


return gradcheck
