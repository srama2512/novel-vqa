function optim.rmsprop_lrscale(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-2
   local alpha = config.alpha or 0.99
   local epsilon = config.epsilon or 1e-8
   local wd = config.weightDecay or 0
   local mfill = config.initialMean or 0
   local lrs = config.learningRates or torch.Tensor(x:size()):fill(1):cuda()
  
   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) initialize mean square values and square gradient storage
   if not state.m then
      state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end

   -- (4) calculate new (leaky) mean squared values
   state.m:mul(alpha)
   state.m:addcmul(1.0-alpha, dfdx, dfdx)

   -- (5) perform update
   state.tmp:sqrt(state.m):add(epsilon)
   local tmpUpdate = torch.cmul(torch.cdiv(dfdx, state.tmp), lrs)
   tmpUpdate:mul(-lr)
   x:add(tmpUpdate)

   -- return x*, f(x) before optimization
   return x, {fx}
end

