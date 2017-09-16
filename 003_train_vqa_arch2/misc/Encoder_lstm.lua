-- Implements an LSTM based Encoder with input as image and text

require 'nn'
local utils = require 'misc.utils'
local LSTM_encoder = require 'misc.LSTM_encoder'

-------------------------------------------------------------------------------
-- Encoder Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.Encoder', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  -- create the core lstm network. note +1 for both the START and END tokens
  self.encoder = LSTM_encoder.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)

  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state_enc then self.init_state_enc = {} end -- lazy init

  for h=1,self.num_layers*2 do -- FOR LSTM
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state_enc[h] then
      if self.init_state_enc[h]:size(1) ~= batch_size then
        self.init_state_enc[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state_enc[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  
  self.num_state = #self.init_state_enc
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the Encoder')
  self.clones_encoder = {self.encoder}
  self.lookup_tables_encoder = {self.lookup_table:clone('weight')}

  for t=2,self.seq_length+2 do
	print('t: '..t)
    self.clones_encoder[t] = self.encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
	self.lookup_tables_encoder[t] = self.lookup_tables_encoder[1]:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.encoder, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.encoder:parameters()
  local p3,g3 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)
  return params, grad_params
end

function layer:training()
  if (self.clones_encoder == nil) then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones_encoder) do
	  v:training()
  end
  for k,v in pairs(self.lookup_tables_encoder) do
	  v:training()
  end
end

function layer:evaluate()
  if (self.clones_encoder == nil) then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones_encoder) do v:evaluate() end
  for k,v in pairs(self.lookup_tables_encoder) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, sents, opt)
  
  local batch_size = sents:size(2)
  self:_createInitState(batch_size)
  local state_enc = self.init_state_enc

  -- encoder forward pass
  for t=1,self.seq_length+2 do

    local xt, it, sampleLogprobs
    
	if t == 1 then
	  -- feed in the images
	  xt = imgs
	elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
      -- take sequence inputs for current time step and feed them in
	  it = sents[t-2]:clone()
      xt = self.lookup_table:forward(it)
    end

    local inputs = {xt,unpack(state_enc)}
	local out = self.encoder:forward(inputs) -- For LSTM, and GRU with more than 1 layers
    state_enc = {}
    for i=1,self.num_state do table.insert(state_enc, out[i]) end
  end
 
  return state_enc

end


--[[
input is:  
   torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length
   and N is batch size

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local imgs = input[1]
  local seq = input[2]
  if (self.clones_encoder == nil) then self:createClones() end -- lazily create clones on first forward pass
  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  
  -- the encoder outputs the hidden states for each layer after all the sequence has been processed
  self.output_enc = {}
    
  self:_createInitState(batch_size)

  self.state_enc = {[0] = self.init_state_enc} 
  self.inputs_enc = {}
  self.lookup_tables_inputs_enc = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  
  -- encoder forward pass
  for t=1,self.seq_length+2 do

    local can_skip = false
    local xt
	if t == 1 then
	  -- feed in the images
	  xt = imgs -- NxK sized tensor
	elseif t == 2 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs_enc[t] = it
      xt = self.lookup_tables_encoder[t]:forward(it) -- NxK sized input (token embedding vectors)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-2]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs_enc[t] = it
        xt = self.lookup_tables_encoder[t]:forward(it)
      end
    end

    if not can_skip then

      -- construct the inputs
	  self.inputs_enc[t] = {xt,unpack(self.state_enc[t-1])}
      -- forward the network
	  --print('t: ' .. t)

      local out = self.clones_encoder[t]:forward(self.inputs_enc[t]) -- For LSTM and GRU with more than 1 layer
	  
	  -- process the outputs
      self.state_enc[t] = {} -- everything is state
      for i=1,self.num_state do table.insert(self.state_enc[t], out[i]) end
	  --print('state_enc @ end of t = ' .. t)
	  --print(self.state_enc[t])
      self.tmax = t
    end
  end

  -- output of the encoder consists of the final layer hidden state at tmax
  self.output_enc = self.state_enc[self.tmax][self.num_state]
  
  return self.output_enc
end

--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)

  -- go backwards and lets compute gradients
  
  -- backward pass for the encoder
  -- the dstate from the decoder at time 1 is wrt to the final state of encoder
  local dstate_enc = {[self.tmax] = self.init_state_enc}
  dstate_enc[self.tmax][self.num_state] = gradOutput
  for t=self.tmax,1,-1 do
    -- only state gradients at time step t for encoder
    local dout = {} -- For LSTM, and GRU with more than 2 layers
    for k=1,#dstate_enc[t] do table.insert(dout, dstate_enc[t][k]) end -- FOR LSTM, and GRU with more than 2 layers
	--dout = dstate_enc[t][1] -- For GRU with 1 layer
    local dinputs = self.clones_encoder[t]:backward(self.inputs_enc[t], dout)
    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate_enc[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate_enc[t-1], dinputs[k]) end
  
    -- continue backprop of xt
	if t == 1 then
	  dimgs = dxt
	else
      local it = self.lookup_tables_inputs_enc[t]
      self.lookup_tables_encoder[t]:backward(it, dxt) -- backprop into lookup table
	end
  end

  -- for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end
