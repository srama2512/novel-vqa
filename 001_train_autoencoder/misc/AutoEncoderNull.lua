-- This LSTM based autoencoder model takes in image and text to produce text

require 'nn'
local utils = require 'misc.utils'
local LSTM_encoder = require 'misc.LSTM_encoder'
local LSTM_decoder = require 'misc.LSTM_decoder'

-------------------------------------------------------------------------------
-- AutoEncoder Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.AutoEncoder', 'nn.Module')
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
  self.decoder = LSTM_decoder.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)

  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state_enc then self.init_state_enc = {} end -- lazy init
  if not self.init_state_dec then self.init_state_dec = {} end -- lazy init

  for h=1,self.num_layers*2 do -- FOR LSTM
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state_enc[h] then
      if self.init_state_enc[h]:size(1) ~= batch_size then
        self.init_state_enc[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state_enc[h] = torch.zeros(batch_size, self.rnn_size)
    end
  
    if self.init_state_dec[h] then
	  if self.init_state_dec[h]:size(1) ~= batch_size then
        self.init_state_dec[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state_dec[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  
  self.num_state = #self.init_state_enc
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the AutoEncoder')
  self.clones_encoder = {self.encoder}
  self.clones_decoder = {self.decoder}
  self.lookup_tables_encoder = {self.lookup_table:clone('weight')}
  self.lookup_tables_decoder = {self.lookup_table:clone('weight')}

  for t=2,self.seq_length+2 do
	print('t: '..t)
    self.clones_encoder[t] = self.encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
	self.lookup_tables_encoder[t] = self.lookup_tables_encoder[1]:clone('weight', 'gradWeight')
    
	-- decoder has only self.seq_length+1 time steps
	if t <= self.seq_length+1 then
      self.clones_decoder[t] = self.decoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
	  self.lookup_tables_decoder[t] = self.lookup_tables_decoder[1]:clone('weight', 'gradWeight')
	end

  end
end

function layer:getModulesList()
  return {self.encoder, self.decoder, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.encoder:parameters()
  local p2,g2 = self.decoder:parameters()
  --local p3,g3 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  --for k,v in pairs(p3) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  --for k,v in pairs(g3) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if (self.clones_encoder == nil or self.clones_decoder == nil) then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones_encoder) do v:training() end
  for k,v in pairs(self.clones_decoder) do v:training() end
  for k,v in pairs(self.lookup_tables_encoder) do v:evaluate() end
  for k,v in pairs(self.lookup_tables_decoder) do v:evaluate() end
end

function layer:evaluate()
  if (self.clones_encoder == nil or self.clones_decoder == nil) then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones_encoder) do v:evaluate() end
  for k,v in pairs(self.lookup_tables_encoder) do v:evaluate() end
  for k,v in pairs(self.clones_decoder) do v:evaluate() end
  for k,v in pairs(self.lookup_tables_decoder) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, sents_input, sents, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  local batch_size = sents:size(2)
  self:_createInitState(batch_size)
  local state_enc = self.init_state_enc

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  
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
	  it = sents_input[t-2]:clone()
      xt = self.lookup_table:forward(it)
    end

    local inputs = {xt,unpack(state_enc)}
	local out = self.encoder:forward(inputs) -- For LSTM, and GRU with more than 1 layers
    state_enc = {}
    for i=1,self.num_state do table.insert(state_enc, out[i]) end
  end

  -- the state of the encoder after the sequence is inputted is the initial state of the decoder
  state_dec = state_enc
  -- decoder forward pass
  for t=1,self.seq_length+1 do

    local xt, it, sampleLogprobs
    if t == 1 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
    end

    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    local inputs = {xt,unpack(state_dec)}
    local out = self.decoder:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state_dec = {}
    for i=1,self.num_state do table.insert(state_dec, out[i]) end
  end


  -- return the samples and their log likelihoods
  return seq, seqLogprobs
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
  local seq_input = input[2]  
  local seq = input[3]
  
  if (self.clones_encoder == nil or self.clones_decoder == nil) then self:createClones() end -- lazily create clones on first forward pass
  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  
  -- the encoder outputs the hidden states for each layer after all the sequence has been processed
  self.output_enc = {}
  -- the decoder outputs the probabilities for each word for each time instance
  if not self.output_dec then
	self.output_dec = torch.Tensor(self.seq_length+1, batch_size, self.vocab_size+1)
	if seq:type() == 'torch.CudaTensor' then
	  self.output_dec = self.output_dec:cuda()
	end
  else
	self.output_dec:resize(self.seq_length+1, batch_size, self.vocab_size+1)
  end
  self:_createInitState(batch_size)

  self.state_enc = {[0] = self.init_state_enc} 
  self.inputs_enc = {}
  self.inputs_dec = {}
  self.lookup_tables_inputs_enc = {}
  self.lookup_tables_inputs_dec = {}
  self.tmax_enc = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  self.tmax_dec = 0
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
      local it = seq_input[t-2]:clone()
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
	  --print('inputs: ')
	  --print(self.inputs_enc[t])
      local out = self.clones_encoder[t]:forward(self.inputs_enc[t]) -- For LSTM and GRU with more than 1 layer
	  
	  -- process the outputs
      self.state_enc[t] = {} -- everything is state
      for i=1,self.num_state do table.insert(self.state_enc[t], out[i]) end
	  --print('state_enc @ end of t = ' .. t)
	  --print(self.state_enc[t])
      self.tmax_enc = t
    end
  end

  -- output of the encoder consists of the hidden state of the layers at tmax
  self.output_enc = self.state_enc[self.tmax_enc]
  self.init_state_dec = self.state_enc[self.tmax_enc]
  -- the final encoder hidden state is used to initialize the decoder
  -- the generation then starts with this hidden state and start token
  self.state_dec = {[0] = self.init_state_dec}
  -- decoder forward pass
  for t=1,self.seq_length+1 do

    local can_skip = false
    local xt
    if t == 1 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs_dec[t] = it
      xt = self.lookup_tables_decoder[t]:forward(it) -- NxK sized input (token embedding vectors)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-1]:clone()
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
        self.lookup_tables_inputs_dec[t] = it
        xt = self.lookup_tables_decoder[t]:forward(it)
      end
    end

    if not can_skip then
      -- construct the inputs
      self.inputs_dec[t] = {xt,unpack(self.state_dec[t-1])}
      -- forward the network
      local out = self.clones_decoder[t]:forward(self.inputs_dec[t])
      -- process the outputs
	  self.output_dec[t] = out[self.num_state+1] -- final output is the probabilities of each word
      self.state_dec[t] = {} -- everything else is saved as state
      for i=1,self.num_state do table.insert(self.state_dec[t], out[i]) end
	  self.tmax_dec = t
    end
  end
  return self.output_dec
end

--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
  local dimgs
  -- go backwards and lets compute gradients
  -- backward pass for the decoder
  local dstate_dec = {[self.tmax_dec] = self.init_state_enc} -- this works when init_state_enc is all zeros, theoretically it has no meaning except that it is all zeros 
  
  -- self.tmax is the maximum length of the decoder+1 (because encoder has image input as well)
  for t=self.tmax_dec,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate_dec[t] do table.insert(dout, dstate_dec[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones_decoder[t]:backward(self.inputs_dec[t], dout)
    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate_dec[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate_dec[t-1], dinputs[k]) end
    
    -- continue backprop of xt
    --local it = self.lookup_tables_inputs_dec[t]
    --self.lookup_tables_decoder[t]:backward(it, dxt) -- backprop into lookup table
  end

  -- backward pass for the encoder
  -- the dstate from the decoder at time 1 is wrt to the final state of encoder
  local dstate_enc = {[self.tmax_enc] = dstate_dec[0]}
  for t=self.tmax_enc,1,-1 do
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
      --local it = self.lookup_tables_inputs_enc[t]
      --self.lookup_tables_encoder[t]:backward(it, dxt) -- backprop into lookup table
	end
  end

  -- for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+1)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time, t = 1 is where the start sequence begins 
      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t is correct, since at t=1 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
