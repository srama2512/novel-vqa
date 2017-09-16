-- Code to train AutoEncoder model on weak paired text and image data for architecture 2

require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoaderWeakPaired'
require 'misc.AutoEncoderNull'
require 'misc.optim_updates'
require 'loadcaffe'
require 'misc/L2Normalize.lua'

local net_utils = require 'misc.net_utils'
math.randomseed(123)

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an AutoEncoder model for sentences')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/data.json','path to the json file containing additional info and vocab')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-start_from_text', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-cnn_model','/home/santhosh/Projects/VQA/CVPR-Work/googlenet/inception-v3.torch/inceptionv3.net','path to inception CNN model')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')

-- Optimization: General
cmd:option('-max_iters', 25001, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',20,'what is the batch size in number of sentences per batch?')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_ae', 0.5, 'strength of dropout in the AutoEncoder RNN')

-- Optimization: for the AutoEncoder Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',3e-5,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay', 1e-6, 'L2 regularization parameter')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_sentences_use', 30000, 'how many sentences to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 10000, 'how often to save a model checkpoint?')
cmd:option('-save_model_every', 12500, 'saving a model every half epoch')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  local ae_modules = protos.ae:getModulesList()
  for k,v in pairs(ae_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
else
  -- create protos from scratch
  -- intialize autoencoder

  --[[  
  local aeOpt = {}
  aeOpt.vocab_size = loader:getVocabSize()
  aeOpt.input_encoding_size = opt.input_encoding_size
  aeOpt.rnn_size = opt.rnn_size
  aeOpt.num_layers = 1
  aeOpt.dropout = opt.drop_prob_ae
  aeOpt.seq_length = loader:getSeqLength()
  aeOpt.batch_size = opt.batch_size
  protos.ae = nn.AutoEncoder(aeOpt)
  --]]
  protos.ae = torch.load(opt.start_from_text).protos.ae
  local ae_modules = protos.ae:getModulesList()
  for k,v in pairs(ae_modules) do net_utils.unsanitize_gradients(v) end

  protos.crit = nn.LanguageModelCriterion()
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = torch.load(opt.cnn_model)
  protos.cnn = net_utils.build_cnn_inception(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
  cnn_raw = nil
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do
	  v:cuda()
	  print('Converted to GPU')
  end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.ae:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in AutoEncoder: ', params:nElement())
assert(params:nElement() == grad_params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

-- Setting layerwise learning rate scales for VGGNetEmbed
local cnnFinalLearningScales = torch.Tensor(cnn_params:nElement()):fill(1)
if opt.gpuid >= 0 then
    cnnFinalLearningScales = cnnFinalLearningScales:cuda()
end

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_ae = protos.ae:clone()
thin_ae.encoder:share(protos.ae.encoder, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_ae.decoder:share(protos.ae.decoder, 'weight', 'bias')
thin_ae.lookup_table:share(protos.ae.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')

-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local ae_modules = thin_ae:getModulesList()
for k,v in pairs(ae_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.ae:createClones()

collectgarbage() -- "yeah, sure why not"

local sizes = 0 

for i = 1,30 do
    local temp = protos.cnn:get(i):getParameters():size()
    if temp:size() ~= 0 then
        print(i .. ' ' .. temp[1])
        sizes = sizes + temp[1]
    end 
end

cnnFinalLearningScales[{{1, sizes}}]:fill(0.01)

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_sentences_use = utils.getopt(evalopt, 'val_sentences_use', true)

  protos.ae:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  local count_sents = 0
  
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, encoding_size = opt.input_encoding_size}
    data.imgs = net_utils.prepro_inception(data.imgs, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
	n = n + data.imgs:size(1)

	if opt.gpuid >= 0 then
	  data.imgs = data.imgs:cuda()
	  data.labels = data.labels:cuda()
    end
    
	-- forward the model to get loss
    local feats = protos.cnn:forward(data.imgs)
	local logprobs = protos.ae:forward{feats, data.labels, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each sentence
	local seq = protos.ae:sample(feats, data.labels, data.labels)
	local sents = net_utils.decode_sequence(vocab, seq)
	local sents_actual = net_utils.decode_sequence(vocab, data.labels)
	for k=1,#sents do
		count_sents = count_sents + 1
      local entry = {seqNo = count_sents, prediction = sents[k], actual = sents_actual[k]}
	  print('Prediction: ' .. sents[k] .. ' ||| Actual: ' .. sents_actual[k])
      table.insert(predictions, entry)
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_sentences_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_sentences_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 1
local function lossFun()
  protos.ae:training()
  grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end
  
  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', encoding_size = opt.input_encoding_size}
  -- data.seq: LxM where L is sequence length upper bound, and M = # of sentences 
  data.imgs = net_utils.prepro_inception(data.imgs, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation

  if opt.gpuid >= 0 then
	  data.imgs = data.imgs:cuda()
	  data.labels = data.labels:cuda()
  end

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.imgs)
  -- forward the auto-encoder model
  local sentInput = data.labels:clone()
  local flag = 0
  if math.random() <= 0.5 then
	  sentInput:fill(0)
	  flag = 1
  end
  local logprobs = protos.ae:forward{feats, sentInput, data.labels}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.labels)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop auto-encoder model
  local dfeats, ddummy = unpack(protos.ae:backward({feats, sentInput, data.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
	
    local dx = protos.cnn:backward(data.imgs, dfeats)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.weight_decay > 0 then
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    grad_params:add(opt.weight_decay, params)
  end
  
-- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, torch.cmul(cnn_params, cnnFinalLearningScales))
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  
  local params_norm = torch.norm(params)
  local num_updates = torch.sum(torch.ge(torch.abs(torch.mul(grad_params, opt.learning_rate)), torch.mul(torch.abs(params), 0.01)))
  local cnn_params_norm = torch.norm(protos.cnn:get(32).weight)
  local num_updates_cnn = torch.sum(torch.ge(torch.abs(torch.mul(cnn_grad_params[{{sizes+1,cnn_params:size(1)}}] , opt.cnn_learning_rate)), torch.mul(torch.abs(cnn_params[{{sizes+1,cnn_params:size(1)}}]), 0.01)))

  print(string.format('iter %d: loss: %.3f | # updates: %4d | paramsNorm: %.4f | # cnn updates: %4d | cnnParamsNorm: %.4f', iter, losses.total_loss, num_updates, params_norm, num_updates_cnn, cnn_params_norm))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_sentences_use = opt.val_sentences_use})
    print('validation loss: ', val_loss)
    print(lang_stats)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.ae = thin_ae -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
		checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote BEST checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
	

  end

  if iter % opt.save_model_every == 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
		local checkpoint = {}

        save_protos.ae = thin_ae -- these are shared clones, and point to correct param storage
		save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
		local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)
        torch.save(checkpoint_path .. iter ..'.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. iter .. '.t7')
  end
  --]]
  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate

  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
	cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate, cnnFinalLearningScales)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state, cnnFinalLearningScales)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state, cnnFinalLearningScales)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
--]]
