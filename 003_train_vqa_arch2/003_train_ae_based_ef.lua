-- Code to finetune multimodal autoencoder architecture 2 trained on BookCorpus on VQA data

require 'nn'
require 'torch'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
cjson=require('cjson') 
LSTM_encoder=require 'misc.LSTM_encoder'
require 'misc.Encoder_lstm'
require 'misc.AutoEncoder'
net_utils = require 'misc.net_utils'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_h5','data_img.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-input_ae_model', '', 'path to AE model')

-- Model parameter settings
cmd:option('-drop_prob_ae', 0.5, 'dropout value')
cmd:option('-learning_rate',1e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-max_iters', 25000, 'max number of iterations to run for ')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

--check point
cmd:option('-save_checkpoint_every', 5000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'models_vqa/', 'folder to save checkpoints')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

opt = cmd:parse(arg)
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------

local ae_path = opt.input_ae_model
local batch_size=opt.batch_size

local modelT = torch.load(ae_path).protos
local encoder = modelT.ae.encoder

local nlstm_layers_q=1
local nhimage=2048+4096
local noutput = opt.num_output
local dummy_output_size=1
local decay_factor = 0.99997592083 -- 50000

local model_path = opt.checkpoint_path

paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()

dataset['question_val'] = h5_file:read('/ques_val'):all()
dataset['lengths_q_val'] = h5_file:read('/ques_length_val'):all()
dataset['img_list_val'] = h5_file:read('/img_pos_val'):all()
dataset['answers_val'] = h5_file:read('/answers_val'):all()

h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_train'):all()
dataset['fv_im_val'] = h5_file:read('/images_val'):all()
h5_file:close()

-- Normalize the image feature
if opt.img_norm == 1 then

	local nm_1 = torch.sqrt( torch.sum( torch.cmul( dataset['fv_im'][{{},{1,2048}}], dataset['fv_im'][{{},{1,2048}}] ),2 ) ) 
    dataset['fv_im'][{{},{1,2048}}] = torch.cdiv( dataset['fv_im'][{{},{1,2048}}], torch.repeatTensor(nm_1,1,2048) ):float() 
    local nm_2 = torch.sqrt( torch.sum( torch.cmul( dataset['fv_im'][{{},{2048+1,2048+4096}}], dataset['fv_im'][{{}, {2048+1,2048+4096}}] ),2 ) ) 
    dataset['fv_im'][{{},{2048+1,2048+4096}}] = torch.cdiv( dataset['fv_im'][{{},{2048+1,2048+4096}}], torch.repeatTensor(nm_2,1,4096) ):float() 

    local nm_val_1 = torch.sqrt( torch.sum( torch.cmul( dataset['fv_im_val'][{{},{1,2048}}], dataset['fv_im_val'][{{},{1,2048}}] ), 2 ) )   
    dataset['fv_im_val'][{{},{1,2048}}] = torch.cdiv(dataset['fv_im_val'][{{},{1,2048}}],torch.repeatTensor(nm_val_1,1,2048)):float() 
    local nm_val_2 = torch.sqrt( torch.sum( torch.cmul( dataset['fv_im_val'][{{},{2048+1,2048+4096}}], dataset['fv_im_val'][{{},{2048+1,2048+4096}}] ), 2 ) )   
    dataset['fv_im_val'][{{},{2048+1,2048+4096}}] = torch.cdiv(dataset['fv_im_val'][{{},{2048+1,2048+4096}}],torch.repeatTensor(nm_val_2,1,4096)):float() 

end

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

collectgarbage() 

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

--Network definitions
--VQA
--embedding: word-embedding

local EncOpt = {}
EncOpt.vocab_size = vocabulary_size_q
EncOpt.input_encoding_size = modelT.ae.input_encoding_size
EncOpt.rnn_size = modelT.ae.rnn_size
EncOpt.num_layers = modelT.ae.num_layers
EncOpt.dropout = opt.drop_prob_ae
EncOpt.seq_length = buffer_size_q

encoder_model = nn.Encoder(EncOpt)
encoder_model.encoder = modelT.ae.encoder:clone()
encoder_model.lookup_table = modelT.ae.lookup_table:clone()

enc_modules = encoder_model:getModulesList()
for k,v in pairs(enc_modules) do net_utils.unsanitize_gradients(v) end
net_utils.unsanitize_gradients(encoder_model)

local backend

if opt.backend == 'cudnn' then
	require 'cudnn'
	backend = cudnn
elseif backend == 'nn' then
	require 'nn'
	backend = nn
end

multimodal_net=nn.Sequential()
				:add(nn.Dropout(0.5))
				:add(nn.Linear(EncOpt.rnn_size,noutput))

cnn_projection = nn.Sequential():add(nn.Linear(nhimage, EncOpt.input_encoding_size))

--criterion
criterion=nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	encoder_model = encoder_model:cuda()
	cnn_projection = cnn_projection:cuda()
	multimodal_net = multimodal_net:cuda()
	criterion = criterion:cuda()
end

--Processings
--embedding_w_q:uniform(-0.08, 0.08) 

cnn_w, cnn_dw=cnn_projection:getParameters()
cnn_w:uniform(-0.08, 0.08)

encoder_w_q,encoder_dw_q=encoder_model:getParameters() 
--encoder_w_q:uniform(-0.08, 0.08) 
multimodal_w,multimodal_dw=multimodal_net:getParameters() 
multimodal_w:uniform(-0.08, 0.08) 

sizes={cnn_w:size(1), encoder_w_q:size(1),multimodal_w:size(1)} 

-- optimization parameter
local optimize={} 
optimize.maxIter=opt.max_iters 
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1 
optimize.beta1 = 0.8
optimize.beta2 = 0.999
optim.epsilon = 1e-8

optimize.winit=join_vector({cnn_w, encoder_w_q,multimodal_w}) 

------------------------------------------------------------------------
-- Next batch for train
-----------------------------------------------------------------------
function dataset:next_batch()

	local qinds=torch.LongTensor(batch_size):fill(0) 
	local iminds=torch.LongTensor(batch_size):fill(0) 	
	
	local nqs=dataset['question']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=1,batch_size do
		qinds[i]=torch.random(nqs) 
		iminds[i]=dataset['img_list'][qinds[i]] 
	end

	local fv_q=dataset['question']:index(1,qinds):t()
	local fv_im=dataset['fv_im']:index(1,iminds) 
	local labels=dataset['answers']:index(1,qinds) 
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_q = fv_q:cuda()
		fv_im = fv_im:cuda()
		labels = labels:cuda()
	end

	return fv_q,fv_im, labels ,batch_size 
end

------------------------------------------------------------------------
-- Next batch for val
------------------------------------------------------------------------
function dataset:next_batch_val(val_count)

	local batch_size_curr = batch_size
	if val_count+batch_size > dataset['img_list_val']:size(1) then
		batch_size_curr = dataset['img_list_val']:size(1)-val_count
	end

	local qinds=torch.LongTensor(batch_size_curr):fill(0) 
	local iminds=torch.LongTensor(batch_size_curr):fill(0) 	
	
	if not val_count then
		val_count = 0
	end

	local nqs=dataset['question_val']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=val_count+1,val_count+batch_size_curr do
		qinds[i-val_count]=i 
		iminds[i-val_count]=dataset['img_list_val'][qinds[i-val_count]] 
	end

	local fv_q=dataset['question_val']:index(1,qinds):t()
	local fv_im=dataset['fv_im_val']:index(1,iminds) 
	local labels=dataset['answers_val']:index(1,qinds) 
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_q = fv_q:cuda()
		fv_im = fv_im:cuda()
		labels = labels:cuda()
	end
	
	val_count = val_count + batch_size_curr
	return fv_q,fv_im, labels ,batch_size_curr 
end
------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- duplicate the RNN
encoder_model:createClones()

-- Objective function
function JdJ(x)

	cnn_projection:training()
	encoder_model:training() 
	multimodal_net:training()

	local params=split_vector(x,sizes) 
	--load x to net parameters--

	if cnn_w ~= params[1] then
		cnn_w:copy(params[1])
	end

	if encoder_w_q~=params[2] then
		encoder_w_q:copy(params[2]) 
	end
	
	if multimodal_w~=params[3] then
		multimodal_w:copy(params[3]) 
	end

	--clear gradients--
	encoder_dw_q:zero()
	cnn_dw:zero()
	multimodal_dw:zero() 
	
	--grab a batch--
	local fv_q,fv_im,labels,batch_size=dataset:next_batch() 
	local question_max_length=fv_q:size(1) 

	--cnn projection forward
	local fv_im_proj = cnn_projection:forward(fv_im)
	--encoder forward--
	local states_q = encoder_model:forward{fv_im_proj, fv_q}
	
	--multimodal/criterion forward--
	local scores=multimodal_net:forward(states_q) 
	local f=criterion:forward(scores,labels) 
	--multimodal/criterion backward--
	local dscores=criterion:backward(scores,labels) 

	local tmp=multimodal_net:backward(states_q,dscores) 
	
	--encoder backward
	local dimgs_proj = encoder_model:backward({fv_img_proj, fv_q}, tmp) 	
	local dimgs = cnn_projection:backward(fv_im, dimgs_proj[1])
	--summarize f and gradient
	local encoder_adw_q=encoder_dw_q:clone()
	
	gradients=join_vector({cnn_dw, encoder_adw_q,multimodal_dw}) 
	gradients:clamp(-10,10) 
	if running_avg == nil then
		running_avg = f
	end
	running_avg=running_avg*0.95+f*0.05 
	return f,gradients 
end

-- Validation function
function validate()

	encoder_model:evaluate() 
	multimodal_net:evaluate()
	cnn_projection:evaluate()

	--grab a batch--
	local count_val = 0
	local f_avg = 0
	local itersVal = 0

	while count_val < dataset['img_list_val']:size(1) do
		
		xlua.progress(count_val, dataset['img_list_val']:size(1))
		local fv_q,fv_im,labels,batch_size=dataset:next_batch_val(count_val) 
		local question_max_length=fv_q:size(1) 

		--cnn projection forward--
		local fv_im_proj = cnn_projection:forward(fv_im)

		--encoder forward--
		local states_q = encoder_model:forward{fv_im_proj, fv_q}
		
		--multimodal/criterion forward--
		local scores=multimodal_net:forward(states_q) 
		local f=criterion:forward(scores,labels) 
		
		if running_avg_val == nil then
			running_avg_val = f
		end
		running_avg_val=running_avg_val*0.95+f*0.05 
		f_avg = f_avg + f
		itersVal = itersVal + 1
		count_val = count_val + batch_size
	end

	encoder_model:training() 
	multimodal_net:training()
	cnn_projection:training()

	f_avg = f_avg/itersVal
	return f_avg
end

----------------------------------------------------------------------------------------------
-- Training
----------------------------------------------------------------------------------------------
-- With current setting, the network seems never overfitting, so we just use all the data to train

paths.mkdir(model_path..'save')
fileLogger = io.open(model_path..'save/logFile.txt', "w")
fileLoggerVal = io.open(model_path..'save/logFileVal.txt', 'w')

local state={}
for iter = 1, opt.max_iters do
	if iter%opt.save_checkpoint_every == 0 then
		--paths.mkdir(model_path..'save')
		local loss_val = validate()
		fileLoggerVal:write('validation loss: ' .. loss_val .. ' validation loss avg: ' .. running_avg_val .. ' on iter: ' .. iter .. '/' .. opt.max_iters .. '\n')
		print('validation loss: ' .. loss_val .. ' validation loss avg: ' .. running_avg_val .. 'on iter: ' .. iter .. '/' .. opt.max_iters)

		torch.save(string.format(model_path..'save/lstm_save_iter%d.t7',iter),
			{cnn_w=cnn_w, encoder_w_q=encoder_w_q, multimodal_w=multimodal_w}) 
	end

	if iter%100 == 0 then
		local cnn_norm = torch.norm(cnn_w)
		local encoder_norm = torch.norm(encoder_w_q)
		local multimodal_norm = torch.norm(multimodal_w)
		fileLogger:write(string.format('iter: %6d train loss: %.3f cnn_norm: %.3f enc_norm: %.3f mm_norm: %.3f\n', iter, running_avg, cnn_norm, encoder_norm, multimodal_norm))
		print(string.format('iter: %6d train loss: %.3f cnn_norm: %.3f enc_norm: %.3f mm_norm: %.3f', iter, running_avg, cnn_norm, encoder_norm, multimodal_norm))
	end
	optim.rmsprop(JdJ, optimize.winit, optimize, state)
	
	optimize.learningRate=optimize.learningRate*decay_factor 
	if iter%50 == 0 then -- change this to smaller value if out of the memory
		collectgarbage()
	end
end

fileLogger:close()
fileLoggerVal:close()

-- Saving the final model
torch.save(string.format(model_path..'lstm.t7',i),
	{cnn_w=cnn_w, encoder_w_q=encoder_w_q, multimodal_w=multimodal_w}) 
