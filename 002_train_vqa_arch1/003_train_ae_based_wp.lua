require 'nn'
require 'torch'
require 'nngraph'
require 'optim'
require 'misc.netdef'
require 'cutorch'
require 'cunn'
require 'hdf5'
cjson=require('cjson') 
LSTM=require 'misc.LSTM'

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

-- Model parameter settings
cmd:option('-model_path', '', 'loading model parameters')
cmd:option('-learning_rate',1e-4,'learning rate for rmsprop')
cmd:option('-lr_scale', 1, 'learning rate scale for the encoder and embedding layer')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-max_iters', 25000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 512, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size', 512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer', 1,'number of the rnn layer')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')

--check point
cmd:option('-save_checkpoint_every', 5000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')

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

local model_path = opt.checkpoint_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=4096
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1
local decay_factor = 0.99997592083 -- 50000
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

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])
dataset['question_val'] = right_align(dataset['question_val'],dataset['lengths_q_val'])

-- Normalize the image feature
if opt.img_norm == 1 then
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2)) 
	dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,nhimage)):float() 
	local nm_val=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_val'],dataset['fv_im_val']),2)) 
	dataset['fv_im_val']=torch.cdiv(dataset['fv_im_val'],torch.repeatTensor(nm_val,1,nhimage)):float() 
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

savedParams = torch.load(opt.model_path)
--Network definitions
--VQA
--embedding: word-embedding
embedding_net_q=nn.Sequential()
				:add(nn.Linear(vocabulary_size_q,embedding_size_q))
				:add(nn.Dropout(0.5))
				:add(nn.Tanh())

--encoder: RNN body
encoder_net_q=LSTM.lstm_conventional(embedding_size_q,lstm_size_q,dummy_output_size,nlstm_layers_q,0.5)

--MULTIMODAL
--multimodal way of combining different spaces
multimodal_net=nn.Sequential()
				:add(netdef.AskipB(2*lstm_size_q*nlstm_layers_q,nhimage,common_embedding_size,0.5))
				:add(nn.Dropout(0.5))
				--:add(nn.Linear(common_embedding_size, noutput))
				
multimodal_w,multimodal_dw=multimodal_net:getParameters() 
multimodal_w:copy(savedParams['multimodal'])
local linearLayer = nn.Linear(common_embedding_size,noutput)
linearLayer.weight:uniform(-0.08, 0.08)
linearLayer.bias:uniform(-0.08, 0.08)
multimodal_net:add(linearLayer:clone())

--criterion
criterion=nn.CrossEntropyCriterion()

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	embedding_net_q = embedding_net_q:cuda()
	encoder_net_q = encoder_net_q:cuda()
	multimodal_net = multimodal_net:cuda()
	criterion = criterion:cuda()
	dummy_state_q = dummy_state_q:cuda()
	dummy_output_q = dummy_output_q:cuda()
end

--Processings

embedding_w_q_, embedding_dw_q_ = embedding_net_q:parameters()

embedding_w_q_[1]:copy(savedParams['lookup'][{{},{1,savedParams['lookup']:size(2)-1}}])
embedding_w_q_[2]:fill(0)

embedding_w_q,embedding_dw_q=embedding_net_q:getParameters() 

encoder_w_q,encoder_dw_q=encoder_net_q:getParameters() 
encoder_w_q:copy(savedParams['encoder'])

multimodal_w,multimodal_dw=multimodal_net:getParameters() 
-- TODO: REMOVE THIS NEXT TIME BEFORE TRAINING
--encoder_w_q:uniform(-0.08, 0.08)
--embedding_w_q:uniform(-0.08, 0.08)

sizes={encoder_w_q:size(1),embedding_w_q:size(1),multimodal_w:size(1)} 


-- optimization parameter
local optimize={} 
optimize.maxIter=opt.max_iters 
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1 

optimize.winit=join_vector({encoder_w_q,embedding_w_q,multimodal_w}) 


------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch()

	local qinds=torch.LongTensor(batch_size):fill(0) 
	local iminds=torch.LongTensor(batch_size):fill(0) 	
	
	local nqs=dataset['question']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=1,batch_size do
		qinds[i]=torch.random(nqs) 
		iminds[i]=dataset['img_list'][qinds[i]] 
	end


	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question']:index(1,qinds),dataset['lengths_q']:index(1,qinds),vocabulary_size_q) 
	local fv_im=dataset['fv_im']:index(1,iminds) 
	local labels=dataset['answers']:index(1,qinds) 
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q[1]=fv_sorted_q[1]:cuda() 
		fv_sorted_q[3]=fv_sorted_q[3]:cuda() 
		fv_sorted_q[4]=fv_sorted_q[4]:cuda() 
		fv_im = fv_im:cuda()
		labels = labels:cuda()
	end

	return fv_sorted_q,fv_im, labels ,batch_size 
end

------------------------------------------------------------------------
-- Next batch for val
------------------------------------------------------------------------
function dataset:next_batch_val(val_count)

	local batch_size_curr = batch_size

	if val_count+batch_size > dataset['question_val']:size(1) then
		batch_size_curr = dataset['question_val']:size(1)-val_count
	end
	--print('Batch Size Curr: ' .. batch_size_curr)
	local qinds=torch.LongTensor(batch_size_curr):fill(0) 
	local iminds=torch.LongTensor(batch_size_curr):fill(0) 	

	local nqs=dataset['question_val']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=val_count+1,val_count+batch_size_curr do
		qinds[i-val_count]=i 
		iminds[i-val_count]=dataset['img_list_val'][qinds[i-val_count]] 
	end
	
	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question_val']:index(1,qinds),dataset['lengths_q_val']:index(1,qinds),vocabulary_size_q) 
	local fv_im=dataset['fv_im_val']:index(1,iminds) 
	local labels=dataset['answers_val']:index(1,qinds) 
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q[1]=fv_sorted_q[1]:cuda() 
		fv_sorted_q[3]=fv_sorted_q[3]:cuda() 
		fv_sorted_q[4]=fv_sorted_q[4]:cuda() 
		
		fv_im = fv_im:cuda()
		labels = labels:cuda()
	end
	return fv_sorted_q,fv_im, labels ,batch_size 
end


------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- duplicate the RNN
local encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q) 

-- Objective function
function JdJ(x)
	local params=split_vector(x,sizes) 
	--load x to net parameters--
	if encoder_w_q~=params[1] then
		encoder_w_q:copy(params[1]) 
		for i=1,buffer_size_q do
			encoder_net_buffer_q[2][i]:copy(params[1]) 
		end
	end
	if embedding_w_q~=params[2] then
		embedding_w_q:copy(params[2]) 
	end
	if multimodal_w~=params[3] then
		multimodal_w:copy(params[3]) 
	end

	--clear gradients--
	for i=1,buffer_size_q do
		encoder_net_buffer_q[3][i]:zero() 
	end
	embedding_dw_q:zero() 
	multimodal_dw:zero() 

	--grab a batch--
	local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch() 
	local question_max_length=fv_sorted_q[2]:size(1) 

	--embedding forward--
	local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]) 

	--encoder forward--
	local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]) 
	
	--multimodal/criterion forward--
	local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]) 
	local scores=multimodal_net:forward({tv_q,fv_im}) 
	local f=criterion:forward(scores,labels) 
	--multimodal/criterion backward--
	local dscores=criterion:backward(scores,labels) 

	local tmp=multimodal_net:backward({tv_q,fv_im},dscores) 
	local dtv_q=tmp[1]:index(1,fv_sorted_q[3]) 
	
	--encoder backward
	local junk4,dword_embedding_q=rnn_backward(encoder_net_buffer_q,dtv_q,dummy_output_q,states_q,word_embedding_q,fv_sorted_q[2]) 

	--embedding backward--
	dword_embedding_q=join_vector(dword_embedding_q) 
	embedding_net_q:backward(fv_sorted_q[1],dword_embedding_q) 
		
	--summarize f and gradient
	local encoder_adw_q=encoder_dw_q:clone():zero()
	for i=1,question_max_length do
		encoder_adw_q=encoder_adw_q+encoder_net_buffer_q[3][i] 
	end

	gradients=join_vector({torch.mul(encoder_adw_q, opt.lr_scale),torch.mul(embedding_dw_q, opt.lr_scale),multimodal_dw}) 
	gradients:clamp(-10,10) 
	if running_avg == nil then
		running_avg = f
	end
	running_avg=running_avg*0.95+f*0.05 
	return f,gradients 
end

function validate()

	encoder_net_q:evaluate()
	embedding_net_q:evaluate()
	multimodal_net:evaluate()

	--grab a batch--
	local count_val = 0
	local f_avg = 0
	local itersVal = 0
	while count_val < dataset['img_list_val']:size(1) do 
	
		xlua.progress(count_val, dataset['img_list_val']:size(1))
		local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch_val(count_val) 
		
		local question_max_length=fv_sorted_q[2]:size(1) 

		--embedding forward--
		local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]) 

		--encoder forward--
		local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]) 
		
		--multimodal/criterion forward--
		local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]) 
		local scores=multimodal_net:forward({tv_q,fv_im}) 
		local f=criterion:forward(scores,labels) 
					
		if running_avg_val == nil then
			running_avg_val = f
		end
		running_avg_val=running_avg_val*0.95+f*0.05
		f_avg = f_avg + f
		itersVal = itersVal + 1
		count_val = count_val + batch_size
		collectgarbage()
	end
	
	encoder_net_q:training()
	embedding_net_q:training()
	multimodal_net:training()

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
for iter = 0, opt.max_iters do
	if iter%opt.save_checkpoint_every == 0 then
		paths.mkdir(model_path..'save')
		
		local loss_val = validate()
		fileLoggerVal:write('validation loss: ' .. loss_val .. ' validation loss avg: ' .. running_avg_val, ' on iter: ' .. iter .. '/' .. opt.max_iters .. '\n')
		print('validation loss: ' .. loss_val .. ' validation loss avg: ' .. running_avg_val .. ' on iter: ' .. iter .. '/' .. opt.max_iters .. '\n')

		
		torch.save(string.format(model_path..'save/lstm_save_iter%d.t7',iter),
			{encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
	end
	if iter%100 == 0 and iter > 0 then
		fileLogger:write('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters .. '\n')
		print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
	end
	optim.rmsprop(JdJ, optimize.winit, optimize, state)
	
	optimize.learningRate=optimize.learningRate*decay_factor 
	if iter%50 == 0 then -- change this to smaller value if out of the memory
		collectgarbage()
	end
end

fileLogger:close()

-- Saving the final model
torch.save(string.format(model_path..'lstm.t7',i),
	{encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
torch.save(string.format(model_path..'params.t7'), opt)
