-------------------------------------------------------------------------------
-- Function to compute the prediction vectors for each question and store it
-------------------------------------------------------------------------------

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
cmd:text('Options')

-- Data input settings
cmd:option('-vgg_img_h5','data_img.h5','path to the h5file containing the vgg image feature')
cmd:option('-inception_img_h5','data_img.h5','path to the h5file containing the inception image feature')
cmd:option('-input_ques_h5','data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro.json','path to the json file containing additional info and vocab')

-- Model parameter settings
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-input_encoding_size_vgg', 200, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size_vgg',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size_incep', 200, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size_incep',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',2,'number of the rnn layer')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-vgg_norm', 1, 'normalize the vgg image feature. 1 = normalize, 0 = not normalize')
cmd:option('-inception_norm', 1, 'normalize the inception image feature. 1 = normalize, 0 = not normalize')
cmd:option('-vgg_model_path', 'model/model_default_params/lstm.t7', 'path to VGG model')
cmd:option('-inception_model_path', 'model/model_inception_default_params/lstm.t7', 'path to Inception model')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-out_path', 'outputVectors.h5', 'output file path')

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
-- Loading Dataset
------------------------------------------------------------------------
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

dataset['question_id'] = h5_file:read('/question_id_train'):all()
dataset['question'] = h5_file:read('/ques_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
dataset['question_id_val'] = h5_file:read('/question_id_val'):all()
dataset['question_val'] = h5_file:read('/ques_val'):all()
dataset['lengths_q_val'] = h5_file:read('/ques_length_val'):all()
dataset['img_list_val'] = h5_file:read('/img_pos_val'):all()
dataset['answers_val'] = h5_file:read('/answers_val'):all()
dataset['question_test'] = h5_file:read('/ques_test'):all()
dataset['lengths_q_test'] = h5_file:read('/ques_length_test'):all()
dataset['img_list_test'] = h5_file:read('/img_pos_test'):all()
dataset['ques_id_test'] = h5_file:read('/question_id_test'):all()
dataset['MC_ans_test'] = h5_file:read('/MC_ans_test'):all()

h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])
dataset['question_val'] = right_align(dataset['question_val'],dataset['lengths_q_val'])
dataset['question_test'] = right_align(dataset['question_test'],dataset['lengths_q_test'])

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

collectgarbage() 

------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch(batchNo)
	
	local nqs=dataset['question']:size(1)
	local qinds=torch.LongTensor(math.min(batch_size, nqs-(batchNo-1)*batch_size)):fill(0) 
	local iminds=torch.LongTensor(math.min(batch_size, nqs-(batchNo-1)*batch_size)):fill(0) 	
	
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=(batchNo-1)*batch_size+1, math.min((batchNo*batch_size), nqs)  do
		qinds[i-(batchNo-1)*batch_size]= i
		iminds[i-(batchNo-1)*batch_size]=dataset['img_list'][qinds[i-(batchNo-1)*batch_size]] 
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
function dataset:next_batch_val(batchNo)
	
	local nqs=dataset['question_val']:size(1)
	local qinds=torch.LongTensor(math.min(batch_size, nqs-(batchNo-1)*batch_size)):fill(0) 
	local iminds=torch.LongTensor(math.min(batch_size, nqs-(batchNo-1)*batch_size)):fill(0) 	
	
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=(batchNo-1)*batch_size+1, math.min((batchNo*batch_size), nqs)  do
		qinds[i-(batchNo-1)*batch_size]= i
		iminds[i-(batchNo-1)*batch_size]=dataset['img_list_val'][qinds[i-(batchNo-1)*batch_size]] 
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



function dataset:next_batch_test(batchNo)
	
	local nqs=dataset['question_test']:size(1)
	local qinds=torch.LongTensor(math.min(batch_size, nqs-(batchNo-1)*batch_size)):fill(0) 
	local iminds=torch.LongTensor(math.min(batch_size, nqs-(batchNo-1)*batch_size)):fill(0) 	
	
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=(batchNo-1)*batch_size+1, math.min((batchNo*batch_size), nqs)  do
		qinds[i-(batchNo-1)*batch_size]= i
		iminds[i-(batchNo-1)*batch_size]=dataset['img_list_test'][qinds[i-(batchNo-1)*batch_size]] 
	end

	local fv_sorted_q=sort_encoding_onehot_right_align(dataset['question_test']:index(1,qinds),dataset['lengths_q_test']:index(1,qinds),vocabulary_size_q) 
	local fv_im=dataset['fv_im_test']:index(1,iminds) 

	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q[1]=fv_sorted_q[1]:cuda() 
		fv_sorted_q[3]=fv_sorted_q[3]:cuda() 
		fv_sorted_q[4]=fv_sorted_q[4]:cuda() 
		fv_im = fv_im:cuda()
	end

	return fv_sorted_q,fv_im,batch_size 
end

------------------------------------------------------------------------
-- Function to compute predictions on the training set
------------------------------------------------------------------------
function predictTrain()
	
	local outputVectors = torch.Tensor(dataset['question']:size(1), 1000)
	for iter = 1, math.ceil(dataset['question']:size(1)/batch_size) do
		--grab a batch--
		local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch(iter) 
		local question_max_length=fv_sorted_q[2]:size(1) 

		--embedding forward--
		local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]) 

		--encoder forward--
		local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]) 
		
		--multimodal/criterion forward--
		local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]) 
		local scores=multimodal_net:forward({tv_q,fv_im}) 
		outputVectors[{{(iter-1)*batch_size+1, math.min(iter*batch_size, dataset['question']:size(1))}, {}}] = scores:float()
		
		xlua.progress(math.min(iter*batch_size, dataset['question']:size(1)), dataset['question']:size(1)) 
	end

	return outputVectors
end

------------------------------------------------------------------------
-- Function to compute predictions on the validation set
------------------------------------------------------------------------
function predictVal()
	
	local outputVectors = torch.Tensor(dataset['question_val']:size(1), 1000)
	for iter = 1, math.ceil(dataset['question_val']:size(1)/batch_size) do
		--grab a batch--
		local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch_val(iter) 
		local question_max_length=fv_sorted_q[2]:size(1) 

		--embedding forward--
		local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]) 

		--encoder forward--
		local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]) 
		
		--multimodal/criterion forward--
		local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]) 
		local scores=multimodal_net:forward({tv_q,fv_im}) 
		outputVectors[{{(iter-1)*batch_size+1, math.min(iter*batch_size, dataset['question_val']:size(1))}, {}}] = scores:float()
		
		xlua.progress(math.min(iter*batch_size, dataset['question_val']:size(1)), dataset['question_val']:size(1)) 
	end

	return outputVectors
end

------------------------------------------------------------------------
-- Function to compute predictions on the testing set
------------------------------------------------------------------------
function predictTest()
	
	local outputVectors = torch.Tensor(dataset['question_test']:size(1), 1000)
	for iter = 1, math.ceil(dataset['question_test']:size(1)/batch_size) do
		--grab a batch--
		local fv_sorted_q,fv_im,batch_size=dataset:next_batch_test(iter) 
		local question_max_length=fv_sorted_q[2]:size(1) 

		--embedding forward--
		local word_embedding_q=split_vector(embedding_net_q:forward(fv_sorted_q[1]),fv_sorted_q[2]) 

		--encoder forward--
		local states_q,junk2=rnn_forward(encoder_net_buffer_q,torch.repeatTensor(dummy_state_q:fill(0),batch_size,1),word_embedding_q,fv_sorted_q[2]) 
		
		--multimodal/criterion forward--
		local tv_q=states_q[question_max_length+1]:index(1,fv_sorted_q[4]) 
		local scores=multimodal_net:forward({tv_q,fv_im}) 
		outputVectors[{{(iter-1)*batch_size+1, math.min(iter*batch_size, dataset['question_test']:size(1))}, {}}] = scores:float()
		
		xlua.progress(math.min(iter*batch_size, dataset['question_test']:size(1)), dataset['question_test']:size(1)) 
	end

	return outputVectors
end

------------------------------------------------------------------------
-- Setting the parameters for VGGNet
------------------------------------------------------------------------

local model_path = opt.vgg_model_path
batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size_vgg
local lstm_size_q=opt.rnn_size_vgg
local nlstm_layers_q=opt.rnn_layer
local nhimage=4096
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1
local decay_factor = 0.99997592083 -- 50000

-----------------------------------------------------------------------
-- Loading image features for VGG model
-----------------------------------------------------------------------

print('DataLoader loading h5 file: ', opt.vgg_img_h5)
local h5_file = hdf5.open(opt.vgg_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_train'):all()
dataset['fv_im_val'] = h5_file:read('/images_val'):all()
dataset['fv_im_test'] = h5_file:read('/images_test'):all()
h5_file:close()

-- Normalize the image feature
if opt.vgg_norm == 1 then
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2)) 
	dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,nhimage)):float()
	local nm_val=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_val'],dataset['fv_im_val']),2)) 
	dataset['fv_im_val']=torch.cdiv(dataset['fv_im_val'],torch.repeatTensor(nm_val,1,nhimage)):float()
	local nm_test=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_test'],dataset['fv_im_test']),2)) 
	dataset['fv_im_test']=torch.cdiv(dataset['fv_im_test'],torch.repeatTensor(nm_test,1,nhimage)):float() 
end

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

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
				:add(netdef.AxB(2*lstm_size_q*nlstm_layers_q,nhimage,common_embedding_size,0.5))
				:add(nn.Dropout(0.5))
				:add(nn.Linear(common_embedding_size,noutput))

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	embedding_net_q = embedding_net_q:cuda()
	encoder_net_q = encoder_net_q:cuda()
	multimodal_net = multimodal_net:cuda()
	dummy_state_q = dummy_state_q:cuda()
	dummy_output_q = dummy_output_q:cuda()
end

-- setting to evaluation
embedding_net_q:evaluate();
encoder_net_q:evaluate();
multimodal_net:evaluate();

-- loading the model
model_param=torch.load(model_path);

embedding_w_q,embedding_dw_q=embedding_net_q:getParameters();
encoder_w_q,encoder_dw_q=encoder_net_q:getParameters();
multimodal_w,multimodal_dw=multimodal_net:getParameters();

embedding_w_q:copy(model_param['embedding_w_q']);
encoder_w_q:copy(model_param['encoder_w_q']);
multimodal_w:copy(model_param['multimodal_w']);

-- duplicate the RNN
encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q) 

outputVectorsVGG = predictTrain()
outputVectorsVGGVal = predictVal()
outputVectorsVGGTest = predictTest()

------------------------------------------------------------------------
-- Setting the parameters for InceptionNet
------------------------------------------------------------------------

local model_path = opt.inception_model_path
batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size_incep
local lstm_size_q=opt.rnn_size_incep
local nlstm_layers_q=opt.rnn_layer
local nhimage=2048
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1
local decay_factor = 0.99997592083 -- 50000

-----------------------------------------------------------------------
-- Loading image features for Inception model
-----------------------------------------------------------------------

print('DataLoader loading h5 file: ', opt.inception_img_h5)
local h5_file = hdf5.open(opt.inception_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_train'):all()
dataset['fv_im_val'] = h5_file:read('/images_val'):all()
dataset['fv_im_test'] = h5_file:read('/images_test'):all()
h5_file:close()

-- Normalize the image feature
if opt.inception_norm == 1 then
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2)) 
	dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,nhimage)):float() 
	local nm_val=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_val'],dataset['fv_im_val']),2)) 
	dataset['fv_im_val']=torch.cdiv(dataset['fv_im_val'],torch.repeatTensor(nm_val,1,nhimage)):float()
	local nm_test=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im_test'],dataset['fv_im_test']),2)) 
	dataset['fv_im_test']=torch.cdiv(dataset['fv_im_test'],torch.repeatTensor(nm_test,1,nhimage)):float() 
end

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

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
				:add(netdef.AxB(2*lstm_size_q*nlstm_layers_q,nhimage,common_embedding_size,0.5))
				:add(nn.Dropout(0.5))
				:add(nn.Linear(common_embedding_size,noutput))

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	embedding_net_q = embedding_net_q:cuda()
	encoder_net_q = encoder_net_q:cuda()
	multimodal_net = multimodal_net:cuda()
	dummy_state_q = dummy_state_q:cuda()
	dummy_output_q = dummy_output_q:cuda()
end

-- setting to evaluation
embedding_net_q:evaluate();
encoder_net_q:evaluate();
multimodal_net:evaluate();

-- loading the model
model_param=torch.load(model_path);

embedding_w_q,embedding_dw_q=embedding_net_q:getParameters();
encoder_w_q,encoder_dw_q=encoder_net_q:getParameters();
multimodal_w,multimodal_dw=multimodal_net:getParameters();

embedding_w_q:copy(model_param['embedding_w_q']);
encoder_w_q:copy(model_param['encoder_w_q']);
multimodal_w:copy(model_param['multimodal_w']);

-- duplicate the RNN
encoder_net_buffer_q=dupe_rnn(encoder_net_q,buffer_size_q) 

outputVectorsInception = predictTrain()
outputVectorsInceptionVal = predictVal()
outputVectorsInceptionTest = predictTest()

local outputFile = hdf5.open(opt.out_path, 'w')
outputFile:write('/VGGOut', outputVectorsVGG)
outputFile:write('/InceptionOut', outputVectorsInception)
outputFile:write('/VGGOutVal', outputVectorsVGGVal)
outputFile:write('/InceptionOutVal', outputVectorsInceptionVal)
outputFile:write('/VGGOutTest', outputVectorsVGGTest)
outputFile:write('/InceptionOutTest', outputVectorsInceptionTest)
