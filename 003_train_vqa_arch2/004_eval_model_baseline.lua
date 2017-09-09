require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'misc.netdef'
require 'hdf5'
cjson=require('cjson');
require 'xlua'
LSTM_encoder = require 'misc.LSTM_encoder'
require 'misc.Encoder_lstm'
require 'misc.AutoEncoder'
net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the Visual Question Answering model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_img_h5','data_img.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', 'model/lstm.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'result/', 'path to save output json file')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-input_encoding_size', 512, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',1,'number of the rnn layer')
cmd:option('-common_embedding_size', 1024, 'size of the common embedding vector')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-img_norm', 1, 'normalize the image feature. 1 = normalize, 0 = not normalize')
cmd:option('-nhimage', 4096, 'Image vector size')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.setDevice(opt.gpuid + 1)
end


------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------

local model_path = opt.model_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=opt.nhimage
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
print('DataLoader loading h5 file: ', opt.input_json)
local dataset = {}

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_test'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_test'):all()
dataset['img_list'] = h5_file:read('/img_pos_test'):all()
dataset['ques_id'] = h5_file:read('/question_id_test'):all()
dataset['MC_ans_test'] = h5_file:read('/MC_ans_test'):all()
h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_img_h5, 'r')
dataset['fv_im'] = h5_file:read('/images_test'):all()
h5_file:close()

-- Normalize the image feature
if opt.img_norm == 1 then
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_im'],dataset['fv_im']),2)) 
	dataset['fv_im']=torch.cdiv(dataset['fv_im'],torch.repeatTensor(nm,1,nhimage)):float() 
end

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count
collectgarbage();

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
buffer_size_q=dataset['question']:size()[2]

--Network definitions
----VQA
----embedding: word-embedding

local EncOpt = {}
EncOpt.vocab_size = vocabulary_size_q
EncOpt.input_encoding_size = opt.input_encoding_size
EncOpt.rnn_size = opt.rnn_size
EncOpt.num_layers = opt.num_layers
EncOpt.dropout = opt.drop_prob_ae
EncOpt.seq_length = buffer_size_q

encoder_model = nn.Encoder(EncOpt)

enc_modules = encoder_model:getModulesList()
for k,v in pairs(enc_modules) do net_utils.unsanitize_gradients(v) end 
net_utils.unsanitize_gradients(encoder_model)

modelT = nil
enc_modules = encoder_model:getModulesList()
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

cnn_projection = nn.Sequential():add(nn.Linear(opt.nhimage, EncOpt.input_encoding_size))

--criterion
criterion=nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	encoder_model = encoder_model:cuda()
	cnn_projection = cnn_projection:cuda()
	multimodal_net = multimodal_net:cuda()
	criterion = criterion:cuda()
end

-- load model and copy parameters

model_loaded = torch.load(opt.model_path)
encoder_w_q,encoder_dw_q=encoder_model:getParameters()
encoder_w_q:copy(model_loaded.encoder_w_q)
multimodal_w,multimodal_dw=multimodal_net:getParameters()
multimodal_w:copy(model_loaded.multimodal_w)
cnn_w, cnn_dw=cnn_projection:getParameters()
cnn_w:copy(model_loaded.cnn_w)
sizes={cnn_w:size(1), encoder_w_q:size(1), multimodal_w:size(1)}

-- duplicate the RNN
encoder_model:createClones()

-- setting to evaluation
encoder_model:evaluate();
multimodal_net:evaluate();
cnn_projection:evaluate();

------------------------------------------------------------------------
--Grab Next Batch--
------------------------------------------------------------------------
function dataset:next_batch_test(test_count)

	local batch_size_curr = batch_size
	if test_count+batch_size > dataset['img_list']:size(1) then
		batch_size_curr = dataset['img_list']:size(1)-test_count
	end

	local qinds=torch.LongTensor(batch_size_curr):fill(0) 
	local iminds=torch.LongTensor(batch_size_curr):fill(0) 	
	
	if not test_count then
		test_count = 0
	end

	local nqs=dataset['question']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)

	for i=test_count+1,test_count+batch_size_curr do
		qinds[i-test_count]=i 
		iminds[i-test_count]=dataset['img_list'][qinds[i-test_count]] 
	end

	local fv_q=dataset['question']:index(1,qinds):t()
	local fv_im=dataset['fv_im']:index(1,iminds) 
	local qids=dataset['ques_id']:index(1,qinds)	
	local batch_length=dataset['lengths_q']:index(1,qinds)
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_q = fv_q:cuda()
		fv_im = fv_im:cuda()
	end
	
	test_count = test_count + batch_size_curr
	return fv_q,fv_im, qids,batch_size_curr, batch_length 
end
------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- Validation function
function evaluateModel()

	nqs=dataset['question']:size(1);
	scores=torch.Tensor(nqs,noutput):double();
	qids=torch.LongTensor(nqs);

	--grab a batch--
	local count_test = 0

	while count_test < dataset['img_list']:size(1) do
		
		xlua.progress(count_test, dataset['img_list']:size(1))
		local fv_q,fv_im,qidsTemp,batch_size, batch_length=dataset:next_batch_test(count_test) 
		local question_max_length=fv_q:size(1) 

		
		--cnn projection forward
		local fv_im_proj = cnn_projection:forward(fv_im)
		
		--encoder forward--
		local states_q = encoder_model:forward{fv_im_proj, fv_q}
		
		--multimodal/criterion forward--
		local tv_q = states_q
		local scoresTemp=multimodal_net:forward(states_q) 
	
		scores[{{count_test+1,count_test+batch_size},{}}] = scoresTemp:double()
		qids[{{count_test+1,count_test+batch_size}}] = qidsTemp

		count_test = count_test + batch_size
	end
	return scores, qids
end

-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------

local scores, qids = evaluateModel()
tmp,pred=torch.max(scores,2);


------------------------------------------------------------------------
-- Write to Json file
------------------------------------------------------------------------
function writeAll(file,data)
    local f = io.open(file, "w")
    f:write(data)
    f:close() 
end

function saveJson(fname,t)
	return writeAll(fname,cjson.encode(t))
end

response={};
for i=1,nqs do
	table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
end

paths.mkdir(opt.out_path)
saveJson(opt.out_path .. 'OpenEnded_mscoco_lstm_results.json',response);

mc_response={};

for i=1,nqs do
	local mc_prob = {}
	local mc_idx = dataset['MC_ans_test'][i]
	local tmp_idx = {}
	for j=1, mc_idx:size()[1] do
		if mc_idx[j] ~= 0 then
			table.insert(mc_prob, scores[{i, mc_idx[j]}])
			table.insert(tmp_idx, mc_idx[j])
		end
	end
	local tmp,tmp2=torch.max(torch.Tensor(mc_prob), 1);
	table.insert(mc_response, {question_id=qids[i],answer=json_file['ix_to_ans'][tostring(tmp_idx[tmp2[1]])]})
end

saveJson(opt.out_path .. 'MultipleChoice_mscoco_lstm_results.json',mc_response);
