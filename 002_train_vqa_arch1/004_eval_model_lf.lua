require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'misc.netdef'
require 'hdf5'
LSTM=require 'misc.LSTM'
cjson=require('cjson');
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the Visual Question Answering model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_ques_h5','data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-input_features_h5','outputVectors.h5', 'path to the h5file containing computed output vectors')
cmd:option('-out_path', 'result/', 'path to save output json file')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-batch_size',500,'batch_size for each iterations')
cmd:option('-num_output', 1000, 'number of output answers')
cmd:option('-weight_vgg', 0.5, 'scaling for vgg in ensemble')
cmd:option('-weight_inception', 0.5, 'scaling for inception in ensemble')
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
local batch_size=opt.batch_size
local noutput=opt.num_output

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

h5_file = hdf5.open(opt.input_features_h5, 'r')
dataset['vgg_out'] = h5_file:read('/VGGOutTest'):all()
dataset['inception_out'] = h5_file:read('/InceptionOutTest'):all()
h5_file:close()

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count
collectgarbage();

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
buffer_size_q=dataset['question']:size()[2]

------------------------------------------------------------------------
--Grab Next Batch--
------------------------------------------------------------------------
function dataset:next_batch_test(s,e)
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=dataset['img_list'][qinds[i]];
	end
	
	local qids=dataset['ques_id']:index(1,qinds);
	local vgg_preds = dataset['vgg_out']:index(1, qinds)
	local inception_preds = dataset['inception_out']:index(1, qinds)

	local inputVectors = torch.Tensor(batch_size, vgg_preds:size(2), 3)
	for i=1,batch_size do
		inputVectors[{{i}, {}, {1}}] = torch.mul(vgg_preds[{{i}, {}}]:t(), opt.weight_vgg)
		inputVectors[{{i}, {}, {2}}] = torch.mul(inception_preds[{{i}, {}}]:t(), opt.weight_inception)
	end

	-- ship to gpu
	if opt.gpuid >= 0 then
		inputVectors = inputVectors:cuda()
	end
	
	--print(string.format('batch_sort:%f',timer:time().real));
	return inputVectors,qids,batch_size;
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
function forward(s,e)
	local timer = torch.Timer();
	
	--grab a batch--
	local inputVectors,qids,batch_size=dataset:next_batch_test(s,e);
	local scores = torch.sum(inputVectors, 3)
	return scores:double(),qids;
end


-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------
nqs=dataset['question']:size(1);
scores=torch.Tensor(nqs,noutput);
qids=torch.LongTensor(nqs);
for i=1,nqs,batch_size do
	xlua.progress(i, nqs)
	r=math.min(i+batch_size-1,nqs);
	scores[{{i,r},{}}],qids[{{i,r}}]=forward(i,r);
end

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
