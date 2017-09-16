-- Preprocessing of images and extraction of feature vectors for VQA model

require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','','path to the image root')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')

cmd:option('-out_name', 'data_img_inc.h5', 'output name')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid+1)
net=torch.load(opt.cnn_model);
net:evaluate()
net=net:cuda()

print(net)

-- input subtraction
local input_sub = 128 
-- Scale input image
local input_scale = 0.0078125
-- input dimension
local input_dim = 299 

function loadim(path)
  local img   = image.load(path, 3)
  local w, h  = img:size(3), img:size(2)
  local min   = math.min(w, h)
  img         = image.crop(img, 'c', min, min)
  img         = image.scale(img, input_dim)
  -- normalize image
  img:mul(255):add(-input_sub):mul(input_scale)
  -- due to batch normalization we must use minibatches
  return img:float():view(1, img:size(1), img:size(2), img:size(3))
end


local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, image_root .. imname)
end

local val_list = {}
for i,imname in pairs(json_file['unique_img_val']) do
    table.insert(val_list, image_root .. imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, image_root .. imname)
end
local ndims=2048
local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.CudaTensor(sz,ndims)

print(string.format('processing %d images...',sz))

for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,input_dim,input_dim)
    for j=1,r-i+1 do
		print(string.format('Image no: %d	Image location: %s', i+j-1, train_list[i+j-1]))
		ims[j]=loadim(train_list[i+j-1]):cuda()
	end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[30].output:clone()

    collectgarbage()
end

feat_train = feat_train:float()
local sz=#val_list
local feat_val=torch.CudaTensor(sz,ndims)

print(string.format('processing %d images...',sz))

for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,input_dim,input_dim)
    for j=1,r-i+1 do
		print(string.format('Image no: %d	Image location: %s', i+j-1, val_list[i+j-1]))
		ims[j]=loadim(val_list[i+j-1]):cuda()
	end
    net:forward(ims)
    feat_val[{{i,r},{}}]=net.modules[30].output:clone()

    collectgarbage()
end

print('DataLoader loading h5 file: ', 'data_train')
local sz=#test_list

feat_val = feat_val:float()

local feat_test=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))

for i=1,sz,batch_size do
    --xlua.progress(i, sz)    
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,input_dim,input_dim)
    for j=1,r-i+1 do
		print(string.format('Image no: %d	Image location: %s', i+j-1, test_list[i+j-1]))
		ims[j]=loadim(test_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_test[{{i,r},{}}]=net.modules[30].output:clone()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:write('/images_val', feat_val:float())
train_h5_file:close()
--]]
