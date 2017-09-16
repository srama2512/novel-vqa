-- Preprocessing of images and extraction of feature vectors for VQA model

require 'torch'
require 'math'
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
cmd:option('-image_map_json', 'image_map.json', 'path to json file containing image to index maps')
cmd:option('-orig_feats_h5', 'data_img_orig.h5', 'path to h5 file containing the original image features')
cmd:option('-out_name', 'data_img.h5', 'output name')
cmd:option('-ndims', 4096, 'number of feature dimensions')

opt = cmd:parse(arg)
print(opt)

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, imname)
end

local val_list = {}
for i,imname in pairs(json_file['unique_img_val']) do
    table.insert(val_list, imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, imname)
end

-- Read image features from original h5 file
orig_feats_h5 = hdf5.open(opt.orig_feats_h5, 'r')
feat_train_orig = orig_feats_h5:read('images_train'):all()
feat_val_orig = orig_feats_h5:read('images_val'):all()
feat_test_orig = orig_feats_h5:read('images_test'):all()

-- Read image map json file 
local file2 = io.open(opt.image_map_json, 'r')
local text2 = file2:read()
file2:close()
image_map_json = cjson.decode(text2)

local ndims=opt.ndims
local sz=#train_list
local feat_train=torch.FloatTensor(sz,ndims)

print(string.format('processing train %d images...',sz))
for i=1,sz do
    xlua.progress(i, sz)
    img_curr = train_list[i]
    dict_element_curr = image_map_json[img_curr]
    -- print (img_curr)
    -- print (dict_element_curr)
    if dict_element_curr['set'] == 'train' then
        feat_train[i] = feat_train_orig[dict_element_curr['idx']]:clone()
    elseif dict_element_curr['set'] == 'test' then
        feat_train[i] = feat_test_orig[dict_element_curr['idx']]:clone()
    else
        feat_train[i] = feat_val_orig[dict_element_curr['idx']]:clone()
    end
end

local sz=#val_list
local feat_val=torch.FloatTensor(sz,ndims)

print(string.format('processing val %d images...',sz))

for i=1,sz do
    xlua.progress(i, sz)
    img_curr = val_list[i]
    dict_element_curr = image_map_json[img_curr]
    if dict_element_curr['set'] == 'train' then
        feat_val[i] = feat_train_orig[dict_element_curr['idx']]:clone()
    elseif dict_element_curr['set'] == 'test' then
        feat_val[i] = feat_test_orig[dict_element_curr['idx']]:clone()
    else
        feat_val[i] = feat_val_orig[dict_element_curr['idx']]:clone()
    end
end


local sz=#test_list
local feat_test=torch.FloatTensor(sz,ndims)
print(string.format('processing test %d images...',sz))

for i=1,sz do
    xlua.progress(i, sz)    
    img_curr = test_list[i]
    dict_element_curr = image_map_json[img_curr]
    if dict_element_curr['set'] == 'train' then
        feat_test[i] = feat_train_orig[dict_element_curr['idx']]:clone()
    elseif dict_element_curr['set'] == 'test' then
        feat_test[i] = feat_test_orig[dict_element_curr['idx']]:clone()
    else
        feat_test[i] = feat_val_orig[dict_element_curr['idx']]:clone()
    end

end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:write('/images_val', feat_val:float())
train_h5_file:close()
--]]
