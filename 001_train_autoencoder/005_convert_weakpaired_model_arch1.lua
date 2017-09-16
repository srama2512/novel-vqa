require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoaderWeakPaired'
require 'misc.AutoEncoder_vqa_arch'
require 'misc.optim_updates'
local net_utils = require 'misc.net_utils'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'misc.L2Normalize'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an AutoEncoder model for sentences')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-model_path','','path to the model file')
cmd:option('-save_path', '', 'path to save parameters')

local opt = cmd:parse(arg)

modelLoad = torch.load(opt.model_path)
lookupTable = modelLoad.protos.ae.lookup_table
multimodalNet = modelLoad.protos.ae.multimodal_net

lookupTable_params, lookupTable_gradparams = lookupTable:parameters()
net_utils.unsanitize_gradients(modelLoad.protos.ae.encoder)
encoder_params, encoder_gradparams = modelLoad.protos.ae.encoder:getParameters()
mm_params, mm_gradparams = multimodalNet:getParameters()

saveModel = {}
-- Transpose because the Linear layer stores params in that way
saveModel['lookup'] = lookupTable_params[1]:t():clone()
saveModel['encoder'] = encoder_params:clone()
saveModel['multimodal'] = mm_params:clone()

torch.save(opt.save_path, saveModel)
