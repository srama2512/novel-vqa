require 'torch'
require 'nn'
require 'nngraph'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.AutoEncoder_text_nostart'
require 'misc.optim_updates'
local net_utils = require 'misc.net_utils'
require 'cutorch'
require 'cunn'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an AutoEncoder model for sentences')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-model_path','','path to the model file')
cmd:option('-save_path', '', 'path to save parameters')
--cmd:option('-save_h5_path', '', 'saving parameters in h5 format')

local opt = cmd:parse(arg)

modelLoad = torch.load(opt.model_path)
lookupTable = modelLoad.protos.ae.lookup_table

lookupTable_params, lookupTable_gradparams = lookupTable:parameters()
net_utils.unsanitize_gradients(modelLoad.protos.ae.encoder)
encoder_params, encoder_gradparams = modelLoad.protos.ae.encoder:getParameters()

saveModel = {}
-- Transpose because the Linear layer stores params in that way
saveModel['lookup'] = lookupTable_params[1]:t():clone()
saveModel['encoder'] = encoder_params:clone()

torch.save(opt.save_path, saveModel)
--local model_h5_file = hdf5.open(opt.save_h5_path, 'w')
--model_h5_file:write('/lookup', saveModel['lookup']:float())
--model_h5_file:write('/encoder', saveModel['encoder']:float())
--model_h5_file:close()
