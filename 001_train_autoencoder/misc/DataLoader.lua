require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.ix_to_word = self.info.ix_to_word
  self.split_count = {}
  self.split_count['train'] = self.info.num_train
  self.split_count['val'] = self.info.num_val
  self.split_count['test'] = self.info.num_test
  self.vocab_size = utils.count_keys(self.ix_to_word)
  print('vocab size is ' .. self.vocab_size)
  
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- load in the sequence data
  local seq_size = self.h5_file:read('/labels/train'):dataspaceSize()
  self.seq_length = seq_size[2]
  self.iterators = {}
  self.iterators['train'] = 1
  self.iterators['val'] = 1
  self.iterators['test'] = 1

  print('max sequence length in data is ' .. self.seq_length)
   
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
  return self.seq_length
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - y (L,M) containing the sentences as columns 
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5) -- how many sentences  get returned at one time
  local encoding_size = utils.getopt(opt, 'encoding_size')
  -- pick an index of the datapoint to load next
  local label_batch = torch.LongTensor(batch_size, self.seq_length)
  local max_index = self.split_count[split]
  local wrapped = false
   
  if self.iterators[split] + batch_size - 1 > max_index then
	  wrapped = true
	  if self.iterators[split] < max_index then
		  local num_left = max_index-self.iterators[split]+1 
		  label_batch[{{1, num_left}, {}}] = self.h5_file:read('/labels/'..split):partial({self.iterators[split], max_index}, {1, self.seq_length})
		  label_batch[{{max_index-self.iterators[split]+2, batch_size}, {}}] = self.h5_file:read('/labels/'..split):partial({1, batch_size-num_left}, {1, self.seq_length})
		  self.iterators[split] = 1
	  else
		label_batch[{{1, batch_size}, {}}] = self.h5_file:read('/labels/'..split):partial({1, batch_size}, {1, self.seq_length})
	     self.iterators[split] = 1 
	  end
  else
	  label_batch[{{1, batch_size}, {}}] = self.h5_file:read('/labels/'..split):partial({self.iterators[split], self.iterators[split]+batch_size-1}, {1, self.seq_length})
	  self.iterators[split] = self.iterators[split]+batch_size
  end

  local data = {}
  data.imgs = torch.Tensor(batch_size, encoding_size):fill(0)
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = max_index, wrapped = wrapped}
  return data
end

