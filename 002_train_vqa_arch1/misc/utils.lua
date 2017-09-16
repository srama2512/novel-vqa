local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.countKeys(tableInput)
  local count = 0
  for k, v in pairs(tableInput) do
	count = count + 1
  end
  return count
end

function utils.read_encoding(path)
  local info = utils.read_json(path)
  local new_table = {}
  local encodingSize = nil
  local numCaps = nil
  for k,v in pairs(info) do
    if encodingSize == nil then
	  encodingSize = utils.countKeys(info[k][1])
	end
	if numCaps == nil then
	  numCaps = utils.countKeys(info[k])
	end

	new_table[tonumber(k)] = torch.Tensor(numCaps, encodingSize)
	
	for k1, v1 in pairs(v) do
 	  for k2, v2 in pairs(v1) do
	    new_table[tonumber(k)][k1][k2] = v2
	  end
	end
  end

  return new_table
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

function utils.getCaptions(data, seq_per_image, caption_encoding_path, encoding_size, negativeCaps)
  local imageIds = {}

  for k, v in pairs(data.infos) do
	table.insert(imageIds, v['id'])
  end
  
  local captionEncodedPositive = torch.Tensor(#imageIds * seq_per_image, encoding_size)
  local captionEncodedNegative = torch.Tensor(#imageIds * seq_per_image, encoding_size)
  local capCount = 0

  for k, v in pairs(imageIds) do
	local temp_table = {}
	if caption_encoding[v] == nil then
		temp_table = utils.read_encoding(caption_encoding_path .. 'coco_skipthoughts_' .. v .. '.json')
		caption_encoding[v] = temp_table[v]
	else
		temp_table[v] = caption_encoding[v]
	end

	captionEncodedPositive[{{capCount+1, capCount+seq_per_image}, {}}] = temp_table[v][{{1, 5}, {}}]

	capCount = capCount + seq_per_image
  end

  capCount = 0 
  
  for k, v in pairs(imageIds) do
	for i = 1, seq_per_image do 
	  --captionEncodedNegative[{{capCount+i}, {}}] = caption_encoding[tostring(negativeCaps[v][i])][math.random(seq_per_image)]
		--[[
	    local randNo1
		while randNo1 == nil or v == imageIds[randNo1] do
			randNo1 = math.random(#imageIds)
		end
		--]]
		
		local temp_table = {}
		v_ = tostring(v)
		if caption_encoding[negativeCaps[v_][i]] == nil then
			temp_table = utils.read_encoding(caption_encoding_path .. 'coco_skipthoughts_' .. negativeCaps[v_][i] .. '.json')
			caption_encoding[negativeCaps[v_][i]] = temp_table[negativeCaps[v_][i]]
		else
			temp_table[negativeCaps[v_][i]] = caption_encoding[negativeCaps[v_][i]]
		end
	  	captionEncodedNegative[{{capCount+i}, {}}] = temp_table[negativeCaps[v_][i]][math.random(seq_per_image)]
	end
	capCount = capCount + seq_per_image
  end

  return captionEncodedPositive, captionEncodedNegative
end

return utils
