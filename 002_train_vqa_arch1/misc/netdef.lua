require 'nn'
require 'nngraph'

netdef={};

function netdef.AxB(nhA,nhB,nhcommon,dropout)
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
	local output=nn.CMulTable()({qc,ic});
	return nn.gModule({q,i},{output});
end

function netdef.AskipB(nhA,nhB,nhcommon,dropout)
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
	local output=nn.CMulTable()({qc,ic});
	local output_skip = nn.CAddTable()({qc, output}) 
	return nn.gModule({q,i},{output_skip});
end

function netdef.A_B(nhA,nhB,nhcommon,dropout)
    dropout = dropout or 0 
    local q=nn.Identity()();
    local i=nn.Identity()();
    local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
    local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
    local output=nn.JoinTable(2)({qc,ic});
    return nn.gModule({q,i},{output});
end

return netdef;
