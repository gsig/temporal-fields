require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

local CRF3, Module = torch.class('nn.CRF3', 'nn.Module')

function CRF3:__init(batchSize,inputSize,nO,nV,nS,indOV,indOS,indVS)
    -- indOV should be Nx2 with all valid (o,v) pairs, etc. should be unique
    Module.__init(self)
    self.inputSize = inputSize
    self.nO = nO
    self.nV = nV
    self.nS = nS

    -- convert sparse index pairs to linear indexes, or gather-like index tables
    self.nOV = indOV:size(1)
    self.nOS = indOS:size(1)
    self.nVS = indVS:size(1)

    -- set up FC layer to predict all those potentials
    self.model = nn.ConcatTable()
    self.model:add(nn.Linear(inputSize,nO):noBias())  
    self.model:add(nn.Linear(inputSize,nV):noBias())  
    self.model:add(nn.Linear(inputSize,nS):noBias())  
    self.model:add(nn.Linear(inputSize,self.nOV):noBias()) 
    self.model:add(nn.Linear(inputSize,self.nOS):noBias()) 
    self.model:add(nn.Linear(inputSize,self.nVS):noBias()) 
end

function CRF3:parameters()
    return self.model:parameters()
end

function CRF3:applyToModules(func)
    for _, module in ipairs(self.model.modules) do
        func(module)
    end
end

function CRF3:updateOutput(input, target)
    return self.model:forward(input)
end

function CRF3:backward(input, gradOutput)
    self.gradInput = self.model:backward(input, gradOutput)
    return self.gradInput
end

function CRF3:accGradParameters(input, gradOutput)
    return self.model:accGradParameters(input, gradOutput)
end

function CRF3:zeroGradParameters()
    return self.model:zeroGradParameters()
end

function CRF3:training()
    return self.model:training()
end

function CRF3:evaluate()
    return self.model:evaluate()
end

function CRF3:reset(stdv)
    return self.model:reset(stdv)
end

function CRF3:__tostring__()
    return torch.type(self) ..
        string.format('(%d -> (%d,%d,%d,%d,%d,%d))', self.inputSize, self.nO,self.nV,self.nS,self.nOV,self.nOS,self.nVS)
end

return nn.CRF3
