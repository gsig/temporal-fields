require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

local CRF3, Module = torch.class('nn.CRF3', 'nn.ConcatTable')

function CRF3:__init(batchSize,inputSize,nO,nV,nS,indOV,indOS,indVS)
    -- indOV should be Nx2 with all valid (o,v) pairs, etc. should be unique
    print('enter')
    Module.__init(self)
    print('paren done')
    self.inputSize = inputSize
    self.nO = nO
    self.nV = nV
    self.nS = nS

    -- convert sparse index pairs to linear indexes, or gather-like index tables
    self.nOV = indOV:size(1)
    --self.nOS = indOS:size(1)
    --self.nVS = indVS:size(1)

    -- set up FC layer to predict all those potentials
    local model = self
    model:add(nn.Linear(inputSize,nO):noBias())  
    model:add(nn.Linear(inputSize,nV):noBias())  
    --self.model:add(nn.Linear(inputSize,nS):noBias())  
    model:add(nn.Linear(inputSize,self.nOV):noBias()) 
    --self.model:add(nn.Linear(inputSize,self.nOS):noBias()) 
    --self.model:add(nn.Linear(inputSize,self.nVS):noBias()) 
    print('done init')
end

--function CRF3:updateOutput(input, target)
--    return Module.updateOutput(self, input, target)
--end
--
--function CRF3:backward(input, gradOutput)
--    print(#gradOutput)
--    print(#input)
--    print(#self.gradInput)
--    return Module.backward(self, input, gradOutput)
--end

function CRF3:__tostring__()
    return torch.type(self) ..
        string.format('(%d -> (%d,%d,%d))', self.inputSize, self.nO,self.nV,self.nOV)
end

return nn.CRF3
