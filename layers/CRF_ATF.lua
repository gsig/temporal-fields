require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')
require('layers/Constant')
require('layers/BlockGradient')

local CRF_ATF, Module = torch.class('nn.CRF_ATF', 'nn.Module')

function CRF_ATF:__init(batchSize,inputSize,condition,nG,nO,nV,nS,indOV,indOS,indVS,indOVG)
    -- Sets up a Deep CRF model
    --
    -- This model models models a CRF over 4 variables: O, V, S, G
    -- This model models the following edges in the CRF
    -- O unary, V unary, and S uniary (omitted in this version)
    -- OV pairwise
    -- OVG ternery potential
    -- O O' where O' is a O in a previous or future frame.
    --
    -- Technically each of the possible values for these terms is enumerated (nO,nOV,nOVG etc)
    -- and then simply a linear layer predicts 4096->N where N is the number of possible values
    -- This version has some variants on this for OVG and OO
    -- 
    -- indOV should be Nx2 with all valid (o,v) pairs, etc. should be unique
    -- See datasets/charades_crf3.lua for details
    Module.__init(self)
    self.inputSize = inputSize
    self.nO = nO
    self.nV = nV
    self.nS = nS
    self.nG = nG
    self.OVGconditionedOnFeatures = condition 

    -- convert sparse index pairs to linear indexes, or gather-like index tables
    self.nOV = indOV:size(1)
    self.nOS = indOS:size(1)
    self.nVS = indVS:size(1)
    self.nOVG = nO*nG

    self.model = nn.ConcatTable()

    -- set up FC layers to predict the unary potentials
    self.model:add(nn.Linear(inputSize,nO):noBias())  
    self.model:add(nn.Linear(inputSize,nV):noBias())  
    --self.model:add(nn.Linear(inputSize,nS):noBias()) -- doesn't help. Dropped

    -- set up FC layers to predict the pairwise potentials
    self.model:add(nn.Linear(inputSize,self.nOV):noBias()) 
    --self.model:add(nn.Linear(inputSize,self.nOS):noBias()) --doesn't help. Dropped
    
    -- set up FC layer to predict the ternery potential
    local m6 = nn.Sequential()
    if self.OVGconditionedOnFeatures then
        --self.model:add(nn.Linear(inputSize,self.nOVG):noBias()) -- Too many parameters
        --This seems to do better
        m6:add(nn.BlockGradient()) -- for stability
        m6:add(nn.Linear(inputSize,100))
        m6:add(nn.ReLU())
        m6:add(nn.Dropout(0.5))
        m6:add(nn.Linear(100,self.nOVG):noBias())
    else
        -- Doesn't do much worse for RGB and easier to train
        m6:add(nn.Constant(1,1))
        m6:add(nn.Linear(1,self.nOVG):noBias())
    end
    self.model:add(m6) 

    -- set up FC layer to predict the across-frame potential
    -- Drop the condition on the features for simplicity
    local mOO = nn.Sequential()
    mOO:add(nn.Constant(1,1))
    mOO:add(nn.Linear(1,nO*nO):noBias())
    self.model:add(mOO)
end

function CRF_ATF:parameters()
    return self.model:parameters()
end

function CRF_ATF:applyToModules(func)
    for _, module in ipairs(self.model.modules) do
        func(module)
    end
end

function CRF_ATF:updateOutput(input, target)
    return self.model:forward(input)
end

function CRF_ATF:backward(input, gradOutput)
    self.gradInput = self.model:backward(input, gradOutput)
    return self.gradInput
end

--function CRF_ATF:updateGradInput(input, gradOutput)
--    self.gradInput = self.model:backward(input, gradOutput)
--    return self.gradInput
--end

function CRF_ATF:accGradParameters(input, gradOutput)
    return self.model:accGradParameters(input, gradOutput)
end

function CRF_ATF:zeroGradParameters()
    return self.model:zeroGradParameters()
end

function CRF_ATF:training()
    return self.model:training()
end

function CRF_ATF:evaluate()
    return self.model:evaluate()
end

function CRF_ATF:reset(stdv)
    return self.model:reset(stdv)
end

function CRF_ATF:cuda()
    return self.model:cuda()
end

function CRF_ATF:__tostring__()
    return torch.type(self) ..
        string.format('(%d -> (%d,%d,%d,%d,%d,%d,%d))', self.inputSize, self.nO,self.nV,self.nS,self.nOV,self.nOS,self.nVS,self.nOVG)
end

return nn.CRF_ATF
