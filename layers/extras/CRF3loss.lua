require 'nn'
require 'cutorch'
torch.setdefaulttensortype('torch.FloatTensor')

local CRF3loss, Criterion = torch.class('nn.CRF3loss', 'nn.Criterion')

local function lse(x,dim)
    local x_max = torch.max(x,dim)
    x_max[x_max:eq(-math.huge)] = 0
    x:csub(x_max:expandAs(x))
    x:exp()
    local out = x:sum(dim):log():add(x_max)
    x:log()
    x:add(x_max:expandAs(x))
    x[x:ne(x)] = -math.huge
    return out
end

local function sub2ind(ind, n, m)
    return m*(ind[{{},1}]-1)+ind[{{},2}]
end

local function imap(ind)
-- for simplicity, ind is the batch x n tensor
    local out = {}
    for i=1,ind:size(2) do
        out[ind[{1,i}]] = i
    end
    return out
end

local function lookup(map,sub,n,m)
    local ind = sub2ind(ind,n,m)
    local out = {}
    for i=1,ind:size(1) do
        table.insert(out,map[ind[i]])
    end
    return out
end

function CRF3loss:uniquerows(x)
    local function eq(x,y)
        assert(x:dim()==1)
        return x[1]==y[1] and x[2]==y[2]
    end
    local keep = torch.ByteTensor(x:size(1)):fill(1)
    for i=1,x:size(1) do
        for j=1,i-1 do
            keep[i] = eq(x[i],x[j]) and 0 or keep[i]
        end
    end
    return x[keep:view(keep:size(1),1):expandAs(x)]:unfold(1,2,2)
end

function CRF3loss:resize(batchSize)
    self.batchSize = batchSize
    self.indOV = self.indOV[{{1},{}}]:expand(batchSize,self.nOV)
    self.indOS = self.indOS[{{1},{}}]:expand(batchSize,self.nOS)
    self.indVS = self.indVS[{{1},{}}]:expand(batchSize,self.nVS)
    self.OV = self.OV:resize(batchSize,self.nO,self.nV):fill(-math.huge)
    self.OS = self.OS:resize(batchSize,self.nO,self.nS):fill(-math.huge)
    self.VS = self.VS:resize(batchSize,self.nV,self.nS):fill(-math.huge)
    self.out = self.out and self.out:resize(batchSize,self.nO,self.nV,self.nS) or torch.CudaTensor(batchSize,self.nO,self.nV,self.nS)
end

function CRF3loss:__init(batchSize,inputSize,nO,nV,nS,indOV,indOS,indVS)
    -- indOV should be Nx2 with all valid (o,v) pairs, etc. should be unique
    Criterion.__init(self)
    self.batchSize = batchSize
    self.inputSize = inputSize
    self.nO = nO
    self.nV = nV
    self.nS = nS

    -- convert sparse index pairs to linear indexes, or gather-like index tables
    self.nOV = indOV:size(1)
    self.nOS = indOS:size(1)
    self.nVS = indVS:size(1)
    self.indOV = sub2ind(indOV,nO,nV):view(1,self.nOV):expand(batchSize,self.nOV)
    self.indOS = sub2ind(indOS,nO,nS):view(1,self.nOS):expand(batchSize,self.nOS)
    self.indVS = sub2ind(indVS,nV,nS):view(1,self.nVS):expand(batchSize,self.nVS)
    self.ind2indOV = imap(self.indOV)
    self.ind2indOS = imap(self.indOS)
    self.ind2indVS = imap(self.indVS)

    -- create placeholders for pair-wise potential functions
    self.OV = torch.Tensor(batchSize,nO,nV):fill(-math.huge)
    self.OS = torch.Tensor(batchSize,nO,nS):fill(-math.huge)
    self.VS = torch.Tensor(batchSize,nV,nS):fill(-math.huge)
end

function CRF3loss:updateOutput(input, target)
    self:resize(input[1]:size(1))
    local unit
    if not target then
        -- unit testing
        target = self.target
        unit = true
    end
    local batchSize,nO,nV,nS = self.batchSize,self.nO,self.nV,self.nS
    local oO = input[1]:view(self.batchSize,nO,1,1):cuda()
    local oV = input[2]:view(self.batchSize,1,nV,1):cuda()
    local oS = input[3]:view(self.batchSize,1,1,nS):cuda()
    
    -- copy predictions back to their respective potential functions
    self.OV:view(batchSize,nO*nV):scatter(2,self.indOV,input[4])
    self.OS:view(batchSize,nO*nS):scatter(2,self.indOS,input[5])
    self.VS:view(batchSize,nV*nS):scatter(2,self.indVS,input[6])

    -- calculate Z, by expanding to singleton dims and naively adding up everything
    self.out = self.out or torch.CudaTensor(batchSize,nO,nV,nS)
    self.out:zero()
    self.out:add(oO:expandAs(self.out))
    self.out:add(oV:expandAs(self.out))
    self.out:add(oS:expandAs(self.out))
    self.out:add(self.OV:cuda():view(batchSize,nO,nV,1):expandAs(self.out))
    self.out:add(self.OS:cuda():view(batchSize,nO,1,nS):expandAs(self.out))
    self.out:add(self.VS:cuda():view(batchSize,1,nV,nS):expandAs(self.out))
    self.batchZ = lse(lse(lse(self.out, 4), 3), 2):squeeze():float()

    -- loss is sum of GT potentials minus the normalization, then sum across batches
    local gt_score = torch.Tensor(batchSize)
    for i=1,batchSize do
        local iO,iV,iS = target[{i,1}],target[{i,2}],target[{i,3}]
        gt_score[i] = self.out[{i,iO,iV,iS}]
    end
    self.output = torch.sum(self.batchZ - gt_score)

    if unit then
        return torch.Tensor{self.output}
    else
        return self.output
    end
end

function CRF3loss:actions(a2ov) -- call forward first
    -- a2ov is 0-index so #a2ov misses the 0th one.
    local out = torch.Tensor(self.batchSize,#a2ov+1)
    for a=1,#a2ov+1 do
        local o,v = unpack(a2ov[a-1]) --0-index
        out[{{},a}] = lse(self.out[{{},{o+1},{v+1},{}}], 4):squeeze():float()
    end
    return out
end

function CRF3loss:marginal(dim) -- call forward first
    if dim==1 then
        return lse(lse(self.out, 4), 3):squeeze():exp()
    elseif dim==2 then
        return lse(lse(self.out, 4), 2):squeeze():exp()
    elseif dim==3 then
        return lse(lse(self.out, 3), 2):squeeze():exp()
    end
end

function CRF3loss:unariesgt(I,n)
    local target = torch.Tensor(self.batchSize,n):zero()
    for i=1,self.batchSize do
        target[{i,I[i]}] = 1
    end
    return target
end

function CRF3loss:pairwisegt(I,n,m,map,nm)
    local linearI = sub2ind(I,n,m):squeeze()
    local target = torch.Tensor(self.batchSize,nm):zero()
    for i=1,self.batchSize do
        target[{i,map[linearI[i]]}] = 1
    end
    return target
end

function CRF3loss:zeroGradParameters()
end

function CRF3loss:updateGradInput(input, target)
    -- target is nBatch x 3
    assert(self.batchZ, "call forward first")

    local targetO = self:unariesgt(target[{{},1}],self.nO)
    local targetV = self:unariesgt(target[{{},2}],self.nV)
    local targetS = self:unariesgt(target[{{},3}],self.nS)
    local targetOV = self:pairwisegt(target[{{},{1,2}}],self.nO,self.nV,self.ind2indOV,self.nOV)
    local targetOS = self:pairwisegt(target:index(2,torch.LongTensor{1,3}),self.nO,self.nS,self.ind2indOS,self.nOS)
    local targetVS = self:pairwisegt(target[{{},{2,3}}],self.nV,self.nS,self.ind2indVS,self.nVS)


    -- sum of all elements of self.out where variables match
    local tmp1 = lse(lse(self.out,3),4):squeeze():float()
    local tmp2 = lse(lse(self.out,2),4):squeeze():float()
    local tmp3 = lse(lse(self.out,2),3):squeeze():float()
    local tmp4 = lse(self.out,4):view(self.batchSize,self.nO*self.nV):index(2,self.indOV[1]:long()):float()
    local tmp5 = lse(self.out,3):view(self.batchSize,self.nO*self.nS):index(2,self.indOS[1]:long()):float()
    local tmp6 = lse(self.out,2):view(self.batchSize,self.nV*self.nS):index(2,self.indVS[1]:long()):float()
    
    self.gradInput = {}

    self.gradInput[1] = torch.exp(tmp1-self.batchZ:view(self.batchSize,1):expandAs(input[1])) - targetO 
    self.gradInput[2] = torch.exp(tmp2-self.batchZ:view(self.batchSize,1):expandAs(input[2])) - targetV 
    self.gradInput[3] = torch.exp(tmp3-self.batchZ:view(self.batchSize,1):expandAs(input[3])) - targetS 
    self.gradInput[4] = torch.exp(tmp4-self.batchZ:view(self.batchSize,1):expandAs(input[4])) - targetOV
    self.gradInput[5] = torch.exp(tmp5-self.batchZ:view(self.batchSize,1):expandAs(input[5])) - targetOS
    self.gradInput[6] = torch.exp(tmp6-self.batchZ:view(self.batchSize,1):expandAs(input[6])) - targetVS

    return self.gradInput
end

function CRF3loss:unit()
    -- test by calling: require('layers/CRF3loss'):unit()
    torch.manualSeed(123)

    local crf3 = require('layers/CRF3')
    local crf3loss = require('layers/CRF3loss') --this

    -- params
    local batchSize = 2 
    local inputSize = 10 
    local nO = 40
    local nV = 30
    local nS = 20
    local fakegt = torch.LongTensor(300):random(1,nO):cat(torch.LongTensor(300):random(1,nV),2):cat(torch.LongTensor(300):random(1,nS),2)
    local indOV = fakegt[{{1,157},{1,2}}]
    local indOS = fakegt:index(2,torch.LongTensor{1,3})
    local indVS = fakegt[{{},{2,3}}]
    local fakegt2 = torch.LongTensor(300):random(1,nO):cat(torch.LongTensor(300):random(1,nV),2):cat(torch.LongTensor(300):random(1,nS),2)
    local input = torch.randn(batchSize, inputSize):float()
    local target = torch.FloatTensor(batchSize, 3):zero()
    for i=1,batchSize do
        target[{i,{}}] = fakegt[i]
    end
    local module = nn.Sequential()
    indOV = CRF3loss:uniquerows(indOV)
    indOS = CRF3loss:uniquerows(indOS)
    indVS = CRF3loss:uniquerows(indVS)
    local crf = crf3(batchSize,inputSize,nO,nV,nS,indOV,indOS,indVS)
    local crfloss = crf3loss(batchSize,inputSize,nO,nV,nS,indOV,indOS,indVS)
    module:add(crf)
    module:add(crfloss)

    -- testing
    local precision = 1e-1
    self.target = target
    local jac = nn.Jacobian.forward(module,input,input,1e-4)
    local output1 = crf:forward(input)
    local output2 = crfloss:forward(output1)
    local gradInput = crfloss:backward(output1, target)
    local jac2 = crf:backward(input, gradInput)
    local tmp = jac2-jac
    local err = tmp:abs():max()
    assert(err<precision, 'Unit test: Error in implementation')

    -- testing params
    module:zeroGradParameters()
    local params,gradParams = crf:getParameters()
    local jac = nn.Jacobian.forward(module,input,params,1e-4)
    local output1 = crf:forward(input)
    local output2 = crfloss:forward(output1)
    local gradInput = crfloss:backward(output1, target)
    crf:backward(input, gradInput)
    local tmp = gradParams-jac
    local err = tmp:abs():max()
    assert(err<precision, 'Unit test: Error in accGradParameters implementation')
    
    print('Unit test: finished without error')
    return err, precision
end

function CRF3loss:__tostring__()
    return torch.type(self) ..
        string.format('((%d,%d,%d,%d,%d,%d) -> 1)',self.nO,self.nV,self.nS,self.nOV,self.nOS,self.nVS)
end

return nn.CRF3loss
