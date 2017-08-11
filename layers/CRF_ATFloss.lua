require 'nn'
require 'cutorch'
local MessagePassing = require 'layers/messagepassing'
torch.setdefaulttensortype('torch.FloatTensor')

local CRF_ATFloss, Criterion = torch.class('nn.CRF_ATFloss', 'nn.Criterion')

local function lse1() end
local function lse(x,dim)
    if dim>10 then return lse1(x,dim) end --shorthand
    -- in place
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

local function lse2(x,dim1,dim2) return lse(lse(x,dim1),dim2) end
local function lse3(x,dim1,dim2,dim3) return lse(lse(lse(x,dim1),dim2),dim3) end
local function lse4(x,dim1,dim2,dim3,dim4) return lse(lse(lse(lse(x,dim1),dim2),dim3),dim4) end
function lse1(x,dim) 
    if     dim==12  then return lse2(x,1,2) 
    elseif dim==23  then return lse2(x,2,3) 
    elseif dim==34  then return lse2(x,3,4) 
    elseif dim==45  then return lse2(x,4,5) 
    elseif dim==123 then return lse3(x,1,2,3) 
    elseif dim==234 then return lse3(x,2,3,4) 
    elseif dim==245 then return lse3(x,2,4,5) 
    elseif dim==345 then return lse3(x,3,4,5) 
    elseif dim==543 then return lse3(x,5,4,3) 
    else assert(false)
    end
end
local function slse(x,dim) return lse(x,dim) end -- might need to use simplelse

local function simplelse(x,dim)
    -- not in place
    local x_max = torch.max(x,dim)
    x_max[x_max:eq(-math.huge)] = 0
    local out = (x + (-x_max):expandAs(x)):exp():sum(dim):log():add(x_max)
    return out
end

local function sme(x,y,dim,sqz)
    -- sum(x*(exp(y)),dim)
    if not (sqz==false) then
        return torch.sum(torch.exp(y:expandAs(x)):cmul(x),dim):squeeze()
    else
        return torch.sum(torch.exp(y:expandAs(x)):cmul(x),dim)
    end
end

local function sub2ind(ind, n, m)
    -- convert subscript to linear index
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

function CRF_ATFloss:uniquerows(x)
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

function CRF_ATFloss:resize(batchSize)
    self.batchSize = batchSize
    local bS,nO,nV,nS,nG = self.batchSize,self.nO,self.nV,self.nS,self.nG
    self.indOV = self.indOV[{{1},{}}]:expand(bS,self.nOV):cuda()
    self.indOS = self.indOS[{{1},{}}]:expand(bS,self.nOS):cuda()
    self.indVS = self.indVS[{{1},{}}]:expand(bS,self.nVS):cuda()
    self.OV = self.OV:resize(bS,nO,nV):fill(-math.huge)
    self.OS = self.OS:resize(bS,nO,nS):fill(-math.huge)
    self.VS = self.VS:resize(bS,nV,nS):fill(-math.huge)
    self.OVG = self.OVG:resize(bS,nO,nG):fill(-math.huge)
    self.out = self.out and self.out:resize(bS,nO,nV,nS,1) or torch.CudaTensor(bS,nO,nV,nS,1)
    self.OO = self.OO:resize(bS,nO,nO):fill(-math.huge)
    self.Qh = self.Qh:resize(bS,nG):fill(-math.huge)
    self.Qi = self.Qi and self.Qi:resize(bS,nO,nV,nS,1) or torch.CudaTensor(bS,nO,nV,nS,1)
end

function CRF_ATFloss:__init(opt,batchSize,inputSize,nG,nO,nV,nS,indOV,indOS,indVS)
    -- indOV should be Nx2 with all valid (o,v) pairs, etc. should be unique
    Criterion.__init(self)
    self.intent_decay = opt.intent_decay
    self.batchSize,self.nO,self.nV,self.nS,self.nG = batchSize,nO,nV,nS,nG
    self.inputSize = inputSize
    print('#goals: ' .. nG)

    -- convert sparse index pairs to linear indexes, or gather-like index tables
    self.nOV = indOV:size(1)
    self.nOS = indOS:size(1)
    self.nVS = indVS:size(1)
    self.nOVG = nO*nG 
    self.indOV = sub2ind(indOV,nO,nV):view(1,self.nOV):expand(batchSize,self.nOV):cuda()
    self.indOS = sub2ind(indOS,nO,nS):view(1,self.nOS):expand(batchSize,self.nOS):cuda()
    self.indVS = sub2ind(indVS,nV,nS):view(1,self.nVS):expand(batchSize,self.nVS):cuda()
    self.ind2indOV = imap(self.indOV:long())
    self.ind2indOS = imap(self.indOS:long())
    self.ind2indVS = imap(self.indVS:long())

    -- create placeholders for pair-wise potential functions
    self.OV = torch.CudaTensor(batchSize,nO,nV):fill(-math.huge)
    self.OS = torch.CudaTensor(batchSize,nO,nS):fill(-math.huge)
    self.VS = torch.CudaTensor(batchSize,nV,nS):fill(-math.huge)
    self.OVG = torch.CudaTensor(batchSize,nO,nG):fill(-math.huge)
    self.OO = torch.CudaTensor(batchSize,nO,nO):fill(-math.huge)
    self.Qh = torch.CudaTensor(batchSize,nG):fill(-math.huge)
    
    self.debug = false
    self.MP = MessagePassing(opt)
end

function CRF_ATFloss:updateOutput(input, target)
    -- Forward pass in CRF
    -- Combines potentials to contstruct distribution over all variables
    -- Calculates loss
    -- Also computes messages
    --
    -- self.out is the 5 dimensional local distribution in a frame
    --
    -- Most operations here implement f(x,y) = g(x,y) * h(y) in logspace by
    --                 f = g:add(h:view(1,Y):expandAs(g))
    -- or accumulating in place: g:add(h:view(1,Y):expandAs(g))
    -- other operations are logsumexp (marginals) and summulexp (expectation)
    --
    self:resize(input[1]:size(1))
    local bS,nO,nV,nS,nG = self.batchSize,self.nO,self.nV,self.nS,self.nG
    local oO = input[1]:view(bS,nO,1,1,1):cuda()
    local oV = input[2]:view(bS,1,nV,1,1):cuda()
    
    -- copy predictions back to their respective potential functions
    self.OV:view(bS,nO*nV):scatter(2,self.indOV,input[3])
    self.OVG:view(bS,nO,nG):copy(input[4])
    self.OO:view(bS,nO*nO):copy(input[5]) -- this is mu
    
    if self.debug then
        print(('  %%%% I1:%+.1e; I2:%+.1e; I3:%+.1e; I4:%+.1e; I5:%+.1e'):format(input[1]:mean(),input[2]:mean(),input[3]:mean(),input[4]:mean(),input[5]:mean()))
        print(('  %%%% I1:%+.1e; I2:%+.1e; I3:%+.1e; I4:%+.1e; I5:%+.1e'):format(input[1]:norm(),input[2]:norm(),input[3]:norm(),input[4]:norm(),input[5]:norm()))
    end

    local Kmessages,Kgtmessages,Pmessages,Kmessages2,Kgtmessages2,Pmessages2,Hmessages,Hgtmessages = self:getMessages(input.ids,input.times)

    if self.debug then
        print(('  ## K:%+.1e; Kgt:%+.1e; K2:%+.1e; Kgt2:%+.1e; \n  ## P:%+.1e; P2:%+.1e; H:%+.1e; Hgt:%+.1e'):format(Kmessages:mean(),Kgtmessages:mean(),Kmessages2:mean(),Kgtmessages:mean(),Pmessages:mean(),Pmessages2:mean(),Hmessages:mean(),Hgtmessages:mean()))
    end

    -- calculate local approximation of the following using mean-field
    -- f(o,v,s,g) = f(o) * f(v) * f(o,v) * f(o,v,g)
    -- by expanding to singleton dims and naively adding up everything
    -- self.out is the "unary" term for the frame clique f(o,v,s,g)
    self.out = self.out or torch.CudaTensor(bS,nO,nV,nS,1)
    self.out:zero()
    self.out:add(oO:expandAs(self.out))
    self.out:add(oV:expandAs(self.out))
    self.out:add(self.OV:cuda():view(bS,nO,nV,1,1):expandAs(self.out))
    
    -- Step 1: Calculate local Mean-Field approximation using messages
    -- Step 1a: initialize Qh, this omits local evidence in Qh
    self.Qh:copy(Hmessages)
    self.Qh:add(-simplelse(self.Qh,2):expandAs(self.Qh)) -- normalize Qh

    -- Step 1b: Calculate Qi, and send some messages between Qi and Qh
    local local_message_passes = 5 -- How many local message passes between Qi and Qh
    for i=1,local_message_passes do
        -- calculate Qi using the current Qh
        local potH = sme(self.OVG:cuda(),self.Qh:view(bS,1,nG),3)
        assert(potH:sum() ~= -math.huge,'weird error in potH')
        self.Qi = self.Qi or torch.CudaTensor(bS,nO,nV,nS,1)
        self.Qi:copy(self.out)
        self.Qi:add(      potH:view(bS,nO,1,1,1):expandAs(self.Qi))
        self.Qi:add( Pmessages:view(bS,nO,1,1,1):expandAs(self.Qi))
        self.Qi:add(Pmessages2:view(bS,nO,1,1,1):expandAs(self.Qi))
        self.Qi_omarginal = slse(self.Qi,543):squeeze()
        self.batchZ = slse(self.Qi_omarginal,2):squeeze():float()
        self.Qi_omarginal = slse(self.Qi,543):squeeze() -- is there still underflow?
        self.Qi_omarginal:add(-self.batchZ:cuda():view(bS,1):expandAs(self.Qi_omarginal)) -- normalize marginal
        self.Qi:add( -self.batchZ:cuda():view(bS,1,1,1,1):expandAs(self.Qi)) -- normalize Qi
        
        -- calculate Qh again using the new Qi
        local newHmessages = sme(self.OVG:cuda(),self.Qi_omarginal:view(bS,nO,1),2)
        self.Qh:copy(Hmessages)
        self.Qh:add(newHmessages)
        self.Qh:add(-slse(self.Qh,2):expandAs(self.Qh)) -- normalize Qh
        -- debug
        if self.debug then
            print(('  ** Qi:%+.1e; Qo:%+.1e; Qh:%+.1e'):format(self.Qi:max(),self.Qi_omarginal:mean(),self.Qh:mean()))
        end
    end
    assert(self.Qi_omarginal:sum() ~= -math.huge,'underflow in Qi')
    assert(self.Qh:sum()           ~= -math.huge,'underflow in Qh')

    -- *** Calculate output messages
    self.newPmessages =  sme(self.OO, self.Qi_omarginal:view(bS,nO,1),2):float()
    self.newPmessages2 = sme(self.OO, self.Qi_omarginal:view(bS,1,nO),3):float()
    self.newHmessages =  sme(self.OVG,self.Qi_omarginal:view(bS,nO,1),2):float()

    self.newKmessages = self.Qi_omarginal:clone():float()
    self.newKgtmessages = torch.Tensor(bS,nO):zero()
    self.newHgtmessages = torch.Tensor(bS,nG):zero()
    for i=1,bS do
        self.newKgtmessages[{i,target[{i,1}]}] = 1
        self.newHgtmessages[{i,{}}] = self.OVG[{i,target[{i,1}],{}}]:float()
    end

    if self.debug then
        print(('  !! Hout:%+.1e(mean); Hout:%+.1e(norm);'):format(self.newHgtmessages:mean(),self.newHgtmessages:norm()))
    end
    -- loss is sum of distribution for GT configuration minus the normalization, then sum across batches
    local gt_score = torch.Tensor(bS)
    for i=1,bS do
        local iO,iV,iS = target[{i,1}],target[{i,2}],target[{i,3}]
        gt_score[i] = lse(self.Qi[{i,iO,iV,iS,{}}],1):squeeze()
    end
    self.output = torch.sum(gt_score)
    assert((self.output~=math.huge) and (self.output==self.output),"error in forward pass")

    -- store messages
    if self.train then
        local m = self:outMessages()
        self.MP:setside(input.ids, input.times, input.iteration, unpack(m)) 
    end

    -- Loss is the log likelihood of the ground truth 
    -- configuration given the current model [-inf,0]
    return self.output
end

function CRF_ATFloss:outMessages()
    return {self.newPmessages:clone(), self.newPmessages2:clone(), self.newKgtmessages:clone(), self.newKmessages:clone(), self.newHgtmessages:clone(), self.newHmessages:clone()}
end

function CRF_ATFloss:getMessages(ids,times)
    -- return the appropriate messages
    local nO, bS, m = self.nO, self.batchSize
    if self.train then 
        -- async messages
        m = self.MP:side(ids,times,nO,bS)
    elseif self:missingMessages() then
        -- zero messages
        m = self.MP:side(nil,nil,nO,bS)
    else
        -- sync messages
        local o = self:outMessages()
        m = self.MP:testside(times,unpack(o))
    end
    for i,x in ipairs(m) do m[i] = x:cuda() end
    for i,x in ipairs(m) do 
        assert(x:sum()~= math.huge,"message overflow in message " .. i )
        assert(x:sum()~=-math.huge,"message underflow in message " .. i)
    end
    return unpack(m)
end

function CRF_ATFloss:zeroMessages()
    self.newPmessages = nil --will be reset by side
end

function CRF_ATFloss:missingMessages()
    return self.newPmessages == nil
end

function CRF_ATFloss:actions(a2ov) -- call forward first
    -- Predicting actions, calculates the marginal of the actions
    -- a2ov is 0-index so #a2ov misses the 0th one.
    local out = torch.Tensor(self.batchSize,#a2ov+1)
    for a=1,#a2ov+1 do
        local o,v = unpack(a2ov[a-1]) --0-index
        local gt_g_marginal = lse(self.Qi[{{},{o+1},{v+1},{},{}}],4):squeeze()
        -- If using 'progress':
        local nO,nV,nS = 38,33,16
        local tmp1 = lse(self.Qi[{{},o+1,v+1,{},{}}],2)[{{},1,{}}]
        local tmp2 = lse(self.Qi[{{},o+1+nO,v+1+nV,{},{}}],2)[{{},1,{}}]
        local tmp3 = lse(self.Qi[{{},o+1+nO*2,v+1+nV*2,{},{}}],2)[{{},1,{}}]
        local gt_g_marginal = lse(tmp1:cat(tmp2,3):cat(tmp3,3), 3)
        out[{{},a}] = (lse(gt_g_marginal,2):squeeze():float()):exp()
        --out[{{},a}] = gt_g_marginal:float():exp():squeeze()
    end
    return out
end

function CRF_ATFloss:unariesgt(I,n)
    -- Just like softmax
    local target = torch.Tensor(self.batchSize,n):zero()
    for i=1,self.batchSize do
        target[{i,I[i]}] = 1
    end
    return target
end

function CRF_ATFloss:pairwisegt(I,n,m,map,nm)
    -- Just like softmax, except 2 dimensions flattened onto one
    local linearI = sub2ind(I,n,m):squeeze()
    local target = torch.Tensor(self.batchSize,nm):zero()
    for i=1,self.batchSize do
        target[{i,map[linearI[i]]}] = 1
    end
    return target
end

function CRF_ATFloss:mugt(I,n,message,message2)
    -- The soft target for the o,o' potential is 
    -- a weighted sum of the ground truth targets for
    -- the surrounding frames depending on their distance
    local target = torch.Tensor(self.batchSize,n,n):zero()
    for i=1,self.batchSize do
        target[{i,{},I[i]}] = target[{i,{},I[i]}] + message[{i,{}}]
        target[{i,I[i],{}}] = target[{i,I[i],{}}] + message2[{i,{}}]
    end
    return target:view(self.batchSize,n*n)
end

function CRF_ATFloss:mflatentgt(I,n,m,messages)
    -- messages is the sum of Hgtmessages and newHgtmessages
    -- The soft target for OVG is f(g) = f(o,v,g|o=o',v=v') 
    -- where o' and v' are the gt values for o and v
    local target = torch.Tensor(self.batchSize,n,m):zero()
    local norm = lse(messages,2):float():squeeze()
    for i=1,I:size(1) do 
        for g=1,self.nG do
            local iO,iV,iS = I[{i,1}],I[{i,2}],I[{i,3}]
            target[{i,iO,g}] = math.exp(messages[{i,g}]-norm[{i}])
        end
    end
    return target:view(self.batchSize,n*m):contiguous()
end


function CRF_ATFloss:zeroGradParameters()
    -- Do nothing
end

function CRF_ATFloss:updateGradInput(input, target)
    -- target is nBatch x 3
    assert(self.batchZ, "call forward first")
    local nO,nV,nS,nG,nOV,bS = self.nO,self.nV,self.nS,self.nG,self.nOV,self.batchSize

    -- Get ground truth "targets" (1 matrices)
    local targetO = self:unariesgt(target[{{},1}],nO)
    local targetV = self:unariesgt(target[{{},2}],nV)
    local targetOV = self:pairwisegt(target[{{},{1,2}}],nO,nV,self.ind2indOV,nOV)

    local Kmessages,Kgtmessages,Pmessages,Kmessages2,Kgtmessages2,Pmessages2,Hmessages,Hgtmessages = self:getMessages(input.ids,input.times)

    -- Get "soft" targets (Targets that are also distributions)
    local targetOVG = self:mflatentgt(target,nO,nG,Hgtmessages+self.newHgtmessages:cuda())
    local targetOO = self:mugt(target[{{},1}],nO,Kgtmessages:float(),Kgtmessages2:float())
    
    -- Get marginals, to compare with targets, i.e. grad ~= f(o) - 1_o
    -- Sum of all elements of self.out where variables match
    -- Marginalize out variables that aren't being considered
    local tmp1 = lse(self.Qi,345):squeeze()
    local tmp2 = lse(self.Qi,245):squeeze()
    local tmp4 = lse(self.Qi,45):view(bS,nO*nV):index(2,self.indOV[1]:long())
    local tmp = lse(self.Qi,345):view(bS,nO,1):expand(bS,nO,nG)
    tmp:add(self.Qh:cuda():view(bS,1,nG):expandAs(tmp))
    local tmp7 = tmp:contiguous():view(bS,nO*nG)
    local tmp8 = (Kmessages:view(bS,nO,1):expand(bS,nO,nO) +
        lse(self.Qi,345):view(bS,1,nO):expand(bS,nO,nO)):view(bS,nO*nO):squeeze()
    local tmp8b = (Kmessages2:view(bS,1,nO):expand(bS,nO,nO) + 
        lse(self.Qi,345):view(bS,nO,1):expand(bS,nO,nO)):view(bS,nO*nO):squeeze()

    self.gradInput = {}
    self.gradInput[1] = torch.exp(tmp1) - targetO:cuda()
    self.gradInput[2] = torch.exp(tmp2) - targetV:cuda()
    self.gradInput[3] = torch.exp(tmp4) - targetOV:cuda()

    -- We use a zero prior on the OVG potential. This seems to be sufficient to avoid instability
    self.gradInput[4] = torch.exp(tmp7) - targetOVG:cuda()  
    self.gradInput[4] = self.gradInput[4] + self.intent_decay*input[4]:clone()

    -- OO' term has tmp8 and tmp8b for past and future
    self.gradInput[5] = torch.exp(tmp8:view(bS,nO*nO)) + torch.exp(tmp8b:view(bS,nO*nO)) - targetOO:cuda() 

    if self.debug then
        print(('  $$ G1:%+.1e; G2:%+.1e; G3:%+.1e; G4:%+.1e; G5:%+.1e'):format(self.gradInput[1]:norm(),self.gradInput[2]:norm(),self.gradInput[3]:norm(),self.gradInput[4]:norm(),self.gradInput[5]:norm()))
    end

    return self.gradInput
end

function CRF_ATFloss:training()
   self.train = true
end

function CRF_ATFloss:evaluate()
   self.train = false
end

function CRF_ATFloss:__tostring__()
    return torch.type(self) ..
        string.format('((%d,%d,%d,%d,%d,%d) -> 1)',self.nO,self.nV,self.nS,self.nOV,self.nOS,self.nVS)
end

return nn.CRF_ATFloss
