require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

local BuildDist, Module = torch.class('nn.BuildDist', 'nn.Module')

function BuildDist:__init(dims,potentials)
    -- dim is a table of the dimensions
    -- potentials is a table of {dims,[inds]}
    Module.__init(self)
    self.dims = dims
    self.potentials = potentials
    self.buffers = {}
    self.gradInput = {}
    self.normalize = true
    self.Z = 0
    for i,d in pairs(potentials) do
        table.insert(self.buffers,torch.CudaTensor())
        table.insert(self.gradInput,torch.CudaTensor())
    end
end

local function notnan(x) return torch.all(x:eq(x)) end

local function simplelse(x,dim)
    -- not in place
    local x_max = torch.max(x,dim)
    x_max[x_max:eq(-math.huge)] = 0
    local out = (x + (-x_max):expandAs(x)):exp():sum(dim):log():add(x_max)
    return out
end

local function collapse(x,dims,fun)
    -- lse over dimensions dims
    dims = type(dims)=='table' and dims or dims:totable()
    if #dims==0 then
        return x
    elseif #dims==1 then
        return fun(x,dims[1])
    else 
        local out = fun(x,dims[1])
        dims = torch.Tensor(dims):narrow(1,2,#dims-1)
        return collapse(out,dims,fun)
    end
end

local function meancollapse(x,dims) return collapse(x,dims,torch.mean) end
local function sumcollapse(x,dims) return collapse(x,dims,torch.sum) end
local function lsecollapse(x,dims) return collapse(x,dims,simplelse) end

local function addto(x,dims,y)
    -- fit x to y given that x corresponds to dimensions of y given by dims
    -- fill in x's dimentions for view, rest is 1 and will be expanded
    local sizes = torch.Tensor(##y):fill(1)
    sizes[1] = y:size(1) -- #batches
    for i,d in ipairs(dims) do
        sizes[d+1] = x:size(i+1) -- +1 for bathes
    end
    return y:add(x:view(unpack(sizes:totable())):expandAs(y))
end

local function sub2ind(ind, n, m)
   -- subscript to linear index
   -- ind is Nx2 tensor
   -- todo generalize to any dims
   return m*(ind[{{},1}]-1)+ind[{{},2}]
end

local function scatteraddto(x,dims,y,ind,buffer)
    -- fit x to y given that x corresponds to dimensions of y given by dims
    -- create a buffer of right size, and fill in the numbers
    -- then call addto as before
    local b = y:size(1)
    local buffersizes = torch.LongTensor(#y):index(1,torch.LongTensor(dims)+1):totable()
    buffer:resize(b,unpack(buffersizes))
    buffer:fill(-math.huge)
    local linearind = sub2ind(ind,unpack(buffersizes))
    buffer:view(b,-1):scatter(2,linearind:view(1,-1):expandAs(x),x)
    buffer:view(b,unpack(buffersizes))
    return addto(buffer,dims,y)
end

local function collapse_except(x,dims,fun)
    -- collapse all dimensions of x except those in dims
    local sizes = torch.range(2,##x) -- collapse dimensions omitting batches
    for i,d in ipairs(dims) do sizes[d] = -1 end
    sizes = sizes[sizes:ne(-1)]:totable()
    return collapse(x,sizes,fun)
end
local function invcollapse(x,dim) return collapse_except(x,dim,torch.sum) end

local function scattercollapse_except(x,dim,ind,fun)
    -- same as collapse except only keep sparse indices defined by ind
    local linearind = sub2ind(ind,unpack(dim))
    return collapse_except(x,dim,fun):view(x:size(1),-1):index(2,linearind)
end
local function invcolscatter(x,dim,ind) return scattercollapse_except(x,dim,ind,torch.sum) end

function BuildDist:updateOutput(input, target)
    -- input should be a table of "potentials" in the same order as __init
    -- potentials are assumed to be in log space
    local b = input[1]:size(1)
    self.output = self.output
        and self.output:resize(b,unpack(self.dims)):zero() 
        or torch.CudaTensor(b,unpack(self.dims)):zero()
    for i,p in ipairs(self.potentials) do
        if #p == 1 then
            -- full potential
            addto(input[i],p[1],self.output)
        else
            -- only certain values in the potential predicted and rest assumed -inf
            scatteraddto(input[i],p[1],self.output,p[2],self.buffers[i])
        end
    end
    if self.normalize then
        -- sum over probability space (excluding batches)
        self.Z = lsecollapse(self.output,torch.range(2,##self.output))
        self.output:add(-self.Z:expandAs(self.output))
    end
    return self.output
end

function BuildDist:updateGradInput(input, gradOutput)
    -- gradient is the exp of the difference between the input and the Dist with all other dimensions "removed" (collapsed)
    assert(notnan(gradOutput), 'nan in build dist, gradOutput ' .. gradOutput:eq(gradOutput):sum()) 
    for i,p in ipairs(self.potentials) do
        self.gradInput[i]:resizeAs(input[i])
        if #p == 1 then
            --self.gradInput[i]:copy(torch.exp(input[i]-collapse(gradOutput,p[1])))
            self.gradInput[i]:copy(invcollapse(gradOutput,p[1]))
        else
            --self.gradInput[i]:copy(torch.exp(input[i]-scattercollapse(gradOutput,p[1],p[2])))
            self.gradInput[i]:copy(invcolscatter(gradOutput,p[1],p[2]))
        end
        assert(notnan(input[i]), 'nan in build dist input, potential ' .. i) 
        assert(notnan(self.gradInput[i]), 'nan in build dist gradInput, potential ' .. i) 
        print('bd grad norm ' .. i .. ': '.. self.gradInput[i]:norm())
    end
    return self.gradInput
end

function BuildDist:__tostring__()
    return torch.type(self) ..
        tostring(torch.Tensor(self.dims))
end

return nn.BuildDist
