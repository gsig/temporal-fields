require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

local LSE, Module = torch.class('nn.LSE', 'nn.Module')

local function simplelse(x,dim)
    -- not in place
    local x_max = torch.max(x,dim)
    x_max[x_max:eq(-math.huge)] = 0
    local out = (x + (-x_max):expandAs(x)):exp():sum(dim):log():add(x_max)
    return out
end
mylse = simplelse

function LSE:__init(dim)
    Module.__init(self)
    self.dim = dim
end

function LSE:parameters()
end

function LSE:updateOutput(input, target)
    self.output = simplelse(input,self.dim)
    assert(torch.all(self.output:eq(self.output)), 'nan in lse, Output ' .. self.output:ne(self.output):sum())
    return self.output
end

function LSE:updateGradInput(input, gradOutput)
    -- todo reuse gradinput
    assert(torch.all(gradOutput:eq(gradOutput)), 'nan in LSE, gradOutput ' .. gradOutput:eq(gradOutput):sum())
    linput = input:clone()
    lgoutput = gradOutput:clone()
    loutput = self.output:clone()
    self.gradInput = torch.exp(input-self.output:expandAs(input))
    --local lse = simplelse(input,3):resizeAs(input)
    --self.gradInput = torch.exp(input-lse)
    assert(self.gradInput:gt(1):sum()==0,'lse bug ' .. self.gradInput:gt(1):sum()==0)
    lginput = self.gradInput:clone()
    assert(torch.all(self.gradInput:eq(self.gradInput)), 'nan in LSE gradInput')
    self.gradInput:cmul(gradOutput:expandAs(self.gradInput))
    assert(torch.all(self.gradInput:eq(self.gradInput)), 'nan in LSE gradInput')
    print('lse grad output ' .. gradOutput:norm())
    print('lse grad norm ' .. self.gradInput:norm())
    return self.gradInput
end

function LSE:accGradParameters(input, gradOutput)
end

function LSE:zeroGradParameters()
end

function LSE:training()
end

function LSE:evaluate()
end

function LSE:reset(stdv)
end

function LSE:__tostring__()
    return torch.type(self) ..
        string.format(' dim %d',self.dim)
end

return nn.LSE
