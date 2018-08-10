--
-- Class that implement message passing for the Asynchronous temporal field
-- Many of these equations do not make much sense without having the supplementary
-- material for Asynchronous temporal fields for reference
--
-- Gunnar Atli Sigurdsson, 2017
--
local M = {}
local MessagePassing = torch.class('MessagePassing', M)


function MessagePassing:__init(opt)
   self.sigma = opt.sigma -- stddev of gaussian kernel for frame-frame potential
   self.videoSize = opt.videoSize and opt.videoSize or 1 -- weight of frame-frame messages
   self.videoSizeGoal = opt.videoSizeGoal -- weight of goal-frame messages
   self.goals = opt.goals 
   self.cache_size = opt.cache_size -- How many recent messages to store for each video
   self.messageDecay = opt.messageDecay and tonumber(opt.messageDecay) or 0.9
   self.cache = {} -- The cache for the messages ("the message server")
end


local function kernel(t,t2,sigma)
    -- Computes the log of a Gaussian kernel between t and t2 with stddev sigma
    return -((t2-t)*(t2-t)/(2*sigma*sigma))
end

local function lse(x,dim)
    -- In-place log-sum-exp
    -- Sensitive to under/overflowing the source matrix
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

function MessagePassing:testside(times, Pmessage, Pmessage2, Kgtmessage, Kmessage, Hgtmessage, Hmessage)
    -- Simplifies MessagePassing:side when the examples
    -- Are seen only once (test time) and are in order
    local I = Pmessage:size(1)
    local N = Pmessage:size(2)
    local M = Hmessage:size(2)

    local newKmessage = torch.Tensor(I,N):zero()
    local newKgtmessage = torch.Tensor(I,N):zero()
    local newPmessage = torch.Tensor(I,N):zero()
    local newKmessage2 = torch.Tensor(I,N):zero()
    local newKgtmessage2 = torch.Tensor(I,N):zero()
    local newPmessage2 = torch.Tensor(I,N):zero()
    local newHmessage = torch.Tensor(I,M):zero()
    local newHgtmessage = torch.Tensor(I,M):zero()
    for i=1,I do
        local c = (i-1)-1+1
        local c2 = I-(i+1)+1 -- after
        if c>0 then
            -- There are messages from the past
            local tmpa = Kmessage[{{1,i-1},{}}]
            local tmpb = Kgtmessage[{{1,i-1},{}}]
            local tmpc = Pmessage[{{1,i-1},{}}]
            for j=1,c do
                tmpa[{j,{}}] = tmpa[{j,{}}] + kernel(times[j],times[i],self.sigma) -- K message is stored as log K
                tmpb[{j,{}}] = tmpb[{j,{}}] * torch.exp(kernel(times[j],times[i],self.sigma))
                tmpc[{j,{}}] = tmpc[{j,{}}] * torch.exp(kernel(times[j],times[i],self.sigma))
            end
            newKmessage[{i,{}}] = lse(tmpa+torch.log(1/tmpa:size(1)),1) * self.videoSize
            newKgtmessage[{i,{}}] = torch.mean(tmpb,1) * self.videoSize
            newPmessage[{i,{}}] = torch.mean(tmpc,1) * self.videoSize
        end
        if c2>0 then
            -- There are messages from the future
            local tmp2a = Kmessage[{{i+1,I},{}}]
            local tmp2b = Kgtmessage[{{i+1,I},{}}]
            local tmp2c = Pmessage2[{{i+1,I},{}}]
            for j=1,c2 do
                tmp2a[{j,{}}] = tmp2a[{j,{}}] + kernel(times[j+i],times[i],self.sigma) -- K message is stored as log K
                tmp2b[{j,{}}] = tmp2b[{j,{}}] * torch.exp(kernel(times[j+i],times[i],self.sigma))
                tmp2c[{j,{}}] = tmp2c[{j,{}}] * torch.exp(kernel(times[j+i],times[i],self.sigma))
            end
            newKmessage2[{i,{}}] = lse(tmp2a+torch.log(1/tmp2a:size(1)),1) * self.videoSize
            newKgtmessage2[{i,{}}] = torch.mean(tmp2b,1) * self.videoSize
            newPmessage2[{i,{}}] = torch.mean(tmp2c,1) * self.videoSize
        end

        -- Calculate messages without direction 
        local tmpd = Hmessage 
        local tmpe = Hgtmessage 
        newHmessage[{i,{}}] = torch.mean(tmpd,1) * self.videoSize
        newHgtmessage[{i,{}}] = torch.mean(tmpe,1) * self.videoSize
        assert((newKmessage2:sum()~=-math.huge) and (newKmessage2:sum()==newKmessage2:sum()),"error in testside")
    end
    return {newKmessage,newKgtmessage,newPmessage,newKmessage2,newKgtmessage2,newPmessage2,newHmessage,newHgtmessage}
end

function MessagePassing:side(ids,times,N,bS)
    -- Calculates incoming messages for a given time point [times] in a given video [ids]
    -- Tries to avoid potential underflow
    local M = self.goals
    local decay = self.messageDecay
    local Kmessage = torch.Tensor(bS,N):zero()
    local Kgtmessage = torch.Tensor(bS,N):zero()
    local sumPmessage = torch.Tensor(bS,N):zero()
    local Kmessage2 = torch.Tensor(bS,N):zero()
    local Kgtmessage2 = torch.Tensor(bS,N):zero()
    local sumPmessage2 = torch.Tensor(bS,N):zero()
    local Hgtmessage = torch.Tensor(bS,M):zero()
    local Hmessage = torch.Tensor(bS,M):zero()
    if ids then
        -- if there are no ideas, this returns zero messages
        for i=1,bS do
            if self.cache[ids[i]] then
                -- We have any messages from that video stored
                

                -- Step 1: Loop through the messages in 
                -- this video and prepare some numbers
                local c,c2,c3 = 0,0,0
                local Ntmp,Ntmp2 = 0,0
                local iter = {}
                local iter2 = {}
                local iter3 = {}
                for t,_ in pairs(self.cache[ids[i]]) do
                    if t <= times[i] then Ntmp = Ntmp + 1
                    else Ntmp2 = Ntmp2 + 1
                    end
                end
                local tmpa = torch.Tensor(Ntmp,N)
                local tmpb = torch.Tensor(Ntmp,N)
                local tmpc = torch.Tensor(Ntmp,N)
                local tmpd = torch.Tensor(Ntmp+Ntmp2,M)
                local tmpe = torch.Tensor(Ntmp+Ntmp2,M)
                local tmp2a = torch.Tensor(Ntmp2,N)
                local tmp2b = torch.Tensor(Ntmp2,N)
                local tmp2c = torch.Tensor(Ntmp2,N)
               
                for t,_ in pairs(self.cache[ids[i]]) do
                    if t <= times[i] then
                        c = c + 1
                        tmpa[{c,{}}] = self.cache[ids[i]][t].K + kernel(t,times[i],self.sigma) -- K is actually stored as log K, so we add
                        tmpb[{c,{}}] = self.cache[ids[i]][t].Kgt * torch.exp(kernel(t,times[i],self.sigma)) -- Kgt is just Kgt however
                        tmpc[{c,{}}] = self.cache[ids[i]][t].P * torch.exp(kernel(t,times[i],self.sigma)) -- so is P
                        table.insert(iter,self.cache[ids[i]][t].iter)
                    else 
                        c2 = c2 + 1
                        tmp2a[{c2,{}}] = self.cache[ids[i]][t].K + kernel(t,times[i],self.sigma)
                        tmp2b[{c2,{}}] = self.cache[ids[i]][t].Kgt * torch.exp(kernel(t,times[i],self.sigma))
                        tmp2c[{c2,{}}] = self.cache[ids[i]][t].P2 * torch.exp(kernel(t,times[i],self.sigma))
                        table.insert(iter2,self.cache[ids[i]][t].iter)
                    end
                    c3 = c3 + 1
                    tmpd[{c3,{}}] = self.cache[ids[i]][t].H
                    tmpe[{c3,{}}] = self.cache[ids[i]][t].Hgt
                    table.insert(iter3,self.cache[ids[i]][t].iter)
                end

                -- Step 2: Calculate messages. See 'Asynchronous Temporal Fields for Action Recognition' for details. Particularly the supplementary material. I do not know about a way to make message passing ever look pretty and not convoluted
                if c>0 then
                    -- If there are messages coming from the past
                    local _,ind = torch.sort(torch.Tensor(iter))
                    local tmp = torch.cumprod(torch.Tensor(ind:size(1)):fill(decay))
                    local decays = torch.Tensor(ind:size(1))
                    for d=1,ind:size(1) do
                        decays[ind[d]] = tmp[d]
                    end
                    decays = decays / decays:sum()
                    decays = decays:view(decays:size(1),1):expandAs(tmpa)
                    Kmessage[{i,{}}] = lse(tmpa+torch.log(decays),1) * self.videoSize -- K message is stored as log K: exp, weighted sum, then log back
                    Kgtmessage[{i,{}}] = torch.sum(tmpb:cmul(decays),1) * self.videoSize
                    sumPmessage[{i,{}}] = torch.sum(tmpc:cmul(decays),1) * self.videoSize
                    assert((Kmessage:sum()~=-math.huge) and (Kmessage:sum()==Kmessage:sum()),"error in side")
                end
                if c2>0 then
                    -- If there are messages from the future
                    local _,ind = torch.sort(torch.Tensor(iter2))
                    local tmp = torch.cumprod(torch.Tensor(ind:size(1)):fill(decay))
                    local decays = torch.Tensor(ind:size(1))
                    for d=1,ind:size(1) do
                        decays[ind[d]] = tmp[d]
                    end
                    decays = decays / decays:sum()
                    decays = decays:view(decays:size(1),1):expandAs(tmp2a)
                    Kmessage2[{i,{}}] = lse(tmp2a+torch.log(decays),1) * self.videoSize -- K message is stored as log K: exp, weighted sum, then log back
                    Kgtmessage2[{i,{}}] = torch.sum(tmp2b:cmul(decays),1) * self.videoSize
                    sumPmessage2[{i,{}}] = torch.sum(tmp2c:cmul(decays),1) * self.videoSize
                    assert((Kmessage2:sum()~=-math.huge) and (Kmessage2:sum()==Kmessage2:sum()),"error in side")
                end
                if c3>0 then
                    -- If there are messages without direction
                    local _,ind = torch.sort(torch.Tensor(iter3))
                    local tmp = torch.cumprod(torch.Tensor(ind:size(1)):fill(decay))
                    local decays = torch.Tensor(ind:size(1))
                    for d=1,ind:size(1) do
                        decays[ind[d]] = tmp[d]
                    end
                    decays = decays / decays:sum()
                    decays = decays:view(decays:size(1),1):expandAs(tmpd)
                    Hmessage[{i,{}}] = torch.sum(tmpd:cmul(decays),1) * self.videoSizeGoal
                    Hgtmessage[{i,{}}] = torch.sum(tmpe:cmul(decays),1) * self.videoSizeGoal
                end
            end
        end
    end
    return {Kmessage,Kgtmessage,sumPmessage,Kmessage2,Kgtmessage2,sumPmessage2,Hmessage,Hgtmessage}
end

function MessagePassing:setside(ids, times, iteration, Pmessage, Pmessage2, Kgtmessage, Kmessage, Hgtmessage, Hmessage)
    -- Cache messages for later use by :side
    for i=1,#ids do
        if not self.cache[ids[i]] then self.cache[ids[i]] = {} end
        self.cache[ids[i]][times[i]] = {
            P = Pmessage[{i,{}}],
            P2 = Pmessage2[{i,{}}],
            Kgt = Kgtmessage[{i,{}}],
            K = Kmessage[{i,{}}],
            Hgt = Hgtmessage[{i,{}}],
            H = Hmessage[{i,{}}],
            iter = iteration,
        } 
        local min = nil
        local argmin
        local count = 0
        for k,v in pairs(self.cache[ids[i]]) do
            count = count + 1
            if (not min) or (v.iter<min) then
                min = v.iter
                argmin = k
            end
        end
        if count > self.cache_size then
            -- this cache is too big, find oldest and remove it
            self.cache[ids[i]][argmin] = nil
        end
    end
end


return M.MessagePassing
