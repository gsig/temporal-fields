--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local MessagePassing = require 'layers/messagepassing'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

-- name of the modules in the same order as model:parameters()
-- assumes a single nn.Sequential
local function layer_names(model)
    local w = {}
    for i=1,#model.modules do
        local name = model.modules[i].name or ""
        local mw,_ = model.modules[i]:parameters()
        if mw then
            for k,_ in pairs(mw) do
                table.insert(w,name)
            end
        end
    end
    return w
end

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   optimState = optimState or {
      originalLR = opt.LR,
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   if opt.solver=='sgd' then
       self.solver = optim.sgd
   elseif opt.solver=='adam' then
       self.solver = optim.adam
   else
       assert(false,"Unknown solver")
   end

   self.opt = opt
   self.params, self.gradParams = model:parameters()
   self.L = #self.params
   self.LR_decay_freq = opt.LR_decay_freq
   self.message_passes = opt.message_passes
   self.optimState = {}
   local names = layer_names(self.model)
   assert(#names==self.L)
   for i=1,self.L do
       local layername = names[i] or ""
       self.optimState[i] = {}
       for k,v in pairs(optimState) do
           self.optimState[i][k] = v
       end
       if string.find(layername, "conv") then
           self.optimState[i].learningRate = opt.LR*opt.convLR
       end
       if string.find(layername, "fc8") then
           self.optimState[i].learningRate = opt.LR*opt.fc8LR
       end
   end
   self.iteration = 0
   self.MP = MessagePassing(opt)
end

function Trainer:train(opt, epoch, dataloader)
   -- Trains the model for a single epoch
   local LRM = self:learningRateModifier(epoch) 
   for l=1,self.L do 
      self.optimState[l].learningRate = self.optimState[l].originalLR*LRM
   end
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval(i)
      return function () return self.criterion.output, self.gradParams[i] end
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   self.criterion:training()
   self.model:zeroGradParameters()
   for n, sample in dataloader:run() do
      self.iteration = self.iteration + 1
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      sample.actiontarget = sample.target
      sample.target = torch.cat(sample.obj, sample.verb, 2):cat(sample.scene, 2)
      self:copyInputs(sample)

      local output = self.model:forward(self.input)

      local batchSize = output[1]:size(1)
      output.ids = sample.ids
      output.times = sample.times
      output.iteration = self.iteration
      local loss = self.criterion:forward(output, self.target:float(), sample.ids, sample.times)

      local gradInput = self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, gradInput)

      if n % opt.accumGrad == 0 then -- accumulate batches
          for i=1,self.L do -- sgd on invdividual layers
              self.solver(feval(i), self.params[i], self.optimState[i])
          end
          self.model:zeroGradParameters()
      end

      local actionoutput = self.criterion:actions(opt.dataopt.a2ov)
      local top1, top5 = self:computeScore(actionoutput, sample.actiontarget, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print(('%s | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         opt.name, epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params[1]:storage() == self.model:parameters()[1]:storage()) 

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(opt, epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   self.criterion:training() -- should use async messages
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      sample.actiontarget = sample.target
      sample.target = torch.cat(sample.obj, sample.verb, 2):cat(sample.scene, 2) 
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      local batchSize = output[1]:size(1) / nCrops

      output.ids = sample.ids
      output.times = sample.times
      output.iteration = self.iteration
      local loss = self.criterion:forward(output, self.target:float())

      local actionoutput = self.criterion:actions(opt.dataopt.a2ov)
      local top1, top5 = self:computeScore(actionoutput, sample.actiontarget, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print(('%s | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         opt.name, epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   self.criterion:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

-- Torch port of THUMOSeventclspr in THUMOS'15
local function mAP(conf, gt)
    local so,sortind = torch.sort(conf, 1, true) --desc order
    local tp = gt:index(1,sortind:view(-1)):eq(1):int()
    local fp = gt:index(1,sortind:view(-1)):eq(0):int()
    local npos = torch.sum(tp)

    fp = torch.cumsum(fp)
    tp = torch.cumsum(tp)
    local rec = tp:float()/npos
    local prec = torch.cdiv(tp:float(),(fp+tp):float())
    
    local ap = 0
    local tmp = gt:index(1,sortind:view(-1)):eq(1):view(-1)
    for i=1,conf:size(1) do
        if tmp[i]==1 then
            ap = ap+prec[i]
        end
    end
    ap = ap/npos

    return rec,prec,ap

end

local function charades_ap(outputs, gt)
   -- approximate version of the charades evaluation function
   -- For precise numbers, use the submission file with the official matlab script
   conf = outputs:clone()
   conf[gt:sum(2):eq(0):expandAs(conf)] = -math.huge -- This is to match the official matlab evaluation code. This omits videos with no annotations 
   ap = torch.Tensor(157,1)
   for i=1,157 do
       _,_,ap[{{i},{}}] = mAP(conf[{{},{i}}],gt[{{},{i}}])
   end
   return ap
end

local function tensor2str(x)
    str = ""
    for i=1,x:size(1) do
        if i == x:size(1) then
            str = str .. x[i]
        else
            str = str .. x[i] .. " "
        end
    end
    return str
end

function Trainer:test2(opt, epoch, dataloader)
   -- Computes the mAP over the whole videos

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = 1
   local N = 0
   local outputs = torch.Tensor(2000,157)
   local gt = torch.Tensor(2000,157)
   local names = {}

   if opt.dumpLocalize then
       frameoutputs = torch.Tensor(25*2000,157)
       framenames = {}
       framenr = {}
       nframe = 0
   end

   self.model:evaluate()
   self.criterion:evaluate() -- sync messages
   n2 = 0
   for n, sample in dataloader:run() do
      n2 = n2 + 1
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      sample.actiontarget = sample.target
      sample.target = torch.cat(sample.obj, sample.verb, 2):cat(sample.scene, 2):fill(1)
      self:copyInputs(sample)

      local output = self.model:forward(self.input)
      local batchSize = 25
      
      self.criterion:zeroMessages() 
      output.times = sample.times
      local loss
      for _=1,self.message_passes do
          loss = self.criterion:forward(output, self.target:float())
      end

      for i=1,batchSize-1 do -- make sure there is no error in the loader, this should be one video
          assert(torch.all(torch.eq(
              sample.actiontarget[{{i},{}}],
              sample.actiontarget[{{i+1},{}}]
          )))
      end

      local actionoutput = self.criterion:actions(opt.dataopt.a2ov)

      if opt.marginal=='max' then
          outputs[{{n2},{}}] = actionoutput:max(1)
      elseif opt.marginal=='mean' then
          outputs[{{n2},{}}] = actionoutput:mean(1)
      else
          assert(false,"wrong marginal option")
      end

      gt[{{n2},{}}] = sample.actiontarget[{{1},{}}]
      table.insert(names,sample.ids[1])

      if opt.dumpLocalize then
          frameoutputs[{{nframe+1,nframe+25},{}}] = actionoutput
          for b=1,25 do
              framenames[nframe+b] = sample.ids[1]
              framenr[nframe+b] = b
          end
          nframe = nframe+25
      end

      print(('%s | Test2: [%d][%d/%d]    Time %.3f  Data %.3f'):format(
         opt.name, epoch, n, size, timer:time().real, dataTime))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   self.criterion:training()
   outputs = outputs[{{1,n2},{}}] 
   gt = gt[{{1,n2},{}}] 
   ap = charades_ap(outputs, gt)

   print((' * Finished epoch # %d     mAP: %7.3f\n'):format(
      epoch, torch.mean(ap)))

   print('dumping output to file')
   local out = assert(io.open(self.opt.save .. "/epoch" .. epoch .. ".txt", "w"))
   for i=1,outputs:size(1) do
      out:write(names[i] .. " " .. tensor2str(outputs[{{i},{}}]:view(-1)) .. "\n")  
   end
   out:close()

   if opt.dumpLocalize then
       print('dumping localization output to file')
       frameoutputs = frameoutputs[{{1,nframe},{}}] 
       local out = assert(io.open(self.opt.save .. "/localize" .. epoch .. ".txt", "w"))
       for i=1,frameoutputs:size(1) do
          f = framenr[i]
          vidid = framenames[i]
          out:write(vidid .. " " .. f .. " " .. tensor2str(frameoutputs[{{i},{}}]:view(-1)) .. "\n")  
       end
       out:close()
   end

   return ap
end


function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Computes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRateModifier(epoch)
   -- Training schedule
   local decay = math.floor((epoch - 1) / self.LR_decay_freq)
   return math.pow(0.1, decay)
end

return M.Trainer
