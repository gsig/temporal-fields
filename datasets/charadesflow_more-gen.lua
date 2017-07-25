--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function parseCSV(filename)
    require 'csvigo'
    print(('Loading csv: %s'):format(filename))
    local all = csvigo.load{path=filename, mode='tidy'}
    local ids,actionss = all['id'],all['actions']
    local N = #ids
    local labels = {}
    for i = 1,#ids do
        local id, actions = ids[i], actionss[i]
        local label = {}
        for a in string.gmatch(actions, '([^;]+)') do -- split on ';'
            local a = string.gmatch(a, '([^ ]+)') -- split on ' '
            table.insert(label,{c=a(), s=tonumber(a()), e=tonumber(a())})
        end
        labels[id] = label
    end
    return labels
end

local function getSceneFromCSV(filename)
    require 'csvigo'
    print(('Loading csv for scenes: %s'):format(filename))
    local all = csvigo.load{path=filename, mode='tidy'}
    local ids, scenes = all['id'],all['scene']
    local labels, sceneMap = {},{}
    local file = io.open("../scene_names.txt")
    local i = 0
    for line in file:lines() do
        sceneMap[line] = i
        i = i + 1
    end
    file:close()
    for i = 1,#ids do 
        labels[ids[i]] = sceneMap[scenes[i]]
    end
    return labels
end

local function loadActionToObjectVerb(filename)
    local file = io.open(filename)
    local action_to_ov = {}
    assert(file,'Action map (Charades_v1_mapping.txt) is missing. [opt.actionmap]')
    for line in file:lines() do
        local a,o,v = line:match('(%S+)%s(%S+)%s(%S+)')
        a = tonumber(string.sub(a,2,-1))
        o = tonumber(string.sub(o,2,-1))
        v = tonumber(string.sub(v,2,-1))
        action_to_ov[a] = {o,v}
    end
    file:close()
    return action_to_ov
end

local function uniquerows(x)
    local keep = torch.ByteTensor(x:size(1)):fill(1)
    local hist = {}
    for i=1,x:size(1) do
        hash = string.format("%d %d", x[i][1], x[i][2])
        if not hist[hash] then
            hist[hash] = ''
        else
            keep[i] = 0
        end
    end
    return x[keep:view(keep:size(1),1):expandAs(x)]:unfold(1,2,2)
end

local function list2tensor(imagePaths)
    local nImages = #imagePaths
    local maxLength = -1
    for _,p in pairs(imagePaths) do
        maxLength = math.max(maxLength, #p + 1)
    end
    local imagePath = torch.CharTensor(nImages, maxLength):zero()
    for i, path in ipairs(imagePaths) do
       ffi.copy(imagePath[i]:data(), path)
    end
    return imagePath
end


local function prepare(opt,labels,split,scenes)
    require 'sys'
    require 'string'
    local imagePath = torch.CharTensor()
    local imageClass = torch.LongTensor()
    local dir = opt.data
    assert(paths.dirp(dir), 'directory not found: ' .. dir)
    local imagePaths,imageClasses,ids = {},{},{}
    local objClasses,verbClasses,sceneClasses= {},{},{}
    local FPS,GAP,testGAP = 24,4,25
    local nO,nV,nS = 38,33,16
    local flowframes = 10

    -- Get action to (object,verb) mapping
    local a2ov = loadActionToObjectVerb(opt.actionmap)
    opt.a2ov = a2ov

    local BUFFER = 4000000
    -- For each video annotation, prepare test files
    local imageClasses2
    if split=='val_video' then
        imageClasses2 = torch.IntTensor(BUFFER, opt.nClasses):zero()
    end
    local e,count = 0,0
    for id,label in pairs(labels) do
        e = e+1
        if e % 100 == 1 then print(e) end
        local scene = scenes[id]
        iddir = dir .. '/' .. id
        local f = io.popen(('find -L %s -iname "*.jpg" '):format(iddir))
        if not f then 
            print('class not found: ' .. id)
            print(('find -L %s -iname "*.jpg" '):format(iddir))
        else
            local lines = {}
            while true do
                local line = f:read('*line')
                if not line then break end
                table.insert(lines,line)
            end
            local N = torch.floor(#lines/2)

            -- Validation annotations
            if split=='val_video' then
                local target = torch.IntTensor(157,1):zero()
                for _,anno in pairs(label) do
                    target[1+tonumber(string.sub(anno.c,2,-1))] = 1 -- 1-index
                end
                local tmp = torch.linspace(1,N-flowframes-1,testGAP)
                for ii = 1,testGAP do
                    local i = torch.floor(tmp[ii])
                    local impath = iddir .. '/' .. id .. '-' .. string.format('%06d',i) .. 'x' .. '.jpg'
                    count = count + 1
                    imageClasses2[count]:copy(target)
                    table.insert(ids,id)
                    table.insert(imagePaths,impath)
                end
            --
            -- Training annotation 
            elseif opt.setup == 'softmax' then
                if #label>0 then 
                    for _,anno in pairs(label) do
                        for i = 1,N,GAP do
                            if (anno.s<(i-1)/FPS) and ((i-1)/FPS<anno.e) then
                                if i+flowframes+1<N then
                                    local impath = iddir .. '/' .. id .. '-' .. string.format('%06d',i) .. 'x' .. '.jpg'
                                    local place = 1+torch.floor(3*((i-1)/FPS-anno.s)/anno.e) -- {1,2,3}
                                    if not (place==1 or place==2 or place==3) then place = 2 end
                                    local a = 1+tonumber(string.sub(anno.c,2,-1))
                                    local o,v = unpack(a2ov[a-1])
                                    table.insert(imagePaths,impath)
                                    table.insert(imageClasses, a) -- 1-index
                                    table.insert(ids,id)
                                    --table.insert(objClasses,o+1)
                                    --table.insert(verbClasses,v+1)
                                    --To use 'progress', just enumerate all combinations, and replace with old obj/verb
                                    table.insert(objClasses,o+1+nO*(place-1))
                                    table.insert(verbClasses,v+1+nV*(place-1))

                                    table.insert(sceneClasses,scene+1)
                                end
                            end
                        end
                    end
                end
            elseif opt.setup == 'sigmoid' then
                -- TODO
            else
                assert(false,'Invalid opt.setup')
            end
            f:close()
        end
    end

    -- Convert the generated list to a tensor for faster loading
    imagePaths = list2tensor(imagePaths) 
    collectgarbage() 
    ids_tensor = list2tensor(ids) 
    collectgarbage() 
    local imageClass, objClass, verbClass, sceneClass
    if split=='val_video' then
        imageClass = imageClasses2[{{1,count},{}}]
    else 
        imageClass = torch.LongTensor(imageClasses)
        objClass = torch.LongTensor(objClasses)
        verbClass = torch.LongTensor(verbClasses)
        sceneClass = torch.LongTensor(sceneClasses)
    end
    assert(imagePaths:size(1)==imageClass:size(1),"Sizes do not match")

    return imagePaths, imageClass, ids_tensor, objClass, verbClass, sceneClass 
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local filename = opt.trainfile
   local filenametest = opt.testfile 

   local scenes = getSceneFromCSV(filename)
   local scenes2 = getSceneFromCSV(filenametest)
   for k,v in pairs(scenes2) do scenes[k] = v end

   local labels = parseCSV(filename)
   print('done parsing train csv')
   local labelstest = parseCSV(filenametest)
   print('done parsing test csv')

   print(" | finding all validation images")
   local valImagePath, valImageClass, valids, valobjClass, valverbClass, valsceneClass = prepare(opt,labelstest,'val',scenes)

   print(" | finding all validation2 images")
   local val2ImagePath, val2ImageClass, val2ids = prepare(opt,labelstest,'val_video',scenes)

   print(" | finding all training images")
   local trainImagePath, trainImageClass, trainids, trainobjClass, trainverbClass, trainsceneClass = prepare(opt,labels,'train',scenes)

   -- only consider combinations that happen in training set
   local indOV = uniquerows(trainobjClass:cat(trainverbClass,2))
   local indOS = uniquerows(trainobjClass:cat(trainsceneClass,2))
   local indVS = uniquerows(trainverbClass:cat(trainsceneClass,2))
   local nO,nV,nS = 3*38,3*33,16
   local a2ov = loadActionToObjectVerb(opt.actionmap)

   local crfopt = {nO,nV,nS,indOV,indOS,indVS}
   crfopt.a2ov = a2ov

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
         ids = trainids,
         objClass = trainobjClass,
         verbClass = trainverbClass,
         sceneClass = trainsceneClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
         ids = valids,
         objClass = valobjClass,
         verbClass = valverbClass,
         sceneClass = valsceneClass,
      },
      val2 = {
         imagePath = val2ImagePath,
         imageClass = val2ImageClass,
         ids = val2ids,
      },
      opt = crfopt,
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
