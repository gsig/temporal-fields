--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local CharadesDataset = torch.class('resnet.CharadesDataset', M)

function CharadesDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function CharadesDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image1 = self:_loadImage(paths.concat(self.dir, path))
   local theimage = torch.Tensor(20,image1:size(2),image1:size(3))
   local prefix = string.sub(path,1,#path-6-5)
   local number = string.sub(path,#path-6-5+1,#path-5)
   for j = 1,10 do
       local thispath1 = prefix .. string.format('%06d',number-1+j) .. 'x' .. '.jpg'
       local thispath2 = prefix .. string.format('%06d',number-1+j) .. 'y' .. '.jpg'
       local image1 = self:_loadImage(paths.concat(self.dir, thispath1))
       local image2 = self:_loadImage(paths.concat(self.dir, thispath2))
       theimage[{(j-1)*2+1,{},{}}] = image1
       theimage[{(j-1)*2+1+1,{},{}}] = image2
   end
   
   local class = self.imageInfo.imageClass[i]
   local id = ffi.string(self.imageInfo.ids[i]:data())
   local obj = self.imageInfo.objClass and self.imageInfo.objClass[i] or 0
   local verb = self.imageInfo.verbClass and self.imageInfo.verbClass[i] or 0
   local scene = self.imageInfo.sceneClass and self.imageInfo.sceneClass[i] or 0
   local time = tonumber(string.gmatch(path, '-([^-]+)x.jpg')())

   return {
      name = path,
      time = time,
      input = theimage,
      target = class,
      id = id,
      obj = obj,
      verb = verb,
      scene = scene,
   }
end

function CharadesDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 1, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function CharadesDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255, 128.0/255 }, --flow vgg16
   std = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, -- I don't think caffe normalizes
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function CharadesDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         --t.ModerateRandomSizedCrop(224),
         --t.RandomCrop(224),
         t.RandomSizedCrop(224),
         --t.ColorJitter({
         --   brightness = 0.4,
         --   contrast = 0.4,
         --   saturation = 0.4,
         --}),
         --t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         --t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   elseif self.split == 'val2' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.CharadesDataset
