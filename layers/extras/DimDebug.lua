local DimDebug, Module = torch.class('nn.DimDebug', 'nn.Module')

function dimprint(x,prefix)
    prefix = prefix and prefix or ''
    if type(x)=='table' then
        print(prefix .. 'Table:')
        for i = 1,#x do
            print(prefix .. i .. ':')
            dimprint(x[i],'  '..prefix)
        end
    else
        -- assume tensor
        print(prefix .. 'Tensor dims:', unpack((#x):totable()))
    end
end

function DimDebug:__init(location)
    Module.__init(self)
    self.location = location
end

function DimDebug:updateOutput(input)
   print('DimDebug Forward')
   if self.location then
       print('Location ' .. self.location)
   end
   dimprint(input)
   self.output = input
   return self.output
end


function DimDebug:updateGradInput(input, gradOutput)
   print('DimDebug Backward')
   if self.location then
       print('Location ' .. self.location)
   end
   dimprint(gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function DimDebug:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif torch.type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end

return nn.DimDebug
