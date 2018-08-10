local BlockGradient, Parent = torch.class('nn.BlockGradient', 'nn.Module')

function BlockGradient:__init()
   Parent.__init(self)
end

function BlockGradient:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   return self.output
end

function BlockGradient:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:zero()
   return self.gradInput
end
