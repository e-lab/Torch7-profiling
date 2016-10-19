
-- Inception-V3 from this paper -
-- https://arxiv.org/pdf/1512.00567v3.pdf
-- and as visualized by http://dgschwend.github.io/netscope/#/preset/inceptionv3
-- Inception uses a 3x299x299 input

-- This module is from Figure 5 of Inception-V3 paper
local function inception_duplicate()
   local SpatialConvolution = nn.SpatialConvolution
   local SpatialMaxPooling = nn.SpatialMaxPooling
   local SpatialAveragePooling = nn.SpatialAveragePooling
   local ReLU = nn.ReLU

   local depth_concat = nn.Concat(2)
   local path1 = nn.Sequential()
   path1:add(SpatialConvolution(288,64,1,1,1,1,0,0)):add(ReLU(true))
   path1:add(SpatialConvolution(64,96,3,3,1,1,1,1)):add(ReLU(true))
   path1:add(SpatialConvolution(96,96,3,3,1,1,1,1)):add(ReLU(true))
   depth_concat:add(path1)

   local path2 = nn.Sequential()
   path2:add(SpatialConvolution(288,48,1,1,1,1,0,0)):add(ReLU(true))
   path2:add(SpatialConvolution(48,64,3,3,1,1,1,1)):add(ReLU(true))
   depth_concat:add(path2)

   local path3 = nn.Sequential()
   path3:add(SpatialAveragePooling(3,3,1,1,1,1))
   path3:add(SpatialConvolution(288,64,1,1,1,1,0,0))
   depth_concat:add(path3)

   local path4 = nn.Sequential()
   path4:add(SpatialConvolution(288,64,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path4)

   return depth_concat
end

-- This module is from Figure 6 of Inception-V3 paper
local function inception_asymmetric()
   local SpatialConvolution = nn.SpatialConvolution
   local SpatialMaxPooling = nn.SpatialMaxPooling
   local SpatialAveragePooling = nn.SpatialAveragePooling
   local ReLU = nn.ReLU

   local depth_concat = nn.Concat(2)
   local path1 = nn.Sequential()
   path1:add(SpatialConvolution(768,128,1,1,1,1,0,0)):add(ReLU(true))
   path1:add(SpatialConvolution(128,128,1,7,1,1,0,3)):add(ReLU(true))
   path1:add(SpatialConvolution(128,128,7,1,1,1,3,0)):add(ReLU(true))
   path1:add(SpatialConvolution(128,128,1,7,1,1,0,3)):add(ReLU(true))
   path1:add(SpatialConvolution(128,192,7,1,1,1,3,0)):add(ReLU(true))
   depth_concat:add(path1)

   local path2 = nn.Sequential()
   path2:add(SpatialConvolution(768,128,1,1,1,1,0,0)):add(ReLU(true))
   path2:add(SpatialConvolution(128,128,1,7,1,1,0,3)):add(ReLU(true))
   path2:add(SpatialConvolution(128,192,7,1,1,1,3,0)):add(ReLU(true))
   depth_concat:add(path2)

   local path3 = nn.Sequential()
   path3:add(SpatialAveragePooling(3,3,1,1,1,1))
   path3:add(SpatialConvolution(768,192,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path3)

   local path4 = nn.Sequential()
   path4:add(SpatialConvolution(768,192,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path4)

   return depth_concat

end

-- This and expanded2 are from Figure 7 of Inception-V3 paper
local function inception_asymmetric_expanded1()
   local SpatialConvolution = nn.SpatialConvolution
   local SpatialMaxPooling = nn.SpatialMaxPooling
   local SpatialAveragePooling = nn.SpatialAveragePooling
   local ReLU = nn.ReLU

   local depth_concat = nn.Concat(2)
   local path1 = nn.Sequential()
   path1:add(SpatialConvolution(1280,448,1,1,1,1,0,0)):add(ReLU(true))
   path1:add(SpatialConvolution(448,384,3,3,1,1,1,1)):add(ReLU(true))
   local path1_depth_concat = nn.Concat(2)
   local path1_1 = nn.Sequential()
   path1_1:add(SpatialConvolution(384,384,1,3,1,1,0,1)):add(ReLU(true))
   path1_depth_concat:add(path1_1)
   local path1_2 = nn.Sequential()
   path1_2:add(SpatialConvolution(384,384,3,1,1,1,1,0)):add(ReLU(true))
   path1_depth_concat:add(path1_2)
   path1:add(path1_depth_concat)
   depth_concat:add(path1)

   local path2 = nn.Sequential()
   path2:add(SpatialConvolution(1280,384,1,1,1,1,0,0)):add(ReLU(true))
   local path2_depth_concat = nn.Concat(2)
   local path2_1 = nn.Sequential()
   path2_1:add(SpatialConvolution(384,384,1,3,1,1,0,1)):add(ReLU(true))
   path2_depth_concat:add(path2_1)
   local path2_2 = nn.Sequential()
   path2_2:add(SpatialConvolution(384,384,3,1,1,1,1,0)):add(ReLU(true))
   path2_depth_concat:add(path2_2)
   path2:add(path2_depth_concat)
   depth_concat:add(path2)

   local path3 = nn.Sequential()
   path3:add(SpatialAveragePooling(3,3,1,1,1,1))
   path3:add(SpatialConvolution(1280,192,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path3)

   local path4 = nn.Sequential()
   path4:add(SpatialConvolution(1280,320,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path4)

   return depth_concat
end

local function inception_asymmetric_expanded2()
   local SpatialConvolution = nn.SpatialConvolution
   local SpatialMaxPooling = nn.SpatialMaxPooling
   local SpatialAveragePooling = nn.SpatialAveragePooling
   local ReLU = nn.ReLU

   local depth_concat = nn.Concat(2)
   local path1 = nn.Sequential()
   path1:add(SpatialConvolution(2048,448,1,1,1,1,0,0)):add(ReLU(true))
   path1:add(SpatialConvolution(448,384,3,3,1,1,1,1)):add(ReLU(true))
   local path1_depth_concat = nn.Concat(2)
   local path1_1 = nn.Sequential()
   path1_1:add(SpatialConvolution(384,384,1,3,1,1,0,1)):add(ReLU(true))
   path1_depth_concat:add(path1_1)
   local path1_2 = nn.Sequential()
   path1_2:add(SpatialConvolution(384,384,3,1,1,1,1,0)):add(ReLU(true))
   path1_depth_concat:add(path1_2)
   path1:add(path1_depth_concat)
   depth_concat:add(path1)

   local path2 = nn.Sequential()
   path2:add(SpatialConvolution(2048,384,1,1,1,1,0,0)):add(ReLU(true))
   local path2_depth_concat = nn.Concat(2)
   local path2_1 = nn.Sequential()
   path2_1:add(SpatialConvolution(384,384,1,3,1,1,0,1)):add(ReLU(true))
   path2_depth_concat:add(path2_1)
   local path2_2 = nn.Sequential()
   path2_2:add(SpatialConvolution(384,384,3,1,1,1,1,0)):add(ReLU(true))
   path2_depth_concat:add(path2_2)
   path2:add(path2_depth_concat)
   depth_concat:add(path2)

   local path3 = nn.Sequential()
   path3:add(SpatialAveragePooling(3,3,1,1,1,1))
   path3:add(SpatialConvolution(2048,192,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path3)

   local path4 = nn.Sequential()
   path4:add(SpatialConvolution(2048,320,1,1,1,1,0,0)):add(ReLU(true))
   depth_concat:add(path4)

   return depth_concat
end


local function inception_grid_reduce(n_input_maps, n_output_maps)
   local SpatialConvolution = nn.SpatialConvolution
   local SpatialMaxPooling = nn.SpatialMaxPooling
   local ReLU = nn.ReLU

   local conv_output_maps = (n_output_maps - n_input_maps)/2

   local depth_concat = nn.Concat(2)

   local path1 = nn.Sequential()
   path1:add(SpatialConvolution(n_input_maps,conv_output_maps,1,1,1,1,0,0)):add(ReLU(true))
   path1:add(SpatialConvolution(conv_output_maps,conv_output_maps,3,3,1,1,1,1)):add(ReLU(true))
   path1:add(SpatialConvolution(conv_output_maps,conv_output_maps,3,3,2,2,0,0)):add(ReLU(true))
   depth_concat:add(path1)

   local path2 = nn.Sequential()
   path2:add(SpatialConvolution(n_input_maps,conv_output_maps,1,1,1,1,0,0)):add(ReLU(true))
   path2:add(SpatialConvolution(conv_output_maps,conv_output_maps,3,3,2,2,0,0)):add(ReLU(true))
   depth_concat:add(path2)

   local path3 = nn.Sequential()
   path3:add(SpatialMaxPooling(3,3,2,2,0,0))
   depth_concat:add(path3)

   return depth_concat
end

local SpatialConvolution = nn.SpatialConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling
local SpatialAveragePooling = nn.SpatialAveragePooling
local ReLU = nn.ReLU

local model = nn.Sequential()
-- Begin Inception "stem"
model:add(SpatialConvolution(3,32,3,3,2,2,0,0)):add(ReLU(true))
model:add(SpatialConvolution(32,32,3,3,1,1,0,0)):add(ReLU(true))
model:add(SpatialConvolution(32,64,3,3,1,1,1,1)):add(ReLU(true))
model:add(SpatialMaxPooling(3,3,2,2,0,0))
model:add(SpatialConvolution(64,80,3,3,1,1,0,0)):add(ReLU(true))
model:add(SpatialConvolution(80,192,3,3,2,2,0,0)):add(ReLU(true))
model:add(SpatialConvolution(192,288,3,3,1,1,1,1)):add(ReLU(true))

model:add(inception_duplicate())
model:add(inception_duplicate())
model:add(inception_duplicate())
model:add(inception_grid_reduce(288,768))

model:add(inception_asymmetric())
model:add(inception_asymmetric())
model:add(inception_asymmetric())
model:add(inception_asymmetric())
model:add(inception_asymmetric())
model:add(inception_grid_reduce(768,1280))

model:add(inception_asymmetric_expanded1())
model:add(inception_asymmetric_expanded2())
model:add(SpatialAveragePooling(8,8,1,1,0,0))

model:add(nn.View(1024):setNumInputDims(3))
model:add(nn.Linear(1024,1000)):add(nn.ReLU(true))

model:get(1).gradInput = nil

return model
