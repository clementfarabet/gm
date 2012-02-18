----------------------------------------------------------------------
--
-- Copyright (c) 2012 Clement Farabet
-- 
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
-- 
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- 
----------------------------------------------------------------------
-- description:
--     gm.examples - a list of example/test functions
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the examples
gm.examples = {}

-- shortcuts
local tensor = torch.Tensor
local zeros = torch.zeros
local ones = torch.ones
local randn = torch.randn
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp
local floor = torch.floor
local ceil = math.ceil
local uniform = torch.uniform

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Simple example, doing decoding and inference
--
function gm.examples.simple()
   -- define graph structure
   nNodes = 10
   adjacency = ones(nNodes,nNodes) - eye(nNodes)
   nEdges = nNodes^2 - nNodes

   -- unary potentials
   nStates = 2
   nodePot = Tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,3},
                    {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

   -- joint potentials
   edgePot = Tensor(nEdges,nStates,nStates)
   basic = Tensor{{2,1}, {1,2}}
   for e = 1,nEdges do
      edgePot[e] = basic
   end

   -- create graph
   g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, 
                nodePot=nodePot, edgePot=edgePot, verbose=true}

   -- exact inference
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   --print('<gm.testme> edge beliefs:')
   --print(edgeBel)
   print('<gm.testme> log(Z):')
   print(logZ)

   -- bp inference
   local bp = g:decode('bp')
   print()
   print('<gm.testme> optimal config with belief propagation:')
   print(bp)

   local nodeBel,edgeBel,logZ = g:infer('bp')
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   --print('<gm.testme> edge beliefs:')
   --print(edgeBel)
   print('<gm.testme> log(Z):')
   print(logZ)
end

----------------------------------------------------------------------
-- Example of how to train a CRF for a simple segmentation task
--
function gm.examples.trainCRF()
   -- make training data
   sample = torch.load(paths.concat(paths.install_lua_path, 'gm', 'X.t7'))
   nRows,nCols = sample:size(1),sample:size(2)
   nNodes = nRows*nCols
   nStates = 2
   nInstances = 100
   -- make labels (MAP):
   y = tensor(nInstances,nRows*nCols)
   for i = 1,nInstances do
      y[i] = sample
   end
   y = y + 1
   -- make noisy training data:
   X = tensor(nInstances,1,nRows*nCols)
   for i = 1,nInstances do
      X[i] = sample
   end
   X = X + randn(X:size())/2
   -- display a couple of input examples
   require 'image'
   image.display{image={X[1]:reshape(32,32),X[2]:reshape(32,32),
                        X[3]:reshape(32,32),X[4]:reshape(32,32)}, 
                 zoom=4, padding=1, nrow=2, legend='training examples'}

   -- define adjacency matrix (4-connexity graph)
   local adj = zeros(nNodes,nNodes)
   for i = 1,nRows do
      for j = 1,nCols do
         local n = (i-1)*nCols + j
         if j < nCols then
            adj[n][n+1] = 1
         end
         if i < nRows then
            adj[n][n+nRows] = 1
         end
      end
   end
   adj:add(adj:t())

   -- create graph
   g = gm.graph{adjacency=adj, nStates=nStates, verbose=true, type='crf', maxIter=10}

   -- create node features (normalized X and a bias)
   Xnode = tensor(nInstances,2,nNodes)
   Xnode:select(2,1):fill(1) -- bias
   -- normalize features:
   nFeatures = X:size(2)
   for f = 1,nFeatures do
      local Xf = X:select(2,f)
      local mu = Xf:mean()
      local sigma = Xf:std()
      Xf:add(-mu):div(sigma)
   end
   Xnode:select(2,2):copy(X) -- features (simple normalized grayscale)
   nNodeFeatures = Xnode:size(2)

   -- tie node potentials to parameter vector
   nodeMap = zeros(nNodes,nStates,nNodeFeatures)
   for f = 1,nNodeFeatures do
      nodeMap:select(3,f):select(2,1):fill(f)
   end

   -- create edge features
   nEdges = g.edgeEnds:size(1)
   nEdgeFeatures = nNodeFeatures*2-1 -- sharing bias, but not grayscale features
   Xedge = zeros(nInstances,nEdgeFeatures,nEdges)
   for i = 1,nInstances do
      for e =1,nEdges do
         local n1 = g.edgeEnds[e][1]
         local n2 = g.edgeEnds[e][2]
         for f = 1,nNodeFeatures do
            -- get all features from node1
            Xedge[i][f][e] = Xnode[i][f][n1]
         end
         for f = 1,nNodeFeatures-1 do
            -- get all features from node1, except bias (shared)
            Xedge[i][nNodeFeatures+f][e] = Xnode[i][f+1][n2]
         end
      end
   end

   -- tie edge potentials to parameter vector
   local f = nodeMap:max()
   edgeMap = zeros(nEdges,nStates,nStates,nEdgeFeatures)
   for ef = 1,nEdgeFeatures do
      edgeMap:select(4,ef):select(3,1):select(2,1):fill(f+ef)
      edgeMap:select(4,ef):select(3,2):select(2,2):fill(f+ef)
   end

   -- initialize parameters
   g:initParameters(nodeMap,edgeMap)

   -- and train, for 3 epochs over the training data
   local learningRate=1e-3
   for iter = 1,nInstances*3 do
      local i = floor(uniform(1,nInstances)+0.5)
      local Xnodei = Xnode:narrow(1,i,1)
      local Xedgei = Xedge:narrow(1,i,1)
      local yi = y:narrow(1,i,1)
      local f,grad = g:nll(Xnodei,Xedgei,yi,'bp')
      g.w:add(-learningRate, grad)
      print('SGD @ iteration ' .. iter .. ': objective = ' .. f)
   end

   -- the model is trained, generate node/edge potentials, and test
   marginals = {}
   labelings = {}
   for i = 1,4 do
      g:makePotentials(Xnode[i],Xedge[i])
      nodeBel = g:infer('bp')
      labeling = g:decode('bp')
      table.insert(marginals,nodeBel:select(2,2):reshape(nRows,nCols))
      table.insert(labelings,labeling:reshape(nRows,nCols))
   end
   image.display{image=marginals, zoom=4, padding=1, nrow=2, legend='marginals'}
   image.display{image=labelings, zoom=4, padding=1, nrow=2, legend='labeling'}
end
