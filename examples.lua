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
   -- define graph
   nNodes = 10
   nStates = 2
   adjacency = gm.adjacency.full(nNodes)
   g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, verbose=true}

   -- unary potentials
   nodePot = tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,1},
                    {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

   -- joint potentials
   edgePot = tensor(g.nEdges,nStates,nStates)
   basic = tensor{{2,1}, {1,2}}
   for e = 1,g.nEdges do
      edgePot[e] = basic
   end

   -- set potentials
   g:setPotentials(nodePot,edgePot)

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
-- Example of how to train an MRF
--
function gm.examples.trainMRF()
   -- define graph:
   nNodes = 10
   nStates = 2
   adjacency = torch.zeros(nNodes,nNodes)
   for i = 1,nNodes-1 do
      adjacency[i][i+1] = 1
      adjacency[i+1][i] = 1
   end
   g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, type='mrf', verbose=true}

   -- define training set:
   nInstances = 100
   Y = tensor(nInstances,nNodes)
   for i = 1,nInstances do
      -- each entry is either 1 or 2, with a probability that
      -- increases with the node index
      for n = 1,nNodes do
         Y[i][n] = torch.bernoulli((n-1)/(nNodes-1)) + 1
      end
      -- create correlation between last two nodes
      Y[i][nNodes-1] = Y[i][nNodes]
   end

   -- NOTE: the 10 training nodes in Y have probability 0, 1/9, ... , 9/9 to be equal
   -- to 2. The node beliefs obtained after training should show that.

   -- tie node potentials to parameter vector
   -- NOTE: we allocate one parameter per node, to properly model
   -- the probability of each node
   nodeMap = zeros(nNodes,nStates)
   for n = 1,nNodes do
      nodeMap[{ n,1 }] = n
   end

   -- tie edge potentials to parameter vector
   -- NOTE: we allocate parameters globally, i.e. parameters model
   -- pairwise relations globally
   nEdges = g.edgeEnds:size(1)
   edgeMap = zeros(nEdges,nStates,nStates)
   edgeMap[{ {},1,1 }] = nNodes+1
   edgeMap[{ {},2,2 }] = nNodes+2
   edgeMap[{ {},1,2 }] = nNodes+3

   -- initialize parameters
   g:initParameters(nodeMap,edgeMap)
   
   -- estimate nll:
   require 'optim'
   optim.lbfgs(function()
      local f,grad = g:nll('exact',Y)
      print('LBFGS – objective = ', f)
      return f,grad
   end, g.w, {maxIter=100, lineSearch=optim.lswolfe})

   -- gen final potentials
   g:makePotentials()

   -- exact decoding:
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   -- exact inference:
   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs (prob that node=2)')
   print(nodeBel[{ {},2 }])
   print('<gm.testme> edge beliefs (prob that node1=2 & node2=2)')
   print(edgeBel[{ {},2,2 }])
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
   require 'gfx.js'
   gfx.image({
      X[1]:reshape(32,32),
      X[2]:reshape(32,32),
      X[3]:reshape(32,32),
      X[4]:reshape(32,32)
   }, {
      zoom=4, legend='training examples'
   })

   -- define adjacency matrix (4-connexity lattice)
   local adj = gm.adjacency.lattice2d(nRows,nCols,4)

   -- create graph
   g = gm.graph{adjacency=adj, nStates=nStates, verbose=true, type='crf', maxIter=10}

   -- create node features (normalized X and a bias)
   Xnode = tensor(nInstances,2,nNodes)
   Xnode[{ {},1 }] = 1 -- bias
   -- normalize features:
   nFeatures = X:size(2)
   for f = 1,nFeatures do
      local Xf = X[{ {},f }]
      local mu = Xf:mean()
      local sigma = Xf:std()
      Xf:add(-mu):div(sigma)
   end
   Xnode[{ {},2 }] = X -- features (simple normalized grayscale)
   nNodeFeatures = Xnode:size(2)

   -- tie node potentials to parameter vector
   nodeMap = zeros(nNodes,nStates,nNodeFeatures)
   for f = 1,nNodeFeatures do
      nodeMap[{ {},1,f }] = f
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
      edgeMap[{ {},1,1,ef }] = f+ef
      edgeMap[{ {},2,2,ef }] = f+ef
   end

   -- initialize parameters
   g:initParameters(nodeMap,edgeMap)

   -- and train on 30 samples
   require 'optim'
   local sgdState = {
      learningRate = 1e-3,
      learningRateDecay = 1e-2,
      weightDecay = 1e-5,
   }
   for iter = 1,100 do
      -- SGD step:
      optim.sgd(function()
         -- random sample:
         local i = torch.random(1,nInstances)
         -- compute f+grad:
         local f,grad = g:nll('bp',y[i],Xnode[i],Xedge[i])
         -- verbose:
         print('SGD @ iteration ' .. iter .. ': objective = ', f)
         -- return f+grad:
         return f,grad
      end, 
      g.w, sgdState)
   end

   -- the model is trained, generate node/edge potentials, and test
   marginals = {}
   labelings = {}
   for i = 1,4 do
      g:makePotentials(Xnode[i],Xedge[i])
      nodeBel = g:infer('bp')
      labeling = g:decode('bp')
      table.insert(marginals,nodeBel[{ {},2 }]:reshape(nRows,nCols))
      table.insert(labelings,labeling:reshape(nRows,nCols))
   end

   -- display
   gfx.image(marginals, {
      zoom=4, legend='marginals'
   })
   for _,labeling in ipairs(labelings) do
      labeling:add(-1)
   end
   gfx.image(labelings, {
      zoom=4, legend='labelings'
   })
end
