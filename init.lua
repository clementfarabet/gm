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
--     gm - a (super simple) graphical model package for Torch.
--          for now, it only provides means of decoding graphical
--          models (i.e. computing their highest potential state)
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

require 'xlua'
require 'torch'

-- package
gm = {}

-- C routines
require 'libgm'

-- extra code
torch.include('gm', 'decode.lua')
torch.include('gm', 'infer.lua')

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local Tensor = torch.Tensor
local sort = torch.sort

----------------------------------------------------------------------
-- creates a graph
--
function gm.graph(...)
   -- usage
   local _, adj, nStates, unaries, joints, verbose = dok.unpack(
      {...},
      'gm.graph',
      'create a graphical model from an adjacency matrix',
      {arg='adjacency', type='torch.Tensor', help='binary adjacency matrix (N x N)', req=true},
      {arg='nStates', type='number | torch.Tensor | table', help='number of states per node (N, or a single number)', default=1},
      {arg='unaries', type='torch.Tensor', help='unary/node potentials (N x nStates)'},
      {arg='joints', type='torch.Tensor', help='joint/edge potentials (N x nStates x nStates)'},
      {arg='verbose', type='boolean', help='verbose mode', default=false}
   )

   -- graph structure
   local graph = {}

   -- construct list of edges
   local nNodes = adj:size(1)
   local nEdges = adj:sum()/2
   local edgeEnds = zeros(nEdges,2)
   local k = 1
   for i = 1,nNodes do
      for j = 1,nNodes do
         if i < j and adj[i][j] == 1 then
            edgeEnds[k][1] = i
            edgeEnds[k][2] = j
            k = k + 1
         end
      end
   end

   -- count incident edges for each variable
   local nNei = zeros(nNodes)
   local nei = zeros(nNodes,nNodes)
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      nNei[n1] = nNei[n1] + 1
      nNei[n2] = nNei[n2] + 1
      nei[n1][nNei[n1]] = e
      nei[n2][nNei[n2]] = e
   end

   -- compute (V,E) with V[i] the sum of the nb of edges connected to 
   -- nodes (1,2,...,i-1) plus 1
   -- and E[i] the indexes of nodes connected to node i
   local V = zeros(nNodes+1)
   local E = zeros(nEdges*2)
   local edge = 1
   for n = 1,nNodes do
      V[n] = edge
      local nodeEdges = sort(nei[n]:narrow(1,1,nNei[n]))
      E:narrow(1,edge,nodeEdges:size(1)):copy(nodeEdges)
      edge = edge + nodeEdges:size(1)
   end
   V[nNodes+1] = edge

   -- create graph structure
   graph.edgeEnds = edgeEnds
   graph.V = V
   graph.E = E
   graph.nNodes = nNodes
   graph.nEdges = nEdges
   if type(nStates) == 'number' then
      graph.nStates = Tensor(nNodes):fill(nStates)
   elseif type(nStates) == 'table' then
      if #nStates ~= nNodes then
         error('#nStates must be equal to nNodes (= adjacency:size(1))')
      end
      graph.nStates = Tensor{nStates}
   end
   graph.adjacency = adj
   graph.verbose = verbose
   graph.timer = torch.Timer()

   -- store unaries/joints if given
   graph.unaries = unaries
   graph.joints = joints

   -- some functions
   graph.getEdgesOf = function(g,node)
      return g.E:narrow(1,g.V[node],g.V[node+1]-g.V[node])
   end

   graph.getNeighborsOf = function(g,node)
      local edges = g:getEdgesOf(node)
      local neighbors = Tensor(edges:size(1))
      local k = 1
      for i = 1,edges:size(1) do
         local edge = g.edgeEnds[edges[i]]
         if edge[1] ~= node then
            neighbors[k] = edge[1]
         else
            neighbors[k] = edge[2]
         end
         k = k + 1
      end
      return neighbors
   end

   graph.setFactors = function(g,unaries,joints)
      if not unaries or not joints then
         print(xlua.usage('setFactors',
               'set factors of an existing graph', nil,
               {type='torch.Tensor', help='unary potentials', req=true},
               {type='torch.Tensor', help='joint potentials', req=true}))
         xlua.error('missing arguments','setFactors')
      end
      g.unaries = unaries
      g.joints = joints
   end

   graph.decode = function(g,method,maxIter)
      if not method or not gm.decode[method] then
         local availmethods = {}
         for k in pairs(gm.decode) do
            table.insert(availmethods,k)
         end
         availmethods = table.concat(availmethods, ' | ')
         print(xlua.usage('decode',
               'compute optimal state of graph', nil,
               {type='string', help='decoding method: ' .. availmethods, req=true},
               {type='number', help='maximum nb of iterations (used by some methods)', default=1}))
         xlua.error('missing/incorrect method','decode')
      end
      graph.timer:reset()
      local state = gm.decode[method](g,maxIter)
      local t = graph.timer:time()
      if g.verbose then
         print('<gm.decode.'..method..'> decoded graph in ' .. t.real .. 'sec')
      end
      return state
   end

   graph.infer = function(g,method,maxIter)
      if not method or not gm.infer[method] then
         local availmethods = {}
         for k in pairs(gm.infer) do
            table.insert(availmethods,k)
         end
         availmethods = table.concat(availmethods, ' | ')
         print(xlua.usage('infer',
               'compute optimal state of graph', nil,
               {type='string', help='inference method: ' .. availmethods, req=true},
               {type='number', help='maximum nb of iterations (used by some methods)', default=1}))
         xlua.error('missing/incorrect method','infer')
      end
      graph.timer:reset()
      local nodeBel,edgeBel,logZ = gm.infer[method](g,maxIter)
      local t = graph.timer:time()
      if g.verbose then
         print('<gm.infer.'..method..'> performed inference on graph in ' .. t.real .. 'sec')
      end
      return nodeBel,edgeBel,logZ
   end

   graph.getPotentialForConfig = function(g,config)
      if not config then
         print(xlua.usage('getPotentialForConfig',
               'get potential for a given configuration', nil,
               {type='torch.Tensor', help='configuration of all nodes in graph', req=true}))
         xlua.error('missing config','getPotentialForConfig')
      end
      -- locals
      local unaries = g.unaries
      local joints = g.joints
      local pot = 1
      -- nodes
      for n = 1,g.nNodes do
         pot = pot * unaries[n][config[n]]
      end
      -- edges
      for e = 1,g.nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         pot = pot * joints[e][config[n1]][config[n2]]
      end
      -- return potential
      return pot
   end

   graph.getLogPotentialForConfig = function(g,config)
      if not config then
         print(xlua.usage('getLogPotentialForConfig',
               'get log potential for a given configuration', nil,
               {type='torch.Tensor', help='configuration of all nodes in graph', req=true}))
         xlua.error('missing config','getPotentialForConfig')
      end
      -- locals
      local unaries = g.unaries
      local joints = g.joints
      local logpot = 1
      -- nodes
      for n = 1,g.nNodes do
         logpot = logpot + math.log(unaries[n][config[n]])
      end
      -- edges
      for e = 1,g.nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         logpot = logpot + math.log(joints[e][config[n1]][config[n2]])
      end
      -- return potential
      return logpot
   end

   local tostring = function(g)
      local str = 'gm.GraphicalModel\n'
      str = str .. ' + nb of nodes: ' .. g.nNodes .. '\n'
      str = str .. ' + nb of edges: ' .. g.nEdges .. '\n'
      str = str .. ' + maximum nb of states per node: ' .. g.nStates:max()
      return str
   end
   setmetatable(graph, {__tostring=tostring})

   -- verbose?
   if graph.verbose then
      print('<gm.graph> created new graphical model:')
      print(tostring(graph))
   end

   -- return result
   return graph
end

----------------------------------------------------------------------
-- example
--
function gm.testme()
   -- define graph structure
   local nNodes = 10
   local adjacency = ones(nNodes,nNodes) - eye(nNodes)
   local nEdges = nNodes^2 - nNodes

   -- unary potentials
   local nStates = 2
   local unaries = Tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,3},
                          {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

   -- joint potentials
   local joints = Tensor(nEdges,nStates,nStates)
   local basic = Tensor{{2,1}, {1,2}}
   for e = 1,nEdges do
      joints[e] = basic
   end

   -- create graph
   local g = gm.graph{adjacency=adjacency, nStates=nStates, 
                      unaries=unaries, joints=joints, verbose=true}

   -- exact inference
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   print('<gm.testme> log(Z):')
   print(logZ)

   -- bp inference
   local bp,nodeBel = g:decode('bp',10)
   print()
   print('<gm.testme> optimal config with belief propagation:')
   print(bp)

   local nodeBel,edgeBel,logZ = g:infer('bp',10)
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   print('<gm.testme> log(Z):')
   print(logZ)

   -- done
   return g
end

