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
--     gm.sample - a list of functions to sample from a given model
--
-- history: 
--     October 2013 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the infer
gm.sample = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local uniform = torch.uniform

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Helpers
--
local function computeZ(g)
   -- Locals
   local Tensor = torch.Tensor
   local nodePot = g.nodePot
   local edgePot = g.edgePot
   local edgeEnds = g.edgeEnds
   local nStates = g.nStates
   local nNodes = g.nNodes
   local nEdges = g.nEdges
   local maxStates = g.nodePot:size(2)
   
   -- Compute Z
   local y = ones(nNodes)
   local Z = 0
   while true do
      local pot = 1

      -- Nodes
      for n = 1,nNodes do
         pot = pot*nodePot[n][y[n]]
      end

      -- Edges
      for e = 1,nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         pot = pot*edgePot[e][y[n1]][y[n2]]
      end

      -- Update Z
      Z = Z + pot

      -- Go to next y
      for yInd = 1,nNodes do
         y[yInd] = y[yInd] + 1
         if y[yInd] <= nStates[yInd] then
            break
         else
            y[yInd] = 1
         end
      end

      -- Stop when we are done all y combinations
      if y:eq(1):sum() == nNodes then
         break
      end
   end

   -- Return Z
   return Z
end

local function sampleY(g,Z)
   -- Locals
   local Tensor = torch.Tensor
   local nodePot = g.nodePot
   local edgePot = g.edgePot
   local edgeEnds = g.edgeEnds
   local nStates = g.nStates
   local nNodes = g.nNodes
   local nEdges = g.nEdges
   local maxStates = g.nodePot:size(2)

   -- Sample...
   local y = ones(nNodes)
   local cumulativePot = 0
   local U = uniform(0,1)
   while true do
      local pot = 1

      -- Nodes
      for n = 1,nNodes do
         pot = pot * nodePot[n][y[n]]
      end

      -- Edges
      for e = 1,nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         pot = pot*edgePot[e][y[n1]][y[n2]]
      end

      -- Update cumulative potential
      cumulativePot = cumulativePot + pot

      if cumulativePot/Z > U then
         -- Take this y
         break
      end

      -- Go to next y
      for yInd = 1,nNodes do
         y[yInd] = y[yInd] + 1
         if y[yInd] <= nStates[yInd] then
            break
         else
            y[yInd] = 1
         end
      end
   end

   -- Samples
   return y
end

----------------------------------------------------------------------
-- exact, brute-force sampling: only adapted to super small graphs
--
function gm.sample.exact(graph, N)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','infer')
   end

   -- verbose
   if graph.verbose then
      print('<gm.sample.exact> doing exact sampling')
   end

   -- local vars
   local Tensor = torch.Tensor
   local nNodes = graph.nNodes
   local maxStates = graph.nodePot:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- Z
   local Z = computeZ(g)

   -- Samples
   local samples = zeros(N,nNodes)
   for i = 1,N do
      samples[i] = sampleY(g,Z)
   end
   return samples
end
