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
--     gm.decode - a list of functions to decode optimal states
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the decode
gm.decode = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local Tensor = torch.Tensor
local sort = torch.sort

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- exact decoding: only adapted to super small graphs
--
function gm.decode.exact(graph)
   -- check args
   if not graph.unaries or not graph.joints then
      xlua.error('missing unaries/joints, please call graph:setFactors(...)','decode')
   end

   -- verbose
   if graph.verbose then
      print('<gm.decode.bp> decoding using exhaustive search')
   end

   -- local vars
   local nNodes = graph.nNodes
   local maxStates = graph.unaries:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates

   -- init
   local config = ones(nNodes)
   local optimalconfig = ones(nNodes)

   -- decode, exactly
   local maxpot = -1
   while true do
      -- get potential for configuration
      local pot = graph:getPotentialForConfig(config)

      -- compare configurations
      if pot > maxpot then
         maxpot = pot
         optimalconfig:copy(config)
      end

      -- next config
      local idx
      for i = 1,nNodes do
         idx = i
         config[i] = config[i] + 1
         if config[i] <= nStates[i] then
            break
         else
            config[i] = 1
         end
      end

      -- stop when tried all configurations
      if idx == nNodes and config[config:size(1)] == 1 then
         break
      end
   end

   -- store and return optimal config
   graph.optimal = optimalconfig
   return optimalconfig
end

----------------------------------------------------------------------
-- belief propagation: if the given graph is loopy, and maxIter is
-- set > 1, then loopy belief propagation is done
--
function gm.decode.bp(graph,maxIter)
   -- check args
   if not graph.unaries or not graph.joints then
      xlua.error('missing unaries/joints, please call graph:setFactors(...)','decode')
   end
   maxIter = maxIter or 1

   -- verbose
   if graph.verbose then
      print('<gm.decode.bp> decoding using belief-propagation')
   end

   -- local vars
   local nNodes = graph.nNodes
   local maxStates = graph.unaries:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds
   local V = graph.V
   local E = graph.E
   local unaries = graph.unaries
   local joints = graph.joints

   -- init
   local product = ones(nNodes,maxStates)
   local nodeBel = ones(nNodes,maxStates)
   local nodeBel_old = nodeBel:clone()
   local msg = zeros(nEdges*2,maxStates)
   local msg_old = zeros(nEdges*2,maxStates)

   -- propagate state normalizations
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      msg[e]:narrow(1,1,nStates[n2]):fill(1/nStates[n2]) -- n1 ==> n2
      msg[e+nEdges]:narrow(1,1,nStates[n1]):fill(1/nStates[n1]) -- n2 ==> n1
   end

   -- do loopy belief propagation (if maxIter = 1, it's regular bp)
   local idx
   for i = 1,maxIter do
      idx = i
      -- pass messages, for all nodes
      for n = 1,nNodes do
         -- find neighbors
         local edges = graph:getEdgesOf(n)

         -- send a message to each neighbor
         for k = 1,edges:size(1) do
            local e = edges[k]
            local n1 = edgeEnds[e][1]
            local n2 = edgeEnds[e][2]

            -- get joint potential
            local pot_ij
            if n == edgeEnds[e][2] then
               pot_ij = joints[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2])
            else
               pot_ij = joints[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2]):t()
            end

            -- compute product of all incoming messages except j
            local temp = unaries[n]:narrow(1,1,nStates[n]):clone()
            for kk = 1,edges:size(1) do
               local e2 = edges[kk]
               if e2 ~= e then
                  if n == edgeEnds[e2][2] then
                     temp:cmul( msg[e2]:narrow(1,1,nStates[n]) )
                  else
                     temp:cmul( msg[e2+nEdges]:narrow(1,1,nStates[n]) )
                  end
               end
            end

            -- compute new message (using max product)
            local new = pot_ij.gm.maxproduct(pot_ij,temp)

            -- normalize message
            if n == edgeEnds[e][2] then
               msg[e+nEdges]:narrow(1,1,nStates[n1]):copy(new):div(new:sum())
            else
               msg[e]:narrow(1,1,nStates[n2]):copy(new):div(new:sum())
            end
         end
      end

      -- check convergence
      if (msg-msg_old):abs():sum() < 1e-4 then
         break
      end
      msg_old:copy(msg)
   end
   if graph.verbose then
      if idx == maxIter then
         warning('<gm.decode.bp> reached max iterations ('..maxIter..') before convergence')
      else
         print('<gm.decode.bp> decoded graph in '..idx..' iterations')
      end
   end

   -- compute nodeBel
   for n = 1,nNodes do
      local edges = graph:getEdgesOf(n)
      product[n] = unaries[n]
      local prod = product[n]:narrow(1,1,nStates[n])
      for i = 1,edges:size(1) do
         local e = edges[i]
         if n == edgeEnds[e][2] then
            prod:cmul(msg[e]:narrow(1,1,nStates[n]))
         else
            prod:cmul(msg[e+nEdges]:narrow(1,1,nStates[n]))
         end
      end
      nodeBel[n]:narrow(1,1,nStates[n]):copy(prod):div(prod:sum())
   end

   -- get argmax of nodeBel: that's the optimal config
   local pot, optimalconfig = nodeBel:max(2)
   optimalconfig = optimalconfig:squeeze(2)

   -- store and return optimal config
   graph.optimal = optimalconfig
   return optimalconfig,nodeBel
end
