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
--     gm.infer - a list of functions to perform inference
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the infer
gm.infer = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local Tensor = torch.Tensor
local sort = torch.sort
local log = torch.log
local eps = 1e-15

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- exact inference: only adapted to super small graphs
--
function gm.infer.exact(graph)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','infer')
   end

   -- verbose
   if graph.verbose then
      print('<gm.infer.bp> doing exact inference')
   end

   -- local vars
   local nNodes = graph.nNodes
   local maxStates = graph.nodePot:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- init
   local config = ones(nNodes)
   local nodeBel = zeros(nNodes,maxStates)
   local edgeBel = zeros(nEdges,maxStates,maxStates)
   local Z = 0

   -- exact inference
   while true do
      -- get potential for configuration
      local pot = graph:getPotentialForConfig(config)

      -- update nodeBel
      for n = 1,nNodes do
         nodeBel[n][config[n]] = nodeBel[n][config[n]] + pot
      end

      -- update edgeBel
      for e = 1,nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         edgeBel[e][config[n1]][config[n2]] = edgeBel[e][config[n1]][config[n2]] + pot
      end

      -- update Z
      Z = Z + pot

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

   -- normalize
   nodeBel:div(Z)
   edgeBel:div(Z)

   -- return marginal beliefs, pairwise beliefs, and negative of free energy
   return nodeBel, edgeBel, log(Z)
end

----------------------------------------------------------------------
-- belief propagation: if the given graph is loopy, and maxIter is
-- set > 1, then loopy belief propagation is done
--
function gm.infer.bp(graph,maxIter)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','decode')
   end
   maxIter = maxIter or 1

   -- verbose
   if graph.verbose then
      print('<gm.infer.bp> inference with belief-propagation')
   end

   -- local vars
   local nNodes = graph.nNodes
   local maxStates = graph.nodePot:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds
   local V = graph.V
   local E = graph.E
   local nodePot = graph.nodePot
   local edgePot = graph.edgePot

   -- init
   local product = ones(nNodes,maxStates)
   local nodeBel = ones(nNodes,maxStates)
   local edgeBel = zeros(nEdges,maxStates,maxStates)
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
               pot_ij = edgePot[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2])
            else
               pot_ij = edgePot[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2]):t()
            end

            -- compute product of all incoming messages except j
            local temp = nodePot[n]:narrow(1,1,nStates[n]):clone()
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

            -- compute new message (using matrix product)
            local new = pot_ij*temp

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
         warning('<gm.infer.bp> reached max iterations ('..maxIter..') before convergence')
      else
         print('<gm.infer.bp> decoded graph in '..idx..' iterations')
      end
   end

   -- compute marginal beliefs
   for n = 1,nNodes do
      local edges = graph:getEdgesOf(n)
      product[n] = nodePot[n]
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

   -- compute edge beliefs
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      local belN1 = nodeBel[n1]:narrow(1,1,nStates[n1]):clone():cdiv(msg[e+nEdges]:narrow(1,1,nStates[n1]))
      local belN2 = nodeBel[n2]:narrow(1,1,nStates[n2]):clone():cdiv(msg[e]:narrow(1,1,nStates[n2]))
      local b1 = Tensor(nStates[n1],nStates[n2])
      local b2 = Tensor(nStates[n1],nStates[n2])
      for i = 1,nStates[n2] do
         b1:select(2,i):copy(belN1)
      end
      for i = 1,nStates[n1] do
         b2:select(1,i):copy(belN2)
      end
      local eb = edgeBel[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2])
      eb:copy(b1):cmul(b2):cmul(edgePot[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2]))
      eb:div(eb:sum())
   end

   -- compute negative free energy
   local eng1 = 0
   local eng2 = 0
   local ent1 = 0
   local ent2 = 0
   nodeBel:add(eps)
   edgeBel:add(eps)
   for n = 1,nNodes do
      local edges = graph:getEdgesOf(n)
      local nNbrs = edges:size(1)
      -- node entropy
      local nb = nodeBel[n]:narrow(1,1,nStates[n])
      ent1 = ent1 + (nNbrs-1) * log(nb):cmul(nb):sum()
      -- node energy
      local np = nodePot[n]:narrow(1,1,nStates[n])
      eng1 = eng1 - log(np):cmul(nb):sum()
   end
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      --  pairwise entropy
      local eb = edgeBel[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2])
      ent2 = ent2 - log(eb):cmul(eb):sum()
      -- pairwise energy
      local ep = edgePot[e]:narrow(1,1,nStates[n1]):narrow(2,1,nStates[n2])
      eng2 = eng2 - log(ep):cmul(eb):sum()
   end
   local F = (eng1+eng2) - (ent1+ent2)
   local logZ = -F

   -- return marginal beliefs, pairwise beliefs, and negative of free energy
   return nodeBel, edgeBel, logZ
end
