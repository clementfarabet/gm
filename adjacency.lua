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
--     gm.adjacency - a list of functions to create adjacency matrices
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains standard functions to create adjacency matrices
gm.adjacency = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Full graph adjacency
--
function gm.adjacency.full(nNodes)
   return ones(nNodes,nNodes) - eye(nNodes)
end

----------------------------------------------------------------------
-- N-connexity 2D lattice (N = 4 or 8)
--
function gm.adjacency.lattice2d(nRows,nCols,connex)
   local nNodes = nRows*nCols
   local adj = {}
   for n = 1,nNodes do
      adj[n] = {}
   end
   if connex == 4 then
      for i = 1,nRows do
         for j = 1,nCols do
            local n = (i-1)*nCols + j
            if j < nCols then
               adj[n][n+1] = 1
               adj[n+1][n] = 1
            end
            if i < nRows then
               adj[n][n+nCols] = 1
               adj[n+nCols][n] = 1
            end
         end
      end
   elseif connex == 8 then
      for i = 1,nRows do
         for j = 1,nCols do
            local n = (i-1)*nCols + j
            if j < nCols then
               adj[n][n+1] = 1
               adj[n+1][n] = 1
            end
            if i < nRows then
               adj[n][n+nCols] = 1
               adj[n+nCols][n] = 1
            end
            if i < nRows and j < nCols then
               adj[n][n+nCols+1] = 1
               adj[n+nCols+1][n] = 1
            end
            if i < nRows and j > 1 then
               adj[n][n+nCols-1] = 1
               adj[n+nCols-1][n] = 1
            end
         end
      end
   else
      sys.error('connexity can only be 4 or 8 on a 2D lattice', 'gm.adjacency.lattice2d')
   end
   return adj
end

----------------------------------------------------------------------
-- N-connexity 3D lattice (N = 6 or 26)
--
function gm.adjacency.lattice3d(nRows,nCols,nLayers,connex)
   sys.error('not implemented yet', 'gm.adjacency.lattice3d')
end
