# Problem 3

This file takes input from the standard input as rows seperated by new line and columns separated by space.

We can think of the given 2d map as a bi-directional graph, where-
  1. each cell is a node
  2. each cell/node has at most 4 edges and is connected to cells immediately to it's right/left or up/down.

Now we need to traverse this 'graph' and find number of connected 'islands'. We achieve this by-
  1. Checking Every starting point(cell)
  2. Marking all land/"1" recursively connected to our starting point.
  3. Count the number of unique starting point that was previously unexplored.

This count is our result.

## In the worst case- 
 The full map can be land, with no water. We'll first traverse the whole map and then again check every cell on the outer loop.
 Leading to a time complexity of- 
 
 ```
   MxN + MxN
 = 2 MxN
 ~ MxN
 Where M and N are the dimensions of teh map
 ```
