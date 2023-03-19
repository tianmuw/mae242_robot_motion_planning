def mediumGrid():
   n, m = 7, 7

   O=[[0,6,0,0],[0,0,0,6],[0,6,6,6],[6,6,0,6],
      [2,2,3,4], [4,4,3,3]]

   START = [1,2]
   DISTANTEXIT = [5,3]
   CLOSEEXIT = [3,3]
   LOSESTATES = [[1,1],[2,1],[3,1],[4,1],[5,1],[4,2]]
   return n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES