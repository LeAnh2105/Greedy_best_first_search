import os
import matplotlib.pyplot as plt
import queue as Q
from queue import PriorityQueue

with open('maze_map1.txt', 'w') as outfile:
  outfile.write('2\n')
  outfile.write('3 6 -3\n')
  outfile.write('5 14 -1\n')
  outfile.write('xxxxxxxxxxxxxxxxxxxxxx\n')
  outfile.write('x   x   xx xx        x\n')
  outfile.write('x     x     xxxxxxxxxx\n')
  outfile.write('x x   +xx  xxxx xxx xx\n')
  outfile.write('  x   x xxxx   xxxx  x\n')
  outfile.write('x          xx +xx  x x\n')
  outfile.write('xxxxxxx x     xxx  x x\n')
  outfile.write('xxxxxxxxx  x         x\n')
  outfile.write('x          x x Sx x  x\n')
  outfile.write('xxxxx x  x x x     x x\n')
  outfile.write('xxxxxxxxxxxxxxxxxxxxxx')

'''
with open('maze_map1.txt', 'w') as outfile:
  outfile.write('0\n')
  outfile.write('xxxxx\n')
  outfile.write('x   x\n')
  outfile.write('x xSx\n')
  outfile.write('    x\n')
  outfile.write('xxxxx\n')
'''
  
def read_file(file_name: str = 'maze.txt'):
  f=open(file_name,'r')
  n_bonus_points = int(next(f)[:-1])
  bonus_points = []
  for i in range(n_bonus_points):
    x, y, reward = map(int, next(f)[:-1].split(' '))
    bonus_points.append((x, y, reward))

  text=f.read()
  matrix=[list(i) for i in text.splitlines()]
  f.close()

  return bonus_points, matrix

bonus_points, matrix = read_file('maze_map1.txt')

def visualize_maze(matrix, bonus, start, end, route=None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='silver')

    plt.text(end[1],-end[0],'EXIT',color='red',
         horizontalalignment='center',
         verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    
    for _, point in enumerate(bonus):
      print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')
nodes = []
edges = []
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j]=='S':
            start=(i,j)

        elif matrix[i][j]==' ':
            if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                end=(i,j)
                
        if matrix[i][j]!='x':
                nodes.append((i, j))
                
                if i + 1 < len(matrix) and matrix[i + 1][j]!='x':
                    edges.append(((i, j), (i + 1, j)))
                
                if j + 1 < len(matrix[i]) and matrix[i][j + 1]!='x':
                    edges.append(((i, j), (i, j + 1)))
                    
                if i - 1 >= 0 and matrix[i-1][j]!='x':
                    edges.append(((i, j), (i-1, j)))
                    
                if j - 1 >=0 and matrix[i][j - 1]!='x':
                    edges.append(((i, j), (i, j - 1)))

#create dictionary
nodes_dict={}
for node in nodes:
    nodes_dict[node]=len(nodes_dict)

#create graph
graph={}

for v in edges:
    graph.setdefault(nodes_dict.get(v[0]),[]).append(nodes_dict.get(v[1]))
    
for node in graph:
    graph[node].sort()

def get_node(val):
    for key, value in nodes_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"



def reconstruct_path(came_from, current):
    final_path = [current]
    while current in came_from:
        current = came_from[current]
        final_path.insert(0,current)
    return final_path


def heuristic_reedy_best_first_search(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)


def greedy_best_first_search(graph, start, goal):
    closed_set = []  # ?????nh ???? ???????c ????nh gi??

    open_set = [start]  # c??c ?????nh ph??t hi???n nh??ng ch??a ???????c ????nh gi??

    came_from = {}  # most efficient path to reach from



    fscore = {}  # chi ph?? t??? ?????nh ???? ?????n ????ch

    for key in nodes:
        fscore[key] = 100

    fscore[start] = heuristic_reedy_best_first_search(start, goal)  # cost for start is only h(x)

    while open_set:
        min_val = 1000  # find node in openset with lowest fscore value
        for node in open_set:
            if fscore[node] < min_val:
                min_val = fscore[node]
                min_node = node

        current = min_node  #?????nh n??y tr??? th??nh ?????nh hi???n t???i
        if current == goal:
           return reconstruct_path(came_from, current)
        open_set.remove(current)  # Lo???i ?????nh ???? ra kh???i danh s??ch ?????nh ch??a ????nh gi??
        closed_set.append(current)  # th??m ?????nh ???? v??o danh s??ch ?????nh ???? ????nh gi??

        for neighbor in graph.get(nodes_dict[current], []): # ki???m tra c??c ?????nh l??n c???n
            Next=get_node(neighbor)
            if Next in closed_set:  # b??? qua n???u n?? ???? n??m trong danh s??ch ?????nh ???? ????nh gi??
                continue
            if Next not in open_set:  # ng?????c l???i th??m n?? v??o danh s??ch ???????c m??? nh??ng ch??a ???????c ch???n
                open_set.append(Next)

           
            came_from[Next] = current  # record the best path untill now
            fscore[Next] = heuristic_reedy_best_first_search(Next, goal)
            

path=greedy_best_first_search(graph, start, end)
print(path)
visualize_maze(matrix,bonus_points,start,end,path)

