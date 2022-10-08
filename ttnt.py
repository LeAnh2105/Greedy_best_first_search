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

def BFS(graph, start, end):
    queue = []
    visited=[]
    queue.append([start])
    while queue:
        path = queue.pop(0)
        
        visited.append(path)
        
        node = path[-1]
        
        #print(node)
        
        if node == end:
            return path
        elif node not in visited:
            for adj in graph.get(node, []):
                new_path = list(path)
                new_path.append(adj)
                queue.append(new_path)
            visited.append(node)
            

def DFS(graph, start, end):
    stack = []
    visited=[]
    stack.append([start])
    while stack:
        path = stack.pop()
        
        visited.append(path)
        
        node = path[-1]
        
        
        if node == end:
            return path
        elif node not in visited:
            for adj in graph.get(node, []):
                new_path = list(path)
                new_path.append(adj)
                stack.append(new_path)
            visited.append(node)


def reconstruct_path(came_from, current):
    final_path = [current]
    while current in came_from:
        current = came_from[current]
        final_path.insert(0,current)
    return final_path



  #Tính khoảng cách giữ 2 tọa độ
def manhattan_distance(node, goal):
    x1, y1 = node
    x2, y2 = goal

    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def heuristic(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return ((x2-x1)**2 + (y2-y1)**2)**0.5
def heuristic_reedy_best_first_search(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal):
    closed_set = []  # nodes already evaluated

    open_set = [start]  # nodes discovered but not yet evaluated

    came_from = {}  # most efficient path to reach from

    gscore = {}  # cost to get to that node from start

    for key in nodes:
        gscore[key] = 100  # intialize cost for every node to inf

    gscore[start] = 0

    fscore = {}  # chi phi đến đích từ 1 đỉnh

    for key in nodes:
        fscore[key] = 100

    fscore[start] = heuristic(start, goal)  # Tính chi phí của đỉnh bắt đầu đến đích

    while open_set:
        min_val = 1000  # Tìm đỉnh trong tập mở có fscore nhỏ nhất
        for node in open_set:
            if fscore[node] < min_val:
                min_val = fscore[node]
                min_node = node

        current = min_node  # set that node to current
        if current == goal:
           return reconstruct_path(came_from, current)
        open_set.remove(current)  # remove node from set to be evaluated and
        closed_set.append(current)  # add it to set of evaluated nodes

        for neighbor in graph.get(nodes_dict[current], []): # check neighbors of current node
            Next=get_node(neighbor)
            if Next in closed_set:  # ignore neighbor node if its already evaluated
                continue
            if Next not in open_set:  # else add it to set of nodes to be evaluated
                open_set.append(Next)

            # dist from start to neighbor through current
            tentative_gscore = gscore[current] + 1
            
            # not a better path to reach neighbor
            if tentative_gscore >= gscore[Next]:
                continue
            came_from[Next] = current  # record the best path untill now
            gscore[Next] = tentative_gscore
            fscore[Next] = gscore[Next] + heuristic(Next, goal)
            
def greedy_best_first_search(graph, start, goal):
    closed_set = []  # đỉnh đã được đánh giá

    open_set = [start]  # các đỉnh phát hiện nhưng chưa được đánh giá

    came_from = {}  # most efficient path to reach from



    fscore = {}  # chi phí từ đỉnh đó đến đích

    for key in nodes:
        fscore[key] = 100

    fscore[start] = heuristic_reedy_best_first_search(start, goal)  # cost for start is only h(x)

    while open_set:
        min_val = 1000  # find node in openset with lowest fscore value
        for node in open_set:
            if fscore[node] < min_val:
                min_val = fscore[node]
                min_node = node

        current = min_node  #Đỉnh này trở thành đỉnh hiện tại
        if current == goal:
           return reconstruct_path(came_from, current)
        open_set.remove(current)  # Loại đỉnh đó ra khỏi danh sách đỉnh chưa đánh giá
        closed_set.append(current)  # thêm đỉnh đó vào danh sách đỉnh đã đánh giá

        for neighbor in graph.get(nodes_dict[current], []): # kiểm tra các đỉnh lân cận
            Next=get_node(neighbor)
            if Next in closed_set:  # bỏ qua nếu nó đã năm trong danh sách đỉnh đã đánh giá
                continue
            if Next not in open_set:  # ngược lại thêm nó vào danh sách được mở nhưng chưa được chọn
                open_set.append(Next)

           
            came_from[Next] = current  # record the best path untill now
            fscore[Next] = heuristic_reedy_best_first_search(Next, goal)
            

path=greedy_best_first_search(graph, start, end)
print(path)
visualize_maze(matrix,bonus_points,start,end,path)

path=a_star_search(graph, start, end)
print(path)
visualize_maze(matrix,bonus_points,start,end,path)

'''
path=DFS(graph, nodes_dict.get(start), nodes_dict.get(end))
route=[]
for n in path:
    route.append(get_node(n))
print(route)
visualize_maze(matrix,bonus_points,start,end,route)
    
    
path=a_star_search(graph, start, end)
print(path)
visualize_maze(matrix,bonus_points,start,end,path)
'''