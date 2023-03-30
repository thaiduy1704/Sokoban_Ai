import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        if len(self.Heap) < 1:
            return -1
        (_, _, item) = heapq.heappop(self.Heap)
        return item


    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)])

    #print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else:
            continue
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""
def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    time_start = time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]]
    temp = []
    while frontier:
        if time.time() - time_start < 990:
            node = frontier.pop()
            node_action = actions.pop()
            if isEndState(node[-1][-1]):
                temp += node_action[1:]
                break
            if node[-1] not in exploredSet:
                exploredSet.add(node[-1])
                for action in legalActions(node[-1][0], node[-1][1]):
                    newPosPlayer, newPosBox = updateState(
                        node[-1][0], node[-1][1], action)
                    if isFailed(newPosBox):
                        continue
                    frontier.append(node + [(newPosPlayer, newPosBox)])
                    actions.append(node_action + [action[-1]])
        else:
            break


    runtime = time.time()-time_start
    return (temp,runtime)



def breadthFirstSearch(gameState):
    # thời gian bắt đầu trò chơi
    time_start = time.time()
    """Implement breadthFirstSearch approach"""
    # trạng thái các vật chướng ban đầu
    beginBox = PosOfBoxes(gameState)
    # trạng thái vị trí bắt đầu của nhân vật
    beginPlayer = PosOfPlayer(gameState)
    # Khởi tạo trạng thái ban đầu
    # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    startState = (beginPlayer, beginBox)
    # khởi tạo hàng đợi
    frontier = collections.deque([[startState]])  # store states
    # khởi tạo mảng hành động với bắt đầu là 0
    actions = collections.deque([[0]])  # store actions
    # khởi tạo tập xét chứa các node đã xét
    exploredSet = set()
    temp = []
    # Implement breadthFirstSearch here
    # Trong khi hàng đợi khác rỗng
    while frontier:
        # Nếu thời gian chơi dưới 990
        if time.time() - time_start < 990:
            # Lấy ra phần tử đầu tiên từ bên trái sang của tập đợi
            node = frontier.popleft()
            # Lấy ra phần tử đầu tiên từ bên trái sang của tập đợi hành động
            node_action = actions.popleft()
            # Nếu trạng thái các vật chướng trùng vị trí đích thì lưu lại hành động này và thoát ra.
            if isEndState(node[-1][-1]):
                temp += node_action[1:]
                break
            #         Nếu node chưa có trong tập xét thì thêm nó vào
            if node[-1] not in exploredSet:
                exploredSet.add(node[-1])
            # Với tất cả hành động hợp lí tại vị trí hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):
                # cập nhật trạng thái mới
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][-1], action)
                # Nếu vị trí vật chướng mới gây lỗi
                if isFailed(newPosBox):
                    # bỏ qua trạng thái đó
                    continue
                # Thêm vào hàng đợi node cha và node trạng thái con
                frontier.append(node + [(newPosPlayer, newPosBox)])
                # Thêm vào mảng hđ hành động hiện tại và hành động dẫn tới node con
                actions.append(node_action + [action[-1]])
        else:
            break
    runtime = time.time() - time_start
    return (temp, runtime)


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def uniformCostSearch(gameState):
    # thời gian bắt đầu trò chơi
    time_start = time.time()
    """Implement uniformCostSearch approach"""
    # Trạng thái vật chướng lúc ban đầu
    beginBox = PosOfBoxes(gameState)
    # trạng thái vị trí khởi đầu nhân vật
    beginPlayer = PosOfPlayer(gameState)
    # khởi tạo trạng thái khởi đầu
    startState = (beginPlayer, beginBox)
    # Khởi tạo hàng đợi
    frontier = PriorityQueue()
    # thêm vào hàng đợi trạng thái ban đầu
    frontier.push([startState], 0)
    # khởi tạo tập xét chứa các node đã xét
    exploredSet = set()
    # Khởi tạo mảng hành động
    actions = PriorityQueue()
    # Thêm vào mảng hành động lúc đầu à bằng 0
    actions.push([0], 0)
    # mảng chứa kết quả chuỗi hành động
    temp = []
    # Implement uniform cost search here
    # trong khi hàng đợi khác rỗng
    while frontier:
        # Nếu thời gian chơi game nhỏ hơn 990
        if time.time() - time_start < 990:
            # lấy ra phần tử từ cuối hàng đợi
            node = frontier.pop()
            # lấy phần tử cuối cùng trong mảng hành động
            node_action = actions.pop()
            # Nếu node rỗng
            if node == -1:
                # thoát
                break
            # Nếu trạng thái các vật chướng trùng vị trí đích
            if isEndState(node[-1][-1]):
                # ta lưu lại chuỗi hành động
                temp += node_action[1:]
                break
             # Nếu node chưa có trong tâp Node đang xét
            if node[-1] not in exploredSet:
                # thêm nó vào-và tick nó là đã xét rồi
                # Sau đó tính lại chi phí cho chuỗi hành động từ đầu đến hiện tại
                exploredSet.add(node[-1])
                print(node_action[1:])
                Cost = cost(node_action[1:])
                # với mỗi hành động có thể thực hiện ở node đang xét
                for action in legalActions(node[-1][0], node[-1][1]):
                    # cập nhật trạng thái mơi
                    newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                    # Nếu trạng thái vật chướng mới gây ra lỗi
                    if isFailed(newPosBox):
                        continue
                    # Thêm vào hàng đợi node theo sự ưu tiên chi phí
                    frontier.push(node + [(newPosPlayer, newPosBox)], Cost)
                    # Thêm vào mảng hành động theo sự ưu tiên chi phí
                    actions.push(node_action + [action[-1]], Cost)
        else:
            break
    runtime = time.time()-time_start
    return (temp, runtime)




"""Read command"""


def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels, "r") as f:
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args


def get_move(layout, player_pos, method):
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)

    if method == 'dfs':
        result, runtime = depthFirstSearch(gameState)
    elif method == 'bfs':
        result, runtime = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result, runtime = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    print('Runtime of %s: %.2f second.' % (method, runtime))
    return (result, runtime)

