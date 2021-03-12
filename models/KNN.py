import numpy as np
from models import ModelBaseClass
import heapq as hq
from utilities import loadConfigWithName


class KNN(ModelBaseClass):
    def train(self, features: np.array, labels: np.array, *args, **dicts):
        super().train(features, labels, *args, **dicts)

    def predict(self, features: np.array):
        super().predict(features)

    def loadPara(self):
        super().loadPara()


# -----------------------UTILITIES-----------------------#

class DisPPair:
    def __init__(self, dis: float, point: np.ndarray):
        self.pair = (dis, point)

    def getValueOfAxis(self, axis: int):
        return self.pair[1][axis]

    # compare
    def __eq__(self, other):
        if self.pair[0] == other.pair[0]:
            return True
        else:
            return False

    def __ne__(self, other):
        if self.pair[0] != other.pair[0]:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.pair[0] < other.pair[0]:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.pair[0] > other.pair[0]:
            return True
        else:
            return False

    def __le__(self, other):
        if self.pair[0] <= other.pair[0]:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.pair[0] >= other.pair[0]:
            return True
        else:
            return False

    # computation
    def __neg__(self):
        newPair = DisPPair(-self.pair[0], self.pair[1].copy())
        return newPair


class maxHeapWithLength:
    def __init__(self, length):
        self.heap = []
        self.length = length

    def push(self, element: DisPPair) -> bool:
        """
        The return value indicates whether the heap is updated
        """
        if len(self.heap) < self.length:
            hq.heappush(self.heap, -element)
            return True
        else:
            if -self.heap[0] <= element:
                return False
            else:
                hq.heappop(self.heap)
                hq.heappush(self.heap, -element)
                return True

    def pop(self) -> DisPPair:
        return -hq.heappop(self.heap)

    def isEmpty(self) -> bool:
        return True if len(self.heap) == 0 else False

    def isFull(self) -> bool:
        return True if len(self.heap) == self.length else False

    def peek(self) -> DisPPair:
        if self.isEmpty():
            raise Exception("The heap is empty")
        else:
            return -self.heap[0]

    def __len__(self):
        return len(self.heap)


class Node:
    def __init__(self, points: np.ndarray = None, father=None, lChild=None, rChild=None, axis: int = None):
        self.points: np.ndarray = points
        self.father: Node = father
        self.lChild: Node = lChild
        self.rChild: Node = rChild
        self.axis: int = axis


class kdTree:
    def __init__(self, root=None):
        self.root = root

    def __medianSplit(self, features: np.ndarray, axis: int):
        if features.shape[0] == 1:
            return None, features, None
        sortedData = np.array(sorted(features, key=lambda x: x[axis]))
        medianIndex = sortedData.shape[0] // 2
        leftSame, rightSame = True
        leftStep = 1
        rightStep = 1
        medianValue = sortedData[medianIndex, axis]

        while leftSame or rightSame:
            if medianIndex - leftStep < 0 or sortedData[medianIndex - leftStep, axis] < medianValue:
                leftSame = False
            else:
                leftStep += 1

            if medianIndex + rightStep >= sortedData.shape[0] or sortedData[
                medianIndex + rightStep, axis] > medianValue:
                rightSame = False
            else:
                rightStep += 1

        leftData = sortedData[:medianIndex - leftStep + 1]
        medianData = sortedData[medianIndex - leftStep + 1:medianIndex + rightStep]
        rightData = sortedData[medianIndex + rightStep:]

        if leftData.shape[0] == 0:
            leftData = None
        if rightData.shape[0] == 0:
            rightData = None

        return leftData, medianData, rightData

    def createKdTree(self, features: np.ndarray):
        """
        Attention:The features here is actually [feature,label]
        """
        self.root = Node()
        assert type(features) == np.ndarray and len(features.shape) == 2
        self.__createChild(features, self.root, 0)

    def __createChild(self, features: np.ndarray, currentNode: Node, depth: int):
        axis = depth % (features.shape[1] - 1)
        currentNode.axis = axis
        leftData, medianData, rightData = self.__medianSplit(features, axis=axis)
        assert len(medianData.shape) == 2
        currentNode.points = medianData
        if leftData == None:
            currentNode.lChild = None
        else:
            currentNode.lChild = Node(father=currentNode)
            self.__createChild(leftData, currentNode.lChild, depth + 1)

        if rightData == None:
            currentNode.rChild = None
        else:
            currentNode.rChild = Node(father=currentNode)
            self.__createChild(rightData, currentNode.rChild, depth + 1)

        return

    def search(self, point: np.ndarray, k: int) -> np.ndarray:
        if self.root == None:
            return np.array([])

        pointHeap = maxHeapWithLength(k)
        calDis = lambda currentPoint, point: np.linalg.norm(currentPoint[:currentPoint.shape[0] - 1] - point)

        currentNode = self.root
        while True:
            judgeValue = currentNode.points[0, currentNode.axis]
            if point[currentNode.axis] < judgeValue:
                if currentNode.lChild == None:
                    break
                else:
                    currentNode = currentNode.lChild

            else:
                if currentNode.rChild == None:
                    break
                else:
                    currentNode = currentNode.rChild

        for currentPoint in currentNode.points:
            newPair = DisPPair(calDis(currentPoint, point), currentPoint)
            pointHeap.push(newPair)

    def __recursiveBackTrack(self, currentNode: Node, pointHeap: maxHeapWithLength):
        fatherNode = currentNode.father
        fatherNodeAxis = fatherNode.axis
        for aPoint in fatherNode.points:
            disWithSuperRectangle = np.abs(pointHeap.peek().getValueOfAxis(fatherNodeAxis) - aPoint[fatherNodeAxis])
            