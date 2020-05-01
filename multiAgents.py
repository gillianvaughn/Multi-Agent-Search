# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        curFood = currentGameState.getFood()
        curFoodList = curFood.asList()
        curPos = currentGameState.getPacmanPosition()
        newFoodList = newFood.asList()

        ghostPositions = successorGameState.getGhostPositions()
        distance = float("inf")  #infinity to find the min
        scared = newScaredTimes[0] > 0

        for ghost in ghostPositions:  #
          tempdist = manhattanDistance(ghost, newPos)
          distance = min(tempdist, distance)
        
        distance2 = float("inf")       #inf to fidn the min dist 
        distance3 = float("-inf") #- because it will be used to find the max

        for food in newFoodList:
          tempdist = manhattanDistance(food, newPos)
          d0 = manhattanDistance(food, curPos)
          distance2 = min(tempdist, distance2)
          distance3 = max(tempdist, distance3) 

        cond = len(newFoodList) < len(curFoodList)
        count = len(newFoodList)
        if cond:
          count = 10000
        if distance < 2:
          distance = -100000
        else:
          distance = 0
        if count == 0:
          count = -1000
        if scared:
          distance = 0
        return distance + 1.0/distance2 + count - successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, depth):
            if agentIndex == 0:
                depth -=1
            
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(state)
            
            actions = gameState.getLegalActions(agentIndex)
            nextState = [state.generateSuccessor(agentIndex,action) for action in actions]

            nextAgentIndex = (agentIndex +1 ) % gameState.numAgents()
            values = [value(nextState, nextAgentIndex, depth) for nextState in nextState]

            if(agentIndex >= 1):
                return min(values)
            else:
                return max(values)

        actions = gameState.getLegalActions(self.index)
        currentScore = -10000
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(gameState,action)
            score = value(nextState,self.index,self.depth)
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, depth):
            if agentIndex == 0:
                depth -= 1
            if depth ==0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(state)
            
            actions = gameState.getLegalActions(agentIndex)
            nextStates = [state.generateSuccessor(agentIndex, action) for action in actions]

            nextAgentIndex = (agentIndex +1) % gameState.numAgents()
            values = [value(next_state, nextAgentIndex, depth) for next_state in nextStates]
            #probability = 1/ len(values)
            #actions = 1/len(actions)
            if(agentIndex>=1):
                sumV = sum(values)
                sumV /= len(values)
                return sumV
            else:
                return max(values)
                """
                agentPolicy = state.getLegalActions(agentIndex)
                expectedValue = 0
                for i, (_action, probability) in enumerate(agentPolicy.list()):
                    value = values[i]
                    expectedValue += value * probability
                return expectedValue
                """
        
        actions = gameState.getLegalActions(self.index)
        currentScore = -10000
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(gameState,action)
            score = value(nextState,self.index,self.depth)
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
      return float("inf")
    if currentGameState.isLose():
        return - float("inf")
    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    foodPos = newFood.asList()
    closestfood = float("inf")
    for pos in foodPos:
        thisdist = util.manhattanDistance(pos, currentGameState.getPacmanPosition())
        if (thisdist < closestfood):
            closestfood = thisdist
    numghosts = currentGameState.getNumAgents() - 1
    i = 1
    disttoghost = float("inf")
    while i <= numghosts:
        nextdist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(i))
        disttoghost = min(disttoghost, nextdist)
        i += 1
    score += max(disttoghost, 4) * 2
    score -= closestfood * 1.5
    capsulelocations = currentGameState.getCapsules()
    score -= 4 * len(foodPos)
    score -= 3.5 * len(capsulelocations)
    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
