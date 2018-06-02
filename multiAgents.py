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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #NEHA_START
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
        #NEHA_START
        def sumFoodProximity(pos, foodPos):
            foodDistances = []
            for food in foodPos:
                foodDistances.append(util.manhattanDistance(food, pos))
            return sum(foodDistances) if sum(foodDistances) > 0 else 1

        def closestFood(pos, foodPos):
            foodDistances = []
            for food in foodPos:
                foodDistances.append(util.manhattanDistance(food, pos))
            return min(foodDistances) if len(foodDistances) > 0 else 1

        score = successorGameState.getScore()
        num_ghosts = 0
        for ghost in newGhostStates:
            if util.manhattanDistance(ghost.getPosition(), newPos) <= 2:
                score -= 30
                num_ghosts += 1
        
        new_food = 1 / sumFoodProximity(newPos, newFood.asList())
        cur_food = 1 / sumFoodProximity(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
        if new_food > cur_food:
            score += (new_food - cur_food) * 3
        else:
            score -= 20
        nextFoodDist = closestFood(newPos, newFood.asList())
        curFoodDist = closestFood(currentGameState.getPacmanPosition(), currentGameState.getFood().asList())
        if nextFoodDist < curFoodDist:
            score += (nextFoodDist - curFoodDist) * 3
        else:
            score -= 20
        return score
        #NEHA_END

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
        """
        "*** YOUR CODE HERE ***"
        #NEHA_START
        def minimizer(state, depth, agents):
            if state.isLose() or state.isWin():
                return state.getScore()
            nextAgent = agents + 1
            if agents == state.getNumAgents() - 1:
                nextAgent = 0
            ret_score = float("inf")
            score = ret_score
            for action in state.getLegalActions(agents):
                if nextAgent == 0: # We are on the last agents and it will be Pacman's turn next.
                    if self.depth == depth + 1:
                        score = self.evaluationFunction(state.generateSuccessor(agents, action))
                    else:
                        score = maximizer(state.generateSuccessor(agents, action), depth + 1)
                else:
                    score = minimizer(state.generateSuccessor(agents, action), depth, nextAgent)
                if score < ret_score:
                    ret_score = score
            return ret_score

        def maximizer(state, depth):
            if state.isLose() or state.isWin():
                return state.getScore()
            ret_score = float("-inf")
            score = ret_score
            ret_action = Directions.STOP
            for action in state.getLegalActions(0):
                score = minimizer(state.generateSuccessor(0, action), depth, 1)
                if score > ret_score:
                    ret_score = score
                    ret_action = action
            if 0 == depth:
                return ret_action
            else:
                return ret_score
        return maximizer(gameState, 0)
        #NEHA_END


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #NEHA_START
        def minimizer(state, depth, agents, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            nextAgent = agents + 1
            if agents == state.getNumAgents() - 1:
                nextAgent = 0
            ret_score = float("inf")
            score = ret_score
            for action in state.getLegalActions(agents):
                if nextAgent == 0:         # We are on the last agents and it will be Pacman's turn next.
                    if self.depth == depth + 1:
                        score = self.evaluationFunction(state.generateSuccessor(agents, action))
                    else:
                        score = maximizer(state.generateSuccessor(agents, action), depth + 1, alpha, beta)
                else:
                    score = minimizer(state.generateSuccessor(agents, action), depth, nextAgent, alpha, beta)
                if score < ret_score:
                    ret_score = score
                beta = min(beta, ret_score)
                if ret_score < alpha:
                    return ret_score
            return ret_score

        def maximizer(state, depth, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            ret_score = float("-inf")
            score = ret_score
            ret_action = Directions.STOP
            for action in state.getLegalActions(0):
                score = minimizer(state.generateSuccessor(0, action), depth, 1, alpha, beta)
                if score > ret_score:
                    ret_score = score
                    ret_action = action
                alpha = max(alpha, ret_score)
                if ret_score > beta:
                    return ret_score
            if 0 == depth:
                return ret_action
            else:
                return ret_score

        return maximizer(gameState, 0, float("-inf"), float("inf"))

        #NEHA_END

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
        #NEHA_START
        def minimizer(state, depth, agents):
            if state.isLose():
                return state.getScore()
            nextAgent = agents + 1
            if agents == state.getNumAgents() - 1:
                nextAgent = 0
            score = 0
            for action in state.getLegalActions(agents):
                num = len(state.getLegalActions(agents))
                if nextAgent == 0: # We are on the last agents and it will be Pacman's turn next.
                    if self.depth == depth + 1:
                        tmp = self.evaluationFunction(state.generateSuccessor(agents, action))
                        score += tmp / num
                    else:
                        tmp = maximizer(state.generateSuccessor(agents, action), depth + 1)
                        score += tmp / num
                else:
                    tmp = minimizer(state.generateSuccessor(agents, action), depth, nextAgent)
                    score += tmp / num
            return score

        def maximizer(state, depth):
            if state.isLose() or state.isWin():
                return state.getScore()
            ret_score = float("-inf")
            score = ret_score
            ret_action = Directions.STOP
            for action in state.getLegalActions(0):
                score = minimizer(state.generateSuccessor(0, action), depth, 1)
                if score > ret_score:
                    ret_score = score
                    ret_action = action
            if 0 == depth:
                return ret_action
            else:
                return ret_score
        return maximizer(gameState, 0)
        #NEHA_END

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

