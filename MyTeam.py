# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_ghost_actions(current_pos):
            walls = game_state.get_walls().as_list()

            max_x = max([wall[0] for wall in walls])
            max_y = max([wall[1] for wall in walls])

            actions = []
            for direction in directions:
                action = directions[direction]
                new_pos = (int(current_pos[0] + action[0]),
                           int(current_pos[1] + action[1]))
                if new_pos not in walls:
                    if (1 <= new_pos[0] < max_x) and (1 <= new_pos[1] < max_y):
                        actions.append(direction.title())
            return actions
            
    def get_new_position(current_pos, action):
            act = directions[[direction for direction in directions if str(
                action).lower() == direction][0]]
            return (current_pos[0] + act[0], current_pos[1] + act[1])
    
    def expectation(game_state, position, legal_actions):
            ghost_dict = {}
            for action in legal_actions:
                new_pos = get_new_position(position, action)
                ghost_dict[action] = self.get_maze_distance(
                    position, newPos) * ghost_weights['distance']

            min_action = min(ghost_dict)

            for action in ghost_dict:
                if ghost_dict[action] == min_action:
                    ghost_dict[action] = .8
                else:
                    ghost_dict[action] = .2/len(legal_actions)
            return ghost_dict
    
    def ghost_eval(game_state, opponents, opponent):
            newPos = opponents[opponent]
            enemy = game_state.get_agent_state(opponent)
            myPos = game_state.get_agent_state(self.index).get_position()

            if enemy.scared_timer != 0:
                distance = - self.get_maze_distance(myPos, newPos) * \
                    ghost_weights['distance']

            else:
                distance = self.get_maze_distance(
                    myPos, newPos)*ghost_weights['distance']
            return distance
           
    def minimax(game_state, depth, agent, opponents, alpha=-float('inf'), beta=float('inf')):
        # Get legal moves per agent
            legal_actions = [action for action in game_state.get_legal_actions(
                self.index) if action != Directions.STOP]

            # Generate optimal action recursively
            actions = {}
            if agent == self.index:
                maxVal = -float('inf')
                for action in legal_actions:
                    eval = self.evaluate(game_state, action)
                    if depth == self.treeDepth:
                        value = eval
                    else:
                        value = eval + \
                            minimax(self.get_successor(game_state, action),
                                    depth, agent+1, opponents, alpha, beta)
                    maxVal = max(maxVal, value)
                    if beta < maxVal:
                        return maxVal
                    else:
                        alpha = max(alpha, maxVal)
                    if depth == 1:
                        actions[value] = action
                if depth == 1:          
                    return actions[maxVal]
                return maxVal
            else:
                minVal = float('inf')
                for opponent in opponents:
                    if game_state.get_agent_state(opponent).get_position() is not None:
                        legal_actions = get_ghost_actions(opponents[opponent])
                        expectations = expectation(
                            game_state, opponents[opponent], legal_actions)
                        for action in legal_actions:
                            new_opponents = opponents.copy()
                            new_opponents[opponent] = get_new_position(
                                opponents[opponent], action)
                            ghost_val = ghost_eval(
                                game_state, new_opponents, opponent)*expectations[action]
                            value = ghost_val + \
                                minimax(game_state, depth+1, self.index,
                                        new_opponents, alpha, beta)
                            minVal = min(minVal, value)
                            if minVal < alpha:
                                return minVal
                            else:
                                beta = min(beta, minVal)
                if minVal == float('inf'):
                    return 0
                return minVal

            return minimax(game_state, 1, self.index, opponents)
            
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  
        
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()##########
        
        #Enemies, ghosts and invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['invader_distance'] = 0.0
        
        #different distances to each type of enemy agents
        if len(invaders) > 0:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]) +1
        
        if len(ghosts) > 0:
            ghost_eval = 0.0
            scared_distance = 0.0
            reg_ghosts = [ghost for ghost in ghosts if ghost.scared_timer == 0]
            scared_ghost = [ghost for ghost in ghosts if ghost.scared_timer > 0]
            if len(reg_ghosts) > 0:
                ghost_eval = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in reg_ghosts])
                if ghost_eval <= 1:
                    ghost_eval = -float("inf")
            
            if len(scared_ghost) > 0:
                scared_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghost])
            if scared_distance < ghost_eval or ghost_eval == 0:
                if scared_distance == 0:
                    features['ghost_scared'] = -10
            features['distance_to_ghost'] = ghost_eval

        # Compute distance to the nearest food
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            features['food_remaining'] = len(food_list)
        
        
        #Avoid stopping or bugging, PAULA ESTO ES MÁS O MENOS LO QUE QUERIAS IM
        if action == Directions.STOP: 
            features['stop'] = 1
        if game_state.get_agent_state(self.index).configuration.direction is not None:
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
        
            
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'invader_distance': -50, 'distance_to_food': -1, 'food_remaining': -1, 'distance_to_ghost': 2, 'ghost_scared': -1, 'stop': -100, 'reverse': -20 }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_ghost_actions(current_pos):
            walls = game_state.get_walls().as_list()

            max_x = max([wall[0] for wall in walls])
            max_y = max([wall[1] for wall in walls])

            actions = []
            for direction in directions:
                action = directions[direction]
                new_pos = (int(current_pos[0] + action[0]),
                           int(current_pos[1] + action[1]))
                if new_pos not in walls:
                    if (1 <= new_pos[0] < max_x) and (1 <= new_pos[1] < max_y):
                        actions.append(direction.title())
            return actions
            
    def get_new_position(current_pos, action):
            act = directions[[direction for direction in directions if str(
                action).lower() == direction][0]]
            return (current_pos[0] + act[0], current_pos[1] + act[1])
    
    def expectation(game_state, position, legal_actions):
            ghost_dict = {}
            for action in legal_actions:
                new_pos = get_new_position(position, action)
                ghost_dict[action] = self.get_maze_distance(
                    position, newPos) * ghost_weights['distance']

            min_action = min(ghost_dict)

            for action in ghost_dict:
                if ghost_dict[action] == min_action:
                    ghost_dict[action] = .8
                else:
                    ghost_dict[action] = .2/len(legal_actions)
            return ghost_dict
    
    def ghost_eval(game_state, opponents, opponent):
            newPos = opponents[opponent]
            enemy = game_state.get_agent_state(opponent)
            myPos = game_state.get_agent_state(self.index).get_position()

            if enemy.scared_timer != 0:
                distance = - self.get_maze_distance(myPos, newPos) * \
                    ghost_weights['distance']

            else:
                distance = self.get_maze_distance(
                    myPos, newPos)*ghost_weights['distance']
            return distance
           
    def minimax(game_state, depth, agent, opponents, alpha=-float('inf'), beta=float('inf')):
        # Get legal moves per agent
            legal_actions = [action for action in game_state.get_legal_actions(
                self.index) if action != Directions.STOP]

            # Generate optimal action recursively
            actions = {}
            if agent == self.index:
                maxVal = -float('inf')
                for action in legal_actions:
                    eval = self.evaluate(game_state, action)
                    if depth == self.treeDepth:
                        value = eval
                    else:
                        value = eval + \
                            minimax(self.get_successor(game_state, action),
                                    depth, agent+1, opponents, alpha, beta)
                    maxVal = max(maxVal, value)
                    if beta < maxVal:
                        return maxVal
                    else:
                        alpha = max(alpha, maxVal)
                    if depth == 1:
                        actions[value] = action
                if depth == 1:          
                    return actions[maxVal]
                return maxVal
            else:
                minVal = float('inf')
                for opponent in opponents:
                    if game_state.get_agent_state(opponent).get_position() is not None:
                        legal_actions = get_ghost_actions(opponents[opponent])
                        expectations = expectation(
                            game_state, opponents[opponent], legal_actions)
                        for action in legal_actions:
                            new_opponents = opponents.copy()
                            new_opponents[opponent] = get_new_position(
                                opponents[opponent], action)
                            ghost_val = ghost_eval(
                                game_state, new_opponents, opponent)*expectations[action]
                            value = ghost_val + \
                                minimax(game_state, depth+1, self.index,
                                        new_opponents, alpha, beta)
                            minVal = min(minVal, value)
                            if minVal < alpha:
                                return minVal
                            else:
                                beta = min(beta, minVal)
                if minVal == float('inf'):
                    return 0
                return minVal

            return minimax(game_state, 1, self.index, opponents)
            
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  
        
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()##########
        
        #Enemies, ghosts and invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['invader_distance'] = 0.0
        
        #different distances to each type of enemy agents
        if len(invaders) > 0:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]) +1
        
        if len(ghosts) > 0:
            ghost_eval = 0.0
            scared_distance = 0.0
            reg_ghosts = [ghost for ghost in ghosts if ghost.scared_timer == 0]
            scared_ghost = [ghost for ghost in ghosts if ghost.scared_timer > 0]
            if len(reg_ghosts) > 0:
                ghost_eval = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in reg_ghosts])
                if ghost_eval <= 1:
                    ghost_eval = -float("inf")
            
            if len(scared_ghost) > 0:
                scared_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghost])
            if scared_distance < ghost_eval or ghost_eval == 0:
                if scared_distance == 0:
                    features['ghost_scared'] = -10
            features['distance_to_ghost'] = ghost_eval

        # Compute distance to the nearest food
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            features['food_remaining'] = len(food_list)
        
        
        #Avoid stopping or bugging, PAULA ESTO ES MÁS O MENOS LO QUE QUERIAS IM
        if action == Directions.STOP: 
            features['stop'] = 1
        if game_state.get_agent_state(self.index).configuration.direction is not None:
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
        
            
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'invader_distance': -50, 'distance_to_food': -1, 'food_remaining': -1, 'distance_to_ghost': 2, 'ghost_scared': -1, 'stop': -100, 'reverse': -20 }
