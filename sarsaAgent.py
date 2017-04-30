
import random,util,math

class SarsaAgent(object):
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=1.0, episodes = 500):
        self.seenState = set()
        self.QValues = util.Counter()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.episodes = int(episodes)
        
        
    def getLegalActions(self, state):
        act = []
        if state == (3,11):
            return(None)
        
        if state[0] != 0:
            act.append("north")
        if state[0] != 3:
            act.append("south")
        if state[1] != 0:
            act.append("west")
        if state[1] != 11:
            act.append("east")
    
        return(act)
    
    def getAction(self, state):
            """
              Compute the action to take in the current state.  With
              probability self.epsilon, we should take a random action and
              take the best policy action otherwise.  Note that if there are
              no legal actions, which is the case at the terminal state, you
              should choose None as the action.
    
              HINT: You might want to use util.flipCoin(prob)
              HINT: To pick randomly from a list, use random.choice(list)
            """
            # Pick Action
            legalActions = self.getLegalActions(state)
            action = None
            "*** YOUR CODE HERE ***"
    #         util.raiseNotDefined()
            
            if util.flipCoin(self.epsilon) and legalActions:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
    
            return action
    
    def getPolicy(self, state):
            return self.computeActionFromQValues(state)
        
    
    def getQValue(self, state, action):
            """
              Returns Q(state,action)
              Should return 0.0 if we have never seen a state
              or the Q node value otherwise
            """
            "*** YOUR CODE HERE ***"
            
            if state not in self.seenState or (state, action) not in self.QValues:
                self.QValues[(state,action)] = 0.0
                self.seenState.add(state)
                return(0.0)
            
            else:
                return(self.QValues[(state,action)])
            
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return(0.0)
        
        max_actionQ = self.getQValue(state,actions[0])
        for action in actions:
            max_actionQ = max(max_actionQ,self.getQValue(state,action))
        return(max_actionQ)
    
    
    def computeActionFromQValues(self, state):
            """
              Compute the best action to take in a state.  Note that if there
              are no legal actions, which is the case at the terminal state,
              you should return None.
            """
            "*** YOUR CODE HERE ***"
            actions = self.getLegalActions(state)
            if not actions:
                return(None)
            res_action = actions[0]
            max_actionQ = self.getQValue(state,res_action)
            for action in actions:
                if max_actionQ < self.getQValue(state,action):
                    max_actionQ = self.getQValue(state,action)
                    res_action = action
                    
            return(res_action)
                
    def update(self, state, action, nextState, nextAction, reward):
            """
              The parent class calls this to observe a
              state = action => nextState and reward transition.
              You should do your Q-Value update here
    
              NOTE: You should never call this function,
              it will be called on your behalf
            """
            "*** YOUR CODE HERE ***"
            
            curr_Qvalue = self.getQValue(state,action)
            new_Qvalue = curr_Qvalue + self.alpha * (int(reward) + self.discount * self.getQValue(nextState, nextAction) - curr_Qvalue)
            self.QValues[(state, action)] = new_Qvalue
            return
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)
    