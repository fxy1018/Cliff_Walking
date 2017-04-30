'''
Created on Apr 26, 2017

@author: fanxueyi
'''
from gridworld import Gridworld
from qAgent import QLearningAgent 
from sarsaAgent import SarsaAgent 
import numpy as np
import pandas as pd

##################
#create grid
##################

def getCliffGrid():
    grid = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            ['S',-100,-100, -100, -100, -100,-100,-100, -100, -100, -100,'TERMINAL_STATE']]
    return Gridworld(grid)


###################
#get next state and reward
###################

def nextStep(state, action, grid):
    grid = grid.grid
    if action == "west":
        newState = [state[0], state[1]-1]
        
    elif action == "east":
        newState = [state[0], state[1]+1]
    
    elif action == "north":
        newState = [state[0]-1, state[1]]
 
    elif action == "south":
        newState = [state[0]+1, state[1]]
             
    if newState == [3,0] or newState == [3,11]:
        reward = -1
    else:
        reward = grid[newState[0]][newState[1]]
    return(tuple(newState), reward)


####################
#get Q-learning rewards
####################

def getQValues():
    Values = []
    for i in range(40):
        QAgent = QLearningAgent()
        Rewards = []
        for e in range(QAgent.episodes):
            state = tuple(grid.getStartState())
            done = False
            sumReward = 0
            while not done:
                action = QAgent.getAction(state)
                if not action:
                    done = True
                elif state[0] == 3 and state[1] != 0:
                    done=True
                else:
                    nextState, reward = nextStep(state, action, grid)
                    QAgent.update(state, action, nextState, reward)
                    state = nextState
                    sumReward += reward 
            Rewards.append(sumReward)
        Values.append(Rewards)
    return(Values)

####################
#get Sarsa rewards
####################

def getSValues():
    Values = []
    for i in range(40):
        sarsaAgent = SarsaAgent()
        SRewards = []
        for e in range(sarsaAgent.episodes):
            state = tuple(grid.getStartState())
            done = False
            sumReward = 0
            action = sarsaAgent.getAction(state)
            while not done:
                if not action:
                    done = True
                elif state[0] == 3 and state[1] != 0:
                    done = True
                else:
                    nextState, reward = nextStep(state, action, grid)
                    nextAction = sarsaAgent.getAction(nextState)
                    sarsaAgent.update(state, action, nextState, nextAction, reward)
                    state = nextState
                    action = nextAction
                    sumReward += reward 
      
            SRewards.append(sumReward)
        Values.append(SRewards)
    return(Values)

####################
#smooth the data
#################### 
def smooth(Values):
    temp_sum = 0
    res = []
    for i in range(0, len(Values)-9):
        res.append(np.mean(Values[i:i+10]))
    return(res) 
  

if __name__ == "__main__":
    grid = getCliffGrid()
    
    QValues = getQValues()
    newQ = zip(*QValues)
    QValues = map(np.mean, newQ)
    
    SValues = getSValues()
    newS = zip(*SValues)
    SValues = map(np.mean, newS)
        

    ts = pd.DataFrame({'Sarsa':smooth(SValues), 'Q-learning': smooth(QValues)}, index= [i for i in range(491)])
    print(ts)
    pg = ts.plot()
     
     
    pg.set_ylim(-100,-15)
    pg.set_xlabel("Episodes")
    pg.set_ylabel("Sum of rewards during episode")
     
    fig = pg.get_figure()
    fig.savefig('result.jpg')
   
    
                
                
            
            
        
        
    
    
    
    
    
    
    