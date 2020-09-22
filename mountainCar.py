import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# Hyperparameters
LR = 0.1        # Learning rate
DISC = 0.95     # Discount
EPS = 25000     # Episodes
EPSIL = 0.5     # Epsilon

START_EPSIL_DECAY = 1
END_EPSIL_DECAY = EPS // 2
EPSIL_DECAY_VAL = EPSIL / (END_EPSIL_DECAY - START_EPSIL_DECAY)
SHOW_EVERY = 2000

obsHigh = env.observation_space.high
obsLow = env.observation_space.low
obsN = env.action_space.n

# Choosing a value from 20 for discrete actions
discreteSize = [20] * len(obsHigh)
discreteWinSize = (obsHigh - obsLow) / discreteSize

qTable = np.random.uniform(low=-2, high=0, size=(discreteSize + [obsN]))

# Helper for constinuous -> discrete states
def getDiscreteState(state):
    discreteState = (state - obsLow) / discreteWinSize
    return tuple(discreteState.astype(np.int))

# Training over a bunch of episodes
for episode in range(EPS):
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
    #print(f"Episode {episode}")
    discreteState = getDiscreteState(env.reset())
    done = False
    while not done:
        # See if random threshold justifies exploration
        if np.random.random() > EPSIL:
            action = np.argmax(qTable[discreteState])
        else:
            action = np.random.randint(0, obsN)
        new_state, reward, done, _ = env.step(action)
        newDiscreteState = getDiscreteState(new_state)
        if render:
            env.render()
        # Q-Table equation
        if not done:
            maxQ = np.max(qTable[newDiscreteState])
            currQ = qTable[discreteState + (action, )]
            newQ = (1- LR) * currQ + LR * (reward + DISC * maxQ)
            qTable[discreteState + (action, )] = newQ
        elif new_state[0] >= env.goal_position:
            print(f"Goal reached at {episode}")
            qTable[discreteState + (action, )] = 0

        discreteState = newDiscreteState
    if END_EPSIL_DECAY >= episode >= START_EPSIL_DECAY:
        EPSIL -= EPSIL_DECAY_VAL


env.close()
