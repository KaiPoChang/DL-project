#!/usr/bin/env python
# coding: utf-8

# In[5]:


import random
from tqdm import tqdm


# In[3]:


class board:
    """simple implementation of 2048 puzzle"""
    
    def __init__(self, tile = None):
        self.tile = tile if tile is not None else [0] * 16
    
    def __str__(self):
        state = '+' + '-' * 24 + '+\n'
        for row in [self.tile[r:r + 4] for r in range(0, 16, 4)]:
            state += ('|' + ''.join('{0:6d}'.format((1 << t) & -2) for t in row) + '|\n')
        state += '+' + '-' * 24 + '+'
        return state
    
    def mirror(self):
        return board([self.tile[r + i] for r in range(0, 16, 4) for i in reversed(range(4))])
    
    def transpose(self):
        return board([self.tile[r + i] for i in range(4) for r in range(0, 16, 4)])
    
    def left(self):
        move, score = [], 0
        for row in [self.tile[r:r+4] for r in range(0, 16, 4)]:
            row, buf = [], [t for t in row if t]
            while buf:
                if len(buf) >= 2 and buf[0] is buf[1]:
                    buf = buf[1:]
                    buf[0] += 1
                    score += 1 << buf[0]
                row += [buf[0]]
                buf = buf[1:]
            move += row + [0] * (4 - len(row))
        return board(move), score if move != self.tile else -1
    
    def right(self):
        move, score = self.mirror().left()
        return move.mirror(), score
    
    def up(self):
        move, score = self.transpose().left()
        return move.transpose(), score
    
    def down(self):
        move, score = self.transpose().right()
        return move.transpose(), score
    
    def popup(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        tile[random.choice(empty)] = random.choice([1] * 9 + [2])
        return board(tile)


# In[7]:


n_episode = 100000

if __name__ == '__main__':
    print('2048 Demo\n')
    score_collection = []
    for episode in tqdm(range(n_episode)):
        state = board().popup().popup()
        score = 0
        step = 0
        while True:
            
            moves = [state.up(), state.right(), state.down(), state.left()]
            
            after, reward = max(moves, key = lambda move: move[1])
            if reward == -1:
                score_collection.append(score)
                break
            state = after.popup()
            score += reward
            step += 1


# In[28]:


import numpy as np
score_freq = [0, 0, 0, 0, 0]
for score in score_collection:
    if score <= 1000:
        score_freq[0] += 1
    elif score > 1000 and score <= 2000:
        score_freq[1] += 1
    elif score > 2000 and score <= 3000:
        score_freq[2] += 1
    elif score > 3000 and score <= 4000:
        score_freq[3] += 1
    elif score > 4000:
        score_freq[4] += 1
print('            Learning score table')
print('Maxium scores: ', np.max(score_freq))
print('Score\t\t\tFrequency\tRatio')
print('0 ~ 1000\t\t{}\t\t{}'.format(score_freq[0], np.round(score_freq[0]/len(score_collection), 4)))
print('1000 ~ 2000\t\t{}\t\t{}'.format(score_freq[1], np.round(score_freq[1]/len(score_collection), 4)))
print('2000 ~ 3000\t\t{}\t\t{}'.format(score_freq[2], np.round(score_freq[2]/len(score_collection), 4)))
print('3000 ~ 4000\t\t{}\t\t{}'.format(score_freq[3], np.round(score_freq[3]/len(score_collection), 4)))
print('4000 ~\t\t\t{}\t\t{}'.format(score_freq[4], np.round(score_freq[4]/len(score_collection), 4)))


# In[26]:


import matplotlib.pyplot as plt

epoch = np.arange(0, n_episode, 1000)
new_score_collection = []
for step, score in enumerate(score_collection):
    if step%1000 == 0:
        new_score_collection.append(score)

plt.figure('2048 ep_training process')
plt.plot(epoch, new_score_collection, color='blue')


plt.title('2048 ep_training process')
plt.xlabel('Epoch')
plt.ylabel('Score')

my_x_ticks = np.arange(0, n_episode, 10000)
plt.xticks(my_x_ticks)

plt.savefig('2048ep_train.png')
plt.show() 

