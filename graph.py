#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#factor_graph[уровень][индекс][направление] = Message(mu, sigma2)
#skill - n, p - n, t - m, u - m, l - k, d - (k - 1)
#вниз - 0, сообщение наверх - 1
import numpy as np
import pandas as pd
import scipy as sp
import itertools
import math
from tqdm import tqdm
from scipy.special import erf, erfc
from scipy.stats import norm, rankdata
from numpy import sqrt, exp
from random import shuffle
import trueskill as ts
import pickle
from Message.py import *
from Sender.py import *

def MakeGraph(players, teams, places): #numbers of entities
  s_p = [[0 for i in range(players)], [0 for i in range(players)]]
  p_I = [[0 for i in range(players)], [0 for i in range(players)]]
  t = [[0 for i in range(teams)], [0 for i in range(teams)]]
  t_I = [[0 for i in range(teams)], [0 for i in range(teams)]]
  I_u = [[0 for i in range(teams)], [0 for i in range(teams)]]
  I_l = [[0 for i in range(teams)], [0 for i in range(teams)]]
  l_I = [[0 for i in range(2 * (places - 1))], [0 for i in range(2 * (places - 1))]]
  I_d = [[0 for i in range(places - 1)], [0 for i in range(places - 1)]]
  return [s_p, p_I, t, t_I, I_u, I_l, l_I, I_d]

def PrintGraph(graph):
  names = ['skill-performance', 'performance-I', 'I-team', 'team-I',           'I-u', 'I-l', 'l-I', 'I-diffrence']
  for i, line in enumerate(graph):
    print(names[i])
    print('↓: ' + ', '.join(map(str, line[0])))
    print('↑: ' + ', '.join(map(str, line[1])))
    
def GetPlaceBorders(places): #хотим удобно находить T(K)
  #places[i] - место которое заняла i-ая команда, массив неубывает
  #поправила и затестила, вроде все ок
  borders = []
  cur = 0
  m = len(places)
  for i in range(1, m):
    if places[i] != places[i - 1]:
      borders.append([cur, i]) # все команды от cur до i - 1 заняли одно и тоже место
      cur = i
  borders.append([cur, m])
  return borders
