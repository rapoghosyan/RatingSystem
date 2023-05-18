#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def Initialization(teams, skills, places, graph, alphas, epsilon, beta = 10): #teams[players], skills - [Message[mu, sigma2]]
  sum_of_players = 0
  for i in range(len(teams)):
    performance_messages = []
    for j in range(len(teams[i])):
      graph[0][0][sum_of_players + j] = SkillToPerformance(skills[sum_of_players + j], beta)
      performance_messages.append(graph[0][0][sum_of_players + j])
    sum_of_players += len(teams[i])
    graph[2][0][i] = PerformanceToTeam(performance_messages, alphas[i])
    graph[4][0][i] = TeamToU(graph[2][0][i])
    graph[4][1][i] = UToTeam(graph[4][0][i], epsilon)
    graph[3][0][i] = TeamToI(graph[2][0][i], graph[4][1][i])
  cur = 0
  for i in range(len(places)):
    team_messages = graph[3][0][places[i][0]:places[i][1]]
    u_messages = graph[4][1][places[i][0]:places[i][1]]
    #graph[5][0][i] = TeamToL(team_messages, u_messages)
    #testing
    messages_to_l = ListTeamsToL(team_messages, u_messages)
    #print(len(messages_to_l), "len")
    for j in range(len(messages_to_l)):
      graph[5][0][cur] = messages_to_l[j]
      cur += 1
      #print(cur, "cur")

  return graph

def ApproximateInference(teams, graph, places, epsilon, convergence_param):

  #вычисляем длины:
  m, d = len(graph[3][0]), len(graph[7][0])
  k = d + 1
  #print(m, k, d, "m k d")

  messages_to_l = []
  borders = GetPlaceBorders(places)
  assert len(borders) == k
  for i in range(k):
    start, end = borders[i]

    #debug
    #print(start, end, "start", "end")
    
    message_from_t_to_l = graph[5][0][start] #считаем что 5 - уровень к L, 4 - к U
    #print(message_from_t_to_l)
    for j in range(start + 1, end):
      #print(graph[5][0][j], j)
      message_from_t_to_l = message_from_t_to_l * graph[5][0][j]
    messages_to_l.append(message_from_t_to_l) #m ->lk

  for i in range(d):
    graph[6][0][2 * i] = LToIndicatorL(messages_to_l[i])
    graph[6][0][2 * i + 1] = LToIndicatorL(messages_to_l[i + 1])

  
  prev_low_ind = []
  prev_high_ind = []

  iter_cnt = 0
  while True:
    #повторяем до сходимости

    if (iter_cnt == 1):
      break
    #чтобы не было бесконечного цикла при плохих параметрах
    iter_cnt = 1

    for i in range(d):
      graph[7][0][i] = IndicatorLToDifference(graph[6][0][2 * i], graph[6][0][2 * i + 1])
      graph[7][1][i] = GetIndicatorGreater(graph[7][0][i], 2 * epsilon)
      #print(graph[7][0][i], graph[7][1][i])
      graph[6][1][2 * i] = IndicatorLToFirstL(graph[7][1][i], graph[6][0][2 * i + 1])
      graph[6][1][2 * i + 1] = IndicatorLToSecondL(graph[7][1][i], graph[6][0][2 * i])


    messages_to_l = [] #m lk->
    #print(k, " k")
    for i in range(k):
      start, end = borders[i]
      #print(start, " start ", end, " end")
      messages_from_u_to_l = [] #m u_i ->l_k
      for j in range(start, end):
        messages_from_u_to_l.append(graph[4][1][j])
      first_mes = []
      if i > 0:
        first_mes = graph[6][1][i - 1]
      second_mes = graph[6][1][i]

      for j in range(start, end):
        graph[5][1][j] = LToIndicatorU(first_mes, second_mes, messages_from_u_to_l, j - start)
        graph[4][0][j] = ToU(graph[6][0][i], graph[3][0][j])
        graph[4][1][j] = FromU(graph[4][0][j], epsilon)
        graph[5][0][j] = UToL(graph[4][1][j], graph[3][0][j]) #need in convergence, commenting for debug
      #print(i, "OK")
      '''
      messages_from_u_to_l = [] #сообщения только что обновились, пересчитываем заново
      for j in range(start, end):
        messages_from_u_to_l.append(graph[5][0][j])
      messages_to_l.append(ToL(messages_from_u_to_l))


    graph[6][0][0] = LToD(0, [], messages_to_l[0]) #m lk->dk
    for i in range(1, d):
      graph[6][0][2 * i] = LToD(i, graph[6][1][2 * i - 1] , messages_to_l[i])

    graph[6][0][2 * d - 1] = LToPreviousD(2 * d - 1, [], messages_to_l[d], d) #m lk->dk-1
    for i in range(1, d):
      graph[6][0][2 * i - 1] = LToPreviousD(i, graph[6][1][2 * i], messages_to_l[i], d)
    

    #now we have to check convergence:

    #debug

    now_low_ind = []
    now_high_ind= []
    for i in range(d):
      now_high_ind.append(graph[7][1][i]) # I (d > 2 eps)
      print(graph[7][1][i])

    for i in range(m):
      now_low_ind.append(graph[4][1][i]) # I(u <= eps)
    
    if iter == 0:
      prev_low_ind, prev_high_ind = now_low_ind, now_high_ind
      iter += 1
    else:
      converged = True

      #здесь скорее всего чушь, думаю надо проверять сходимость по другому
      for i in range(d):
        if abs(now_high_ind[i].mu - prev_high_ind[i].mu) + abs(now_high_ind[i].sigma2 - prev_high_ind[i].sigma2) > convergence_param:
          converged = False
          break
      for i in range(m):
        if abs(now_low_ind[i].mu - prev_low_ind[i].mu) + abs(now_low_ind[i].sigma2 - prev_low_ind[i].sigma2) > convergence_param:
          converged = False
          break
      
      if converged:
        break
      
      iter += 1
      '''

  #returning m->tj
  messages_to_teams = []
  for i in range(m):
    graph[3][1][i] = graph[5][1][i]  - graph[4][1][i]
    messages_to_teams.append(graph[5][1][i]  - graph[4][1][i])
  #return messages_to_teams
  return graph


def Propagating(teams, graph, places, alphas, skills, beta = 10):
  sum_of_players = 0
  for i in range(len(teams)):
    for j in range(len(teams[i])):
      other_teams_messages = graph[0][0][sum_of_players:sum_of_players + j]
      other_teams_messages.extend(graph[0][0][sum_of_players + j + 1: sum_of_players + len(teams[i])])
      current_alphas = alphas[i][:j]
      current_alphas.extend(alphas[i][j + 1:])
      graph[1][1][sum_of_players + j] = TeamToPerformance(graph[3][1][i], other_teams_messages, alphas[i][j  + 1], current_alphas)
      graph[0][1][sum_of_players + j] = PerformanceToSkill(graph[1][1][sum_of_players + j], skills[i], beta)
    sum_of_players += len(teams[i])
  return graph

def GetFinalResult(graph):
  skills = []
  for i in range(len(graph[0][0])):
    skills.append(graph[0][0][i] * graph[0][1][i])
  return skills
