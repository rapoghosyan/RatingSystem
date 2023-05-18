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

#класс, задающий сообщение, которое передается по ребру фактор-графа
#храним только mu, sigma2
class Message():
  def __init__(self, mu, sigma2):
    self.mu, self.sigma2 = mu, sigma2

  def __add__(self, other):
    mes = Message(0, 0)
    mes.mu = self.mu + other.mu
    mes.sigma2 = self.sigma2 + other.sigma2
    return mes

  def __mul__(self, other):
    mes = Message(0, 0)
    mes.mu = (self.mu * other.sigma2 + other.mu * self.sigma2) / (other.sigma2 + self.sigma2)
    mes.sigma2 = (self.sigma2 * other.sigma2) / (self.sigma2 + other.sigma2)
    return mes

  #adding sub and div operators

  def __sub__(self, other):
    mes = Message(0, 0)
    mes.mu = self.mu - other.mu
    mes.sigma2 = self.sigma2 + other.sigma2
    return mes

  def __truediv__(self, other):
    mes = Message(0, 0)
    mes.mu = (self.mu * other.sigma2 - other.mu * self.sigma2) / (other.sigma2 - self.sigma2)
    mes.sigma2 = (self.sigma2 * other.sigma2) / (abs(other.sigma2 - self.sigma2))
    return mes

  def __str__(self):
    return '(mu = ' + str(round(self.mu, 2)) + ', sigma = ' + str(round(self.sigma2 ** (1 / 2), 2)) + ')'

  def __str__(self):
    return '(mu = ' + str(round(self.mu, 2)) + ', sigma = ' + str(round(self.sigma2 ** (1 / 2), 2)) + ')'

  #adding comparison operators
  def __eq__(self, other):
      return self.mu == other.mu and self.sigma2 == other.sigmas2
  def __lt__(self, other):
      if self.mu < other.mu:
          return True
      return False
    
    
def ProdMessages(messages): #список сообщений
  ret_mes = Message(messages[0].mu, messages[0].sigma2)
  for i in range(1, len(messages)):
    ret_mes.mu = (ret_mes.mu * messages[i].sigma2 + messages[i].mu * ret_mes.sigma2) / (messages[i].sigma2 + ret_mes.sigma2)
    ret_mes.sigma2 = (ret_mes.sigma2 * messages[i].sigma2) / (ret_mes.sigma2 + messages[i].sigma2)
  return ret_mes

def SumMessages(messages, alphas=0):
  messages.sort(reverse = True) #adding sort
  if alphas == 0:
    alphas = [0]
    alphas.extend([1 for x in range(len(messages))])
  ret_mes = Message(alphas[0], 0)
  for i, message in enumerate(messages):
    ret_mes.mu += alphas[i + 1] * message.mu
    ret_mes.sigma2 += (alphas[i + 1] ** 2) * message.sigma2
  return ret_mes

def AlphaMessage(message, alpha):
  return Message(message.mu * alpha, message.sigma2 * (alpha ** 2)) 
def GetExponent(mu, sigma2, t):
  return exp(- (((t - mu) * (t - mu)) / sigma2))

def GetMoment0(mu, sigma2, t):
  return 0.5 * sqrt(sigma2 * np.pi) * erfc((t - mu) / sqrt(sigma2))

def GetMoment1(mu, sigma2, t):
  return mu * GetMoment0(0, sigma2, t - mu) + 0.5 * sigma2 * GetExponent(mu, sigma2, t)

def GetMoment2(mu, sigma2, t):
  return mu * mu * GetMoment0(0, sigma2, t - mu) + 2 * mu * GetMoment1(0, sigma2, t - mu) + sigma2 * 0.25 * (
      2 * GetExponent(mu, sigma2, t) * (t - mu) + sqrt(sigma2 * np.pi) * erfc((t - mu) / sqrt(sigma2)))

def GetAlphaGreater(message, epsilon): #α>
  mu = message.mu
  sigma2 = message.sigma2
  return GetMoment0(mu, sigma2, epsilon)

def GetMuGreater(message, epsilon): #mu>
  alpha_gr = GetAlphaGreater(message, epsilon)
  return (1 / alpha_gr) * GetMoment1(message.mu, message.sigma2, epsilon)

def GetSigma2Greater(message, epsilon): #σ^2>
  alpha_gr = GetAlphaGreater(message, epsilon)
  mu_gr = GetMuGreater(message, epsilon)
  return (1 / alpha_gr) * GetMoment2(message.mu, message.sigma2, epsilon) - mu_gr * mu_gr

def GetIndicatorGreater(message, epsilon):
  if GetAlphaGreater(message, epsilon) < 1e-5: #low precision
    return Message(epsilon, message.sigma2 / 2) / message
  return Message(GetMuGreater(message, epsilon), GetSigma2Greater(message, epsilon)) / message


# for indicator I <= epsilon:

def GetAlphaLower(message, epsilon): #α≤
  mu = message.mu
  sigma2 = message.sigma2
  return GetMoment0(mu, sigma2, -epsilon) - GetMoment0(mu, sigma2, epsilon)


def GetMuLower(message, epsilon): #µ≤
  alpha_lo = GetAlphaLower(message, epsilon)
  return (1 / alpha_lo) * (GetMoment1(message.mu, message.sigma2, -epsilon) - GetMoment1(message.mu, message.sigma2, epsilon))

def GetSigma2Lower(message, epsilon): #σ^2<=
  alpha_lo = GetAlphaLower(message, epsilon)
  mu_lo = GetMuLower(message, epsilon)
  return (1 / alpha_lo) * (GetMoment2(message.mu, message.sigma2, -epsilon) - GetMoment2(message.mu, message.sigma2, epsilon)) - mu_lo * mu_lo

def GetIndicatorLower(message, epsilon):
  if GetAlphaLower(message, epsilon) < 1e-5: #low precision
    return Message(0, 1/3) / message
  return Message(GetMuLower(message, epsilon), GetSigma2Lower(message, epsilon)) / message
