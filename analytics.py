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
from Graph.py import *

def extract_players(match_results, players):
  res = []
  indexes = []
  for i in match_results:
    res += match_results[i]['teamMembers']
    #for pl in match_results[i]['teamMembers']:
    #  indexes.append(players.index(pl))
  return res

def make_data(match, players):
  sorted_tuples = sorted(match.items(), key=lambda item: item[1]['position'])
  match = {k: v for k, v in sorted_tuples}
  places = [match[x]['position'] for x in match]
  ret_players = extract_players(match, players)
  teams = [match[x]['teamMembers'] for x in match]
  return ret_players, teams, places

def predict_res(match, pl_skills, eps=0, ts_flag=False):
  predict_team_per = {}
  for key in match:
    pl_skills_pred = sorted([pl_skills[x].mu for x in match[key]['teamMembers']])
    pl_skills_pred =  pl_skills_pred[:6] if len(pl_skills_pred) > 6 else pl_skills_pred
    predict_team_per[key] = sum(pl_skills_pred)
  results = sorted(predict_team_per.items(), key=lambda x: x[1], reverse=ts_flag)
  num = 0
  ret_results = {}
  prev_val = 0
  for i in results:
    if abs(i[1] - prev_val) >= eps:
      num += 1
    ret_results[i[0]] = num
    prev_val = i[1]
  return ret_results

def predict_res_average(match, pl_skills, eps=0, ts_flag=False):
  predict_team_per = {}
  for key in match:
    pl_skills_pred = sorted([pl_skills[x].mu for x in match[key]['teamMembers']])
    pl_skills_pred =  pl_skills_pred[:6] if len(pl_skills_pred) > 6 else pl_skills_pred
    predict_team_per[key] = np.mean(pl_skills_pred)
  results = sorted(predict_team_per.items(), key=lambda x: x[1], reverse=ts_flag)
  num = 0
  ret_results = {}
  prev_val = 0
  for i in results:
    if abs(i[1] - prev_val) >= eps:
      num += 1
    ret_results[i[0]] = num
    prev_val = i[1]
  return ret_results

def predict_random(match, pl_skills, eps, ts_flag):
  teams_nums = list(match.keys())
  shuffle(teams_nums)
  results = {x: teams_nums.index(x) for x in teams_nums}
  return results

def predict_by_api(match, pl_skills, eps, ts_flag):
  results = {x: match[x]['predictedPosition'] for x in match}
  return results

def get_match_relults(match):
  return {x: match[x]['position'] for x in match}

def metric(y_pred, y_true):
  counter = 0
  for key_1 in y_pred:
    for key_2 in y_pred:
      if key_1 != key_2:
        if (y_pred[key_1] > y_pred[key_2] and y_true[key_1] < y_true[key_2]) or\
           (y_pred[key_1] < y_pred[key_2] and y_true[key_1] > y_true[key_2]):
           #(y_pred[key_1] != y_pred[key_2] and y_true[key_1] == y_true[key_2]):
           counter += 1
  return counter / (len(y_pred) * (len(y_pred) - 1))

def calc_probs(y_pred, y_true, teams_): #{teams: [ratings]}
  win_probs_pos, win_probs_neg = [], []
  for key_1 in y_pred:
    for key_2 in y_pred:
      if key_1 != key_2:
        if y_true[key_1] < y_true[key_2]:
          win_probs_pos.append(my_win_probability(teams_[key_2], teams_[key_1]))
          win_probs_neg.append(my_win_probability(teams_[key_1], teams_[key_2]))
        if y_true[key_1] > y_true[key_2]:
          win_probs_neg.append(my_win_probability(teams_[key_2], teams_[key_1]))
          win_probs_pos.append(my_win_probability(teams_[key_1], teams_[key_2]))
  return win_probs_pos, win_probs_neg

def calc_probs_ts(y_pred, y_true, teams_): #{teams: [ratings]}
  win_probs_pos, win_probs_neg = [], []
  for key_1 in y_pred:
    for key_2 in y_pred:
      if key_1 != key_2:
        if y_true[key_1] < y_true[key_2]:
          win_probs_pos.append(win_probability(teams_[key_2], teams_[key_1]))
          win_probs_neg.append(win_probability(teams_[key_1], teams_[key_2]))
        if y_true[key_1] > y_true[key_2]:
          win_probs_neg.append(win_probability(teams_[key_2], teams_[key_1]))
          win_probs_pos.append(win_probability(teams_[key_1], teams_[key_2]))
  return win_probs_pos, win_probs_neg

def simp_metric(y_pred, y_true):
  summ = 0
  for key_1 in y_pred:
    summ += abs(y_pred[key_1] - y_true[key_1])
  return summ / (len(y_pred) ** 2)

def calc_metric(test_results, skills, pred_fun, eps=0, what_metric=metric, ts_flag=False, train_type='ts'):
  metric_arr = []
  for match in test_results:
    metric_arr.append(metric(pred_fun(test_results[match], skills, eps, ts_flag), get_match_relults(test_results[match])))
    if train_type == 'ts_mode':
      epsilon = 60  #Optimise value!!!
      cur_players, cur_teams, cur_places = make_data(test_results[match], players)
      cur_skills = [skills[x] for x in cur_players]
      alphas = [[0 if i == 0 else 1 for i in range(len(x))] + [1] for x in cur_teams]
      game_graph = MakeGraph(len(cur_players), len(cur_teams), len(np.unique(cur_places)))
      game_graph = Initialization(cur_teams, cur_skills, GetPlaceBorders(cur_places), game_graph, alphas, epsilon)
      game_graph = ApproximateInference(cur_teams, game_graph, cur_places, epsilon, 500)
      game_graph = Propagating(cur_teams, game_graph, GetPlaceBorders(places), alphas, cur_skills)
      return_skills = GetFinalResult(game_graph)
      for i, current_player in enumerate(cur_players):
        skills[current_player] = return_skills[i]
  
  return round(np.mean(metric_arr), 5)

def calc_win_prob(test_results, skills, pred_fun, eps=0, what_metric=metric, ts_flag=False):
  win_probs_pos, win_probs_neg = [], []
  for match in tqdm(test_results):
    format_teams = {x: [skills[y] for y in test_results[match][x]['teamMembers']] for x in test_results[match]}
    cur_win_probs_pos, cur_win_probs_neg = calc_probs(pred_fun(test_results[match], skills, eps, ts_flag), get_match_relults(test_results[match]), format_teams)
    win_probs_pos.extend(cur_win_probs_pos)
    win_probs_neg.extend(cur_win_probs_neg)
  return win_probs_pos, win_probs_neg

def calc_win_prob_ts(test_results, skills, pred_fun, eps=0, what_metric=metric, ts_flag=False):
  win_probs_pos, win_probs_neg = [], []
  for match in tqdm(test_results):
    format_teams = {x: [skills[y] for y in test_results[match][x]['teamMembers']] for x in test_results[match]}
    cur_win_probs_pos, cur_win_probs_neg = calc_probs_ts(pred_fun(test_results[match], skills, eps, ts_flag), get_match_relults(test_results[match]), format_teams)
    win_probs_pos.extend(cur_win_probs_pos)
    win_probs_neg.extend(cur_win_probs_neg)
  return win_probs_pos, win_probs_neg