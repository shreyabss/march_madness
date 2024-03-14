# This is a sample Python script.
import os
import random

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Slider, Dropdown, Button
from bokeh.plotting import output_file, show
from bokeh.io import show
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker, HoverTool, ColumnDataSource
from bokeh.palettes import Viridis256, Magma256
from bokeh.plotting import figure
from bokeh.transform import transform
# first neural network with keras tutorial
from numpy import loadtxt
from pandas.errors import PerformanceWarning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from itertools import combinations
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OrdinalEncoder
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import cv

import warnings
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def load_data():
    csvs = {}
    directory = os.path.join("C:\\Users\\shreya\\MarchMadnessProject\\march-machine-learning"
                             "-mania-2023")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csvs[file] = pd.read_csv(f'C:\\Users\\Shreya\\MarchMadnessProject\\march-machine-learning-mania-2023\\{file}', encoding='cp1252')
                print(f'Read file {file}, it has columns {csvs[file].columns}, it has shape {csvs[file].shape}')
    return csvs

def calculate_averages(s, csvs, file):
    season = s
    regular_season = csvs[file]
    season_s = regular_season.loc[regular_season['Season'] == season]
    teams = csvs['MTeams.csv'].iloc[:, 0:2].copy()
    teams['OPPFGM_W'] = season_s.groupby('WTeamID')['LFGM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPPOINTS_W'] = season_s.groupby('WTeamID')['LScore'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFGA_W'] = season_s.groupby('WTeamID')['LFGA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFGM3_W'] = season_s.groupby('WTeamID')['LFGM3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFGA3_W'] = season_s.groupby('WTeamID')['LFGA3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPOR_W'] = season_s.groupby('WTeamID')['LOR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPDR_W'] = season_s.groupby('WTeamID')['LDR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPAST_W'] = season_s.groupby('WTeamID')['LAst'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPTO_W'] = season_s.groupby('WTeamID')['LTO'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPSTL_W'] = season_s.groupby('WTeamID')['LStl'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPBLK_W'] = season_s.groupby('WTeamID')['LBlk'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPPF_W'] = season_s.groupby('WTeamID')['LPF'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFTM_W'] = season_s.groupby('WTeamID')['LFTM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFTA_W'] = season_s.groupby('WTeamID')['LFTA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGM_W'] = season_s.groupby('WTeamID')['WFGM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['POINTS_W'] = season_s.groupby('WTeamID')['WScore'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGA_W'] = season_s.groupby('WTeamID')['WFGA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGM3_W'] = season_s.groupby('WTeamID')['WFGM3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGA3_W'] = season_s.groupby('WTeamID')['WFGA3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FTM_W'] = season_s.groupby('WTeamID')['WFTM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FTA_W'] = season_s.groupby('WTeamID')['WFTA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OR_W'] = season_s.groupby('WTeamID')['WOR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['DR_W'] = season_s.groupby('WTeamID')['WDR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['AST_W'] = season_s.groupby('WTeamID')['WAst'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['TO_W'] = season_s.groupby('WTeamID')['WTO'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['STL_W'] = season_s.groupby('WTeamID')['WStl'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['BLK_W'] = season_s.groupby('WTeamID')['WBlk'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['PF_W'] = season_s.groupby('WTeamID')['WPF'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values

    teams['OPPFGM_L'] = season_s.groupby('LTeamID')['WFGM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPPOINTS_L'] = season_s.groupby('LTeamID')['WScore'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFGA_L'] = season_s.groupby('LTeamID')['WFGA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFGM3_L'] = season_s.groupby('LTeamID')['WFGM3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFGA3_L'] = season_s.groupby('LTeamID')['WFGA3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPOR_L'] = season_s.groupby('LTeamID')['WOR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPDR_L'] = season_s.groupby('LTeamID')['WDR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPAST_L'] = season_s.groupby('LTeamID')['WAst'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPTO_L'] = season_s.groupby('LTeamID')['WTO'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPSTL_L'] = season_s.groupby('LTeamID')['WStl'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPBLK_L'] = season_s.groupby('LTeamID')['WBlk'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPPF_L'] = season_s.groupby('LTeamID')['WPF'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFTM_L'] = season_s.groupby('LTeamID')['WFTM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OPPFTA_L'] = season_s.groupby('LTeamID')['WFTA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGM_L'] = season_s.groupby('LTeamID')['LFGM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['POINTS_L'] = season_s.groupby('LTeamID')['LScore'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGA_L'] = season_s.groupby('LTeamID')['LFGA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGM3_L'] = season_s.groupby('LTeamID')['LFGM3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FGA3_L'] = season_s.groupby('LTeamID')['LFGA3'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FTM_L'] = season_s.groupby('LTeamID')['LFTM'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['FTA_L'] = season_s.groupby('LTeamID')['LFTA'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['OR_L'] = season_s.groupby('LTeamID')['LOR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['DR_L'] = season_s.groupby('LTeamID')['LDR'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['AST_L'] = season_s.groupby('LTeamID')['LAst'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['TO_L'] = season_s.groupby('LTeamID')['LTO'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['STL_L'] = season_s.groupby('LTeamID')['LStl'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['BLK_L'] = season_s.groupby('LTeamID')['LBlk'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['PF_L'] = season_s.groupby('LTeamID')['LPF'].sum().reindex(np.arange(1101, 1478, 1), fill_value=0).values

    teams['OPPFGM'] = teams['OPPFGM_W'] + teams['OPPFGM_L']
    teams['OPPPOINTS'] = teams['OPPPOINTS_W'] + teams['OPPFGM_L']
    teams['OPPFGA'] = teams['OPPFGA_W'] + teams['OPPFGA_L']
    teams['OPPFGM3'] = teams['OPPFGM3_W'] + teams['OPPFGM3_L']
    teams['OPPFGA3'] = teams['OPPFGA3_W'] + teams['OPPFGA3_L']
    teams['OPPOR'] = teams['OPPOR_L'] + teams['OPPOR_W']
    teams['OPPDR'] = teams['OPPDR_L'] + teams['OPPDR_W']
    teams['OPPAST'] = teams['OPPAST_L'] + teams['OPPAST_W']
    teams['OPPTO'] = teams['OPPTO_L'] + teams['OPPTO_W']
    teams['OPPSTL'] = teams['OPPSTL_L'] + teams['OPPSTL_W']
    teams['OPPBLK'] = teams['OPPBLK_L'] + teams['OPPBLK_W']
    teams['OPPPF'] = teams['OPPPF_L'] + teams['OPPPF_W']
    teams['OPPFTM'] = teams['OPPFTM_L'] + teams['OPPFTM_W']
    teams['OPPFTA'] = teams['OPPFTA_W'] + teams['OPPFTA_L']
    teams['FGM'] = teams['FGM_W'] + teams['FGM_L']
    teams['POINTS'] = teams['POINTS_W'] + teams['FGM_L']
    teams['FGA'] = teams['FGA_W'] + teams['FGA_L']
    teams['FGM3'] = teams['FGM3_W'] + teams['FGM3_L']
    teams['FGA3'] = teams['FGA3_W'] + teams['FGA3_L']
    teams['OR'] = teams['OR_L'] + teams['OR_W']
    teams['DR'] = teams['DR_L'] + teams['DR_W']
    teams['AST'] = teams['AST_L'] + teams['AST_W']
    teams['TO'] = teams['TO_L'] + teams['TO_W']
    teams['STL'] = teams['STL_L'] + teams['STL_W']
    teams['BLK'] = teams['BLK_L'] + teams['BLK_W']
    teams['PF'] = teams['PF_L'] + teams['PF_W']
    teams['FTM'] = teams['FTM_L'] + teams['FTM_W']
    teams['FTA'] = teams['FTA_W'] + teams['FTA_L']

    teams['WINS'] = season_s['WTeamID'].value_counts().sort_index().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['LOSSES'] = season_s['LTeamID'].value_counts().sort_index().reindex(np.arange(1101, 1478, 1), fill_value=0).values
    teams['GAMES'] = teams['WINS'] + teams['LOSSES']
    teams['POSSESSIONS'] = 0.5 * (teams['FGA'] - teams['OR'] + teams['TO'] + (0.475 * teams['FTA']))
    teams['OPPPOSSESSIONS'] = 0.5 * (teams['OPPFGA'] - teams['OPPOR'] + teams['OPPTO'] + (0.475 * teams['OPPFTA']))
    teams['ORTG'] = (100 / (teams['POSSESSIONS'] + teams['OPPPOSSESSIONS'])) * teams['POINTS']
    teams['DRTG'] = (100 / (teams['POSSESSIONS'] + teams['OPPPOSSESSIONS'])) * teams['OPPPOINTS']
    teams['OPPFGMPG'] = teams['OPPFGM'] / teams['GAMES']
    teams['OPPPOINTSPG'] = teams['OPPPOINTS'] / teams['GAMES']
    teams['OPPFGAPG'] = teams['OPPFGA'] / teams['GAMES']
    teams['OPPFGM3PG'] = teams['OPPFGM3'] / teams['GAMES']
    teams['OPPFGA3PG'] = teams['OPPFGA3'] / teams['GAMES']
    teams['OPPORPG'] = teams['OPPOR'] / teams['GAMES']
    teams['OPPDRPG'] = teams['OPPDR'] / teams['GAMES']
    teams['OPPASTPG'] = teams['OPPAST'] / teams['GAMES']
    teams['OPPTOPG'] = teams['OPPTO'] / teams['GAMES']
    teams['OPPSTLPG'] = teams['OPPSTL'] / teams['GAMES']
    teams['OPPBLKPG'] = teams['OPPBLK'] / teams['GAMES']
    teams['OPPPFPG'] = teams['OPPPF'] / teams['GAMES']
    teams['OPPFTMPG'] = teams['OPPFTM'] / teams['GAMES']
    teams['OPPFTAPG'] = teams['OPPFTA'] / teams['GAMES']
    teams['FGMPG'] = teams['FGM'] / teams['GAMES']
    teams['POINTSPG'] = teams['POINTS'] / teams['GAMES']
    teams['FGAPG'] = teams['FGA'] / teams['GAMES']
    teams['FGM3PG'] = teams['FGM3'] / teams['GAMES']
    teams['FGA3PG'] = teams['FGA3'] / teams['GAMES']
    teams['ORPG'] = teams['OR'] / teams['GAMES']
    teams['DRPG'] = teams['DR'] / teams['GAMES']
    teams['ASTPG'] = teams['AST'] / teams['GAMES']
    teams['TOPG'] = teams['TO'] / teams['GAMES']
    teams['STLPG'] = teams['STL'] / teams['GAMES']
    teams['BLKPG'] = teams['BLK'] / teams['GAMES']
    teams['PFPG'] = teams['PF'] / teams['GAMES']
    teams['FTMPG'] = teams['FTM'] / teams['GAMES']
    teams['FTAPG'] = teams['FTA'] / teams['GAMES']

    regular_season = csvs['MNCAATourneySeeds.csv']
    season_s = regular_season.loc[regular_season['Season'] == season]
    return teams
    #df.loc[df['a'] == 1, 'b'].sum()
    #df.groupby('a')['b']df.reindex(np.arange(214200), fill_value=0)

def calculate_tourney_wins_and_graph(reg, tou):
    reg['TOURNWINS'] = tou['WINS']
    reg['TOURNLS'] = tou['LOSSES']

    menu = []
    for x in reg:
        menu.append((x, x))

    output_file("js_on_change.html")
    reg['x'] = reg['ORTG']
    reg['y'] = reg['DRTG']
    reg['c'] = reg['POINTS']

    label = {'xl': 'Offensive Rating', 'yl': 'Defensive Rating', 'sl': 'Tournament Wins'}

    source = ColumnDataSource(data=reg)

    p = figure(x_axis_label=label['xl'], y_axis_label=label['yl'])

    mapper = LinearColorMapper(palette=Viridis256, low=min(reg['c']),
                               high=max(reg['c']))
    color_bar = ColorBar(color_mapper=mapper,
                         location=(0, 0),
                         ticker=BasicTicker())
    p.add_layout(color_bar, 'right')

    p.scatter(x='x', y='y', size=10,
              fill_color=transform('c', mapper),
              source=source)
    p.add_tools(HoverTool(
        tooltips=[('Name', '@{TeamName}'), ('Tournament Wins', '@{TOURNWINS}'), ('Regular Season Wins', '@{WINS}'),
                  ('Regular Season Losses', '@{LOSSES}'),
                  ("Team ID", '@{TeamID}')]))

    maps = {}
    for x in reg:
        if x == 'TeamName' or x == 'TeamID':
            continue
        if '_' in x:
            continue
        maps[x] = LinearColorMapper(palette=Viridis256, low=min(reg[x]),
                                    high=max(reg[x]))

    callbackX = CustomJS(args=dict(source=source, label=label, axis=p.xaxis[0]), code="""
                console.log(label['xl']);
                const data = source.data;
                source.data['x'] = data[this.item]
                label['xl'] = this.item;
                axis.axis_label = this.item;
                source.change.emit();

                console.log('dropdown: ' + this.item + source.data['x'], this.toString())  
            """)

    callbackY = CustomJS(args=dict(source=source, label=label, axis=p.yaxis[0]), code="""
                    const data = source.data;
                    source.data['y'] = data[this.item];
                    source.change.emit();
                    label['yl'] = this.item;
                    axis.axis_label = this.item;
                    console.log('dropdown: ' + this.item + source.data['y'], this.toString())  
                """)
    val = 1

    callbackC = CustomJS(args=dict(source=source, mapper=mapper, maps=maps, label=label), code="""
                    const data = source.data;
                    source.data['c'] = data[this.item];
                    mapper.low = maps[this.item].low;
                    mapper.high = maps[this.item].high;

                    source.change.emit();
                    mapper.change.emit();
                    console.log('dropdown: ' + mapper.palette + maps, this.toString())  
                """)

    callback = CustomJS(args=dict(source=source), code="""
            const data = source.data;
            for (let i = 0; i < 149; i++) {
                data['s'][i] = data['s'][i]*0.01;
            }

            source.change.emit();
        """)

    dropdownX = Dropdown(label="Change X", button_type="warning", menu=menu)
    dropdownX.js_on_event("menu_item_click", callbackX)
    dropdownY = Dropdown(label="Change Y", button_type="warning", menu=menu)
    dropdownY.js_on_event("menu_item_click", callbackY)
    dropdownC = Dropdown(label="Change Color", button_type="warning", menu=menu)
    dropdownC.js_on_event("menu_item_click", callbackC)
    button = Button(label="Divide by 10!", button_type="success")
    button.js_on_click(callback)


    layout = column(dropdownX, dropdownY, dropdownC, p)

    show(layout)
    return reg

def make_training_data(csvs, file):
    szns = np.arange(2003, 2019, 1)
    dataset = {}
    c = 0
    for i in szns:
        print(i)
        temp = calculate_averages(i, csvs, file)
        training_data_cols = temp.columns[-35:].values.tolist()
        training_data_cols.append('Team1')
        training_data_cols.append('Team2')
        training_data_cols.append('Team1Wins?')
        tourNCAA = csvs['MNCAATourneyCompactResults.csv']
        tourNCAA_s = tourNCAA.loc[tourNCAA['Season'] == i]
        tourSec = csvs['MSecondaryTourneyCompactResults.csv']
        tourSec_s = tourSec.loc[tourSec['Season'] == i]
        dataset[i] = []

        for index, row in tourNCAA_s.iterrows():
            #print(row['c1'], row['c2'])
            a = temp.loc[temp['TeamID'] == row['WTeamID']].iloc[:, -35:]
            b = temp.loc[temp['TeamID'] == row['LTeamID']].iloc[:, -35:]
            num1 = random.randint(0, 1)
            if num1 == 1:
                x = (a.values - b.values).tolist()[0]
                x = [i * -1 for i in x]
                x.append(row['LTeamID'])
                x.append(row['WTeamID'])
                x.append(1)
            else :
                x = (a.values - b.values).tolist()[0]
                x.append(row['WTeamID'])
                x.append(row['LTeamID'])
                x.append(0)
            dataset[i].append(x)
        for index, row in tourSec_s.iterrows():
            # print(row['c1'], row['c2'])
            a = temp.loc[temp['TeamID'] == row['WTeamID']].iloc[:, -35:]
            b = temp.loc[temp['TeamID'] == row['LTeamID']].iloc[:, -35:]
            num1 = random.randint(0, 1)
            if num1 == 1:
                x = (a.values - b.values).tolist()[0]
                x = [i * -1 for i in x]
                x.append(row['LTeamID'])
                x.append(row['WTeamID'])
                x.append(1)
            else :
                x = (a.values - b.values).tolist()[0]
                x.append(row['WTeamID'])
                x.append(row['LTeamID'])
                x.append(0)
            dataset[i].append(x)
            #print(x)
        if i == 2003:
            c = np.array(dataset[i])
        else:
            c = np.concatenate((c, np.array(dataset[i])), axis=0)

    return pd.DataFrame(data=c, columns=training_data_cols)

def make_training_data(csvs, file):
    szns = np.append(np.arange(2003, 2019, 1), [2021, 2022])
    dataset = {}
    c = 0
    for i in szns:
        print(i)
        temp = calculate_averages(i, csvs, file)
        training_data_cols = temp.columns[-35:].values.tolist()
        training_data_cols.append('Team1')
        training_data_cols.append('Team2')
        training_data_cols.append('Team1Wins?')
        training_data_cols.append('Team1Name')
        training_data_cols.append('Team2Name')
        tourNCAA = csvs['MNCAATourneyCompactResults.csv']
        tourNCAA_s = tourNCAA.loc[tourNCAA['Season'] == i]
        tourSec = csvs['MSecondaryTourneyCompactResults.csv']
        tourSec_s = tourSec.loc[tourSec['Season'] == i]
        dataset[i] = []
        team_names = csvs['MTeams.csv']

        for index, row in tourNCAA_s.iterrows():
            a = temp.loc[temp['TeamID'] == row['WTeamID']].iloc[:, -35:]
            b = temp.loc[temp['TeamID'] == row['LTeamID']].iloc[:, -35:]
            num1 = random.randint(0, 1)
            if num1 == 1:
                x = (a.values - b.values).tolist()[0]
                x = [i * -1 for i in x]
                x.append(row['LTeamID'])
                x.append(row['WTeamID'])
                x.append(0)
                x.append(team_names.loc[team_names['TeamID'] == row['LTeamID']].iloc[:, 1].values[0])
                x.append(team_names.loc[team_names['TeamID'] == row['WTeamID']].iloc[:, 1].values[0])
            else :
                x = (a.values - b.values).tolist()[0]
                x.append(row['WTeamID'])
                x.append(row['LTeamID'])
                x.append(1)
                x.append(team_names.loc[team_names['TeamID'] == row['WTeamID']].iloc[:, 1].values[0])
                x.append(team_names.loc[team_names['TeamID'] == row['LTeamID']].iloc[:, 1].values[0])
            dataset[i].append(x)
        for index, row in tourSec_s.iterrows():
            # print(row['c1'], row['c2'])
            a = temp.loc[temp['TeamID'] == row['WTeamID']].iloc[:, -35:]
            b = temp.loc[temp['TeamID'] == row['LTeamID']].iloc[:, -35:]
            num1 = random.randint(0, 1)
            if num1 == 1:
                x = (a.values - b.values).tolist()[0]
                x = [i * -1 for i in x]
                x.append(row['LTeamID'])
                x.append(row['WTeamID'])
                x.append(0)
                x.append(team_names.loc[team_names['TeamID'] == row['LTeamID']].iloc[:, 1].values[0])
                x.append(team_names.loc[team_names['TeamID'] == row['WTeamID']].iloc[:, 1].values[0])
            else :
                x = (a.values - b.values).tolist()[0]
                x.append(row['WTeamID'])
                x.append(row['LTeamID'])
                x.append(1)
                x.append(team_names.loc[team_names['TeamID'] == row['WTeamID']].iloc[:, 1].values[0])
                x.append(team_names.loc[team_names['TeamID'] == row['LTeamID']].iloc[:, 1].values[0])
            dataset[i].append(x)
            #print(x)
        if i == 2003:
            c = np.array(dataset[i])
        else:
            c = np.concatenate((c, np.array(dataset[i])), axis=0)
        print(c.shape)
    return pd.DataFrame(data=c, columns=training_data_cols)

def make_2023_data(csvs, file):
    i = 2023
    temp = calculate_averages(i, csvs, file)
    training_data_cols = temp.columns[-35:].values.tolist()
    training_data_cols.append('Team1')
    training_data_cols.append('Team2')
    training_data_cols.append('Team1Wins?')
    games = list(combinations(list(range(1101, 1478, 1)), 2))
    dataset = []

    for game in games:
        a = temp.loc[temp['TeamID'] == game[0]].iloc[:, -35:]
        b = temp.loc[temp['TeamID'] == game[1]].iloc[:, -35:]
        x = (a.values - b.values).tolist()[0]
        x.append(game[0])
        x.append(game[1])
        dataset.append(x)
    c = np.array(dataset)
    return pd.DataFrame(data=c, columns=training_data_cols)


def corelation_coefficient_calculation(teams, tou):
    print("WELCOME TO CORRELATION_COEFFICIENT_CALCULATOR!")
    print("We find what variables are a bop and what variables are a flop!")

    print(teams.dropna())
    teams = teams.dropna()

    teams['TOURNWINS'] = tou['WINS']
    print(teams.head())

    dict = {}

    for col in teams.columns:
        print(col)
        if (col != "TeamID" and col != "TeamName" and not(col.endswith('_W')) and not(col.endswith('_L'))):
            print(f"Correlation Coefficient Calculated between {col} and Tournament Wins: {np.corrcoef(teams[col], teams['TOURNWINS'])[0,1]}")
            dict[col] = np.corrcoef(teams[col], teams['TOURNWINS'])[0,1]

    keys = list(dict.keys())
    values = list(dict.values())
    sorted_value_index = np.argsort(values)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

    print(sorted_dict)

    df = pd.DataFrame(sorted_dict.items(), columns=["PG statistics", "Correlation Coefficient"])
    print(df)
    print(df.to_string())

    #remove nas from dataset
    #remove_w and _l just only have statistics ending in pg





def neural_net(data):
    test = data.iloc[2105:, ]
    training = data.iloc[2104:, ]
    X = training.iloc[:, 0:-5].values
    y = training.iloc[:, -3].values
    X_test = test.iloc[:, 0:-5].values
    y_test = test.iloc[:, -3].values


    model = Sequential()
    model.add(Dense(12, input_shape=(36,), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])


    model.fit(X, y, epochs=150, batch_size=10)
    _, accuracy, mse = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))
    print('MSE: %.2f' % (mse * 100))
    predictions = model.predict(X_test)
    for i in range(129):
        print('%s => %f (expected %d)' % (X_test[i, -2:].tolist(), predictions[i], y_test[i]))

def xgboostclassifier(training):
    data = training
    # Split the data
    test = data.iloc[2105:, ]
    train = data.iloc[:2104, ]
    X_train = train.iloc[:, 0:-5].values
    y_train = train.iloc[:, -3].values
    X_test = test.iloc[:, 0:-5].values
    y_test = test.iloc[:, -3].values
    # Create classification matrices

    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    for i in range(129):
        print('%s => %f (expected %d)' % (test.iloc[i, -2:].tolist(), predictions[i], y_test[i]))
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))




#model = svm.SVC()
#model = svm.SVR()
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def svm_svc_model(training):
    data = training
    # Split the data
    test = data.iloc[2105:, ]
    train = data.iloc[:2104, ]
    X_train = train.iloc[:, 0:-5].values
    y_train = train.iloc[:, -3].values
    X_test = test.iloc[:, 0:-5].values
    y_test = test.iloc[:, -3].values

    model = svm.SVC()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    #for i in range(129):
        #print('%s => %f (expected %d)' % (test.iloc[i, -2:].tolist(), predictions[i], y_test[i]))
    accuracy = accuracy_score(y_test, predictions)
    print("SVM svc Accuracy: %.2f%%" % (accuracy * 100.0))


def svm_svr_model(training):
    data = training
    # Split the data
    test = data.iloc[2105:, ]
    train = data.iloc[:2104, ]
    X_train = train.iloc[:, 0:-5].values
    y_train = train.iloc[:, -3].values
    X_test = test.iloc[:, 0:-5].values
    y_test = test.iloc[:, -3].values

    model = svm.SVR()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    #for i in range(129):
        #print('%s => %f (expected %d)' % (test.iloc[i, -2:].tolist(), predictions[i], y_test[i]))
    accuracy = accuracy_score(y_test, predictions)
    print("SVM svr Accuracy: %.2f%%" % (accuracy * 100.0))



from sklearn.neighbors import KNeighborsClassifier

def KNeighborsClassifier1(training):
    data = training
    # Split the data
    test = data.iloc[2105:, ]
    train = data.iloc[:2104, ]
    X_train = train.iloc[:, 0:-5].values
    y_train = train.iloc[:, -3].values
    X_test = test.iloc[:, 0:-5].values
    y_test = test.iloc[:, -3].values

    if sklearn.__version__ >= '0.24':
        model = KNeighborsClassifier(n_neighbors=39, weights='uniform')
    else:
        model = KNeighborsClassifier(n_neighbors=39)

    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    #for i in range(129):
        #print('%s => %f (expected %d)' % (test.iloc[i, -2:].tolist(), predictions[i], y_test[i]))
    accuracy = accuracy_score(y_test, predictions)
    print("knn Accuracy: %.2f%%" % (accuracy * 100.0))






if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    csvs = load_data()
    reg_szn_2016 = calculate_averages(2016, csvs, 'MRegularSeasonDetailedResults.csv')
    print(reg_szn_2016)
    tourn_2016 = calculate_averages(2016, csvs, 'MNCAATourneyDetailedResults.csv')
    reg = calculate_tourney_wins_and_graph(reg_szn_2016, tourn_2016)
    make_training_data(csvs, 'MRegularSeasonDetailedResults.csv').to_csv('TrainingData.csv')
    data = pd.read_csv("/content/TrainingData.csv" ,encoding='cp1252')
    neural_net(data)
    xgboostclassifier(data)
    print(sklearn.__version__)
    svm_svc_model(data)
    svm_svr_model(data)
    KNeighborsClassifier1(data)
    #make_2023_data(csvs, 'MRegularSeasonDetailedResults.csv')
