import pandas as pd, requests, os, collections as co, json, datetime, sys, shutil, unidecode
from bs4 import BeautifulSoup
import numpy as np
from scipy.stats import uniform
import statsmodels.api as sm
import statsmodels.formula.api as smf

rosterdict = json.load(open('/home/Jon2Anderson/nba/filestore/rosterdict.json'))
namechange = json.load(open('/home/Jon2Anderson/nba/filestore/playernamechange.json'))
poschange = json.load(open('/home/Jon2Anderson/nba/filestore/poschangedict.json'))
offense = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masteroff.csv')
defense = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterdef.csv')
os.chdir('/home/Jon2Anderson/nba/')
month = datetime.date.today().strftime("%b")
day = datetime.date.today().strftime("%d")
date = month+day

def getOffensiveAverages():
    masteroffdf = pd.DataFrame(columns=['Team','Poss','FGA','3PA','TRB','AST','STL','BLK','PTS','FP'])
    teams = list(offense['Team'].unique())
    for team in teams:
        df = offense[offense['Team']==team]
        mp = sum(df['MP']) / 5
        poss = (sum(df['Poss']) / mp) * 48
        fga = (sum(df['FGA']) / mp) * 48
        threepa = (sum(df['3PA']) / mp) * 48
        trb = (sum(df['TRB']) / mp) * 48
        ast = (sum(df['AST']) / mp) * 48
        stl = (sum(df['STL']) / mp) * 48
        blk = (sum(df['BLK']) / mp) * 48
        pts = (sum(df['PTS']) / mp) * 48
        fp = (sum(df['FP']) / mp) * 48
        dfdict = {'Team':team,'Poss':poss,'FGA':fga, '3PA': threepa, 'TRB': trb, 'AST': ast, 'STL': stl, 'BLK': blk, 'PTS': pts, 'FP':fp}
        teamdf = pd.DataFrame(dfdict,index=[0])
        masteroffdf = pd.concat([masteroffdf,teamdf])

    masteroffdf = masteroffdf[['Team','Poss','FGA','3PA','TRB','AST','STL','BLK','PTS','FP']]
    return(masteroffdf)

def salDiff():
    sals = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/DK_SalsMaster.csv')
    sals = sals[['Player','Date','Sal']]

    thepast = sals[sals['Date']!=date]
    today = sals[sals['Date']==date]
    today = today[today['Sal']>3000]
    todaysals = dict(zip(today.Player, today.Sal))
    players = list(today['Player'].unique())

    saldiffdict = {}
    for player in players:
        print(player)
        pdf = thepast[thepast['Player']==player]
        avgsal = round(np.mean(pdf['Sal']),-2)
        saldiff = todaysals.get(player) - avgsal
        saldiffdict.update({player: saldiff})


    saldiffdf = pd.DataFrame.from_dict(saldiffdict, orient='index')
    saldiffdf.columns=['SalDiff']
    saldiffdf.to_csv('/home/Jon2Anderson/nba/filestore/saldiff/%s_saldiff.csv' %date)

def valByClus():
    boxes = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterboxscores.csv')
    boxes['Date'] = pd.to_datetime(boxes['Date'])
    boxes['Date'] = boxes['Date'].dt.strftime('%b%d')
    sals = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/DK_SalsMaster.csv')
    sals = sals[['Player','Date','Sal']]
    today = pd.read_csv('/home/Jon2Anderson/nba/filestore/salaries/DK/%s_DKSals.csv' %date)
    today['Team'] = today['Team'].str.upper()
    teams = list(today['Team'].unique())
    teams = [x for x in teams if str(x) != 'nan']
    clusters = pd.read_csv('/home/Jon2Anderson/nba/filestore/clusters/nba_clusters.csv')
    clusdict = dict(zip(clusters.Player, clusters.Cluster))

    df = pd.merge(boxes, sals, on=['Player','Date'])
    df = df[['Player','MP','Sal','Date','Opp','FP', 'FPPM']]
    df = df[df['MP'] > 20]
    df['Val'] = (df['FP'] / (df['Sal']/1000))
    df['Sal'] = df.Sal.round(-3)
    df.loc[df['Sal'] > 8000, 'SalGroup'] = 'High'
    df.loc[df['Sal'] < 6000, 'SalGroup'] = 'Low'
    df.loc[(df['Sal'] > 5000) & (df['Sal'] < 9000), 'SalGroup'] = 'Mid'
    df['Clus'] = df['Player'].map(clusdict)
    df = df.dropna()

    #pricegroups = ['High', 'Low', 'Mid']
    #clusters = [0, 1, 2, 3, 4, 5]
    high0dict={}
    high1dict={}
    high2dict={}
    high3dict={}
    high4dict={}
    high5dict={}
    mid0dict={}
    mid1dict={}
    mid2dict={}
    mid3dict={}
    mid4dict={}
    mid5dict={}
    low0dict={}
    low1dict={}
    low2dict={}
    low3dict={}
    low4dict={}
    low5dict={}

    for team in teams:
        xdf = df[(df['SalGroup']=='High') & (df['Clus'] == 0.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        high0dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Mid') & (df['Clus'] == 0.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        mid0dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Low') & (df['Clus'] == 0.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        low0dict.update({team:answer})
        #######################################################
        xdf = df[(df['SalGroup']=='High') & (df['Clus'] == 1.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        high1dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Mid') & (df['Clus'] == 1.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        mid1dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Low') & (df['Clus'] == 1.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        low1dict.update({team:answer})
        #######################################################
        xdf = df[(df['SalGroup']=='High') & (df['Clus'] == 2.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        high2dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Mid') & (df['Clus'] == 2.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        mid2dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Low') & (df['Clus'] == 2.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        low2dict.update({team:answer})
        #######################################################
        xdf = df[(df['SalGroup']=='High') & (df['Clus'] == 3.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        high3dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Mid') & (df['Clus'] == 3.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        mid3dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Low') & (df['Clus'] == 3.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        low3dict.update({team:answer})
        #######################################################
        xdf = df[(df['SalGroup']=='High') & (df['Clus'] == 4.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        high4dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Mid') & (df['Clus'] == 4.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        mid4dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Low') & (df['Clus'] == 4.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        low4dict.update({team:answer})
        #######################################################
        xdf = df[(df['SalGroup']=='High') & (df['Clus'] == 5.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        high5dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Mid') & (df['Clus'] == 5.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        mid5dict.update({team:answer})

        xdf = df[(df['SalGroup']=='Low') & (df['Clus'] == 5.0)]
        xdf = xdf[xdf['Opp']==team]
        answer = np.mean(xdf['Val'])
        if np.isnan(answer):
            answer=0
        low5dict.update({team:answer})
        #######################################################
        high0df = pd.DataFrame.from_dict(high0dict, orient='index')
        high0df['SalGroup'] = 'High'
        high0df['Clus'] = 0.0

        mid0df = pd.DataFrame.from_dict(mid0dict, orient='index')
        mid0df['SalGroup'] = 'Mid'
        mid0df['Clus'] = 0.0

        low0df = pd.DataFrame.from_dict(low0dict, orient='index')
        low0df['SalGroup'] = 'Low'
        low0df['Clus'] = 1.0
        ##########################################################
        high1df = pd.DataFrame.from_dict(high1dict, orient='index')
        high1df['SalGroup'] = 'High'
        high1df['Clus'] = 1.0

        mid1df = pd.DataFrame.from_dict(mid1dict, orient='index')
        mid1df['SalGroup'] = 'Mid'
        mid1df['Clus'] = 1.0

        low1df = pd.DataFrame.from_dict(low1dict, orient='index')
        low1df['SalGroup'] = 'Low'
        low1df['Clus'] = 1.0
        ##########################################################
        high2df = pd.DataFrame.from_dict(high2dict, orient='index')
        high2df['SalGroup'] = 'High'
        high2df['Clus'] = 2.0

        mid2df = pd.DataFrame.from_dict(mid2dict, orient='index')
        mid2df['SalGroup'] = 'Mid'
        mid2df['Clus'] = 2.0

        low2df = pd.DataFrame.from_dict(low2dict, orient='index')
        low2df['SalGroup'] = 'Low'
        low2df['Clus'] = 2.0
        ##########################################################
        high3df = pd.DataFrame.from_dict(high3dict, orient='index')
        high3df['SalGroup'] = 'High'
        high3df['Clus'] = 3.0

        mid3df = pd.DataFrame.from_dict(mid3dict, orient='index')
        mid3df['SalGroup'] = 'Mid'
        mid3df['Clus'] = 3.0

        low3df = pd.DataFrame.from_dict(low3dict, orient='index')
        low3df['SalGroup'] = 'Low'
        low3df['Clus'] = 3.0
        ##########################################################
        high4df = pd.DataFrame.from_dict(high4dict, orient='index')
        high4df['SalGroup'] = 'High'
        high4df['Clus'] = 4.0

        mid4df = pd.DataFrame.from_dict(mid4dict, orient='index')
        mid4df['SalGroup'] = 'Mid'
        mid4df['Clus'] = 4.0

        low4df = pd.DataFrame.from_dict(low4dict, orient='index')
        low4df['SalGroup'] = 'Low'
        low4df['Clus'] = 4.0
        ##########################################################
        high5df = pd.DataFrame.from_dict(high5dict, orient='index')
        high5df['SalGroup'] = 'High'
        high5df['Clus'] = 5.0

        mid5df = pd.DataFrame.from_dict(mid5dict, orient='index')
        mid5df['SalGroup'] = 'Mid'
        mid5df['Clus'] = 5.0

        low5df = pd.DataFrame.from_dict(low5dict, orient='index')
        low5df['SalGroup'] = 'Low'
        low5df['Clus'] = 5.0
        ##########################################################


        finaldf = pd.concat([high0df, high1df, high2df, high3df, high4df, high5df,
                            mid0df, mid1df, mid2df, mid3df, mid4df, mid5df,
                            low0df, low1df, low2df, low2df, low4df, low5df])
        #finaldf.columns = ['AvgVal','Pos']
        finaldf.to_csv('/home/Jon2Anderson/nba/filestore/clusters/AvgValByClus.csv')

def valByPos():
    boxes = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterboxscores.csv')
    boxes['Date'] = pd.to_datetime(boxes['Date'])
    boxes['Date'] = boxes['Date'].dt.strftime('%b%d')
    sals = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/DK_SalsMaster.csv')
    sals = sals[['Player','Date','Sal']]
    today = pd.read_csv('/home/Jon2Anderson/nba/filestore/salaries/DK/%s_DKSals.csv' %date)
    today['Team'] = today['Team'].str.upper()
    teams = list(today['Team'].unique())
    teams = [x for x in teams if str(x) != 'nan']

    df = pd.merge(boxes, sals, on=['Player','Date'])
    df = df[['Player','MP','Pos','Sal','Date','Opp','FP', 'FPPM']]
    df = df[df['MP'] > 20]
    df['Sal'] = df.Sal.round(-3)

    df['Val'] = (df['FP'] / (df['Sal']/1000))

    positions = ['Big','G','Wing']
    masterdf = pd.DataFrame()
    bigdict={}
    wingdict={}
    gdict={}

    for team in teams:
        xdf = df[df['Pos']=='Big']
        xdf = xdf[xdf['Opp']==team]
        bigdict.update({team:np.mean(xdf['Val'])})

    for team in teams:
        xdf = df[df['Pos']=='Wing']
        xdf = xdf[xdf['Opp']==team]
        wingdict.update({team:np.mean(xdf['Val'])})

    for team in teams:
        xdf = df[df['Pos']=='G']
        xdf = xdf[xdf['Opp']==team]
        gdict.update({team:np.mean(xdf['Val'])})

    gdf = pd.DataFrame.from_dict(gdict, orient='index')
    gdf['Pos'] = 'g'
    wingdf = pd.DataFrame.from_dict(wingdict, orient='index')
    wingdf['Pos'] = 'wing'
    bigdf = pd.DataFrame.from_dict(bigdict, orient='index')
    bigdf['Pos'] = 'big'

    finaldf = pd.concat([gdf,wingdf,bigdf])
    finaldf.columns = ['AvgVal','Pos']
    finaldf.to_csv('/home/Jon2Anderson/nba/filestore/defRtg/AvgValByPos.csv')

def defStatAllowances():
    masteroff = getOffensiveAverages()
    teams = list(offense['Team'].unique())
    masterdefdf = pd.DataFrame()
    for team in teams:
        teamdf = defense[defense['Team']==team]
        games = teamdf.shape[0]

        possdiff=0
        fgadiff=0
        threepadiff=0
        trbdiff=0
        astdiff=0
        stldiff=0
        blkdiff=0
        ptsdiff=0
        fpdiff=0

        for i in range(games):
            oppdf = teamdf.iloc[i,:]
            oppname = oppdf.get('Opp')
            oppmp = oppdf.get('MP') / 5
            oppposs = (oppdf.get('Poss') / oppmp) * 48
            oppfga = (oppdf.get('FGA') / oppmp) * 48
            opp3pa = (oppdf.get('3PA') / oppmp) * 48
            opptrb = (oppdf.get('TRB') / oppmp) * 48
            oppast = (oppdf.get('AST') / oppmp) * 48
            oppstl = (oppdf.get('STL') / oppmp) * 48
            oppblk = (oppdf.get('BLK') / oppmp) * 48
            opppts = (oppdf.get('PTS') / oppmp) * 48
            oppfp = (oppdf.get('FP') / oppmp) * 48

            checkdf = masteroff[masteroff['Team']==oppname]
            possdiff = (possdiff + (checkdf.get('Poss') - oppposs)).values[0]
            fgadiff = (fgadiff + (checkdf.get('FGA') - oppfga)).values[0]
            threepadiff = (threepadiff + (checkdf.get('3PA') - opp3pa)).values[0]
            trbdiff = (trbdiff + (checkdf.get('TRB') - opptrb)).values[0]
            astdiff = (astdiff + (checkdf.get('AST') - oppast)).values[0]
            stldiff = (stldiff + (checkdf.get('STL') - oppstl)).values[0]
            blkdiff = (blkdiff + (checkdf.get('BLK') - oppblk)).values[0]
            ptsdiff = (ptsdiff + (checkdf.get('PTS') - opppts)).values[0]
            fpdiff = (fpdiff + (checkdf.get('FP') - oppfp)).values[0]
            i = i + 1

        possdiff = -possdiff/games
        fgadiff = -fgadiff/games
        threepadiff = -threepadiff/games
        trbdiff = -trbdiff/games
        astdiff = -astdiff/games
        stldiff = -stldiff/games
        blkdiff = -blkdiff/games
        ptsdiff = -ptsdiff/games
        fpdiff = -fpdiff/games
        tmdefdict = {'Team': team, 'Poss': possdiff, 'FGA': fgadiff, '3PA': threepadiff, 'TRB': trbdiff, 'AST': astdiff, 'STL': stldiff, 'BLK': blkdiff, 'PTS': ptsdiff, 'FP': fpdiff}
        masterdefdf=masterdefdf.append(tmdefdict, ignore_index=True)

    masterdefdf = masterdefdf[['Team','Poss','FGA','3PA','TRB','AST','STL','BLK','PTS','FP']]
    masterdefdf.to_csv('/home/Jon2Anderson/nba/filestore/defRtg/defStatAllowances.csv')
    print('...finished')

def getRosters():
    teams=['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU',
    'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL',
    'PHI', 'PHO', 'POR', 'SAC' , 'SAS', 'TOR', 'UTA', 'WAS']

    masterdf = pd.DataFrame(columns=['Player','Pos','Team'])
    for team in teams:
        url='https://www.basketball-reference.com/teams/%s/2020.html' %team
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        table = soup.find('table', {'class': 'sortable stats_table', 'id': 'roster'})
        table_body = table.find('tbody')

        playerlist = []
        for td in table_body.findAll('td', {'data-stat': 'player'}):
            player=unidecode.unidecode(td.text)
            player=player.replace("(TW)","")
            player=player.rstrip()
            player=player.lower()
            playerlist.append(player)

        poslist = []
        for td in table_body.findAll('td', {'data-stat': 'pos'}):
            pos = td.text
            pos = pos.strip()
            poslist.append(pos)

        df = pd.DataFrame(columns=['Player','Pos','Team'])
        df['Player'] = playerlist
        df['Pos'] = poslist
        df['Pos'] = df['Pos'].replace(poschange)

        df['Team'] = team.lower()
        #df.columns=['Player','Pos','Team']
        masterdf = pd.concat([masterdf,df])

    rosterdict = dict(zip(masterdf.Player,masterdf.Team))
    posdict = dict(zip(masterdf.Player,masterdf.Pos))

    with open('/home/Jon2Anderson/nba/filestore/rosterdict.json', 'w') as fp:
        json.dump(rosterdict,fp)

    with open('/home/Jon2Anderson/nba/filestore/posdict.json', 'w') as fp:
        json.dump(posdict,fp)

def projFP():
    offense = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masteroff.csv')
    offense['Team'] = offense['Team'].str.lower()
    offense['Opp'] = offense['Opp'].str.lower()
    defense = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterdef.csv')
    defense['Team'] = defense['Team'].str.lower()
    defense['Opp'] = defense['Opp'].str.lower()
    lgaveragefp = (sum(offense['FP']) / sum(offense['MP'])) * 240

    lines = pd.read_csv('/home/Jon2Anderson/nba/filestore/gamelines/dailylines/%s_lines.csv' %date)
    matchups = dict(zip(lines.team,lines.opp))
    matchups2 = dict(zip(lines.opp,lines.team))
    matchups.update(matchups2)

    teams = list(matchups.keys())

    projdict={}
    for team in teams:
        opp = matchups.get(team)
        teamoffense = offense[offense['Team']==team]
        teamdefense = defense[defense['Team']==opp]
        teamavgfp = (sum(teamoffense['FP']) / sum(teamoffense['MP'])) * 240
        oppavgfp = (sum(teamdefense['FP']) / sum(teamdefense['MP'])) * 240

        oppdiff = (oppavgfp - lgaveragefp) / lgaveragefp
        teamproj = teamavgfp + (teamavgfp * oppdiff)
        projdict.update({team: teamproj})

    projdf = pd.DataFrame()
    projdf = projdf.append(projdict, ignore_index=True)
    projdf = projdf.T
    projdf.columns=['ProjFP']

    projdf.to_csv('/home/Jon2Anderson/nba/filestore/projections/teamFPproj/%s_FPProj.csv' %date)

def projFP2():
    offense = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masteroff.csv')
    offense['Date'] = pd.to_datetime(offense['Date'])
    offense['Date'] = offense['Date'].dt.strftime('%b%d')
    offense['Team'] = offense['Team'].str.lower()
    offense['Opp'] = offense['Opp'].str.lower()

    linemaster = pd.read_csv('/home/Jon2Anderson/nba/filestore/gamelines/linemaster.csv')
    linemaster = linemaster[['team','date','proj']]
    linemaster.columns=['Team','Date','Proj']
    linedict = dict(zip(linemaster.Team, linemaster.Proj))

    todaylines = pd.read_csv('/home/Jon2Anderson/nba/filestore/gamelines/dailylines/%s_lines.csv' %date)
    matchups = dict(zip(todaylines.team,todaylines.opp))
    matchups2 = dict(zip(todaylines.opp,todaylines.team))
    matchups.update(matchups2)

    defrtg = pd.read_csv('/home/Jon2Anderson/nba/filestore/defRtg/defStatAllowances.csv')
    defrtg = defrtg[['Team','FP']]
    defrtg.columns=['Opp','FP']
    defrtg['Opp'] = defrtg['Opp'].str.lower()
    defdict = dict(zip(defrtg.Opp,defrtg.FP))

    offense = offense[['Team','Date','Opp','FP']]
    offense['OppDef'] = offense['Opp'].map(defdict)

    modeldf = pd.merge(offense, linemaster, on=['Team','Date'])
    modeldf = modeldf[['Date','Team','Opp', 'OppDef','Proj','FP']]
    np.random.seed(1234)
    modeldf = modeldf.sample(frac=1)
    mymodel = str('FP ~ OppDef + Proj')
    trainmodel_fit = smf.ols(mymodel, data=modeldf).fit()

    todaydf = todaylines[['team','opp','proj']]
    todaydf.columns=['Team','Opp','Proj']
    todaydf['OppDef'] = todaydf['Opp'].map(defdict)

    todaydf['PredFP'] = trainmodel_fit.predict(todaydf)
    todaydf = todaydf[['Team','PredFP']]
    todaydf.columns = ['Team', 'ProjFP']

    todaydf.to_csv('/home/Jon2Anderson/nba/filestore/projections/teamFPproj/%s_LM_TeamProjFP.csv' %date)

def dubdub(df):
    if df['PTS'] > 9 and df['REB'] > 9:
        return(1.5)
    elif df['PTS'] > 9 and df['AST'] > 9:
        return(1.5)
    elif df['AST'] > 9 and df['REB'] > 9:
        return(1.5)
    else:
        return(0)

def tripdub(df):
    if df['PTS'] > 9 and df['REB'] > 9 and df['AST'] > 9:
        return(3)
    else:
        return(0)

def getDKSals():
    url='https://www.fantasypros.com/daily-fantasy/nba/draftkings-salary-changes.php'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    table = soup.find('table', {'class': 'table table-full table-bordered', 'id': 'data-table'})
    tbody = table.find('tbody')
    playerlist=[]

    for a in tbody.findAll('a'):
        player=a.text.lower()
        playerlist.append(player)
    playerlist = playerlist[::2]

    sallist=[]
    for td in tbody.findAll('td', {'class': 'salary center'}):
        sal = td.text
        sal = sal.replace('$','')
        sal = sal.replace(',','')
        sallist.append(sal)

    saldf = pd.DataFrame(columns=['Player','Team','Sal'])

    saldf['Player']=playerlist
    saldf['Player'] = saldf['Player'].replace(namechange)
    saldf['Sal']=sallist
    saldf['Team'] = saldf['Player'].map(rosterdict)
    saldf = saldf[['Player','Team','Sal']]
    saldf['Sal'] = pd.to_numeric(saldf['Sal'])
    saldf = saldf.sort_values(by='Sal', ascending=False)
    saldf.to_csv('/home/Jon2Anderson/nba/filestore/salaries/DK/%s_DKSals.csv' %date)

def getFDSals():
    url='https://www.fantasypros.com/daily-fantasy/nba/fanduel-salary-changes.php'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    table = soup.find('table', {'class': 'table table-full table-bordered', 'id': 'data-table'})
    tbody = table.find('tbody')
    playerlist=[]

    for a in tbody.findAll('a'):
        player=a.text.lower()
        playerlist.append(player)
    playerlist = playerlist[::2]

    sallist=[]
    for td in tbody.findAll('td', {'class': 'salary center'}):
        sal = td.text
        sal = sal.replace('$','')
        sal = sal.replace(',','')
        sallist.append(sal)

    saldf = pd.DataFrame(columns=['Player','Team','Sal'])

    saldf['Player']=playerlist
    saldf['Player'] = saldf['Player'].replace(namechange)
    saldf['Sal']=sallist
    saldf['Team'] = saldf['Player'].map(rosterdict)
    saldf = saldf[['Player','Team','Sal']]
    saldf['Sal'] = pd.to_numeric(saldf['Sal'])
    saldf = saldf.sort_values(by='Sal', ascending=False)
    saldf.to_csv('/home/Jon2Anderson/nba/filestore/salaries/FD/%s_FDSals.csv' %date)

def scrapeFP():
    url = 'https://secure.fantasypros.com/accounts/login/?next=https://www.fantasypros.com/nba/projections/daily-overall.php?'
    client = requests.session()

    client.get(url)  # sets cookie
    if 'csrftoken' in client.cookies:
        csrftoken = client.cookies['csrftoken']
    else:
        csrftoken = client.cookies['csrf']

    login_data = dict(username='jon2anderson@comcast.net', password='rockChalk22!fa', csrfmiddlewaretoken=csrftoken, next='/')
    r = client.post(url, data=login_data, headers=dict(Referer=url))

    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find('table', {'class': 'table table-bordered table-striped table-hover player-table'})

    table_body = table.find('tbody')
    table_head = table.find('thead')

    header=[]
    for th in table_head.findAll('th'):
        key = th.get_text()
        header.append(key)

    trlist=[]
    for tr in table_body.findAll('tr'):
        trlist.append(tr)

    listofdicts=[]

    for row in trlist:
        the_row=[]
        for td in row.findAll('td'):
            the_row.append(td.text)
        od = co.OrderedDict(zip(header, the_row))
        listofdicts.append(od)

    projdf = pd.DataFrame(listofdicts)
    projdf['Player'] = projdf['Player'].replace(namechange)
    projdf['Player'] = projdf['Player'].str.split("(").str[0]
    projdf['Player'] = projdf['Player'].str.rstrip()
    projdf['Player'] = projdf['Player'].str.lower()

    projdf['Team'] = projdf['Player'].map(rosterdict)
    projdf['PTS'] = pd.to_numeric(projdf['PTS'])
    projdf['REB'] = pd.to_numeric(projdf['REB'])
    projdf['AST'] = pd.to_numeric(projdf['AST'])
    projdf['BLK'] = pd.to_numeric(projdf['BLK'])
    projdf['STL'] = pd.to_numeric(projdf['STL'])
    projdf['TO'] = pd.to_numeric(projdf['TO'])
    projdf['MIN'] = pd.to_numeric(projdf['MIN'])
    projdf['3PM'] = pd.to_numeric(projdf['3PM'])

    projdf['DubDub'] = projdf.apply(dubdub, axis=1)
    projdf['TripDub'] = projdf.apply(tripdub, axis=1)
    projdf['FP'] = (projdf['PTS']) + (projdf['3PM'] * .5) + (projdf['REB'] * 1.25) + (projdf['AST'] * 1.5) + (projdf['STL'] * 2) + (projdf['BLK'] * 2) - (projdf['TO'] * .5) + projdf['DubDub'] + projdf['TripDub']
    #projdf['FP'] = round(projdf['FP'],2)
    projdf['Opponent'] = projdf['Opponent'].str.replace('at ','')
    projdf['Opponent'] = projdf['Opponent'].str.replace('vs ','')
    projdf['Opponent'] = projdf['Opponent'].str.lower()

    projdf = projdf[['Player','Team','Opponent','PTS','REB','AST','BLK','STL','FG%','FT%','3PM','GP','MIN','TO','DubDub','TripDub','FP']]

    projdf.to_csv('/home/Jon2Anderson/nba/filestore/projections/FP/%s_FP_Proj.csv' %date)
    return(projdf)

def getDFSCafeProjScores():
    url = 'https://www.dailyfantasycafe.com/tools/minutes/nba'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    divstr = str(soup.find('div', {'id': 'minutes-tool'}))
    divstr = divstr.replace("&quot;","\"")

    begin = divstr.find("data-players=\"")
    end = divstr.find("data-sites")
    cut = divstr[begin+14:end-2]

    data = json.loads(cut)
    playerdata = data.get('data')
    playerlist=[]
    scorelist=[]

    for player in playerdata:
        score = player.get('projections').get('draftkings_cash')
        playerlist.append(player.get('full_name'))
        scorelist.append(score)

    projscoredf = pd.DataFrame(columns=['Player','Team','ProjScore'])
    projscoredf['Player'] = playerlist
    projscoredf['Player'] = projscoredf['Player'].str.lower()
    projscoredf['Team'] = projscoredf['Player'].map(rosterdict)
    projscoredf['ProjScore'] = scorelist
    projscoredf['ProjScore'] = pd.to_numeric(projscoredf['ProjScore'])

    projscoredf = projscoredf.sort_values(by='ProjScore',ascending=False)
    projscoredf.to_csv('/home/Jon2Anderson/nba/filestore/projections/DFSCafe/%s_DFSCafeProj.csv' %date)

def scrapeRG():
    dkdata = requests.get('https://rotogrinders.com/projected-stats/nba-player.csv?site=draftkings')
    dkdata.raise_for_status()
    dkfile = open('/home/Jon2Anderson/nba/filestore/dkdata/%s_DKData.csv' %date, 'wb')

    for chunk in dkdata.iter_content(100000):
        dkfile.write(chunk.lower())

    dkfile.close()

    dkdata = pd.read_csv('/home/Jon2Anderson/nba/filestore/dkdata/%s_DKData.csv' %date, names=['Player', 'Sal', 'Team', 'Pos', 'Opp', 'Ceil', 'Floor', 'FP'])
    dkdata['Player'] = dkdata['Player'].replace(namechange)
    dkdata = dkdata[['Player', 'Sal', 'Team', 'Pos', 'Opp', 'Ceil', 'Floor', 'FP']]
    sals = dkdata[['Player', 'Team', 'Sal']]
    rgproj = dkdata[['Player', 'Team', 'FP']]
    rgproj = rgproj.sort_values(by='FP', ascending=False)
    sals.to_csv('/home/Jon2Anderson/nba/filestore/salaries/%s_Sals.csv' %date)
    rgproj.to_csv('/home/Jon2Anderson/nba/filestore/projections/%s_RG_Proj.csv' %date)

def getLines():
    url='https://rotogrinders.com/schedules/nba/dfs'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    scripts = soup.findAll('script')
    raw = str(scripts[-1])
    raw = raw.replace("\n",'')
    begin = raw.find("[")
    end = raw.rfind("}];")
    cut = raw[begin:end+2]
    data = json.loads(cut)

    mykeys=['team','opponent','line','moneyline','overunder','projected']
    masterdf = pd.DataFrame(columns=['team','opponent','line','moneyline','overunder','projected'])

    for i in range(len(data)):
        team = data[i]
        newdict = {}

        for key,value in team.items():
            if key in mykeys:
                newdict[key]=value

        df = pd.DataFrame(newdict, index=[0])
        masterdf = pd.concat([df, masterdf])

    masterdf['opponent'] = masterdf['opponent'].str.replace('@ ','')
    masterdf['opponent'] = masterdf['opponent'].str.replace('vs. ','')
    masterdf['opponent'] = masterdf['opponent'].str.lower()
    masterdf['team'] = masterdf['team'].str.lower()
    masterdf['team'] = masterdf['team'].str.replace('bkn','brk')
    masterdf['team'] = masterdf['team'].str.replace('cha','cho')
    masterdf['opponent'] = masterdf['opponent'].str.replace('cha','cho')
    masterdf['opponent'] = masterdf['opponent'].str.replace('bkn','brk')
    masterdf['line'] = pd.to_numeric(masterdf['line'])
    masterdf['projected'] = pd.to_numeric(masterdf['projected'])

    masterdf = masterdf[['team','opponent','line','moneyline','overunder','projected']]
    masterdf.columns=['team','opp','line','moneyline','ou','proj']
    masterdf.to_csv('/home/Jon2Anderson/nba/filestore/gamelines/dailylines/%s_lines.csv' %date)

    '''
    # Combine Lines
    linemaster = pd.DataFrame(columns=['team','opp','date','line','moneyline','ou','proj'])
    mydir = '/home/Jon2Anderson/nba/filestore/gamelines'
    os.chdir(mydir)
    files = os.listdir(mydir)
    for f in files:
        linedf = pd.read_csv(f)
        thedate = f.split('_')[0]
        linedf['date']=thedate
        linedf = linedf[['team','opp','date','line','moneyline','ou','proj']]
        linemaster = pd.concat([linemaster,linedf])

    #linemaster.to_csv('/home/Jon2Anderson/nba/filestore/gamelines/linemaster/linemaster.csv')
    '''
def combineLines():
    linemaster = pd.DataFrame(columns=['team','opp','date','line','moneyline','ou','proj'])
    mydir = '/home/Jon2Anderson/nba/filestore/gamelines/dailylines'
    os.chdir(mydir)
    files = os.listdir(mydir)
    for f in files:
        linedf = pd.read_csv(f)
        thedate = f.split('_')[0]
        linedf['date']=thedate
        linedf = linedf[['team','opp','date','line','moneyline','ou','proj']]
        linemaster = pd.concat([linemaster,linedf])

    linemaster.to_csv('/home/Jon2Anderson/nba/filestore/gamelines/linemaster.csv')

def dfsCafeMinutes():
    url = 'https://www.dailyfantasycafe.com/tools/minutes/nba'
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'lxml')
    divstr = str(soup.find('div', {'id': 'minutes-tool'}))
    divstr = divstr.replace("&quot;","\"")

    begin = divstr.find("data-players=\"")
    end = divstr.find("data-sites")
    cut = divstr[begin+14:end-2]

    data = json.loads(cut)
    playerdata = data.get('data')

    playerslist=[]
    minuteslist=[]

    for player in playerdata:
        playerslist.append(player.get('full_name'))
        minuteslist.append(player.get('minutes'))

    pmindf = pd.DataFrame(columns=['Player','PMin'])
    pmindf['Player'] = playerslist
    pmindf['PMin'] = minuteslist
    pmindf['Player'] = pmindf['Player'].str.lower()
    pmindf = pmindf[['Player','PMin']]
    pmindf['PMin'] = pd.to_numeric(pmindf['PMin'])
    pmindf = pmindf.sort_values(by='PMin', ascending=False)
    pmindf['Player'] = pmindf['Player'].replace(namechange)
    pmindf['Team'] = pmindf['Player'].map(rosterdict)

    pmindf.to_csv('/home/Jon2Anderson/nba/filestore/proj_minutes/dfscafe_minutes/%s_ProjMins.csv' %date)

def getNumFireProjMins():
    url = 'https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections#_=_'
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find('table', {'class': 'stat-table fixed-head'})
    table_body = table.find('tbody')
    trlist=[]
    for tr in table_body.findAll('tr'):
        trlist.append(tr)

    players = []
    for row in trlist:
        for a in row.findAll('a', {'class': 'full'}):
            playername = a.text.rstrip()
            playername = playername.lstrip()
            players.append(playername)

    pmins = []
    for row in trlist:
        for td in row.findAll('td', {'class': 'min'}):
            pmin = td.text.rstrip()
            pmin = pmin.lstrip()
            pmins.append(pmin)

    pmindict = dict(zip(players, pmins))
    pmindf = pd.DataFrame(list(pmindict.items()), columns=['Player', 'MIN'])
    pmindf['Player'] = pmindf['Player'].str.lower()
    pmindf['Player'] = pmindf['Player'].replace(namechange)
    pmindf['Team'] = pmindf['Player'].map(rosterdict)

    pmindf.to_csv('/home/Jon2Anderson/nba/filestore/proj_minutes/numfire_minutes/%s_ProjMins.csv' %date)

def getDefRtg():
    boxes = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterboxscores.csv')
    teams = list(boxes['Opp'].unique())

    ####### league averages ########
    guards = boxes[boxes['Pos']=='G']
    wings = boxes[boxes['Pos']=='Wing']
    bigs = boxes[boxes['Pos']=='Big']

    avg_guards = sum(guards['FP']) / sum(guards['MP'])
    avg_wings = sum(wings['FP']) / sum(wings['MP'])
    avg_bigs = sum(bigs['FP']) / sum(bigs['MP'])
    avg_all = sum(boxes['FP']) / sum(boxes['MP'])
    ###############################

    defrtgdf = pd.DataFrame(columns=['Team', 'vsAll', 'vsG', 'vsWing', 'vsBig'])

    for team in teams:

        # Guards
        g_df = boxes[(boxes['Opp']==team) & (boxes['Pos'] == 'G')]
        g_mp = sum(g_df['MP'])
        g_fp = sum(g_df['FP'])
        g_fppm = g_fp/g_mp
        g_diff = g_fppm - avg_guards
        g_pct = g_diff / avg_guards

        # Wings
        wing_df = boxes[(boxes['Opp']==team) & (boxes['Pos'] == 'Wing')]
        w_mp = sum(wing_df['MP'])
        w_fp = sum(wing_df['FP'])
        w_fppm = w_fp/w_mp
        w_diff = w_fppm - avg_wings
        w_pct = w_diff / avg_wings

        # Guards
        b_df = boxes[(boxes['Opp']==team) & (boxes['Pos'] == 'Big')]
        b_mp = sum(b_df['MP'])
        b_fp = sum(b_df['FP'])
        b_fppm = b_fp/b_mp
        b_diff = b_fppm - avg_bigs
        b_pct = b_diff / avg_bigs

        # All
        all_df = boxes[boxes['Opp']==team]
        all_mp = sum(all_df['MP'])
        all_fp = sum(all_df['FP'])
        all_fppm = all_fp/all_mp
        all_diff = all_fppm - avg_all
        all_pct = all_diff / avg_all

        teamdict = {'Team': team, 'vsAll': all_pct, 'vsG': g_pct, 'vsWing': w_pct, 'vsBig': b_pct}
        teamdf = pd.DataFrame(teamdict, index=[0])
        defrtgdf = pd.concat([defrtgdf,teamdf])
    defrtgdf['Team'] = defrtgdf['Team'].str.lower()
    defrtgdf.to_csv('/home/Jon2Anderson/nba/filestore/defRtg/defRtgs.csv')

def combineSals():
    os.chdir('/home/Jon2Anderson/nba/filestore/salaries/FD')
    files = os.listdir('/home/Jon2Anderson/nba/filestore/salaries/FD')

    fdmaster = pd.DataFrame(columns=['Player','Team','Date','Sal'])
    for file in files:
        df = pd.read_csv(file)
        df['Date'] = file.split('_')[0]
        df = df[['Player','Team','Date','Sal']]
        fdmaster = pd.concat([fdmaster,df])
    fdmaster.to_csv('/home/Jon2Anderson/nba/filestore/todownload/FD_SalsMaster.csv')

    os.chdir('/home/Jon2Anderson/nba/filestore/salaries/DK')
    files = os.listdir('/home/Jon2Anderson/nba/filestore/salaries/DK')

    dkmaster = pd.DataFrame(columns=['Player','Team','Date','Sal'])
    for file in files:
        df = pd.read_csv(file)
        df['Date'] = file.split('_')[0]
        df = df[['Player','Team','Date','Sal']]
        dkmaster = pd.concat([dkmaster,df])
    dkmaster.to_csv('/home/Jon2Anderson/nba/filestore/todownload/DK_SalsMaster.csv')

def dfsCafe():
    url = 'https://www.dailyfantasycafe.com/tools/minutes/nba'
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'lxml')
    divstr = str(soup.find('div', {'id': 'minutes-tool'}))
    divstr = divstr.replace("&quot;","\"")

    begin = divstr.find("data-players=\"")
    end = divstr.find("data-sites")
    cut = divstr[begin+14:end-2]

    data = json.loads(cut)
    playerdata = data.get('data')

    playerslist=[]
    minuteslist=[]

    for player in playerdata:
        playerslist.append(player.get('full_name'))
        minuteslist.append(player.get('minutes'))

    pmindf = pd.DataFrame(columns=['Player','DC_Mins'])
    pmindf['Player'] = playerslist
    pmindf['DC_Mins'] = minuteslist
    pmindf['Player'] = pmindf['Player'].str.lower()
    pmindf = pmindf[['Player','DC_Mins']]
    pmindf['DC_Mins'] = pd.to_numeric(pmindf['DC_Mins'])
    pmindf = pmindf.sort_values(by='DC_Mins', ascending=False)
    pmindf['Player'] = pmindf['Player'].replace(namechange)
    pmindf['Team'] = pmindf['Player'].map(rosterdict)
    return(pmindf)

def numFire():
    url = 'https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections#_=_'
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'lxml')
    table = soup.find('table', {'class': 'stat-table fixed-head'})
    table_body = table.find('tbody')
    trlist=[]
    for tr in table_body.findAll('tr'):
        trlist.append(tr)

    players = []
    for row in trlist:
        for a in row.findAll('a', {'class': 'full'}):
            playername = a.text.rstrip()
            playername = playername.lstrip()
            players.append(playername)

    pmins = []
    for row in trlist:
        for td in row.findAll('td', {'class': 'min'}):
            pmin = td.text.rstrip()
            pmin = pmin.lstrip()
            pmins.append(pmin)

    pmindict = dict(zip(players, pmins))
    pmindf = pd.DataFrame(list(pmindict.items()), columns=['Player', 'NF_Mins'])
    pmindf['Player'] = pmindf['Player'].str.lower()
    pmindf['Player'] = pmindf['Player'].replace(namechange)
    pmindf['Team'] = pmindf['Player'].map(rosterdict)
    return(pmindf)

def sportsline():
    url = 'https://www.sportsline.com/sportsline-web/service/v1/fantasy/projections/nba/simulation'
    data = requests.get(url)
    data_json = data.json()['projections']
    df = pd.DataFrame(data_json)
    df = df[['player', 'min']]
    df['player'] = df['player'].str.lower()
    df.columns=['Player','SL_Mins']
    df['Player'] = df['Player'].replace(namechange)
    return(df)

def getMins():
    print('DFS Cafe...')
    try:
        dfsCafe_mins = dfsCafe()
    except:
        dfsCafe_mins=0
        print('Problem with DFS Cafe Projections')

    print('Number Fire...')
    try:
        numFire_mins = numFire()
    except:
        print('Problem with Number Fire Projections')

    print('Sportsline...')
    try:
        sportsline_mins = sportsline()
    except:
        print('Problem with Sportsline Projections')


    if len(dfsCafe_mins)==0:
        dfsCafe_mins = numFire_mins.copy()
        dfsCafe_mins.columns=['Player','DC_Mins','Team']
        print('Problem with DFS Cafe')

    if len(numFire_mins)==0:
        numFire_mins = dfsCafe_mins.copy()
        numFire_mins.columns=['Player','NF_Mins','Team']
        print('Problem with Number Fire')

    if len(sportsline_mins)==0:
        sportsline_mins = dfsCafe_mins.copy()
        sportsline_mins.columns=['Player','NF_Mins','Team']
        print('Problem with Sportsline')


    #print('Merging...')
    minsdf = pd.merge(dfsCafe_mins,numFire_mins, on='Player', how='outer')
    minsdf = pd.merge(minsdf, sportsline_mins, on='Player', how='outer')
    minsdf = minsdf.fillna(0)
    minsdf = minsdf[['Player','DC_Mins','NF_Mins', 'SL_Mins']]
    minsdf.columns=['Player','DC','NF','SL']
    minsdf['DC'] = pd.to_numeric(minsdf['DC'])
    minsdf['NF'] = pd.to_numeric(minsdf['NF'])
    minsdf['SL'] = pd.to_numeric(minsdf['SL'])
    minsdf['Avg'] = minsdf.mean(axis=1, skipna=True)
    minsdf['Low'] = minsdf.min(axis=1, skipna=True)
    minsdf.to_csv('/home/Jon2Anderson/nba/filestore/projections/minutes/%s_Mins.csv' %date)

def playerSim():
    sals = pd.read_csv('/home/Jon2Anderson/nba/filestore/salaries/DK/%s_DKSals.csv' %date)
    mins = pd.read_csv('/home/Jon2Anderson/nba/filestore/projections/minutes/%s_Mins.csv' %date)
    boxes = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterboxscores.csv')

    sals = sals.dropna()
    saldict = dict(zip(sals.Player,sals.Sal))
    minsdict = dict(zip(mins.Player,mins.Avg))
    players = list(sals['Player'].unique())

    playerprojdict = {}
    trials=1000
    for player in players:
        print(player)
        sal = saldict.get(player)
        minsproj = minsdict.get(player)
        df = boxes[boxes['Player']==player]
        df = df[df['MP'] >= 20]

        if (len(df) < 5) or (minsproj <= 20) or (minsproj == None):
            print('***Skipping ' + player)
        else:

            pts = list(df['PTS'])
            ast = list(df['AST'])
            reb = list(df['TRB'])
            stl = list(df['STL'])
            blk = list(df['BLK'])
            tov = list(df['TOV'])
            thr = list(df['3P'])

            scores=[]
            for i in range(trials):
                ptsproj = np.random.normal(np.mean(pts), np.std(pts))
                astproj = np.random.normal(np.mean(ast), np.std(ast))
                rebproj = np.random.normal(np.mean(reb), np.std(reb))
                stlproj = np.random.normal(np.mean(stl), np.std(stl))
                blkproj = np.random.normal(np.mean(blk), np.std(blk))
                tovproj = np.random.normal(np.mean(tov), np.std(tov))
                thrproj = np.random.normal(np.mean(thr), np.std(thr))
                theproj = ptsproj + (astproj * 1.5) + (rebproj * 1.25) + (stlproj * 2) + (blkproj * 2) + (thrproj * .5) - (tovproj * .5)
                if (ptsproj > 9) and (astproj > 9):
                    theproj = theproj + 1.5
                elif (ptsproj > 9) and (rebproj > 9):
                    theproj = theproj + 1.5
                elif (astproj > 9) and (rebproj > 9):
                    theproj = theproj + 1.5

                if (ptsproj > 9) and (astproj > 9) and (rebproj > 9):
                    theproj = theproj + 3

                scores.append(theproj)
                avgscore = np.mean(scores)
                avgval = (avgscore) / (sal/1000)
                playerprojdict.update({player: avgval})

    finaldf = pd.DataFrame.from_dict(playerprojdict, orient='index')
    finaldf.columns=['AvgVal']
    finaldf = finaldf.sort_values(by='AvgVal', ascending=False)
    finaldf.to_csv('/home/Jon2Anderson/nba/filestore/projections/avgValProjections/%s_AvgValProj.csv' %date)

print('Starting data collection..')

print('Getting Rosters')
#getRosters()

print('Get DK Salaries...')
getDKSals()

print('Get FD Salaries...')
getFDSals()

print('Combining Salary Info...')
combineSals()

print('Collecting game lines...')
getLines()

print('Combining game line data...')
combineLines()

#print('Collecting Number Fire Minutes Projections...')
#getNumFireProjMins()

#print('Collecting DFS Cafe Minutes Projection...')
#dfsCafeMinutes()

#print('Collecting Fantasy Pros Projections...')
#scrapeFP()

print('Get minutes projections...')
getMins()

print('Collecting DFS Cafe Projections...')
getDFSCafeProjScores()

print('Get Defensive Ratings...')
getDefRtg()

print('Get Position Average Value by Opponent')
valByPos()

print('Get Salary Diff...')
salDiff()

print('Projecting team FP output...')
projFP()
projFP2()

print('Get defensive stat allowances...')
defStatAllowances()

print('Player sim...')
playerSim()

print('Finished.')
