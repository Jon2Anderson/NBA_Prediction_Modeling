import pandas as pd, requests, os, collections as co, json, sys, shutil, unidecode
import datetime
from bs4 import BeautifulSoup
import numpy as np
from datetime import date, timedelta
from scipy.stats import uniform
import statsmodels.api as sm
import statsmodels.formula.api as smf

print('Starting...')
boxes = pd.read_csv('/home/Jon2Anderson/nba/filestore/todownload/masterboxscores.csv')
boxes = boxes.sort_values(by='Date')
rosterdict = json.load(open('/home/Jon2Anderson/nba/filestore/rosterdict.json'))
namechange = json.load(open('/home/Jon2Anderson/nba/filestore/playernamechange.json'))
os.chdir('/home/Jon2Anderson/nba/')
month = datetime.date.today().strftime("%b")
day = datetime.date.today().strftime("%d")
date = month+day

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
def sportsline():
    url = 'https://www.sportsline.com/sportsline-web/service/v1/fantasy/projections/nba/simulation'
    data = requests.get(url)
    data_json = data.json()['projections']
    df = pd.DataFrame(data_json)
    df = df[['player', 'min']]
    return(df)

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

def analyzeMinutes():
    try:
        dfsCafe_mins = dfsCafe()
        numFire_mins = numFire()
    except:
        print('problem with minutes projections.')

    if len(dfsCafe_mins)==0:
        dfsCafe_mins = numFire_mins.copy()
        dfsCafe_mins.columns=['Player','DC_Mins','Team']
        print('Problem with DFS Cafe')

    if len(numFire_mins)==0:
        numFire_mins = dfsCafe_mins.copy()
        numFire_mins.columns=['Player','NF_Mins','Team']
        print('Problem with Number Fire')

    minsdf = pd.merge(dfsCafe_mins,numFire_mins, on='Player')
    minsdf = minsdf[['Player','Team_x','DC_Mins','NF_Mins']]
    minsdf.columns=['Player','Team','DC_Mins','NF_Mins']
    minsdf['DC_Mins'] = pd.to_numeric(minsdf['DC_Mins'])
    minsdf['NF_Mins'] = pd.to_numeric(minsdf['NF_Mins'])

    minsdf['Avg'] = minsdf.mean(axis=1, skipna=True)
    minsdf['Avg'] = round(minsdf['Avg'],0)
    minsdf['Diff'] = abs(minsdf['DC_Mins'] - minsdf['NF_Mins'])
    minsdf = minsdf.sort_values(by='Diff', ascending=False)

    minsdf.to_csv('/home/Jon2Anderson/nba/filestore/proj_minutes/master_minutes/%s_ProjMins.csv' %date)
'''

def FP():
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

    projdf = projdf[['Player','Team','MIN']]
    projdf.columns=['Player','Team','FP_Mins']
    projdf['Player'] = projdf['Player'].replace(namechange)
    projdf = projdf[projdf['Team']!='']
    return(projdf)

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
    #print('Fantasy pros...')
    #try:
        #fp_mins = FP()
    #except:
        #print('Problem with Fantasy Pros Projections')


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

    #if len(fp_mins)==0:
        #fp_mins = dfsCafe_mins.copy()
        #fp_mins.columns=['Player','NF_Mins','Team']
        #print('Problem with FantasyPros')

    if len(sportsline_mins)==0:
        sportsline_mins = dfsCafe_mins.copy()
        sportsline_mins.columns=['Player','NF_Mins','Team']
        print('Problem with Sportsline')


    print('Merging...')
    minsdf = pd.merge(dfsCafe_mins,numFire_mins, on='Player', how='outer')
    #minsdf = pd.merge(minsdf, fp_mins, on='Player', how='outer')
    minsdf = pd.merge(minsdf, sportsline_mins, on='Player', how='outer')
    minsdf = minsdf.fillna(0)
    #minsdf = minsdf[['Player','DC_Mins','NF_Mins','FP_Mins', 'SL_Mins']]
    minsdf = minsdf[['Player','DC_Mins','NF_Mins', 'SL_Mins']]
    #minsdf.columns=['Player','DC','NF','FP','SL']
    minsdf.columns=['Player','DC','NF','SL']
    minsdf['DC'] = pd.to_numeric(minsdf['DC'])
    minsdf['NF'] = pd.to_numeric(minsdf['NF'])
    minsdf['SL'] = pd.to_numeric(minsdf['SL'])
    minsdf['Avg'] = minsdf.mean(axis=1, skipna=True)
    minsdf['Low'] = minsdf.min(axis=1, skipna=True)
    minsdf.to_csv('/home/Jon2Anderson/nba/filestore/projections/minutes/%s_Mins.csv' %date)
    print('finished')

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

def getFPPMseason():
    seasonfppmdict = {}
    players = boxes['Player'].unique()
    for player in players:
        df = boxes[boxes['Player']==player]
        df = df.sort_values(by='Date', ascending=True)
        df = df.head(10)
        fppm = sum(df['FP']) / sum(df['MP'])
        seasonfppmdict.update({player: fppm})
    with open('/home/Jon2Anderson/nba/filestore/fppm_dict.json', 'w') as json_file:
        json.dump(seasonfppmdict, json_file)
    return(seasonfppmdict)

def getFPPMdict():
    fppmdict = {}
    players = boxes['Player'].unique()
    for player in players:
        df = boxes[boxes['Player']==player]
        df = df.sort_values(by='Date', ascending=True)
        df = df.head(10)
        fppm = sum(df['FP']) / sum(df['MP'])
        fppmdict.update({player: fppm})
    with open('/home/Jon2Anderson/nba/filestore/fppm_dict.json', 'w') as json_file:
        json.dump(fppmdict, json_file)
    return(fppmdict)

def getSeasonUSGdict():
    seasonusgdict = {}
    players = boxes['Player'].unique()
    for player in players:
        df = boxes[boxes['Player']==player]
        df = df.sort_values(by='Date', ascending=True)
        usg = np.mean(df['USG'])
        seasonusgdict.update({player: usg})
    with open('/home/Jon2Anderson/nba/filestore/usg_dict.json', 'w') as json_file:
        json.dump(seasonusgdict, json_file)
    return(seasonusgdict)

def getUSGdict():
    usgdict = {}
    players = boxes['Player'].unique()
    for player in players:
        df = boxes[boxes['Player']==player]
        df = df.sort_values(by='Date', ascending=True)
        df = df.head(10)
        usg = np.mean(df['USG'])
        usgdict.update({player: usg})
    with open('/home/Jon2Anderson/nba/filestore/usg_dict.json', 'w') as json_file:
        json.dump(usgdict, json_file)
    return(usgdict)

def makeToday():
    usgdict = getUSGdict()
    fppmdict = getFPPMdict()
    posdict = json.load(open('/home/Jon2Anderson/nba/filestore/posdict.json'))
    rosterdict = json.load(open('/home/Jon2Anderson/nba/filestore/rosterdict.json'))
    namechange = json.load(open('/home/Jon2Anderson/nba/filestore/playernamechange.json'))
    os.chdir('/home/Jon2Anderson/nba/')
    month = datetime.date.today().strftime("%b")
    day = datetime.date.today().strftime("%d")
    date = month+day

    teamproj = pd.read_csv('/home/Jon2Anderson/nba/filestore/projections/teamFPproj/%s_FPProj.csv' %date)
    teamproj.columns=['Team','TmFP']
    teamfpdict = dict(zip(teamproj.Team, teamproj.TmFP))

    teamproj2 = pd.read_csv('/home/Jon2Anderson/nba/filestore/projections/teamFPproj/%s_LM_TeamProjFP.csv' %date)
    teamproj2 = teamproj2[['Team','ProjFP']]
    teamproj2.columns = ['Team','TmFP']
    teamfpdict2 = dict(zip(teamproj2.Team, teamproj2.TmFP))

    dfscafe = pd.read_csv('/home/Jon2Anderson/nba/filestore/projections/DFSCafe/%s_DFSCafeProj.csv' %date)
    dfscafedict = dict(zip(dfscafe.Player,dfscafe.ProjScore))

    dksals = pd.read_csv('/home/Jon2Anderson/nba/filestore/salaries/DK/%s_DKSals.csv' %date)
    dksaldict = dict(zip(dksals.Player,dksals.Sal))

    fdsals = pd.read_csv('/home/Jon2Anderson/nba/filestore/salaries/FD/%s_FDSals.csv' %date)
    fdsaldict = dict(zip(fdsals.Player,fdsals.Sal))

    gamelines = pd.read_csv('/home/Jon2Anderson/nba/filestore/gamelines/dailylines/%s_lines.csv' %date)
    oppdict = dict(zip(gamelines.team,gamelines.opp))
    linedict = dict(zip(gamelines.team,gamelines.line))
    oudict = dict(zip(gamelines.team,gamelines.ou))
    teamprojdict = dict(zip(gamelines.team,gamelines.proj))

    projmins = pd.read_csv('/home/Jon2Anderson/nba/filestore/projections/minutes/%s_Mins.csv' %date)
    minsdict = dict(zip(projmins.Player,projmins.Avg))
    minslowdict = dict(zip(projmins.Player,projmins.Low))

    saldiffdf = pd.read_csv('/home/Jon2Anderson/nba/filestore/saldiff/%s_saldiff.csv' %date)
    saldiffdf.columns=['Player','SalDiff']
    saldiffdict = dict(zip(saldiffdf.Player,saldiffdf.SalDiff))

    avgvalbysim = pd.read_csv('/home/Jon2Anderson/nba/filestore/projections/avgValProjections/%s_AvgValProj.csv' %date)
    avgvalbysim.columns=['Player','AvgSimVal']
    avgvalbysimdict = dict(zip(avgvalbysim.Player, avgvalbysim.AvgSimVal))

    defrtg = pd.read_csv('/home/Jon2Anderson/nba/filestore/defRtg/defStatAllowances.csv')
    defrtg['Team'] = defrtg['Team'].str.lower()
    defrtg = defrtg[['Team','FP']]
    defrtg.columns=['Opp','FP']
    defrtgdict = dict(zip(defrtg.Opp, defrtg.FP))

    posval = pd.read_csv('/home/Jon2Anderson/nba/filestore/defRtg/AvgValByPos.csv')
    posval.columns=['Opp','Def_AvgVal','Pos']
    posval['Opp'] = posval['Opp'].str.lower()

    masterdf = pd.DataFrame(columns=['Player', 'Pos', 'Team','Opp','DKSal','FDSal','ProjMin','FPPM', 'USG', 'Line','GameTot','TeamTot','DefRtg'])
    masterdf['Player'] = dksals['Player']
    masterdf['Pos'] = masterdf['Player'].map(posdict)
    masterdf['Pos'] = masterdf['Pos'].str.lower()
    masterdf['Team'] = masterdf['Player'].map(rosterdict)
    masterdf['Opp'] = masterdf['Team'].map(oppdict)
    masterdf['TmFP'] = masterdf['Team'].map(teamfpdict)
    masterdf['TmFP_LM'] = masterdf['Team'].map(teamfpdict2)
    masterdf['DKSal'] = masterdf['Player'].map(dksaldict)
    masterdf['FDSal'] = masterdf['Player'].map(fdsaldict)
    masterdf['SalDiff'] = masterdf['Player'].map(saldiffdict)
    masterdf['MP'] = masterdf['Player'].map(minsdict)
    masterdf['LowMP'] = masterdf['Player'].map(minslowdict)
    masterdf['FPPM'] = masterdf['Player'].map(fppmdict)
    masterdf['USG'] = masterdf['Player'].map(usgdict)
    masterdf['Line'] = masterdf['Team'].map(linedict)
    masterdf['GameTot'] = masterdf['Team'].map(oudict)
    masterdf['TeamTot'] = masterdf['Team'].map(teamprojdict)
    masterdf['DefRtg'] = masterdf['Opp'].map(defrtgdict)
    masterdf['ValBySim'] = masterdf['Player'].map(avgvalbysimdict)
    masterdf['DFSCafe'] = masterdf['Player'].map(dfscafedict)
    masterdf['DC_Val'] = (masterdf['DFSCafe']) / (masterdf['DKSal'] / 1000)

    masterdf = masterdf[['Player','Pos','Team','Opp','TmFP','TmFP_LM','DKSal','FDSal','SalDiff','MP','LowMP','FPPM','USG','TeamTot','DefRtg', 'ValBySim', 'DFSCafe', 'DC_Val']]
    masterdf = pd.merge(masterdf, posval, on=['Opp','Pos'])
    masterdf = masterdf[['Player','Pos','Team','Opp','TmFP','TmFP_LM','DKSal','FDSal','SalDiff','MP','LowMP','FPPM','USG','TeamTot','DefRtg', 'ValBySim', 'Def_AvgVal','DFSCafe','DC_Val']]
    return(masterdf)

def getModelData():
    linemaster = pd.read_csv('/home/Jon2Anderson/nba/filestore/gamelines/linemaster.csv')
    linemaster = linemaster[['team','date','proj']]
    linemaster.columns=['Team','Date','TeamTot']
    seasonfppmdict = getFPPMseason()
    seasonusgdict = getSeasonUSGdict()

    newboxes = boxes
    newboxes['Opp'] = newboxes['Opp'].str.lower()
    newboxes['FPPM'] = newboxes['Player'].map(seasonfppmdict)
    newboxes['USG'] = newboxes['Player'].map(seasonusgdict)
    newboxes['Date'] = pd.to_datetime(newboxes['Date'])
    newboxes['Date'] = newboxes['Date'].dt.strftime('%b%d')
    newboxes = pd.merge(newboxes, linemaster, on=['Team','Date'])

    defrtg = pd.read_csv('/home/Jon2Anderson/nba/filestore/defRtg/defStatAllowances.csv')
    defrtg['Team'] = defrtg['Team'].str.lower()
    defrtg = defrtg[['Team','FP']]
    defrtg.columns=['Opp','FP']
    defrtgdict = dict(zip(defrtg.Opp, defrtg.FP))
    newboxes['DefRtg'] = newboxes['Opp'].map(defrtgdict)
    newboxes = newboxes[newboxes['MP'] > 5]
    return(newboxes)

def runModel(modeldata,today):
    np.random.seed(1234)
    modeldata = modeldata.sample(frac=1)
    modeldata['runiform'] = uniform.rvs(loc=0,scale=1,size=len(modeldata))

    mymodel = str('FP ~ MP + USG + FPPM + TeamTot + DefRtg')
    trainmodel_fit = smf.ols(mymodel, data=modeldata).fit()

    modeldata['MyProj'] = trainmodel_fit.fittedvalues
    today['MyProj'] = trainmodel_fit.predict(today)
    today['MyVal'] = today['MyProj'] / (today['DKSal']/1000)

    today['AvgProj'] = today[['DFSCafe', 'MyProj']].mean(axis=1)
    today['AvgVal'] = today['AvgProj'] / (today['DKSal']/1000)

    today = today.sort_values(by='AvgVal', ascending=False)
    return(today)


print('Updating Vegas lines...')
getLines()

print('Updating Team FP Projections...')
projFP2()

print('Collecting current minutes projections...')
getMins()

print('Building model to use...')
modeldata = getModelData()
modeldata.to_csv('/home/Jon2Anderson/nba/filestore/todownload/dailysheets/%s_modeldata.csv' %date)

print('Getting today\'s information...')
todaydf = makeToday()
todaydf = todaydf[todaydf['MP']>5]

print('Make predictions and write file...')
predictionsdf = runModel(modeldata,todaydf)

predictionsdf.to_csv('/home/Jon2Anderson/nba/filestore/todownload/dailysheets/projsheets/%s.csv' %date)

print('finished')