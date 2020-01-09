import pandas as pd, requests, os, collections as co, json, sys, shutil, unidecode, datetime
from bs4 import BeautifulSoup
from datetime import date, timedelta

os.chdir('/home/Jon2Anderson/nba/')
posdict = json.load(open('/home/Jon2Anderson/nba/filestore/posdict.json'))
namechange = json.load(open('/home/Jon2Anderson/nba/filestore/playernamechange.json'))

def combineAllData():
    teams=['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU',
    'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL',
    'PHI', 'PHO', 'POR', 'SAC' , 'SAS', 'TOR', 'UTA', 'WAS']

    master = pd.DataFrame(columns=["Player", "Team", "Pos", "Date", "Opp", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "ORB",
                                    "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-"])
    for team in teams:
        print(team)
        os.chdir("/home/Jon2Anderson/nba/filestore/boxscores/%s" %team)
        thefiles = os.listdir("/home/Jon2Anderson/nba/filestore/boxscores/%s" %team)
        for file in thefiles:
            df = pd.read_csv(file)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            master = pd.concat([master, df])

    master['Player'] = master['Player'].str.lower()
    master['Player'] = master['Player'].replace(namechange)
    master['Team'] = master['Team'].str.lower()
    master['Pos'] = master['Player'].map(posdict)
    master = master[["Player", "Team", "Pos", "Date", "Opp", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "ORB",
                        "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "USG", "FP"]]
    master['FPPM'] = master['FP'] / master['MP']
    master.to_csv('/home/Jon2Anderson/nba/filestore/todownload/masterboxscores.csv')


    def_master = pd.DataFrame(columns=["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%",
                                        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"])
    for team in teams:
        print(team)
        os.chdir("/home/Jon2Anderson/nba/filestore/boxscores/defense/%s" %team)
        thefiles = os.listdir("/home/Jon2Anderson/nba/filestore/boxscores/defense/%s" %team)
        for file in thefiles:
            df = pd.read_csv(file)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            def_master = pd.concat([def_master, df])
    def_master = def_master[["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%",
                                        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"]]

    def_master.to_csv('/home/Jon2Anderson/nba/filestore/todownload/masterdef.csv')


    off_master = pd.DataFrame(columns=["Team", "Date", "Opp",  "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%",
                                        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"])
    for team in teams:
        print(team)
        os.chdir("/home/Jon2Anderson/nba/filestore/boxscores/offense/%s" %team)
        thefiles = os.listdir("/home/Jon2Anderson/nba/filestore/boxscores/offense/%s" %team)
        for file in thefiles:
            df = pd.read_csv(file)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            off_master = pd.concat([off_master, df])
    off_master = off_master[["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%",
                                        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"]]
    off_master.to_csv('/home/Jon2Anderson/nba/filestore/todownload/masteroff.csv')

def clearAllDirs():
    os.chdir("/home/Jon2Anderson/nba/filestore/boxscores")
    teams=['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU',
    'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL',
    'PHI', 'PHO', 'POR', 'SAC' , 'SAS', 'TOR', 'UTA', 'WAS']

    for team in teams:
        shutil.rmtree('%s' %team)
        shutil.rmtree('defense/%s' %team)
        shutil.rmtree('offense/%s' %team)
        os.mkdir('%s' %team)
        os.mkdir('defense/%s' %team)
        os.mkdir('offense/%s' %team)

def getDailyURLS(date):
    month=date.split('-')[0]
    day=date.split('-')[1]
    year=date.split('-')[2]
    url = 'https://www.basketball-reference.com/boxscores/?month=%s&day=%s&year=%s' %(month,day,year)

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    # Find which teams are playing #
    teamtables = soup.findAll('table', {'class': 'teams'})
    teamlist=[]
    for tbl in teamtables:
        tbody = tbl.find('tbody')
        trlist=[]
        for tr in tbody.findAll('tr'):
            trlist.append(tr)

        for row in trlist:
            for a in row.findAll('a'):
                teamlist.append(a.text)
    teamlist[:] = (value for value in teamlist if value != 'Final')

    with open('/home/Jon2Anderson/nba/boxScoreScrape/teamnames/nbateamnames.json') as json_file:
        teamnamedict = json.load(json_file)
    teamsthatplayed = [teamnamedict.get(item,item) for item in teamlist]
    listofteams=teamsthatplayed

    gamelist=[]
    while len(teamsthatplayed)>1:
        gamelist.append(teamsthatplayed[0:2])
        teamsthatplayed.pop(0)
        teamsthatplayed.pop(0)

    data = soup.findAll('p', {'class': 'links'})
    i=0
    urls=[]
    for game in data:
        first=str(data[i])
        start="href=\""
        end="\">Box"
        url=(first[first.find(start)+len(start):first.rfind(end)])
        base="https://www.basketball-reference.com"
        url = base + url
        urls.append(url)
        i=i+1

    tuplelist=[]
    for i in range(len(urls)):
        tuplelist.append((urls[i],gamelist[i]))

    return(tuplelist)

def dubdub(df):
    if df['PTS'] > 9 and df['TRB'] > 9:
        return(1.5)
    elif df['PTS'] > 9 and df['AST'] > 9:
        return(1.5)
    elif df['AST'] > 9 and df['TRB'] > 9:
        return(1.5)
    else:
        return(0)

def tripdub(df):
    if df['PTS'] > 9 and df['TRB'] > 9 and df['AST'] > 9:
        return(3)
    else:
        return(0)

def getBox(date, url, team1, team2):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    id1 = 'box-%s-game-basic' %team1
    id2 = 'box-%s-game-basic' %team2

    table1 = soup.find('table', {'class': 'sortable stats_table', 'id': id1})
    table2 = soup.find('table', {'class': 'sortable stats_table', 'id': id2})

    try:
        table1_body = table1.find('tbody')
        table2_body = table2.find('tbody')
    except:
        print('URL exception, trying different table ID')
        id1 = 'box_%s_basic' %team1
        id2 = 'box_%s_basic' %team2
        table1 = soup.find('table', {'class': 'sortable stats_table', 'id': id1})
        table2 = soup.find('table', {'class': 'sortable stats_table', 'id': id2})
        table1_body = table1.find('tbody')
        table2_body = table2.find('tbody')

    playerlist1=[]
    playerlist2=[]
    for a in table1_body.findAll('a'):
        playerlist1.append(unidecode.unidecode(a.text))

    for a in table2_body.findAll('a'):
        playerlist2.append(unidecode.unidecode(a.text))

    trlist1=[]
    for tr in table1_body.findAll('tr'):
        trlist1.append(tr)

    trlist2=[]
    for tr in table2_body.findAll('tr'):
        trlist2.append(tr)

    header = ["MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-"]

    listofdicts1=[]
    for row in trlist1:
        the_row=[]
        for td in row.findAll('td'):
            the_row.append(td.text)
        od = co.OrderedDict(zip(header,the_row))
        listofdicts1.append(od)

    listofdicts2=[]
    for row in trlist2:
        the_row=[]
        for td in row.findAll('td'):
            the_row.append(td.text)
        od = co.OrderedDict(zip(header,the_row))
        listofdicts2.append(od)

    #colnames = ["MP", "Opp", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-"]
    df1 = pd.DataFrame(listofdicts1)
    df1 = df1.dropna()
    df2 = pd.DataFrame(listofdicts2)
    df2 = df2.dropna()

    df1['Date'] = date
    df1['Team'] = team1
    df1['Opp'] = team2
    df1['FG'] = df1['FG'].astype('int32')
    df1['FGA'] = df1['FGA'].astype('int32')
    df1['3P'] = df1['3P'].astype('int32')
    df1['3PA'] = df1['3PA'].astype('int32')
    df1['FT'] = df1['FT'].astype('int32')
    df1['FTA'] = df1['FTA'].astype('int32')
    df1['ORB'] = df1['ORB'].astype('int32')
    df1['DRB'] = df1['DRB'].astype('int32')
    df1['TRB'] = df1['TRB'].astype('int32')
    df1['AST'] = df1['AST'].astype('int32')
    df1['STL'] = df1['STL'].astype('int32')
    df1['BLK'] = df1['BLK'].astype('int32')
    df1['TOV'] = df1['TOV'].astype('int32')
    df1['PF'] = df1['PF'].astype('int32')
    df1['PTS'] = df1['PTS'].astype('int32')

    df1['+/-'] = df1['+/-'].apply(lambda x: 0 if x == '' else x)
    df1['+/-'] = df1['+/-'].astype('int32')
    df1['FG%'] = (df1['FG']/df1['FGA']).fillna(0)
    df1['3P%'] = (df1['3P']/df1['3PA']).fillna(0)
    df1['FT%'] = (df1['FT']/df1['FTA']).fillna(0)
    try:
        df1['FG%'] = round(df1['FG%'],3)
        df1['3P%'] = round(df1['3P%'],3)
        df1['FT%'] = round(df1['FT%'],3)
    except:
        print('rounding failed, moving ... ')
    min_sec_split = df1['MP'].str.split(":",n=1,expand=True)
    min_sec_split[0] = min_sec_split[0].astype('int32')
    min_sec_split[1] = min_sec_split[1].astype('int32')
    Minutes = min_sec_split[0] + (min_sec_split[1]/60)
    df1['MP'] = Minutes
    try:
        df1['MP'] = round(df1['MP'],1)
    except:
        print('rounding failed, moving ... ')
    df1['USG'] = (((df1['FGA']) + (.44*df1['FTA']) + (df1['TOV'])) * (sum(df1['MP']))) / (((sum(df1['FGA'])) + (.44 * (sum(df1['FTA']))) + (sum(df1['TOV']))) * ((df1['MP']) * 5))
    try:
        df1['USG'] = round(df1['USG'],3)
    except:
        print('rounding failed, moving ... ')
    df1['DubDub'] = df1.apply(dubdub, axis=1)
    df1['TripDub'] = df1.apply(tripdub, axis=1)
    df1['FP'] = (df1['PTS']) + (df1['3P'] * .5) + (df1['TRB'] * 1.25) + (df1['AST'] * 1.5) + (df1['STL'] * 2) + (df1['BLK'] * 2) - (df1['TOV'] * .5) + df1['DubDub'] + df1['TripDub']

    df1len = df1.shape[0]
    playerlist1 = playerlist1[0:df1len]
    df1['Player'] = playerlist1
    df1 = df1[["Player", "Team", "Date", "Opp", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "USG", "FP"]]

    df2['Date'] = date
    df2['Team'] = team2
    df2['Opp'] = team1
    df2['FG'] = df2['FG'].astype('int32')
    df2['FGA'] = df2['FGA'].astype('int32')
    df2['3P'] = df2['3P'].astype('int32')
    df2['3PA'] = df2['3PA'].astype('int32')
    df2['FT'] = df2['FT'].astype('int32')
    df2['FTA'] = df2['FTA'].astype('int32')
    df2['ORB'] = df2['ORB'].astype('int32')
    df2['DRB'] = df2['DRB'].astype('int32')
    df2['TRB'] = df2['TRB'].astype('int32')
    df2['AST'] = df2['AST'].astype('int32')
    df2['STL'] = df2['STL'].astype('int32')
    df2['BLK'] = df2['BLK'].astype('int32')
    df2['TOV'] = df2['TOV'].astype('int32')
    df2['PF'] = df2['PF'].astype('int32')
    df2['PTS'] = df2['PTS'].astype('int32')
    df2['+/-'] = df2['+/-'].apply(lambda x: 0 if x == '' else x)
    df2['+/-'] = df2['+/-'].astype('int32')
    df2['FG%'] = (df2['FG']/df2['FGA']).fillna(0)
    df2['3P%'] = (df2['3P']/df2['3PA']).fillna(0)
    df2['FT%'] = (df2['FT']/df2['FTA']).fillna(0)
    try:
        df2['FG%'] = round(df2['FG%'],3)
        df2['3P%'] = round(df2['3P%'],3)
        df2['FT%'] = round(df2['FT%'],3)
    except:
        print('rounding failed, moving ... ')
    min_sec_split = df2['MP'].str.split(":",n=1,expand=True)
    min_sec_split[0] = min_sec_split[0].astype('int32')
    min_sec_split[1] = min_sec_split[1].astype('int32')
    Minutes = min_sec_split[0] + (min_sec_split[1]/60)
    df2['MP']=Minutes
    try:
        df2['MP'] = round(df2['MP'],1)
    except:
        print('rounding failed, moving ... ')
    df2['USG'] = (((df2['FGA']) + (.44*df2['FTA']) + (df2['TOV'])) * (sum(df2['MP']))) / (((sum(df2['FGA'])) + (.44 * (sum(df2['FTA']))) + (sum(df2['TOV']))) * ((df2['MP']) * 5))
    try:
        df2['USG'] = round(df2['USG'],3)
    except:
        print('rounding failed, moving ... ')
    df2['DubDub'] = df2.apply(dubdub, axis=1)
    df2['TripDub'] = df2.apply(tripdub, axis=1)
    df2['FP'] = (df2['PTS']) + (df2['3P'] * .5) + (df2['TRB'] * 1.25) + (df2['AST'] * 1.5) + (df2['STL'] * 2) + (df2['BLK'] * 2) - (df2['TOV'] * .5) + df2['DubDub'] + df2['TripDub']

    df2len = df2.shape[0]
    playerlist2 = playerlist2[0:df2len]
    df2['Player'] = playerlist2
    df2 = df2[["Player", "Team", "Date", "Opp", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "USG", "FP"]]

    df1.to_csv('filestore/boxscores/%s/%s_%s.csv' %(team1, date, team1))
    df2.to_csv('filestore/boxscores/%s/%s_%s.csv' %(team2, date, team2))

    # defensive stats block #
    defdict1 = {
                    'MP': sum(df2['MP']),
                    'FG': sum(df2['FG']),
                    'FGA': sum(df2['FGA']),
                    'FG%': sum(df2['FG']) / sum(df2['FGA']),
                    '3P': sum(df2['3P']),
                    '3PA': sum(df2['3PA']),
                    '3P%': sum(df2['3P']) / sum(df2['3PA']),
                    'FT': sum(df2['FT']),
                    'FTA': sum(df2['FTA']),
                    'FT%': sum(df2['FT']) / sum(df2['FTA']),
                    'ORB': sum(df2['ORB']),
                    'DRB': sum(df2['DRB']),
                    'TRB': sum(df2['TRB']),
                    'AST': sum(df2['AST']),
                    'STL': sum(df2['STL']),
                    'BLK': sum(df2['BLK']),
                    'TOV': sum(df2['TOV']),
                    'PF': sum(df2['PF']),
                    'PTS': sum(df2['PTS']),
                    '+/-': sum(df2['+/-'])
                }
    defdf1 = pd.DataFrame(defdict1, index=[0])
    defdf1['Team'] = team1
    defdf1['Date'] = date
    defdf1['Opp'] = team2

    defdf1['FP'] = defdf1['PTS'] + (defdf1['3P'] * .5) + (defdf1['AST'] * 1.5) + (defdf1['TRB'] * 1.25) + (defdf1['STL'] * 2) + (defdf1['BLK'] * 2) - (defdf1['TOV'] * .5)

    # team 2 #
    defdict2 = {
                'MP': sum(df1['MP']),
                'FG': sum(df1['FG']),
                'FGA': sum(df1['FGA']),
                'FG%': sum(df1['FG']) / sum(df1['FGA']),
                '3P': sum(df1['3P']),
                '3PA': sum(df1['3PA']),
                '3P%': sum(df1['3P']) / sum(df1['3PA']),
                'FT': sum(df1['FT']),
                'FTA': sum(df1['FTA']),
                'FT%': sum(df1['FT']) / sum(df1['FTA']),
                'ORB': sum(df1['ORB']),
                'DRB': sum(df1['DRB']),
                'TRB': sum(df1['TRB']),
                'AST': sum(df1['AST']),
                'STL': sum(df1['STL']),
                'BLK': sum(df1['BLK']),
                'TOV': sum(df1['TOV']),
                'PF': sum(df1['PF']),
                'PTS': sum(df1['PTS']),
                '+/-': sum(df1['+/-'])
            }
    defdf2 = pd.DataFrame(defdict2, index=[0])
    defdf2['Team'] = team2
    defdf2['Date'] = date
    defdf2['Opp'] = team1
    defdf2['FP'] = defdf2['PTS'] + (defdf2['3P'] * .5) + (defdf2['AST'] * 1.5) + (defdf2['TRB'] * 1.25) + (defdf2['STL'] * 2) + (defdf2['BLK'] * 2) - (defdf2['TOV'] * .5)

    defdf1['Poss'] = (defdf1['FGA'] - defdf1['ORB'] + defdf1['TOV'] + (.4 * defdf1['FTA']))
    defdf2['Poss'] = (defdf2['FGA'] - defdf2['ORB'] + defdf2['TOV'] + (.4 * defdf2['FTA']))

    defdf1 = defdf1[["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA",
            "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"]]
    defdf2 = defdf2[["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA",
            "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"]]

    defdf1.to_csv('filestore/boxscores/defense/%s/%s_%s.csv' %(team1, date, team1))
    defdf2.to_csv('filestore/boxscores/defense/%s/%s_%s.csv' %(team2, date, team2))

    offdf1 = defdf1
    offdf1['Team'] = team2
    offdf1['Opp'] = team1

    offdf2 = defdf2
    offdf2['Team'] = team1
    offdf2['Opp'] = team2

    offdf1 = offdf1[["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA",
            "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"]]
    offdf2 = offdf2[["Team", "Date", "Opp", "MP", "Poss", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA",
            "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "+/-", "FP"]]

    offdf1.to_csv('filestore/boxscores/offense/%s/%s_%s.csv' %(team2, date, team2))
    offdf2.to_csv('filestore/boxscores/offense/%s/%s_%s.csv' %(team1, date, team1))

    return(df1,df2)


print('starting ... ')

today = date.today()
yesterday = today - timedelta(days=1)
date = yesterday.strftime('%m-%d-%Y')

x=getDailyURLS(date)
for i in range(len(x)):
    getBox(date, x[i][0], x[i][1][0].upper(), x[i][1][1].upper())

print('combining data')
combineAllData()

print('... finished')