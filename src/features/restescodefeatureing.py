singlezonedf=df.copy()
singlezonedf=df.copy().drop(['multizone_mesT','multizone_setT','multizone_nextT','number_D'],axis=1)
multizonedf=df.copy()
multizonedf=df.copy().drop(['singlezone_mesT','singlezone_setT','singlezone_nextT','number_D'],axis=1)


UniqueDays = singlezonedf.Id_D.unique()
dict1 = {elem : pd.DataFrame for elem in UniqueDays}

for key in dict1.keys():
    dict1[key] = singlezonedf[:][singlezonedf.Id_D == key].drop(['Id_D'],axis=1)
    dict1[key].to_csv('C:/Users/DimiP/Documents/GitHub/ULG/COURS/basic-MPC/data/processed/singlezone/'+key+'.csv')

UniqueDays2 = multizonedf.Id_D.unique()
dict2 = {elem : pd.DataFrame for elem in UniqueDays2}

for key in dict2.keys():
    dict2[key] = multizonedf[:][multizonedf.Id_D == key].drop(['Id_D'],axis=1)
    dict2[key].to_csv('C:/Users/DimiP/Documents/GitHub/ULG/COURS/basic-MPC/data/processed/multizone/'+key+'.csv')

