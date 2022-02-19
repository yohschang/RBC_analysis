# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:08:46 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from glob import glob
import pandas as pd
from sqlalchemy import create_engine

normalize = lambda x : (x - np.min(x)) / np.max((x - np.min(x))) 
normalize2 = lambda x, minn, maxx : (x - minn) / (maxx-minn)

#%%
db_connection_str = 'mysql+pymysql://root:asdzxc7856@localhost/rbc_paras'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM rbc_db', con = db_connection)

print(df.columns)
#%%
# normal = df.loc[(df['person']=="YX")]
# normal.to_pickle(r"C:\Users\YH\Desktop\dataframe\normal")
# glu001 = df.loc[df['person'] == 'YX w/ 0.01% glu']
# glu001.to_pickle(r"C:\Users\YH\Desktop\dataframe\glu001")
# glu0001 = df.loc[df['person'] == 'YX w/ 0.001% glu']
# glu0001.to_pickle(r"C:\Users\YH\Desktop\dataframe\glu0001")

# glu00005 = df.loc[df['person'] == 'YX w/ 0.0005% glu']
# glu00005.to_pickle(r"C:\Users\YH\Desktop\dataframe\glu00005")

#%%
# chang= df.loc[(df['person']=="YX")]
# chang.to_pickle(r"C:\Users\YH\Desktop\dataframe\chang")
# xui= df.loc[(df['person']=="yixiang")]
# xui.to_pickle(r"C:\Users\YH\Desktop\dataframe\xui")
# xie= df.loc[(df['person']=="xinyuan")]
# xie.to_pickle(r"C:\Users\YH\Desktop\dataframe\xie")
# kao= df.loc[(df['person']=="bankao")]
# kao.to_pickle(r"C:\Users\YH\Desktop\dataframe\kao")
# whan= df.loc[(df['person']=="WH")]
# whan.to_pickle(r"C:\Users\YX\Desktop\dataframe\wh")

# glu05 = df.loc[(df['person']=='YX w/ 0.05% glu')]
# glu05.to_pickle(r"C:\Users\YH\Desktop\dataframe\glu05")
# glu01 = df.loc[(df['person']=='YX w/ 0.01% glu')]
# glu01.to_pickle(r"C:\Users\YH\Desktop\dataframe\glu01")

#%%

def MCD(SA , V):
    SA = SA[SA!=0]+10
    V = V[V!=0]
    
    sa_v = SA / V
    # return 0.56*(sa_v)**2 - 2.54*(sa_v) + 5.68
    return sa_v

# normal = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\normal")
glu01 = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\glu01")
# glu01 = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\glu0001")
glu05 = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\glu05")

#%%
from scipy.stats import ttest_ind
def modi(dfs , column ,a = 0 ,b  = -1):
    DF = dfs[column]
    DF = np.nan_to_num(DF)
    DF = np.sort(DF[DF!=0])
    # print(len(DF))
    
    # head = np.sort(DF)[0:12]
    DF = np.sort(DF)[a:b]

    # DF = pd.DataFrame({"a": np.hstack((head , DF))})  
    return DF
    
    
chang = pd.read_pickle(r"C:\Users\YX\Desktop\dataframe\disease\new2_chang")
xui = pd.read_pickle(r"C:\Users\YX\Desktop\dataframe\disease\new2_xui")
xie = pd.read_pickle(r"C:\Users\YX\Desktop\dataframe\disease\new2_xie")
kao = pd.read_pickle(r"C:\Users\YX\Desktop\dataframe\disease\new2_kao")
wh = pd.read_pickle(r"C:\Users\YX\Desktop\dataframe\disease\new2_wh")
# glu01 = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\glu01")


# for i in chang.columns[3:]:
for i in ['c2']:
    
    Ch = modi(chang , i, 5,-5 )
    Xu = modi(xui , i ,2 )
    Xi = modi(xie , i )
    Ka = modi(kao , i ,3)
    WH = modi(wh , i , 0,-5 )
    
    maxx = np.max(np.hstack((Ch, Xu, Xi, Ka, WH)))
    minn = np.min(np.hstack((Ch, Xu, Xi, Ka, WH)))
    
    Chc2 = normalize2(Ch , minn , maxx)
    Xuc2 = normalize2(Xu , minn , maxx)
    Xic2 = normalize2(Xi , minn , maxx)
    Kac2 = normalize2(Ka , minn , maxx)
    WHc2 = normalize2(WH , minn , maxx)
    
    Ch = modi(chang , "c4" , 5,-5)
    Xu = modi(xui , "c4" ,2 )
    Xi = modi(xie , "c4" )
    Ka = modi(kao , "c4" ,3)
    WH = modi(wh , "c4" , 0,-5 )
    
    maxx = np.max(np.hstack((Ch, Xu, Xi, Ka, WH)))
    minn = np.min(np.hstack((Ch, Xu, Xi, Ka, WH)))
    
    Chc4 = normalize2(Ch , minn , maxx)
    Xuc4 = normalize2(Xu , minn , maxx)
    Xic4 = normalize2(Xi , minn , maxx)
    Kac4 = normalize2(Ka , minn , maxx)
    WHc4 = normalize2(WH , minn , maxx)

    Ch = pd.DataFrame({"a":Chc2+Chc4})
    Xu = pd.DataFrame({"a":Xuc2+Xuc4})
    Xi = pd.DataFrame({"a":Xic2+Xic4})
    Ka = pd.DataFrame({"a":Kac2+Kac4})
    WH = pd.DataFrame({"a":WHc2+WHc4})
    
    # G01 = modi(glu01 , i ,0,-5)
    # G05 = modi(glu05 , i)
    
    # n = normal[i]
    # g1 = MCD( glu1["curvature_0"],glu1[i] )
    # g01 = MCD( glu01["curvature_0"], glu01[i] )
    # g005 = MCD( glu005["curvature_0"],glu005[i] )
    # n =  MCD( normal["curvature_0"],normal[i] )
    
    
    plt.figure(dpi = 300)
    data = pd.DataFrame({ "chang": Ch["a"],
                          "xui": Xu["a"],
                          "xie" : Xi["a"],
                          "kao" : Ka["a"],
                          "wh" : WH["a"]})
    # plt.figure(dpi = 300)
    # data = pd.DataFrame({"0.05% Glu" :G05["a"],
    #                      "0.01% Glu" : G01["a"],
    #                      "chang": Ch["a"]})
    
    medianprops = dict(linestyle='-', linewidth=2, color='g')
    data.boxplot(medianprops=medianprops)
    
    
    for j,d in enumerate(data):
        y = data[d]
        x = np.random.normal(j+1, 0.04, len(y))
        plt.plot(x, y, mfc = ["red","orange","blue","yellow","green"][j], mec='k', ms=7, marker="o", linestyle="None")
        # plt.plot(x, y, mfc = ["blue","yellow","red"][j], mec='k', ms=7, marker="o", linestyle="None")
    plt.grid(b=None)
    plt.title(i)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.ylim(0.5,2)
    plt.show()
    
    for i, name in zip([Ch , Xu , Xi , Ka], ["ch" , "xui" , "xie" , "ka"]):
        t, p = ttest_ind(i , WH, equal_var = False)
        print(name + " : " + str(p))
    
    # t, p = ttest_ind(G05 , G01, equal_var = False)
    # print("g01-g05 : " + str(p*2))
        
    # t, p = ttest_ind(G05 , Ch, equal_var = False)
    # print("g05-ch : " + str(p*2))
        
    # t, p = ttest_ind(G01 , Ch, equal_var = False)
    # print("g01-ch : " + str(p*2))
    
    # break

#%%

normal = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\normal")
glu1 = pd.read_pickle(r"C:\Users\YH\Desktop\dataframe\glu001")
print(normal.columns[3:])
#%%
from scipy.stats import ttest_ind
for i in ['asymmetry_value']:
    
    g1 = glu1[i]
    g1 = np.nan_to_num(g1)
    g1 = g1[g1!=0]
    g1 = np.sort(g1)[12:]
    # print(g1[:10] , g1[-10:])
    
    n = normal[i]
    n = np.nan_to_num(n)
    n = n[n!=0]
    n = np.sort(n)[:-12]
    # print(n[:10] , n[-10:])

    n = pd.DataFrame({"aa": n})
    g1 = pd.DataFrame({"aa": g1})

    t, p = ttest_ind(n, g1, equal_var = False)
    print("t = " + str(t) +", p = " +str(p))
    print("med_g1 " + str(np.median(g1)) +", med_n = " +str(np.median(n)))
    print("len_g1 " + str(len(g1)) +", len_n = " +str(len(n)))
    
    
    plt.figure(figsize = (4,5) ,dpi = 300)
    data = pd.DataFrame({"0.01%": g1["aa"],
                         "saline" : n["aa"]})
    

    
    for j,d in enumerate(data):
        y = data[d]
        x = np.random.normal(j+1, 0.04, len(y))
        plt.plot(x, y, mfc = ["orange","red"][j], mec='k', ms=7, marker="o", linestyle="None")
    medianprops = dict(linestyle='-', linewidth=2, color='g')
    data.boxplot(medianprops=medianprops)
    plt.grid(b=None)
    # plt.title(i)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.ylim([0.58,0.71])

    plt.show()
    

    



