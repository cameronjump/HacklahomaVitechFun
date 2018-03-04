import urllib.request, json
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import threading
import queue
import time

start = time.time()

q = queue.Queue()
bigNumber = 0
timesWritten = 0

def retrieveByID(idNumStart, idNumStop):
    urls={}
    urls[0]='https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&q=id:['+str(idNumStart)+'%20TO%20'+str(idNumStop)+']&wt=json&rows=1000'
    urls[1]='https://v3v10.vitechinc.com/solr/v_us_participant_detail/select?indent=on&q=id:['+str(idNumStart)+'%20TO%20'+str(idNumStop)+']&wt=json&rows=1000'
    urls[2]='https://v3v10.vitechinc.com/solr/v_us_quotes/select?indent=on&q=id:['+str(idNumStart)+'%20TO%20'+str(idNumStop)+']&wt=json&rows=1000'
    data={}
    df={}
    for i  in range(0,3):
       with urllib.request.urlopen(urls[i]) as url:
           data = json.loads(url.read().decode())
           to_df = data['response']['docs']
           df[i]=pd.DataFrame(to_df)
    df1=pd.merge(df[0],df[1],on='id')
    df_final=pd.merge(df1,df[2],on='id')
    return df_final

def writeRange(idNumStart, idNumStop):
    print('welcome')
    count = idNumStart
    while (True):
        if(count > idNumStop):
            if(count>bigNumber):
                bigNumber = count
            print('break')
            break
        q.put(retrieveByID(count, count + 1000).to_string())
        count = count + 1000
        print('step')
    afile.close()

number = 1
for i in range(0,100):
    #writeRange(number, number + 14000)
    thread = threading.Thread(target=writeRange, args=(number,number+14000))
    thread.daemon = True
    thread.start()
    number = number + 14000

fileA = open('big.txt', 'a')
while(bigNumber <1400000):
    try:
        fileA.write(q.get())
        timesWritten = timesWritten + 1
        print(str(timesWritten))
    except Exception as e:
        print('nothing there')

fileA.close()
print('done')
print(str(time.time()-start))
