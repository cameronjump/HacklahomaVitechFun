import urllib.request, json
import pandas as pd
import threading
import time

states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada",'New%20Hampshire',"New%20Jersey","New%20Mexico","New%20York",
  "North%20Carolina","North%20Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode%20Island","South%20Carolina","South%20Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West%20Virginia","Wisconsin","Wyoming"]

def getStateID(state, retrieve):
    url = 'https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&wt=json&rows=1000&q=state:' + state +'&fl=' + retrieve
    connection = urllib.request.urlopen(url)
    dataFrame = pd.DataFrame(json.load(connection)['response']['docs'])
    return dataFrame

def getParam(id):
    url = 'https://v3v10.vitechinc.com/solr/v_us_participant_detail/select?indent=on&wt=json&rows=100000&q=id:'+ str(id) +'&fl=HEIGHT,WEIGHT,TOBACCO'
    connection = urllib.request.urlopen(url)
    dataFrame = pd.DataFrame(json.load(connection)['response']['docs'])
    return dataFrame

def dip(x):
    if x=='Yes':
        dippers=1
    else:
        dippers=0
    return dippers

def calcbmi(height, weight):
    return 703 * weight / (height * height)




file = open("statetest.txt","w")
completed = 0

def calculateStateHealth(state):
    dataFrame = getStateID(state, 'id')

    height = 0
    weight = 0
    tobacco = 0
    bmi = 0
    number = 0
    for i in dataFrame['id']:
        hT = 0
        wT = 0
        data = getParam(str(i))
        for j in data['HEIGHT']:
            height = j  + height
            hT = j
        for j in data['WEIGHT']:
            weight = j  + weight
            wT = j
        for j in data['TOBACCO']:
            tobacco = dip(j) + tobacco
        bmi = bmi + calcbmi(hT, wT)
        number = number + 1
    print(state + ' height: '+ str(height/number) + ' weight: '+ str(weight/number)+" tobacco percent: " + str(tobacco/number) +' bmi: ' + str(bmi/number))
    file.write(state + ' height: '+ str(height/number) + ' weight: '+ str(weight/number)+" tobacco percent: " + str(tobacco/number)+ ' bmi: ' + str(bmi/number)+'\n')
    complete = completed + 1
    return

for state in states:
    thread = threading.Thread(target=calculateStateHealth, args=(state,))
    thread.daemon = True
    thread.start()
    time.sleep(.1)

while(True):
    if(completed == 50):
        file.close()
        break
