import tkinter as tk
from tkinter import *
from tkinter import ttk
import numpy as np
import urllib.request, json
import pandas as pd
import scipy
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import xlrd




win = tk.Tk()

win.title("Vitech")

ml2 = pd.read_excel('ml.xlsx')
ml2_train = ml2.drop('Gold', axis=1)
algo, name = SVR(kernel='rbf', C=1e3, gamma=0.1), 'SVR'
x_train, x_test, y_train, y_test = train_test_split(ml2_train, ml2['Gold'], test_size=.1, random_state=11)
algo.fit(x_train, y_train)



#name = 'DELUCCA,SHALEE V'


def getParam(id):
     url = 'https://v3v10.vitechinc.com/solr/v_us_quotes/select?indent=on&wt=json&rows=100000&q=id:'+ str(id)
     connection = urllib.request.urlopen(url)
     return pd.DataFrame(json.load(connection)['response']['docs'])

def find_customer():
    #query to find customer
    name = last.get() + "," + first.get()
    name = name.upper()
    url = 'https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&wt=json&rows=1000&q=name:' + name + '&fl=name,DOB,sex'
    connection = urllib.request.urlopen(url)
    data = pd.DataFrame(json.load(connection)['response']['docs'])
    #print(data.to_string())
    ID_returned.set(next(iter(data['id'])))
    DOB_returned.set(next(iter(data['DOB']))[0:10])
    gender_returned.set(next(iter(data['sex'])))
    address_returned.set(next(iter(data['address'])) + "\n" + next(iter(data['city'])).upper() + ", " + next(iter(data['state'])).upper())

    quoteDataframe = getParam(next(iter(data['id'])))
    #print(quoteDataframe.to_string())

    bronze.set(next(iter(quoteDataframe['BRONZE'])))
    silver.set(next(iter(quoteDataframe['SILVER'])))
    gold.set(next(iter(quoteDataframe['GOLD'])))
    platinum.set(next(iter(quoteDataframe['PLATINUM'])))
    purchased.set(next(iter(quoteDataframe['PURCHASED'])).upper())

    calculate_quote()



def calculate_quote():
    #use machine learning to get price
    a = age.get()
    inc = income.get()/10000
    ppl = num_people.get()
    if married == "Single":
        mar = 0
    else:
        mar=1
    if tobacco == "No":
        dip = 0
    else:
        dip =1
    if employment == "Unemployed":
        job = 0
    else:
        job =1
    if gender == "F":
        sex = 0
    else:
        sex = 1

    l = low.get()
    m = med.get()
    h = high.get()
    pred1 = algo.predict([[a, inc, ppl, mar, dip, job, sex, l, m, h]])[0]
    pred1


    bronze_quote.set("TBD")
    silver_quote.set("TBD")
    gold_quote.set(str(pred1))
    platinum_quote.set("TBD")
    plan_quote.set("TBD")


#CREATE TABS
tabControl = ttk.Notebook(win)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tab4 = ttk.Frame(tabControl)
tabControl.add(tab1, text="Request Quote")
tabControl.add(tab2, text="Search")
tabControl.add(tab3, text="Coverage Map")
tabControl.add(tab4, text="Statistics")
tabControl.pack(expand=1, fill="both")


#age, income/10000, #people on plan, marriage status (1=married), tobacco, employment status, sex, #low risk, #med risk, #high
#QUOTE TAB
quote = ttk.LabelFrame(tab1)
quote.grid(column=0,row=0,padx=8,pady=4)
ttk.Label(quote, text="Gender:").grid(column=0,row=1,sticky=tk.W)
ttk.Label(quote, text="Age:").grid(column=0,row=2,sticky=tk.W)
ttk.Label(quote, text="Number of High Risk Conditions:").grid(column=0,row=3,sticky=tk.W)
ttk.Label(quote, text="Number of Medium Risk Conditions:").grid(column=0,row=4,sticky=tk.W)
ttk.Label(quote, text="Number of Low Risk Conditions:").grid(column=0,row=5,sticky=tk.W)
ttk.Label(quote, text="Income:").grid(column=0,row=6,sticky=tk.W)
ttk.Label(quote, text="Number of People on Plan:").grid(column=0,row=7,sticky=tk.W)
ttk.Label(quote, text="Marital Status:").grid(column=0,row=8,sticky=tk.W)
ttk.Label(quote, text="Tobacce Use:").grid(column=0,row=9,sticky=tk.W)
ttk.Label(quote, text="Employment Status:").grid(column=0,row=10,sticky=tk.W)
get_quote = ttk.Button(quote, text="Get Quote", command = calculate_quote).grid(column=0,row=11,pady=20,sticky=tk.W)
ttk.Label(quote, text="Bronze Plan Quote:").grid(column=0,row=12,sticky=tk.W)
ttk.Label(quote, text="Silver Plan Quote").grid(column=0,row=13,sticky=tk.W)
ttk.Label(quote, text="Gold Plan Quote:").grid(column=0,row=14,sticky=tk.W)
ttk.Label(quote, text="Platinum Plan Quote:").grid(column=0,row=15,sticky=tk.W)
ttk.Label(quote, text="Plan Recommendation:").grid(column=0,row=16,sticky=tk.W)


gender = tk.StringVar()
age = tk.IntVar()
high = tk.IntVar()
med = tk.IntVar()
low = tk.IntVar()
income = tk.DoubleVar()
num_people = tk.IntVar()
married = tk.StringVar()
tobacco = tk.StringVar()
employment = tk.StringVar()
bronze_quote = StringVar()
silver_quote = StringVar()
gold_quote = StringVar()
platinum_quote = StringVar()
plan_quote = StringVar()


gender_entered = ttk.Combobox(quote, width=12, textvariable=gender, state='readonly')
gender_entered['values']=("F", "M")
gender_entered.grid(column=1, row=1,sticky=tk.W)
ttk.Entry(quote, width=12, textvariable=age).grid(column=1,row=2,sticky=tk.W)
num_high_cond = ttk.Entry(quote, width = 12, textvariable=high).grid(column=1, row=3,sticky=tk.W)
num_med_cond = ttk.Entry(quote, width = 12, textvariable=med).grid(column=1, row=4,sticky=tk.W)
num_low_cond = ttk.Entry(quote, width = 12, textvariable=low).grid(column=1, row=5,sticky=tk.W)
ttk.Entry(quote, width = 12, textvariable=income).grid(column=1, row=6,sticky=tk.W)
ttk.Entry(quote, width = 12, textvariable=num_people).grid(column=1, row=7,sticky=tk.W)
married_entered = ttk.Combobox(quote, width=12, textvariable=married, state='readonly')
married_entered['values']=("Single", "Married")
married_entered.grid(column=1, row=8,sticky=tk.W)
tobacco_entered = ttk.Combobox(quote, width=12, textvariable=tobacco, state='readonly')
tobacco_entered['values']=("No", "Yes")
tobacco_entered.grid(column=1, row=9,sticky=tk.W)
employ_entered = ttk.Combobox(quote, width=12, textvariable=employment, state='readonly')
employ_entered['values']=("Unemployed", "Employed")
employ_entered.grid(column=1, row=10,sticky=tk.W)
ttk.Label(quote, textvariable = bronze_quote).grid(column=1,row=12,sticky=tk.W)
ttk.Label(quote, textvariable = silver_quote).grid(column=1,row=13,sticky=tk.W)
ttk.Label(quote, textvariable = gold_quote).grid(column=1,row=14,sticky=tk.W)
ttk.Label(quote, textvariable = platinum_quote).grid(column=1,row=15,sticky=tk.W)
ttk.Label(quote, textvariable = plan_quote).grid(column=1,row=16,sticky=tk.W)


#SEARCH TAB
search = ttk.LabelFrame(tab2)
search.grid(column=0,row=0,padx=8,pady=4)


ttk.Label(search, text="Customer Last Name:").grid(column=0,row=1,sticky=tk.W)
ttk.Label(search, text="Customer First Name:").grid(column=0,row=2,sticky=tk.W)
search_button = ttk.Button(search, text="Search", command=find_customer).grid(column=0,row=3,pady=12,sticky=tk.W)
ttk.Label(search, text="Customer ID:").grid(column=0,row=4,sticky=tk.W)
ttk.Label(search, text="Date of Birth:").grid(column=0,row=5,sticky=tk.W)
ttk.Label(search, text="Gender:").grid(column=0,row=6,sticky=tk.W)
ttk.Label(search, text="Address:").grid(column=0,row=7,pady=12,sticky=tk.W)
ttk.Label(search, text="ACTUAL").grid(column=1,row=8, pady=4, sticky=tk.W)
ttk.Label(search, text="PREDICTED").grid(column=2,row=8, padx=16,pady=4, sticky=tk.W)
ttk.Label(search, text="Bronze Plan Price:").grid(column=0,row=9,sticky=tk.W)
ttk.Label(search, text="Silver Plan Price").grid(column=0,row=10,sticky=tk.W)
ttk.Label(search, text="Gold Plan Price:").grid(column=0,row=11,sticky=tk.W)
ttk.Label(search, text="Platinum Plan Price:").grid(column=0,row=12,sticky=tk.W)
ttk.Label(search, text="Purchased Plan:").grid(column=0,row=13,sticky=tk.W)

first = tk.StringVar()
last = tk.StringVar()
ID_returned = tk.StringVar()
DOB_returned = tk.StringVar()
gender_returned = tk.StringVar()
address_returned = tk.StringVar()
bronze = tk.StringVar()
silver = tk.StringVar()
gold = tk.StringVar()
platinum = tk.StringVar()
purchased = tk.StringVar()


first_entered = ttk.Entry(search, width=12, textvariable=first).grid(column=1,row=1,sticky=tk.W)
last_entered = ttk.Entry(search, width=12, textvariable=last).grid(column=1,row=2,sticky=tk.W)
ttk.Label(search, textvariable = ID_returned).grid(column=1,row=4,sticky=tk.W)
ttk.Label(search, textvariable = DOB_returned).grid(column=1,row=5,sticky=tk.W)
ttk.Label(search, textvariable = gender_returned).grid(column=1,row=6,sticky=tk.W)
ttk.Label(search, textvariable = address_returned).grid(column=1,row=7,sticky=tk.W)
ttk.Label(search, textvariable = bronze).grid(column=1,row=9,sticky=tk.W)
ttk.Label(search, textvariable = silver).grid(column=1,row=10,sticky=tk.W)
ttk.Label(search, textvariable = gold).grid(column=1,row=11,sticky=tk.W)
ttk.Label(search, textvariable = platinum).grid(column=1,row=12,sticky=tk.W)
ttk.Label(search, textvariable = purchased).grid(column=1,row=13,sticky=tk.W)

#compare quote
ttk.Label(search, textvariable = bronze_quote).grid(column=2,row=9,padx=16,sticky=tk.W)
ttk.Label(search, textvariable = silver_quote).grid(column=2,row=10,padx=16,sticky=tk.W)
ttk.Label(search, textvariable = gold_quote).grid(column=2,row=11,padx=16,sticky=tk.W)
ttk.Label(search, textvariable = platinum_quote).grid(column=2,row=12,padx=16,sticky=tk.W)
ttk.Label(search, textvariable = plan_quote).grid(column=2,row=13,padx=16,sticky=tk.W)


#MAP TAB
map_tab = ttk.LabelFrame(tab3)
map_tab.grid(column=0,row=0,padx=8,pady=4)

map_image = PhotoImage(file='/Users/Maddie/Desktop/populationdensity.gif')
map_button = ttk.Button(map_tab, image=map_image)
map_button.pack()


#STAT TAB
stats = ttk.LabelFrame(tab4)
stats.grid(column=0,row=0,padx=8,pady=4)

stat_image = PhotoImage(file='/Users/Maddie/Desktop/graph.gif')
stat_button= ttk.Button(stats, image=stat_image)
stat_button.pack()





win.mainloop()
