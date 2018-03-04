import tkinter as tk
from tkinter import *
from tkinter import ttk
import numpy as np

win = tk.Tk()

win.title("Bingo Was His Name-O")

cards = []
tabs = []

def new_card():
    a = np.arange(1, 15)
    b = np.arange(16, 30)
    c = np.arange(31, 45)
    d = np.arange(46, 60)
    e = np.arange(61, 75)

    np.random.shuffle(a)
    np.random.shuffle(b)
    np.random.shuffle(c)
    np.random.shuffle(d)
    np.random.shuffle(e)

    card = [[a[0], b[0], c[0], d[0], e[0]],
            [a[1], b[1], c[1], d[1], e[1]],
            [a[2], b[2], 0, d[2], e[2]],
            [a[3], b[3], c[3], d[3], e[3]],
            [a[4], b[4], c[4], d[4], e[4]]]

    global cards
    cards.append(card)

    card_count = len(cards)
    card_tab = "card" + str(card_count)
    card_name = "Card " + str(card_count)
    card_tab = ttk.Frame(tabControl)
    tabControl.add(card_tab,text=card_name)

    global tabs
    tabs.append(card_tab)

    #create card numbers
    for i in range(5):
        for j in range(5):
            ttk.Label(card_tab, text=str(card[i][j])).grid(column=i, row=j)

def number():
    for k in range(len(cards)):
        current_card = cards[k]
        # card_tab = "card" + str(k+1)
        # card_tab = ttk.Frame(tabControl)
        for i in range(0, 5):
            for j in range(0, 5):
                if current_card[i][j] == num.get():
                    current_card[i][j] = 0
                    ttk.Label(tabs[k+1], text="X").grid(column=i, row=j)
                    has_bingo = bingo(current_card)
                    if(has_bingo == 1):
                        ttk.Label(tabs[0], text="BINGO: Card " + str(k+1)).grid(column=0, row=3)


def bingo(card):
    b = 0
    b_row = np.zeros(5)
    for i in range(0, 5):
        for j in range(0, 5):
            if card[i][j] == 0:
                b_row[i] = b_row[i] + 1

    b_col = np.zeros(5)
    for j in range(0, 5):
        for i in range(0, 5):
            if card[i][j] == 0:
                b_col[j] = b_col[j] + 1

    b_dia = 0
    for i in range(0, 5):
        if card[i][i] == 0:
            b_dia = b_dia + 1

    b_dia1 = 0
    for i in range(0, 5):
        if card[i][4 - i] == 0:
            b_dia1 = b_dia1 + 1

    b_cor = 1
    if card[0][0] == 0:
        b_cor = b_cor + 1
    if card[4][0] == 0:
        b_cor = b_cor + 1
    if card[0][4] == 0:
        b_cor = b_cor + 1
    if card[4][4] == 0:
        b_cor = b_cor + 1

    for i in range(0, 5):
        if b_row[i] == 5:
            b = 1
        if b_col[i] == 5:
            b = 1
    if b_dia == 5:
        b = 1
    if b_dia1 == 5:
        b = 1
    if b_cor == 5:
        b = 1

    return b





tabControl = ttk.Notebook(win)
controls = ttk.Frame(tabControl)
graph = ttk.Frame(tabControl)
tabControl.add(controls, text="Main Menu")
tabControl.pack(expand=1, fill="both")


#control panel menu
control_menu = ttk.LabelFrame(controls)
control_menu.grid(column=0,row=0,padx=8,pady=4)
button_frame = ttk.LabelFrame(controls)
button_frame.grid(column=0,row=1,padx=8,pady=4)
num = tk.IntVar()
ttk.Label(control_menu, text="Enter number:").grid(column=0,row=0)
ttk.Entry(control_menu, width =12, textvariable=num).grid(column=0, row=1)
ttk.Button(control_menu,text="Check Cards",command=number).grid(column=0,row=2)
ttk.Button(button_frame,text="Add New Card",command=new_card).grid(column=0,row=0)

tabs.append(control_menu)

win.mainloop()