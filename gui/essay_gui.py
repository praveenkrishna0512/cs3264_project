# zzc: Nov 2022
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import tkinter 
from tkinter import ttk

import grade_models

win = tkinter.Tk()
wd = 800
ht = 600
grades_labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

def myclick():
    tkinter.message= "Hello "+ entry.get()
    label= tkinter.Label(frame, text= "aavv", font= ('Times New Roman', 14, 'italic'))
    entry.delete(0, 'end')
    label.pack(pady=30)

def get_data():
    frame.config(text= entry.get("1.0",'end-1c'), font= ('Helvetica 13'))

def text_processing():
    my_essay = grade_models.essay(entry.get("1.0",'end-1c'))
    #my_essay = grade_models.essay(entry.get())
    grades = list(my_essay.rand_grading())    
    for i in range(0, len(grades)):
        grades[i] = '{:.1f}'.format(grades[i])

    canv0.create_text(60, 60, text=grades[0], font=("Helvetica", 18))
    canv1.create_text(60, 60, text=grades[1], font=("Helvetica", 18))
    canv2.create_text(60, 60, text=grades[2], font=("Helvetica", 18))
    canv3.create_text(60, 60, text=grades[3], font=("Helvetica", 18))
    canv4.create_text(60, 60, text=grades[4], font=("Helvetica", 18))
    canv5.create_text(60, 60, text=grades[5], font=("Helvetica", 18))

    new_essay, err_cnt, wrong_wds, right_wds = my_essay.correct_spellings()
    if err_cnt == 0:
        entry2.insert(tkinter.END, "No wrong spellings!")
    else:
        info_to_show = "Here're the wrongly spelled words: \n"
        for wrong in wrong_wds:
            info_to_show += (wrong + ', ')
        info_to_show += '\n\n'
        info_to_show += 'And their correct spellings could probably be: '
        for rgt in right_wds:
            info_to_show += (rgt + ', ')

        entry2.insert(tkinter.END, info_to_show)

    
    
#Creates a Frame
frame = tkinter.LabelFrame(win, width= wd, height= ht)
#frame.grid()
frame.pack()
#Stop the frame from propagating the widget to be shrink or fit
frame.pack_propagate(False)

canv0 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv0.config(bg='dark goldenrod')
canv0.place(relx=1./8, rely=5./60)
lbl = canv0.create_text(60, 10, text=grades_labels[0], font=("Helvetica", 15))

canv1 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv1.config(bg='dark goldenrod')
canv1.place(relx=3./8, rely=5./60)
lbl = canv1.create_text(60, 10, text=grades_labels[1], font=("Helvetica", 15))

canv2 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv2.config(bg='dark goldenrod')
canv2.place(relx=5./8, rely=5./60)
lbl = canv2.create_text(60, 10, text=grades_labels[2], font=("Helvetica", 15))

canv3 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv3.config(bg='dark goldenrod')
canv3.place(relx=1./8, rely=45./60)
lbl = canv3.create_text(60, 10, text=grades_labels[3], font=("Helvetica", 15))

canv4 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv4.config(bg='dark goldenrod')
canv4.place(relx=3./8, rely=45./60)
lbl = canv4.create_text(60, 10, text=grades_labels[4], font=("Helvetica", 15))

canv5 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv5.config(bg='dark goldenrod')
canv5.place(relx=5./8, rely=45./60)
lbl = canv5.create_text(60, 10, text=grades_labels[5], font=("Helvetica", 15))

#Create an Entry widget in the Frame
'''
entry = ttk.Entry(frame)
entry.insert(0, "Please enter the essay here...")
entry.pack(ipady = ht/5, ipadx = wd/4)
entry.place(relx = 0.2, rely = 0.5)
'''

# The text box
textframe = tkinter.Frame(frame, height= ht/2.5, width= wd/2)
textframe.grid(row=0, column=1)
textframe.columnconfigure(0, weight=10)  
textframe.grid_propagate(False)
textframe.place(relx = 0.1, rely = 0.25)

entry = tkinter.Text(textframe, wrap=tkinter.WORD, font=("Helvetica", 12))
entry.insert(tkinter.END, "Please enter your essay here ...")
entry.grid(sticky="we")

# The box for spelling check

textframe2 = tkinter.Frame(frame, height= ht/3.5, width= wd/4)
textframe2.grid(row=0, column=1)
textframe2.columnconfigure(0, weight=10)  
textframe2.grid_propagate(False)
textframe2.place(relx = 0.7, rely = 0.3)

sp_label = tkinter.Label(frame, text="Spelling Checks", font=("Helvetica", 14))#, relief=tkinter.RAISED )
sp_label.place(relx = 0.7, rely = 0.25)
entry2 = tkinter.Text(textframe2, wrap=tkinter.WORD, font=("Helvetica", 11))
#entry2.insert(tkinter.END, "Please enter your essay here ...")
entry2.grid(sticky="we")

#Create a Button
button_fonts = tkinter.font.Font(family='Helvetica', size=17, weight=tkinter.font.BOLD)
tkinter.Button(win, text= "Get your AI graded score!", command=text_processing, foreground= "OrangeRed3", background= "White", font=button_fonts).place(relx = 0.3, rely = 0.91)
win.geometry("%dx%d" % (wd, ht))

win.mainloop()
