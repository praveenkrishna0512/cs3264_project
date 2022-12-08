# zzc: Nov 2022
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import tkinter 
from tkinter import ttk
from tkinter import scrolledtext
import grade_models
from PIL import ImageTk, Image  

win = tkinter.Tk()
win.title("hickory's essay grader")
win.geometry("%dx%d" % (1200, 900))
win.update_idletasks() 

grades_labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
clr_ls = ['dark goldenrod', 'light coral', 'LightCyan3', 'purple1', 'aquamarine2', 'lime green']
def myclick():
    tkinter.message= "Hello "+ entry.get()
    label= tkinter.Label(frame, text= "aavv", font= ('Times New Roman', 14, 'italic'))
    entry.delete(0, 'end')
    label.pack(pady=30)

def get_data():
    frame.config(text= entry.get("1.0",'end-1c'), font= ('Helvetica 13'))

def text_processing():
    # clear the output first
    entry2.delete('1.0', tkinter.END)
    canv0.delete("all")
    canv1.delete("all")
    canv2.delete("all")
    canv3.delete("all")
    canv4.delete("all")
    canv5.delete("all")
    canv0.create_text(75, 15, text=grades_labels[0], font=("Helvetica", 17))
    canv1.create_text(75, 15, text=grades_labels[1], font=("Helvetica", 17))
    canv2.create_text(75, 15, text=grades_labels[2], font=("Helvetica", 17))
    canv3.create_text(75, 15, text=grades_labels[3], font=("Helvetica", 17))
    canv4.create_text(75, 15, text=grades_labels[4], font=("Helvetica", 17))
    canv5.create_text(75, 15, text=grades_labels[5], font=("Helvetica", 17))

    my_essay = grade_models.essay(entry.get("1.0",'end-1c'))
    #my_essay = grade_models.essay(entry.get())
    grades = list(my_essay.svm_grading())    
    for i in range(0, len(grades)):
        grades[i] = '{:.1f}'.format(grades[i])

    g0 = canv0.create_text(75, 60, text=grades[0], font=("Helvetica", 22))
    g1 = canv1.create_text(75, 60, text=grades[1], font=("Helvetica", 22))
    g2 = canv2.create_text(75, 60, text=grades[2], font=("Helvetica", 22))
    g3 = canv3.create_text(75, 60, text=grades[3], font=("Helvetica", 22))
    g4 = canv4.create_text(75, 60, text=grades[4], font=("Helvetica", 22))
    g5 = canv5.create_text(75, 60, text=grades[5], font=("Helvetica", 22))

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
wd = win.winfo_width()
ht = win.winfo_height()
frame = tkinter.LabelFrame(win, width= wd, height= ht)
#frame.grid(sticky="we")
frame.pack(fill="both", expand=1)
frame.pack_propagate(False)

canv0 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv0.config(bg=clr_ls[0])
canv0.place(relx=1./8, rely=8./60)
lbl = canv0.create_text(75, 15, text=grades_labels[0], font=("Helvetica", 17))

canv1 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv1.config(bg=clr_ls[1])
canv1.place(relx=3.5/8, rely=8./60)
lbl = canv1.create_text(75, 15, text=grades_labels[1], font=("Helvetica", 17))

canv2 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv2.config(bg=clr_ls[2])
canv2.place(relx=6./8, rely=8./60)
lbl = canv2.create_text(75, 15, text=grades_labels[2], font=("Helvetica", 17))

canv3 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv3.config(bg=clr_ls[3])
canv3.place(relx=1./8, rely=45./60)
lbl = canv3.create_text(75, 15, text=grades_labels[3], font=("Helvetica", 17))

canv4 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv4.config(bg=clr_ls[4])
canv4.place(relx=3.5/8, rely=45./60)
lbl = canv4.create_text(75, 15, text=grades_labels[4], font=("Helvetica", 17))

canv5 = tkinter.Canvas(win, width=wd/7, height=ht/7)
canv5.config(bg=clr_ls[5])
canv5.place(relx=6./8, rely=45./60)
lbl = canv5.create_text(75, 15, text=grades_labels[5], font=("Helvetica", 17))

g0 = canv0.create_text(75, 60, text="", font=("Helvetica", 22))
g1 = canv1.create_text(75, 60, text="", font=("Helvetica", 22))
g2 = canv2.create_text(75, 60, text="", font=("Helvetica", 22))
g3 = canv3.create_text(75, 60, text="", font=("Helvetica", 22))
g4 = canv4.create_text(75, 60, text="", font=("Helvetica", 22))
g5 = canv5.create_text(75, 60, text="", font=("Helvetica", 22))

# The text box
textframe = tkinter.Frame(frame, height= ht/2, width= wd/2.5)
textframe.grid(row=0, column=1)
textframe.columnconfigure(0, weight=1)  
textframe.grid_propagate(False)
textframe.place(relx = 0.25, rely = 0.35)

entry = scrolledtext.ScrolledText(textframe, width=50, height=18, font=("Helvetica", 12))
entry.insert(tkinter.END, "Please enter your essay here ...")
entry.grid(row=0, column=0, sticky="news")

# The box for spelling check

textframe2 = tkinter.Frame(frame, height= ht/3, width= wd/4)
textframe2.grid(row=0, column=1)
textframe2.columnconfigure(0, weight=10)  
textframe2.grid_propagate(False)
textframe2.place(relx = 0.7, rely = 0.4)

sp_label = tkinter.Label(frame, text="Spelling Checks", font=("Helvetica", 14))#, relief=tkinter.RAISED )
sp_label.place(relx = 0.7, rely = 0.35)
entry2 = tkinter.Text(textframe2, width=40, height=15, font=("Helvetica", 11))
entry2.grid(sticky="we")

title = tkinter.Label(frame, text="English Essay Auto Grader", fg='SteelBlue4', font=("Times New Roman", 30))#, relief=tkinter.RAISED )
title.place(relx = 0.3, rely = 0.03)


#Create a Button
button_fonts = tkinter.font.Font(family='Helvetica', size=19, weight=tkinter.font.BOLD)
tkinter.Button(win, text= "Get your AI graded score!", command=text_processing, foreground= "OrangeRed3", background= "White", font=button_fonts).place(relx = 0.4, rely = 0.91)

image1 = Image.open("hemingway2.png")
image1 = image1.resize((250, 350), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)
fig = tkinter.Label(frame, image=test)
fig.pack()
fig.place(relx=0.01, rely=0.3)

win.resizable(False, False) 
win.mainloop()
