from tkinter import *
import os
from PIL import ImageTk, Image


window = Tk()


window.title("Computational practicum")
window.geometry('1285x970')

lbl0 = Label(window, text="Computational practicum. \n Margarita Peregudova BS18-05 \n \n \n ",
             font=("Arial Bold", 20))
lbl0.grid(column=0, columnspan=4, row=0)


lbl1 = Label(window, text="x0 = ")
lbl1.grid(column=0, row=1)

txt1 = Entry(window, width=10)
txt1.grid(column=1, row=1)


lbl2 = Label(window, text="y0 = ")
lbl2.grid(column=0, row=2)

txt2 = Entry(window, width=10)
txt2.grid(column=1, row=2)


lbl3 = Label(window, text="X = ")
lbl3.grid(column=0, row=3)

txt3 = Entry(window, width=10)
txt3.grid(column=1, row=3)


lbl4 = Label(window, text="N = ")
lbl4.grid(column=0, row=4)

txt4 = Entry(window, width=10)
txt4.grid(column=1, row=4)


lbl5 = Label(window, text="Number of interval for the error:")
lbl5.grid(columnspan=4, row=5)


lbl6 = Label(window, text="from")
lbl6.grid(column=0, row=6)

txt6 = Entry(window, width=10)
txt6.grid(column=1, row=6)

lbl7 = Label(window, text="to")
lbl7.grid(column=2, row=6)

txt7 = Entry(window, width=10)
txt7.grid(column=3, row=6)

first = Image.open("exact.png")
exact = ImageTk.PhotoImage(first)
label1 = Label(image=exact)
label1.image = exact
label1.grid(column=4, row=0, rowspan=8)

second = Image.open("errors.png")
errors = ImageTk.PhotoImage(second)
label2 = Label(image=errors)
label2.image = errors
label2.grid(column=0, columnspan=4, row=8)

third = Image.open("interval.png")
interval = ImageTk.PhotoImage(third)
label3 = Label(image=interval)
label3.image = interval
label3.grid(column=4, row=8)


def clicked():
    x0 = int(txt1.get())
    y0 = int(txt2.get())
    b = int(txt3.get())
    n = int(txt4.get())
    initial_number_of_interval = int(txt6.get())
    finite_number_if_interval = int(txt7.get())

    os.system('python3 plots-update.py ' + str(x0) + ' ' + str(y0) + ' ' + str(b) + ' ' + str(n) +
              ' ' + str(initial_number_of_interval) + ' ' + str(finite_number_if_interval))
    
    first = Image.open("exact.png")
    exact = ImageTk.PhotoImage(first)
    label1 = Label(image=exact)
    label1.image = exact
    label1.grid(column=4, row=0, rowspan=8)
    
    second = Image.open("errors.png")
    errors = ImageTk.PhotoImage(second)
    label2 = Label(image=errors)
    label2.image = errors
    label2.grid(column=0, columnspan=4, row=8)

    third = Image.open("interval.png")
    interval = ImageTk.PhotoImage(third)
    label3 = Label(image=interval)
    label3.image = interval
    label3.grid(column=4, row=8)

    #lbl.configure(text=res)


btn = Button(window, text="  Run  ", command=clicked , font=("Arial Bold", 20))
btn.grid(column=2, columnspan=2, row=7)


window.mainloop()
