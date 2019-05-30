from tkinter import *
from tkinter import scrolledtext, filedialog
import PIL.Image

from PIL import ImageTk

from function import *

text = '10'
process_running = '\n\n\n\n\t\t\t    Running\n'

def chose_file(title):
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title=title,
                                               filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    ret = root.filename
    root.destroy()
    return ret


def chose_directory(title):
    root = Tk()
    root.title('pics')
    root.filename = filedialog.askdirectory(initialdir=os.getcwd(), title=title)
    ret = root.filename
    root.destroy()
    return ret + '/'


def showImg(name):
    root = Toplevel()
    root.title('pics')
    root.geometry('600x600')
    load = PIL.Image.open(name)
    render = ImageTk.PhotoImage(load)
    img = Label(root, image=render)
    img.image = render
    img.place(x=0, y=0)


def read_val(title):
    root = Tk()
    root.title(title)
    root.geometry('400x100')
    e1 = Entry(root, width=10, font=('Calibri', 15))
    e1.pack(anchor=N)

    def clicked2():
        global text
        text = e1.get()
        root.destroy()
        return

    btn = Button(root, text='OK', width=10, command=clicked2, font=('Calibri', 15))
    btn.pack(anchor=N)


if __name__ == '__main__':

    window = Tk()
    window.title('骆老师最帅')
    window.geometry('600x600')

    bg = PIL.Image.open('bg.jpg')
    render = ImageTk.PhotoImage(bg)
    img = Label(window, image=render)
    img.image = render
    img.place(x=0, y=0)

    options = [('Basic Analysis', 1), ('Moving Average', 2), ('Similar Stocks', 3), ('Compare Stocks', 4),
               ('Get Rules', 5), ('Query Patterns', 6), ('DGIM', 7)]

    selected = IntVar()
    selected.set(1)
    i = 0
    fm1 = Frame()
    fm1.pack(side = TOP)
    fm2 = Frame(fm1)
    fm2.pack(side=LEFT)
    fm3 = Frame(fm1)
    fm3.pack(side=RIGHT)
    for algo, num in options:
        if i > 3:
            Radiobutton(fm1, text=algo, variable=selected, value=num, font=('Calibri', 15)).pack(anchor=W)
        else:
            Radiobutton(fm2, text=algo, variable=selected, value=num, font=('Calibri', 15)).pack(anchor=W)
        i += 1

    txt = scrolledtext.ScrolledText(window, width=600, height=40, font=('Calibri', 15))


    def clicked():
        global text
        chosen = selected.get()
        if chosen == 1:
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path = chose_file('Choose target stock')
            ret = fun_basic_analysis(path)
            txt.delete(1.0, END)
            txt.insert(INSERT, ret)
        if chosen == 2:
            read_val('Please give the win size')
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path = chose_file('Choose target stock')
            name1, name2 = mavg_default(path, int(text))
            showImg(name1)
            showImg(name2)
            txt.delete(1.0, END)
            txt.insert(INSERT, 'Finish Analysis Moving Average')
        if chosen == 3:
            read_val('How many stocks do you want?')
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path = chose_file('Chose target stock file')
            fp = chose_directory('Choose compare set directory')
            ret = fun_knn(path, fp, int(text))
            txt.delete(1.0, END)
            txt.insert(INSERT, ret)
        if chosen == 4:
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path1 = chose_file('Choose the first stock')
            path2 = chose_file('Choose the second stock')
            ret, name = func_compare(path1, path2)
            txt.delete(1.0, END)
            txt.insert(INSERT, ret)
            showImg(name)
        if chosen == 5:
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path1 = chose_directory('Choose the directory')
            ret1 = fun_associate_rules(path1)
            txt.delete(1.0, END)
            txt.insert(INSERT, ret1)
        if chosen == 6:
            read_val('Please give the pattern you want to query')
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path = chose_directory('Chose the direcotry you wan to query')
            ret1 = fun_query_pattern(path, text)
            txt.delete(1.0, END)
            txt.insert(INSERT, ret1)
        if chosen == 7:
            txt.delete(1.0, END)
            txt.insert(INSERT, process_running)
            path = chose_directory('Chose the directory you wan to query')
            ret1 = DGIM(path)
            txt.delete(1.0, END)
            txt.insert(INSERT, ret1)


    btn_run = Button(window, text='Run', width=10, command=clicked, font=('Calibri', 15))
    btn_run.pack(anchor=N)

    txt.pack(anchor=N)

    window.mainloop()
