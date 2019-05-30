from tkinter import *
from tkinter import filedialog
import tkinter as tk

from tkinter.ttk import *



def chose_file():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return root.filename


if __name__ == '__main__':

    root = tk.Tk()
    root.title("Data Mining Project")
    canvas = tk.Canvas(root, background="white")
    button_frame = tk.Frame(root)

    button_frame.pack(side="bottom", fill="x", expand=False)
    canvas.pack(side="top", fill="both", expand=True)

    mavg_btn = tk.Button(button_frame, text="Moving Average")
    simar_btn = tk.Button(button_frame, text="Similar Stocks")
    patter_btn = tk.Button(button_frame, text="Find Patterns")
    dgim_btn = tk.Button(button_frame, text="DGIM")
    button_frame.grid_columnconfigure(0, weight=1)
    mavg_btn.grid(row=0, column=1, sticky="ew")
    simar_btn.grid(row=0, column=2, sticky="ew")
    patter_btn.grid(row=1, column=1, sticky="ew")
    dgim_btn.grid(row=1, column=2, sticky="ew")
    root.mainloop()
# if __name__ == '__main__':
#     window = Tk()
#
#     window.title("Welcome to LikeGeeks app")
#     window.geometry("300x200+300+300")
#
#     selected = IntVar()
#
#     rad1 = Radiobutton(window, text='First', value=1, variable=selected)
#
#     rad2 = Radiobutton(window, text='Second', value=2, variable=selected)
#
#     rad3 = Radiobutton(window, text='Third', value=3, variable=selected)
#     mabtn = Button(window, text="Calculate Moving Average", command=clicked)
#
#     rad1.grid(column=0, row=0)
#
#     rad2.grid(column=1, row=0)
#
#     rad3.grid(column=2, row=0)
#
#     mabtn.grid(column=3, row=0)
#
#     window.mainloop()