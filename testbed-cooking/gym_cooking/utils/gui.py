from tkinter import *

def popup_text(description: str = '', fill_in: str = '') -> str | None:
    root = Tk()
    # root.geometry("300x300")
    root.title("Input Window")

    ret = None

    def take_input():
        i = inputtxt.get("1.0", "end-1c")
        nonlocal ret
        ret = i
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 15, 'bold'))
    inputtxt = Text(root, height=15, width=30, bg="light yellow", font=('Times New Roman', 15, 'bold'))
    inputtxt.insert("end-1c", fill_in)
    display = Button(root, height=2, width=5, text="Submit", command=lambda: take_input())

    label.pack()
    inputtxt.pack()
    display.pack()

    mainloop()

    return ret


def popup_choice(description: str, choices: list[str]) -> str | None:
    root = Tk()
    root.geometry("300x200")
    root.title("Choose Window")

    ret = None

    def take_input():
        nonlocal ret
        ret = var.get()
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 15, 'bold'))
    var = StringVar(root)
    var.set(choices[0])
    option = OptionMenu(root, var, *choices)
    display = Button(root, height=2, width=5, text="Submit", command=lambda: take_input())

    label.pack()
    option.pack(expand=True)
    display.pack()

    mainloop()

    return ret

def popup_box(description: str = '') -> bool | None:
    root = Tk()
    # root.geometry("300x300")
    root.title("Box Window")

    ret = None

    def take_yes():
        nonlocal ret
        ret = True
        root.destroy()

    def take_no():
        nonlocal ret
        ret = False
        root.destroy()

    label = Label(text=description, font=('Times New Roman', 15, 'bold'))
    display_yes = Button(root, height=2, width=5, text="Yes", command=lambda: take_yes())
    display_no = Button(root, height=2, width=5, text="No", command=lambda: take_no())

    label.pack()
    display_yes.pack()
    display_no.pack()

    mainloop()

    return ret

if __name__ == '__main__':
    print(popup_text("test", "test"))
    print(popup_choice("tes11111t", ["test1", "test2"]))
    print(popup_box("tes22222"))

