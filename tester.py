# /usr/bin/env python2
""" GUI to test new files against a previously saved model """

import pickle

import Tkinter
import ttk
import tkFont
import tkFileDialog

import code


class MainGUI(ttk.Frame):
    """ Main Graphical User Interface class, using Tkinter """

    def __init__(self, parent, backend):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        self.backend = backend(self)
        self.init_ui()   # pylint: disable=invalid-name
        self.center_window(280, 220)
        self.audio_file = None

    def init_ui(self):
        """ Initialize UI layout """
        self.grid()
        self.heading = tkFont.Font(family='Roboto', size=12, weight='bold')
        self.body = tkFont.Font(family='Roboto', size=10)

        self.parent.title("Audio Classifier - Tester")
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.grid(sticky=(Tkinter.N, Tkinter.W, Tkinter.E, Tkinter.S))

        ttk.Label(
            self, text='Init', font=self.heading).grid(
                column=0, row=0)

        ttk.Label(
            self, text='Sample Rate:', font=self.body).grid(
                column=0, row=1, sticky=Tkinter.W)

        Tkinter.Entry(
            self, textvariable=self.backend.sample_rate, width=12).grid(
                column=0, row=2, sticky=Tkinter.E)

        ttk.Button(
            self, text='Load Model', command=self.load_model).grid(
                column=0, row=3)

        ttk.Button(
            self, text='Load Audio File', command=self.set_audio_file).grid(
                column=0, row=4)

        ttk.Button(
            self, text='Classify',
            command=self.classify).grid(
                column=0, row=5)

        ttk.Label(
            self, text='Results', font=self.heading).grid(
                column=1, row=0)
        ttk.Label(
            self, textvariable=self.backend.categ, font=self.heading).grid(
                column=1, row=2)
        ttk.Label(
            self, textvariable=self.backend.proba, font=self.heading).grid(
                column=1, row=3)

        quit_button = ttk.Button(self, text='Quit',
                                 command=self.quit)
        quit_button.grid(column=1, row=99)

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=2)

        self.columnconfigure('all', minsize=140)
        self.columnconfigure(5, minsize=70)

    def load_model(self):
        """ Call the backend function to load a model """
        model_name = tkFileDialog.askopenfilename()
        self.backend.load_package(model_name)

    def set_audio_file(self):
        """ Choose an audio file for prediction """
        self.audio_file = tkFileDialog.askopenfilename()

    def classify(self):
        """ Call the backend function for classification """
        self.backend.classify(self.audio_file)

    def center_window(self, width, height):
        """ Center the window """
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        target_width = (screen_width - width)/2
        target_height = (screen_height - height)/2
        self.parent.geometry("{0}x{1}+{2}+{3}".format(
            width, height, target_width, target_height))


class Backend(object):
    """ Operations for the GUI """
    def __init__(self, main_gui):
        """ Initialize some basic variables """
        self.Main = main_gui
        self.sample_rate = Tkinter.StringVar()
        self.audio_file = Tkinter.StringVar()
        self.categ = Tkinter.StringVar()
        self.proba = Tkinter.StringVar()
        self.package = None

    def load_package(self, filename):
        """ Load a pickled model """
        self.package = pickle.load(open(filename, 'rb'))

    def classify(self, audiofile):
        """ Predict target from given audio file """
        _, proba, label = code.classification.query(
            audiofile, int(self.sample_rate.get()), self.package
            )
        self.categ.set(label.capitalize())
        self.proba.set(proba)


def main():
    """ Run the gui """
    root = Tkinter.Tk()
    app = MainGUI(root, Backend)
    app.mainloop()
    return 0


if __name__ == '__main__':
    main()
