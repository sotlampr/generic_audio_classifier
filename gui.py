#!/usr/bin/env python2
'''Generic Audio Classifier - Graphical User Interface
'''
import os
from time import time
import pickle

import numpy as np
from matplotlib import pyplot as plt
import Tkinter
import ttk
import tkFont
import tkFileDialog
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize

import code


class MainGUI(ttk.Frame):   # pylint: disable=too-many-ancestors
    """ GUI Object

    """

    def __init__(self, parent, backend):
        """ Initializes the frame, backend, window size and centers the window
        """
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        self.backend = backend(self)
        self.init_user_interface()
        self.center_window(630, 460)

    def init_user_interface(self):
        ''' Initializes the User Interface'''
        self.grid()
        self.heading = tkFont.Font(family='Roboto', size=12, weight='bold')
        self.body = tkFont.Font(family='Roboto', size=10)

        self.parent.title("Generic Audio Classifier")
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.grid(sticky=(Tkinter.N, Tkinter.W, Tkinter.E, Tkinter.S))

        ttk.Label(
            self, text='Init', font=self.heading).grid(
                column=0, row=0)

        ttk.Button(
            self, text='Read Files', command=self.backend.run_db).grid(
                column=0, row=1)

        ttk.Label(
            self, text='Classes (subdirectories):', font=self.body).grid(
                column=0, row=2)
        ttk.Label(
            self, textvariable=self.backend.n_subdirs,
            font=self.heading).grid(column=0, row=3)

        ttk.Label(
            self, text='Files:', font=self.body).grid(
                column=0, row=4)
        ttk.Label(
            self, textvariable=self.backend.n_files,
            font=self.heading).grid(column=0, row=5)

        ttk.Separator(self, orient=Tkinter.HORIZONTAL).grid(
            column=0, row=6, sticky="ew")
        ttk.Label(
            self, text='Process', font=self.heading).grid(column=0, row=8)

        ttk.Label(
            self, text='Sample Rate:\n(Hz)', font=self.body).grid(
                column=0, row=9)

        Tkinter.Entry(
            self, textvariable=self.backend.sample_rate, width=12).grid(
                column=0, row=10)

        ttk.Button(
            self, text='Process Files',
            command=self.backend.process).grid(column=0, row=11)

        ttk.Progressbar(
            self, mode='determinate', orient=Tkinter.HORIZONTAL,
            variable=self.backend.process_progress).grid(column=0, row=12)

        ttk.Label(
            self, textvariable=self.backend.process_dur, font=self.body).grid(
                column=0, row=13)

        ttk.Label(
            self, text='Evaluation', font=self.heading).grid(column=1, row=0)

        ttk.Button(
            self, text='Evaluate Features',
            command=self.backend.evaluate).grid(column=1, row=1)

        ttk.Progressbar(
            self, mode='determinate', orient=Tkinter.HORIZONTAL,
            variable=self.backend.evaluation_progress).grid(
                column=1, row=2)

        ttk.Button(
            self, text='Run comparison',
            command=self.backend.run_eval).grid(column=1, row=3)

        ttk.Progressbar(
            self, mode='determinate', orient=Tkinter.HORIZONTAL,
            variable=self.backend.comparison_progress).grid(
                column=1, row=4)

        self.evaluation_results = Tkinter.Text(self, width=20, height=12)
        self.evaluation_results.grid(column=1, row=5, rowspan=7)
        results_scrollbar = Tkinter.Scrollbar(
            self, orient=Tkinter.VERTICAL,
            command=self.evaluation_results.yview
            )
        self.evaluation_results.config(yscrollcommand=results_scrollbar.set)
        results_scrollbar.grid(column=1, row=7, sticky=Tkinter.E)

        ttk.Label(
            self, text='Chosen Model', font=self.heading).grid(column=2, row=0)
        ttk.Label(
            self, text='Desired K:', font=self.body).grid(
                column=2, row=1, sticky=Tkinter.W)

        Tkinter.Entry(
            self, textvariable=self.backend.final_K, width=12).grid(
                column=2, row=2, sticky=Tkinter.E)

        self.backend.final_K.trace('w', self.backend.generate_model)

        ttk.Label(
            self, textvariable=self.backend.gen_progress).grid(
                column=2, row=3)

        ttk.Button(
            self, text='Save Model',
            command=self.model_save_prompt).grid(column=2, row=4)

        ttk.Label(
            self, text='Metrics', font=self.heading).grid(column=3, row=0)

        ttk.Button(
            self, text='Save\n Confusion Matrix',
            command=self.fig_save_prompt).grid(column=3, row=2)

        ttk.Button(
            self, text='Save Text Report',
            command=self.report_save_prompt).grid(column=3, row=1)

        ttk.Button(self, text='Quit',
                   command=self.quit).grid(column=3, row=99)

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=2)

        self.columnconfigure('all', minsize=140)
        self.columnconfigure(5, minsize=70)
        # self.rowconfigure('all', minsize=30)

    def center_window(self, width, height):
        """Resize and center the parent Window"""
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        x_coordinate = (screen_width - width) / 2
        y_coordinate = (screen_height - height) / 2
        self.parent.geometry("{0}x{1}+{2}+{3}".format(
            width, height, x_coordinate, y_coordinate))

    def model_save_prompt(self):
        """Save the model"""
        file_name = tkFileDialog.asksaveasfilename()
        self.backend.save(file_name)

    def fig_save_prompt(self):
        """Save the Figure"""
        fig_name = tkFileDialog.asksaveasfilename()
        self.backend.draw(fig_name)

    def report_save_prompt(self):
        """Save Classification Report"""
        report_name = tkFileDialog.asksaveasfilename()
        self.backend.txt_report(report_name)


class Backend(object):
    """Back-end support for connecting the GUI with the components"""
    def __init__(self, Main):
        """Messy...."""
        self.main = Main
        self.n_subdirs = Tkinter.StringVar()
        self.n_files = Tkinter.StringVar()
        self.file_type = Tkinter.StringVar()
        self.sample_rate = Tkinter.IntVar()
        self.base_directory = os.path.join(os.getcwd(), 'files')
        self.process_progress = Tkinter.DoubleVar()
        self.evaluation_progress = Tkinter.DoubleVar()
        self.comparison_progress = Tkinter.DoubleVar()
        self.process_dur = Tkinter.StringVar()
        self.final_K = Tkinter.IntVar()  # pylint: disable=invalid-name
        self.gen_progress = Tkinter.StringVar()
        self.save_progress = Tkinter.StringVar()
        self.audio_file = Tkinter.StringVar()
        self.categ = Tkinter.StringVar()
        self.proba = Tkinter.StringVar()

        # Dummy variables
        self.y_pred = None
        self.evaluator = None
        self.yaafe = None
        self.file_ext = None
        self.db = None    # pylint: disable=invalid-name
        self.length = None
        self.y_true = None
        self.new_plan = None

    @staticmethod
    def get_file_ext(file_type):
        """Trivial"""
        return "*.{}".format(file_type)

    def run_db(self):
        """Populate subdirectories and entries"""
        self.file_ext = self.get_file_ext(self.file_type.get())
        self.db = code.toolbox.Database(self.base_directory)    # pylint: disable=invalid-name
        self.db.populate()
        self.n_files.set(str(len(self.db.entries)))
        self.n_subdirs.set(str(len(self.db.subdirs)))

    def process(self):
        """Process files on the database"""
        start = time()
        self.yaafe = code.toolbox.Yaafe(int(self.sample_rate.get()))
        self.db.process(self.yaafe, self.process_progress, self.main)
        self.process_dur.set(
            "Took {} minutes".format(round((time() - start) / 60, 2)))

    def evaluate(self):
        """Get scoring for different Features"""
        self.evaluator = code.classification.FeatureEvaluator(
            self.db, self.yaafe
            )
        self.evaluator.evaluate(self.evaluation_progress, self.main)

    def run_eval(self):
        """Display the score for increasing features combination"""
        self.evaluator.run(12, self.main, self.comparison_progress, 3)

    def generate_model(self, *_):
        """Generate the Classifier Model (package)"""
        self.gen_progress.set("Please Wait")
        self.main.update_idletasks()
        self.y_true, self.y_pred, self.length = self.evaluator.run_k(
            self.final_K.get(), 10)
        self.new_plan = \
            self.evaluator.get_K_feature_plan(int(self.final_K.get()))
        self.gen_progress.set("Done!")

    def save(self, filename):
        """Save the package"""
        self.save_progress.set("Please wait...")
        self.main.update_idletasks()
        package = self.evaluator.packer(self.final_K.get())
        pickle.dump(package, open("{}.pickle".format(filename), 'wb'))
        self.save_progress.set("Done!")

    def draw(self, fig_name):
        """Draw and save the Confusion Matrix"""
        categories = self.db.subdirs.values()
        temp_matrix = np.array(
            confusion_matrix(self.y_true, self.y_pred)).astype(np.float64)
        conf_matrix = normalize(temp_matrix, axis=1, norm='l1')
        np.set_printoptions(precision=2)
        plt.figure(figsize=(9, 9))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greys)
        plt.colorbar()
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=45)
        plt.yticks(tick_marks, categories)
        plt.tight_layout()
        plt.subplots_adjust(left=0.175)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(fig_name)
        with open("{}.cm.txt".format(fig_name), 'wb') as text_file:
            text_file.write("Confusion Matrix\n---------\n" + str(conf_matrix))
            text_file.write("Categories\n----------\n" + str(categories))

    def txt_report(self, file_name):
        """Write The Classification Report"""
        categories = self.db.subdirs.values()
        report = classification_report(
            self.y_true, self.y_pred, target_names=categories
            )
        feats = self.new_plan.keys()
        with open("{}.txt".format(file_name), 'wb') as text_file:
            text_file.write(report)
            text_file.write("\nChosen Features are:")
            text_file.write('\n'.join(feats))
            text_file.write("\nThe combined vector length is: {}".format(self.length))


def main():
    """ Main function, load the tkinter gui """
    root = Tkinter.Tk()
    app = MainGUI(root, Backend)
    app.mainloop()
    return 0

if __name__ == '__main__':
    main()
