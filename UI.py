import tkinter as tk
from tkinter import ttk
import random
from pprint import pprint
import pickle
import nltk
nltk.download('punkt')
from Tokenizer import unfiltered_tokenize, useless_tokenize

root = tk.Tk()
style = ttk.Style(root)
style.configure('TCheckbutton', font=("Arial", 12))
style.configure('TRadiobutton', font=("Arial", 12))
root.title("Sentiment Predictor")

# [DATA PROCESS]
dp_abbv = ["BoW", "BoW L1", "TF-IDF", "TF-IDF L1"]
dp_dict = {
    "Bag of Words" : 0, 
    "Bag of Words L1" : 1,
    "TF-IDF" : 2,
    "TF-IDF L1" : 3
}

algo_abbv = ["kNN", "NB", "DT", "SVM", "LR", "RF", "SR"]
algo_dict = {
    "k-Nearest Neighbors" : 0, 
    "Naive Bayes" : 1,
    "Decision tree" : 2, 
    "Support vector machine" : 3,
    "Logistic regression (OvR)" : 4,
    "Random forest" : 5,
    "Softmax regression" : 6
}

score = [[0 for i in range(len(algo_abbv))] for j in range(len(dp_abbv))]
pipelines = list(list())
for i in range(len(dp_abbv)):
    row = list()
    for j in range(len(algo_abbv)):
        with open('data/models/' + dp_abbv[i] + '/' + algo_abbv[j] + '.pkl', 'rb') as f:
            loaded_pipeline = pickle.load(f)
            row.append(loaded_pipeline)
    pipelines.append(row)

# [USER INTERFACE BUILD]
# Used class
class CheckboxList(tk.Frame):
    def __init__(self, parent, title, options, default_option, single_selection=False):
        super().__init__(parent)
        self.single_selection = single_selection
        self.checkboxes = []
        self.selected_options = []
        
        # Variable to hold the selected option in single selection mode
        self.selected_single_option = tk.StringVar() if single_selection else None

        # Set default
        self.selected_options = default_option

        # Create and pack title label
        title_label = ttk.Label(self, text=title, font=("Arial", 16))
        title_label.pack(anchor="w")

        # Create checkboxes or radio buttons
        for option in options:
            if self.single_selection:
                radiobutton = ttk.Radiobutton(self, text=option, variable=self.selected_single_option, value=option, command=self.on_radiobutton_clicked)
                radiobutton.pack(anchor="w")
                self.checkboxes.append(radiobutton)
            else:
                var = tk.StringVar()
                var.set(option in default_option)

                checkbox = ttk.Checkbutton(self, text=option, variable=var, command=self.on_checkbox_clicked)
                checkbox.var = var
                checkbox.pack(anchor="w")
                self.checkboxes.append(checkbox)

        if self.single_selection == True:
            self.selected_single_option.set(default_option)

    def on_checkbox_clicked(self):
        self.selected_options = [checkbox.cget("text") for checkbox in self.checkboxes if checkbox.var.get() == '1']
        print("Selected options:", self.selected_options)

    def on_radiobutton_clicked(self):
        self.selected_options = [self.selected_single_option.get()]
        print("Selected option:", self.selected_options)

class Table:
    def __init__(self, root, data, font=("Arial, 12")):
        self.root = root
        self.frame = tk.Frame(root, borderwidth=2, relief=tk.SUNKEN)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.data = data
        self.font = font
        self.cells = []

    def create_table(self, cell_colors=None):
        if cell_colors is None:
            cell_colors = {}

        rows = len(self.data)
        cols = len(self.data[0])

        # Clear previous cells
        for cell_row in self.cells:
            for cell in cell_row:
                cell.destroy()

        self.cells = []

        for i in range(rows):
            cell_row = []
            for j in range(cols):
                cell = tk.Label(self.frame, text=self.data[i][j], borderwidth=1, relief='solid', padx=5, pady=5, font=self.font)

                # Apply background color if specified
                if (i, j) in cell_colors:
                    cell.configure(bg=cell_colors[(i, j)])

                cell.grid(row=i, column=j, sticky='nsew')
                cell_row.append(cell)
            self.cells.append(cell_row)

        # Make the table cells resize with the window
        for i in range(rows):
            self.frame.grid_rowconfigure(i, weight=1)
        for j in range(cols):
            self.frame.grid_columnconfigure(j, weight=1)

    def update_data(self, new_data, cell_colors=None):
        self.data = new_data
        self.create_table(cell_colors)


# [CREATE COLUMN 1]
# Create data process method list
frame_1 = tk.Frame(root, borderwidth=0, relief=tk.SUNKEN)
frame_1.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

options_1 = ["Bag of Words", "Bag of Words L1", "TF-IDF", "TF-IDF L1"]
checkbox_list_1 = CheckboxList(frame_1, "Data process method:", options_1, options_1)
checkbox_list_1.grid(row=0, column=0, sticky='nsew')

# Create algorithm list
options_2 = ["k-Nearest Neighbors", 
             "Naive Bayes",
             "Decision tree", 
             "Support vector machine",
             "Logistic regression (OvR)",
             "Random forest",
             "Softmax regression"]
checkbox_list_2 = CheckboxList(frame_1, "Algorithm:", options_2, options_2)
checkbox_list_2.grid(row=1, column=0, pady=30, sticky='nsew')




# [CREATE COLUMN 2]
# Create a Frame widget
frame_2 = tk.Frame(root, borderwidth=0, relief=tk.SUNKEN)
frame_2.grid(row=0, column=1, sticky='nsew', padx=5)

# Create input frame
input_frame = tk.Frame(frame_2, borderwidth=3)
input_frame.pack()

textbox = tk.Text(input_frame, wrap=tk.WORD, font=("Arial", 20))
textbox.configure(height=6, width=40)
textbox.grid(row=0, column=0)

label = ["None", "Anger", "Joy", "Sadness", "Love", "Fear", "Surprise"]
label_choice = CheckboxList(input_frame, "True label", label, ["None"], True)
label_choice.grid(row=0, column=1, padx=10)

# Create result table
result_table = Table(frame_2, None)

# Create button predict
predict_button = tk.Button(frame_2, text="Predict", font=("Arial", 15), width=10)
predict_button.pack()



# [CREATE COLUMN 3]
frame_3 = tk.Frame(root, borderwidth=0, relief=tk.SUNKEN)
frame_3.grid(row=0, column=3, sticky='nsew')

def get_all_score():
    def get_score(row, column):
        return [algo_abbv[column] + '-' + dp_abbv[row]  + ": ", str(score[row][column])]
    
    index = [(i, j) for j in range(len(algo_abbv)) for i in range(len(dp_abbv))]
    index.sort(key= lambda t : score[t[0]][t[1]], reverse=True)

    res = []
    for i, j in index:
        res.append(get_score(i, j))

    return res

def on_configure(event):
    # Update the scroll region to encompass the inner frame
    canvas.configure(scrollregion=canvas.bbox("all"))

table_frame = tk.Frame(frame_3)
table_frame.pack()

canvas = tk.Canvas(table_frame, width=180)
canvas.grid(row=0, column=0)

# Create a scrollbar and attach it to the canvas
scrollbar = tk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame to contain the scrollable content
scrollable_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Bind the frame to the canvas so it expands properly
scrollable_frame.bind("<Configure>", on_configure)

print(get_all_score())
result_dashboard = Table(scrollable_frame, get_all_score())
result_dashboard.create_table()

def reset():
    for i in range(len(score)):
        for j in range(len(score[j])):
            score[i][j] = 0

    result_table.update_data(score)

reset_button = tk.Button(frame_3, text="Reset", font=("Arial", 15), width=10)
reset_button.pack(pady=20)

# [SET FUNCTIONALITY]
def predict():
    sdp_index = [dp_dict[sdp] for sdp in checkbox_list_1.selected_options]
    salgo_index = [algo_dict[sa] for sa in checkbox_list_2.selected_options]

    data = [[""] + [algo_abbv[i] for i in range(len(algo_abbv)) if i in salgo_index]]
    color = {}
    true_label = label_choice.selected_options[0]
    
    for i in sdp_index:
        row = [dp_abbv[i]]
        for j in salgo_index:
            result = pipelines[i][j].predict([textbox.get("1.0", tk.END)])[0].capitalize()
            score[i][j] += int(result == true_label)
            row.append(result)
                
        data.append(row)

    # Color up
    for i in range(1, len(data)):
        for j in range(1, len(data[i])):
            if true_label != "None":
                color[(i, j)] = "#DAFFD5" if data[i][j] == true_label else "#FFEAEB" 

    if (len(sdp_index) > 0 and len(salgo_index) > 0):
        result_table.update_data(data, color)
        result_dashboard.update_data(get_all_score())

predict_button.config(command=predict)


# Run the main event loop
root.mainloop()
