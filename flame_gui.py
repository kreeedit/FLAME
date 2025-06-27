import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import queue
import sys
from flame import Flame, DEFAULT_PARAMS, SimilarityVisualizer


class FlameGUI(tk.Tk):
    """
    A Tkinter-based graphical user interface for the FLAME analysis tool.

    This class creates a window with input fields for all analysis parameters,
    a button to start the analysis, and a log window to display progress
    and results, making the tool accessible without using the command line.
    """
    def __init__(self):
        """Initializes the main application window and its components."""
        super().__init__()
        self.title("FLAME - Formulaic Language Analysis in Medieval Expressions")
        self.geometry("900x750")

        # Create Tkinter StringVars to hold the values of the parameters from the GUI.
        # This allows easy access and modification of GUI input values.
        self.params = {key: tk.StringVar(value=str(val)) for key, val in DEFAULT_PARAMS.items()}

        # Build the user interface.
        self.create_widgets()

        # A queue is used for thread-safe communication between the analysis thread
        # and the main GUI thread. Print statements will be put into this queue.
        self.log_queue = queue.Queue()
        # Periodically check the queue for new messages to display.
        self.after(100, self.process_log_queue)

    def create_widgets(self):
        """Creates and arranges all the widgets in the main window."""
        # Use a main frame with padding for better aesthetics.
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Section 1: A frame for all input parameters.
        params_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding="10")
        params_frame.pack(fill=tk.X, expand=True, side=tk.TOP)

        # --- File Path Inputs ---
        self.create_path_entry(params_frame, "input_path", "Primary Corpus Path:", 0)
        self.create_path_entry(params_frame, "input_path2", "Secondary Corpus Path (Optional):", 1)

        # --- Core Algorithm Parameter Inputs ---
        self.create_param_entry(params_frame, "ngram", "N-gram size:", 2, 0)
        self.create_param_entry(params_frame, "n_out", "N-out size:", 2, 2)
        self.create_param_entry(params_frame, "min_text_length", "Min. Text Length:", 3, 0)

        # --- Threshold Input ---
        ttk.Label(params_frame, text="Similarity Threshold:").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        threshold_entry = ttk.Entry(params_frame, textvariable=self.params['similarity_threshold'], width=5)
        threshold_entry.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)

        # Section 2: The main button to start the analysis.
        self.run_button = ttk.Button(main_frame, text="Run Analysis", command=self.start_analysis_thread)
        self.run_button.pack(pady=10)

        # Section 3: A text area for logging output.
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        # The log should be read-only for the user.
        self.log_text.configure(state='disabled')

    def create_path_entry(self, parent, param_name, label_text, row):
        """Helper function to create a labeled entry field for a directory path with a 'Browse' button."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=self.params[param_name], width=60)
        entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        browse_button = ttk.Button(parent, text="Browse...", command=lambda: self.browse_directory(param_name))
        browse_button.grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)

    def create_param_entry(self, parent, param_name, label_text, row, col):
        """Helper function to create a standard labeled entry field for a parameter."""
        ttk.Label(parent, text=label_text).grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=self.params[param_name], width=5)
        entry.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)

    def browse_directory(self, param_name):
        """Opens a system dialog to select a directory and updates the corresponding entry field."""
        directory = filedialog.askdirectory(title="Select a Folder")
        if directory:
            self.params[param_name].set(directory)

    def start_analysis_thread(self):
        """
        Starts the main analysis process in a separate thread to prevent the GUI from freezing.
        """
        # Disable the run button to prevent multiple simultaneous runs.
        self.run_button.config(state="disabled", text="Analysis in Progress...")
        # Clear the log window for the new run.
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

        # Create and start a new thread that will execute the `run_analysis` method.
        # `daemon=True` ensures the thread will close when the main window is closed.
        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()

    def run_analysis(self):
        """
        This is the main worker function. It collects parameters from the GUI,
        runs the entire FLAME analysis pipeline, and redirects print statements
        to the GUI log window.
        """
        try:
            # Convert GUI string parameters to their correct types (int, float, etc.).
            args_for_flame = {}
            for key, var in self.params.items():
                val = var.get()
                if key in ['keep_texts', 'ngram', 'n_out', 'min_text_length', 'char_norm_min_freq']:
                    args_for_flame[key] = int(val)
                elif key == 'similarity_threshold' and val.lower() != 'auto':
                    args_for_flame[key] = float(val)
                else:
                    args_for_flame[key] = val

            # Initialize the main analysis class with the parameters from the GUI.
            analyzer = Flame(params=args_for_flame)

            # --- Redirect stdout/stderr to the GUI log window ---
            # This is a trick to capture all `print()` statements from the engine
            # and display them in the GUI instead of the console.
            sys.stdout = self
            sys.stderr = self

            # --- Execute the full analysis pipeline ---
            # This logic is identical to the original `main()` function.
            if (analyzer.args.ngram - analyzer.args.n_out) < 1:
                raise ValueError(f"N-gram size ({analyzer.args.ngram}) minus n-out ({analyzer.args.n_out}) must be at least 1.")

            analyzer.load_corpus()
            if not analyzer.corpus:
                raise RuntimeError("Execution halted because no documents were loaded.")

            analyzer.compute_similarity_matrix()
            if analyzer.dist_mat is None:
                raise RuntimeError("Execution halted because the similarity matrix could not be computed.")

            if str(analyzer.args.similarity_threshold).lower() == 'auto':
                final_threshold = analyzer._determine_auto_threshold(method=analyzer.args.auto_threshold_method)
            else:
                final_threshold = float(analyzer.args.similarity_threshold)

            visualizer = SimilarityVisualizer()

            if analyzer.dist_mat.shape[0] < 2000 and analyzer.dist_mat.shape[1] < 2000:
                visualizer.plot_similarity_heatmap(analyzer)
            else:
                print(f"Skipping heatmap generation for large matrix ({analyzer.dist_mat.shape[0]}x{analyzer.dist_mat.shape[1]}).")

            visualizer.generate_comparison_html(analyzer, similarity_threshold=final_threshold)
            visualizer.generate_similarity_summary_tsv(analyzer, similarity_threshold=final_threshold)
            visualizer.generate_linguistic_summary_tsv(analyzer, similarity_threshold=final_threshold)

            print("\n--- ANALYSIS COMPLETE ---")

        except Exception as e:
            # If any error occurs, display it in the log window.
            self.log(f"\n--- AN ERROR OCCURRED ---\n{e}\n")
        finally:
            # This block runs whether the analysis succeeded or failed.
            # Re-enable the run button.
            self.run_button.config(state="normal", text="Run Analysis")
            # Restore the original stdout and stderr.
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def write(self, text):
        """
        This method is required for stdout redirection. It receives any text
        from `print()` calls and puts it into the thread-safe queue.
        """
        self.log_queue.put(text)

    def flush(self):
        """Required for stdout redirection, but doesn't need to do anything here."""
        pass

    def process_log_queue(self):
        """
        Periodically checks the queue for new messages from the analysis thread
        and, if any are found, calls the `log` method to display them in the GUI.
        """
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log(text)
        except queue.Empty:
            # If the queue is empty, schedule this method to run again after 100ms.
            self.after(100, self.process_log_queue)

    def log(self, text):
        """Inserts text into the GUI's log widget"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END) # Auto-scroll to the bottom
        self.log_text.configure(state='disabled')


if __name__ == "__main__":
    # This is the entry point for the application.
    app = FlameGUI()
    # `mainloop()` starts the Tkinter event loop, showing the window and
    # waiting for user interaction.
    app.mainloop()
