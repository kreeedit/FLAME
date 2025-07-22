import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import queue
import sys
import os
import webbrowser
from flame import Flame, DEFAULT_PARAMS, SimilarityVisualizer


class FlameGUI(tk.Tk):
    """
    A Tkinter-based graphical user interface for the FLAME analysis tool.
    """
    def __init__(self):
        """Initializes the main application window and its components."""
        super().__init__()
        self.title("FLAME - Formulaic Language Analysis in Medieval Expressions")
        self.geometry("900x850") # Növeltem az ablak magasságát az új vezérlők miatt

        self.params = {}
        for key, val in DEFAULT_PARAMS.items():
            if isinstance(val, bool):
                self.params[key] = tk.BooleanVar(value=val)
            else:
                self.params[key] = tk.StringVar(value=str(val))

        self.create_widgets()

        self.log_queue = queue.Queue()
        self.after(100, self.process_log_queue)

    def create_widgets(self):
        """Creates and arranges all the widgets in the main window."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        params_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding="10")
        params_frame.pack(fill=tk.X, side=tk.TOP)

        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)

        self.create_path_entry(params_frame, "input_path", "Primary Corpus Path:", 0)
        self.create_path_entry(params_frame, "input_path2", "Secondary Corpus Path (Optional):", 1)

        core_params_frame = ttk.LabelFrame(params_frame, text="Core Settings", padding="5")
        core_params_frame.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=5)
        self.create_param_entry(core_params_frame, "ngram", "N-gram size:", 0, 0)
        self.create_param_entry(core_params_frame, "n_out", "N-out size:", 0, 2)
        self.create_param_entry(core_params_frame, "min_text_length", "Min. Text Length:", 1, 0)
        ttk.Label(core_params_frame, text="Similarity Threshold:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        threshold_entry = ttk.Entry(core_params_frame, textvariable=self.params['similarity_threshold'], width=10)
        threshold_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        vocab_params_frame = ttk.LabelFrame(params_frame, text="Tokenizer & Vocabulary Settings", padding="5")
        vocab_params_frame.grid(row=3, column=0, columnspan=4, sticky=tk.EW, pady=5)
        self.create_param_entry(vocab_params_frame, "vocab_size", "Vocab Size (or 'auto'):", 0, 0)
        self.create_param_entry(vocab_params_frame, "vocab_min_word_freq", "Vocab Min. Word Freq.:", 0, 2)
        self.create_param_entry(vocab_params_frame, "vocab_coverage", "Vocab Coverage %:", 0, 4)

        reports_frame = ttk.LabelFrame(params_frame, text="Report & Output Settings", padding="5")
        reports_frame.grid(row=4, column=0, columnspan=4, sticky=tk.EW, pady=5)

        self.chk_no_reports = ttk.Checkbutton(reports_frame, text="Disable All Reports", variable=self.params['no_reports'], command=self.toggle_report_checkboxes)
        self.chk_no_reports.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.chk_html = ttk.Checkbutton(reports_frame, text="Generate Comparison HTML", variable=self.params['gen_comparison_html'])
        self.chk_html.grid(row=1, column=0, sticky=tk.W, padx=20, pady=2)

        self.chk_summary = ttk.Checkbutton(reports_frame, text="Generate Summary TSV", variable=self.params['gen_summary_tsv'])
        self.chk_summary.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        self.chk_linguistic = ttk.Checkbutton(reports_frame, text="Generate Linguistic TSV", variable=self.params['gen_linguistic_tsv'])
        self.chk_linguistic.grid(row=2, column=0, sticky=tk.W, padx=20, pady=2)

        self.chk_heatmap = ttk.Checkbutton(reports_frame, text="Generate Heatmap HTML", variable=self.params['gen_heatmap'])
        self.chk_heatmap.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        self.toggle_report_checkboxes() # Kezdeti állapot beállítása

        self.run_button = ttk.Button(main_frame, text="Run Analysis", command=self.start_analysis_thread)
        self.run_button.pack(pady=10)

        results_frame = ttk.LabelFrame(main_frame, text="Open Results", padding="10")
        results_frame.pack(fill=tk.X, pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        results_frame.columnconfigure(3, weight=1)

        self.html_button = ttk.Button(results_frame, text="Comparison Report (HTML)", state=tk.DISABLED,
                                      command=lambda: self.open_result_file('text_comparisons_01.html'))
        self.html_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.EW)

        self.heatmap_button = ttk.Button(results_frame, text="Heatmap (HTML)", state=tk.DISABLED,
                                         command=lambda: self.open_result_file('similarity_heatmap.html'))
        self.heatmap_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        self.summary_button = ttk.Button(results_frame, text="Summary (TSV)", state=tk.DISABLED,
                                         command=lambda: self.open_result_file('similarity_summary.tsv'))
        self.summary_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.EW)

        self.linguistic_button = ttk.Button(results_frame, text="Variations (TSV)", state=tk.DISABLED,
                                            command=lambda: self.open_result_file('linguistic_variations.tsv'))
        self.linguistic_button.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, bg="black", fg="white")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

    def toggle_report_checkboxes(self):
        """Disables individual report checkboxes if 'Disable All' is checked."""
        is_disabled = self.params['no_reports'].get()
        new_state = tk.DISABLED if is_disabled else tk.NORMAL

        self.chk_html.config(state=new_state)
        self.chk_summary.config(state=new_state)
        self.chk_linguistic.config(state=new_state)
        self.chk_heatmap.config(state=new_state)

    def create_path_entry(self, parent, param_name, label_text, row):
        """Helper function to create a labeled entry field for a directory path with a 'Browse' button."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=self.params[param_name], width=60)
        entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        browse_button = ttk.Button(parent, text="Browse...", command=lambda: self.browse_directory(param_name))
        browse_button.grid(row=row, column=3, sticky=tk.W, padx=5, pady=5)

    def create_param_entry(self, parent, param_name, label_text, row, col):
        """Helper function to create a standard labeled entry field for a parameter."""
        ttk.Label(parent, text=label_text).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Entry(parent, textvariable=self.params[param_name], width=10)
        entry.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=2)

    def browse_directory(self, param_name):
        """Opens a system dialog to select a directory and updates the corresponding entry field."""
        directory = filedialog.askdirectory(title="Select a Folder")
        if directory:
            self.params[param_name].set(directory)

    def open_result_file(self, filename):
        """Opens a given file with the system's default application."""
        if os.path.exists(filename):
            webbrowser.open(os.path.realpath(filename))
        else:
            self.log(f"\nFile not found: {filename}\n")

    def start_analysis_thread(self):
        """Starts the analysis in a separate thread."""
        self.run_button.config(state="disabled", text="Analysis in Progress...")

        self.html_button.config(state=tk.DISABLED)
        self.heatmap_button.config(state=tk.DISABLED)
        self.summary_button.config(state=tk.DISABLED)
        self.linguistic_button.config(state=tk.DISABLED)

        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()

    def run_analysis(self):
        """The main worker function for the analysis."""
        analysis_successful = False
        try:
            args_for_flame = {}
            for key, var in self.params.items():
                val = var.get()
                if isinstance(var, tk.BooleanVar):
                    args_for_flame[key] = val
                elif key in ['keep_texts', 'ngram', 'n_out', 'min_text_length',
                             'char_norm_min_freq', 'vocab_min_word_freq']:
                    args_for_flame[key] = int(val)
                elif key in ['similarity_threshold', 'vocab_coverage'] and str(val).lower() != 'auto':
                    args_for_flame[key] = float(val)
                else:
                    args_for_flame[key] = val

            analyzer = Flame(params=args_for_flame)
            sys.stdout = self
            sys.stderr = self

            if (analyzer.args.ngram - analyzer.args.n_out) < 1:
                raise ValueError(f"N-gram size ({analyzer.args.ngram}) minus n-out ({analyzer.args.n_out}) must be at least 1.")

            analyzer.load_corpus()
            if not analyzer.corpus:
                raise RuntimeError("Execution halted because no documents were loaded.")

            analyzer.compute_similarity_matrix()
            if analyzer.dist_mat is None:
                raise RuntimeError("Execution halted because the similarity matrix could not be computed.")

            if analyzer.args.no_reports:
                print("\n--- Report generation skipped as per GUI setting. ---")
            else:
                print("\n--- Generating Reports ---")

                if str(analyzer.args.similarity_threshold).lower() == 'auto':
                    final_threshold = analyzer._determine_auto_threshold(method=analyzer.args.auto_threshold_method)
                else:
                    final_threshold = float(analyzer.args.similarity_threshold)

                if analyzer.args.gen_heatmap:
                    if analyzer.dist_mat.shape[0] < 2000 and analyzer.dist_mat.shape[1] < 2000:
                        SimilarityVisualizer.plot_similarity_heatmap(analyzer)
                    else:
                        print(f"Skipping heatmap generation for large matrix ({analyzer.dist_mat.shape[0]}x{analyzer.dist_mat.shape[1]}).")
                else:
                    print("Skipping heatmap generation as per GUI setting.")

                if analyzer.args.gen_comparison_html:
                    SimilarityVisualizer.generate_comparison_html(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping interactive HTML generation as per GUI setting.")

                if analyzer.args.gen_summary_tsv:
                    SimilarityVisualizer.generate_similarity_summary_tsv(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping summary TSV generation as per GUI setting.")

                if analyzer.args.gen_linguistic_tsv:
                    SimilarityVisualizer.generate_linguistic_summary_tsv(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping linguistic TSV generation as per GUI setting.")

            print("\n--- ANALYSIS COMPLETE ---")
            analysis_successful = True

        except Exception as e:
            self.log_queue.put(f"\n--- AN ERROR OCCURRED ---\n{type(e).__name__}: {e}\n")
        finally:
            def final_gui_update():
                self.run_button.config(state="normal", text="Run Analysis")
                if analysis_successful:
                    self.update_results_buttons()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
            self.after(0, final_gui_update)

    def update_results_buttons(self):
        """Enables result buttons if their corresponding files were generated."""
        if self.params['no_reports'].get():
            return # Ha minden riport ki volt kapcsolva, ne csinálj semmit

        if self.params['gen_comparison_html'].get() and os.path.exists('text_comparisons_01.html'):
            self.html_button.config(state=tk.NORMAL)
        if self.params['gen_heatmap'].get() and os.path.exists('similarity_heatmap.html'):
            self.heatmap_button.config(state=tk.NORMAL)
        if self.params['gen_summary_tsv'].get() and os.path.exists('similarity_summary.tsv'):
            self.summary_button.config(state=tk.NORMAL)
        if self.params['gen_linguistic_tsv'].get() and os.path.exists('linguistic_variations.tsv'):
            self.linguistic_button.config(state=tk.NORMAL)

    def write(self, text):
        """This method is required for stdout redirection."""
        self.log_queue.put(text)

    def flush(self):
        """Required for stdout redirection."""
        pass

    def process_log_queue(self):
        """Periodically checks the queue for new messages and updates the GUI."""
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log(text)
        except queue.Empty:
            self.after(100, self.process_log_queue)

    def log(self, text):
        """Inserts text into the GUI's log widget"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')


if __name__ == "__main__":
    app = FlameGUI()
    app.mainloop()
