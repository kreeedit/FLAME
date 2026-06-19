import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import queue
import sys
import os
import webbrowser
from flame import Flame, DEFAULT_PARAMS, SimilarityVisualizer
from argparse import Namespace

class FlameGUI(tk.Tk):
    """
    A Tkinter-based graphical user interface for the FLAME analysis tool.
    Designed for users who prefer an intuitive UI over terminal commands.
    """
    def __init__(self):
        """Initializes the main application window and structures layout tabs."""
        super().__init__()
        self.title("FLAME - Formulaic Language Analysis in Medieval Expressions")
        self.geometry("950x900")
        self.minsize(850, 750)

        self.params = {}
        for key, val in DEFAULT_PARAMS.items():
            if isinstance(val, bool):
                self.params[key] = tk.BooleanVar(value=val)
            else:
                self.params[key] = tk.StringVar(value=str(val))

        # REACTIVE LOGIC: Trace changes on similarity_threshold to auto-update method dropdown
        self.params['similarity_threshold'].trace_add("write", self.on_threshold_change)

        self.create_widgets()

        self.log_queue = queue.Queue()
        self.after(100, self.process_log_queue)

    def on_threshold_change(self, *args):
        """Automatically switches threshold method based on auto vs numerical input."""
        val = self.params['similarity_threshold'].get().strip().lower()
        if val == 'auto':
            self.params['auto_threshold_method'].set('otsu')
        else:
            self.params['auto_threshold_method'].set('percentile')

    def create_widgets(self):
        """Creates and arranges all widgets inside tabbed logical containers."""
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Tab container layout control
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.X, side=tk.TOP, pady=5)

        tab_core = ttk.Frame(self.notebook, padding="10")
        tab_nlp = ttk.Frame(self.notebook, padding="10")
        tab_reports = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(tab_core, text="1. Data & Core Setup")
        self.notebook.add(tab_nlp, text="2. Normalization & Philology")
        self.notebook.add(tab_reports, text="3. Auto-Tune & Outputs")

        # ==================== TAB 1: CORE SETUP ====================
        paths_frame = ttk.LabelFrame(tab_core, text="Corpus Directory Selection", padding="10")
        paths_frame.pack(fill=tk.X, pady=5)
        paths_frame.columnconfigure(1, weight=1)
        self.create_path_entry(paths_frame, "input_path", "Primary Corpus Path:", 0)
        self.create_path_entry(paths_frame, "input_path2", "Secondary Corpus Path (Optional):", 1)

        core_params_frame = ttk.LabelFrame(tab_core, text="Core Windowing & Matching Settings", padding="10")
        core_params_frame.pack(fill=tk.X, pady=5)
        self.create_param_entry(core_params_frame, "ngram", "N-gram window size:", 0, 0)
        self.create_param_entry(core_params_frame, "n_out", "N-out positions (Gaps):", 0, 2)
        self.create_param_entry(core_params_frame, "min_text_length", "Min. Text Length (chars):", 1, 0)
        self.create_param_entry(core_params_frame, "file_suffix", "Target File Suffix:", 1, 2)
        self.create_param_entry(core_params_frame, "keep_texts", "Max Documents to Load:", 2, 0)

        threshold_frame = ttk.LabelFrame(tab_core, text="Global Threshold Selection", padding="10")
        threshold_frame.pack(fill=tk.X, pady=5)
        self.create_param_entry(threshold_frame, "similarity_threshold", "Similarity Threshold ('auto' or 0-1):", 0, 0)

        # DROPDOWN IMPLEMENTATION: Replacing entry with a reactive readonly Combobox
        ttk.Label(threshold_frame, text="Auto Threshold Method:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=4)
        method_dropdown = ttk.Combobox(
            threshold_frame,
            textvariable=self.params['auto_threshold_method'],
            values=['otsu', 'percentile'],
            state='readonly',
            width=12
        )
        method_dropdown.grid(row=0, column=3, sticky=tk.W, padx=5, pady=4)

        # ==================== TAB 2: NLP & PHILOLOGY ====================
        char_frame = ttk.LabelFrame(tab_nlp, text="Character Normalization Layers", padding="10")
        char_frame.pack(fill=tk.X, pady=5)
        char_frame.columnconfigure(1, weight=1)
        ttk.Label(char_frame, text="Target Norm Alphabet:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(char_frame, textvariable=self.params['char_norm_alphabet'], width=50).grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        self.create_param_entry(char_frame, "char_norm_strategy", "Norm Strategy:", 1, 0)
        self.create_param_entry(char_frame, "char_norm_min_freq", "Min. Norm Frequency:", 1, 2)

        bpe_frame = ttk.LabelFrame(tab_nlp, text="Subword Tokenizer (BPE) Heuristics", padding="10")
        bpe_frame.pack(fill=tk.X, pady=5)
        self.create_param_entry(bpe_frame, "vocab_size", "Vocab Size ('auto' or integer):", 0, 0)
        self.create_param_entry(bpe_frame, "vocab_min_word_freq", "Min. Word Freq for Affixes:", 0, 2)
        self.create_param_entry(bpe_frame, "vocab_coverage", "Target Morphological Coverage (0-1):", 1, 0)

        alignment_frame = ttk.LabelFrame(tab_nlp, text="Philological Variation & Bridge Words Evaluation", padding="10")
        alignment_frame.pack(fill=tk.X, pady=5)
        self.create_param_entry(alignment_frame, "fuzz_threshold", "Bridge Word Fuzzy Sensitivity (0-1):", 0, 0)
        self.create_param_entry(alignment_frame, "max_gap_words", "Max Bridge Word Length Gap:", 0, 2)

        # ==================== TAB 3: AUTO-TUNE & REPORTS ====================
        autotune_frame = ttk.LabelFrame(tab_reports, text="Trial Digging (Autonomous Hyperparameter Auto-Tune)", padding="10")
        autotune_frame.pack(fill=tk.X, pady=5)
        chk_tune = ttk.Checkbutton(autotune_frame, text="Enable Self-Supervised Auto-Tune (Finds best N-gram & Gap setup)", variable=self.params['auto_tune'])
        chk_tune.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        self.create_param_entry(autotune_frame, "auto_tune_sample_size", "Auto-Tune Training Sample Size:", 1, 0)
        ttk.Label(autotune_frame, text="Note: Auto-tuning executes a temporary synthetic noise sweep to maximize signal separation.", font=("TkDefaultFont", 9, "italic")).grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)

        reports_frame = ttk.LabelFrame(tab_reports, text="Report & Output File Generation", padding="10")
        reports_frame.pack(fill=tk.X, pady=5)

        self.chk_no_reports = ttk.Checkbutton(reports_frame, text="Disable All Document Reports", variable=self.params['no_reports'], command=self.toggle_report_checkboxes)
        self.chk_no_reports.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        self.chk_html = ttk.Checkbutton(reports_frame, text="Generate Interactive Side-by-Side HTML", variable=self.params['gen_comparison_html'])
        self.chk_html.grid(row=1, column=0, sticky=tk.W, padx=20, pady=2)

        self.chk_summary = ttk.Checkbutton(reports_frame, text="Generate Spreadsheet Summary TSV", variable=self.params['gen_summary_tsv'])
        self.chk_summary.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        self.chk_linguistic = ttk.Checkbutton(reports_frame, text="Generate Granular Linguistic Variants TSV", variable=self.params['gen_linguistic_tsv'])
        self.chk_linguistic.grid(row=2, column=0, sticky=tk.W, padx=20, pady=2)

        self.chk_heatmap = ttk.Checkbutton(reports_frame, text="Generate Dynamic Cluster Heatmap HTML", variable=self.params['gen_heatmap'])
        self.chk_heatmap.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        self.toggle_report_checkboxes()

        # ==================== RUN & LOG ENGINE (BOTTOM) ====================
        self.run_button = ttk.Button(main_frame, text="Run FLAME Pipeline", command=self.start_analysis_thread)
        self.run_button.pack(pady=15)

        results_frame = ttk.LabelFrame(main_frame, text="Direct Action: Open Analysis Results", padding="10")
        results_frame.pack(fill=tk.X, pady=5)
        for i in range(4): results_frame.columnconfigure(i, weight=1)

        self.html_button = ttk.Button(results_frame, text="Open Aligned HTML", state=tk.DISABLED, command=lambda: self.open_result_file('text_comparisons_01.html'))
        self.html_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.EW)

        self.heatmap_button = ttk.Button(results_frame, text="Open Heatmap Chart", state=tk.DISABLED, command=lambda: self.open_result_file('similarity_heatmap.html'))
        self.heatmap_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        self.summary_button = ttk.Button(results_frame, text="Open Summary TSV", state=tk.DISABLED, command=lambda: self.open_result_file('similarity_summary.tsv'))
        self.summary_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.EW)

        self.linguistic_button = ttk.Button(results_frame, text="Open Linguistic Variants TSV", state=tk.DISABLED, command=lambda: self.open_result_file('linguistic_variations.tsv'))
        self.linguistic_button.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        log_frame = ttk.LabelFrame(main_frame, text="Pipeline Execution Standard Output Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=12, bg="#1e1e1e", fg="#ffffff", insertbackground="white", font=("Courier", 9))
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
        """Creates a standard path field workspace wrapper configuration."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=self.params[param_name], width=55)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        browse_button = ttk.Button(parent, text="Browse Folder...", command=lambda: self.browse_directory(param_name))
        browse_button.grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)

    def create_param_entry(self, parent, param_name, label_text, row, col):
        """Utility wrapper to render clean execution properties inside grids."""
        ttk.Label(parent, text=label_text).grid(row=row, column=col, sticky=tk.W, padx=5, pady=4)
        entry = ttk.Entry(parent, textvariable=self.params[param_name], width=12)
        entry.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=4)

    def browse_directory(self, param_name):
        directory = filedialog.askdirectory(title="Select Corpus Target Directory Folder")
        if directory:
            self.params[param_name].set(directory)

    def open_result_file(self, filename):
        if os.path.exists(filename):
            webbrowser.open(os.path.realpath(filename))
        else:
            self.log_queue.put(f"\nTarget execution file asset not ready or found: {filename}\n")

    def start_analysis_thread(self):
        """Spawns processing workers in background layers safely."""
        self.run_button.config(state="disabled", text="Pipeline Analysis Running...")

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
        """Executes full operational processing sequence mapping values safely."""
        analysis_successful = False
        try:
            args_for_flame = {}
            for key, var in self.params.items():
                val = var.get()
                if isinstance(var, tk.BooleanVar):
                    args_for_flame[key] = val
                elif key in ['keep_texts', 'ngram', 'n_out', 'min_text_length',
                             'char_norm_min_freq', 'vocab_min_word_freq',
                             'max_gap_words', 'auto_tune_sample_size']:
                    args_for_flame[key] = int(val)
                elif key in ['vocab_coverage', 'fuzz_threshold']:
                    args_for_flame[key] = float(val)
                elif key == 'similarity_threshold' and str(val).lower() != 'auto':
                    args_for_flame[key] = float(val)
                else:
                    args_for_flame[key] = val

            args_object = Namespace(**args_for_flame)
            analyzer = Flame(args=args_object)

            sys.stdout = self
            sys.stderr = self

            analyzer.load_corpus()
            if not analyzer.corpus:
                raise RuntimeError("Execution halted because no valid target documents loaded.")

            if analyzer.args.auto_tune:
                analyzer.auto_tune_parameters()

            if (analyzer.args.ngram - analyzer.args.n_out) < 1:
                raise ValueError(f"N-gram size ({analyzer.args.ngram}) minus n-out ({analyzer.args.n_out}) must be at least 1.")

            analyzer.compute_similarity_matrix()
            if analyzer.dist_mat is None:
                raise RuntimeError("Execution processing failure: global matrix could not be resolved.")

            if analyzer.args.no_reports:
                print("\n--- Report generation skipped as per GUI configuration toggle. ---")
            else:
                print("\n--- Generating Scholar Report Assets ---")

                if str(analyzer.args.similarity_threshold).lower() == 'auto':
                    final_threshold = analyzer._determine_auto_threshold(method=analyzer.args.auto_threshold_method)
                else:
                    final_threshold = float(analyzer.args.similarity_threshold)

                if analyzer.args.gen_heatmap:
                    if analyzer.dist_mat.shape[0] < 2000 and analyzer.dist_mat.shape[1] < 2000:
                        SimilarityVisualizer.plot_similarity_heatmap(analyzer)
                    else:
                        print(f"Skipping heatmap generation for large matrix bounds ({analyzer.dist_mat.shape[0]}x{analyzer.dist_mat.shape[1]}).")
                else:
                    print("Skipping heatmap generation as per configuration.")

                if analyzer.args.gen_comparison_html:
                    SimilarityVisualizer.generate_comparison_html(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping interactive HTML generation as per configuration.")

                if analyzer.args.gen_summary_tsv:
                    SimilarityVisualizer.generate_similarity_summary_tsv(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping summary TSV generation as per configuration.")

                if analyzer.args.gen_linguistic_tsv:
                    SimilarityVisualizer.generate_linguistic_summary_tsv(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping linguistic TSV variants generation as per configuration.")

            print("\n--- PIPELINE EXECUTION COMPLETELY FINISHED WITH SUCCESS ---")
            analysis_successful = True

        except Exception as e:
            self.log_queue.put(f"\n--- AN EXCEPTION ERROR PIPELINE RUN ENCOUNTERED ---\n{type(e).__name__}: {e}\n")
        finally:
            def final_gui_update():
                self.run_button.config(state="normal", text="Run FLAME Pipeline")
                if analysis_successful:
                    self.update_results_buttons()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
            self.after(0, final_gui_update)

    def update_results_buttons(self):
        """Enables individual reporting buttons conditionally based on outputs matching criteria."""
        if self.params['no_reports'].get():
            return

        if self.params['gen_comparison_html'].get() and os.path.exists('text_comparisons_01.html'):
            self.html_button.config(state=tk.NORMAL)
        if self.params['gen_heatmap'].get() and os.path.exists('similarity_heatmap.html'):
            self.heatmap_button.config(state=tk.NORMAL)
        if self.params['gen_summary_tsv'].get() and os.path.exists('similarity_summary.tsv'):
            self.summary_button.config(state=tk.NORMAL)
        if self.params['gen_linguistic_tsv'].get() and os.path.exists('linguistic_variations.tsv'):
            self.linguistic_button.config(state=tk.NORMAL)

    def write(self, text):
        self.log_queue.put(text)

    def flush(self):
        pass

    def process_log_queue(self):
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log(text)
        except queue.Empty:
            self.after(100, self.process_log_queue)

    def log(self, text):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

if __name__ == "__main__":
    app = FlameGUI()
    style = ttk.Style()
    style.theme_use('clam')
    app.mainloop()
