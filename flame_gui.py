import numpy as np
from itertools import combinations
import fargv
import re
import plotly.graph_objects as go
import os
import pathlib
from difflib import SequenceMatcher
from IPython.display import display # Keep for original compatibility, but unused in GUI

# GUI Imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import webbrowser
import sys

# Fargv
DEFAULT_PARAMS = {
    'input_path': './',
    'file_suffix': '.txt',
    'keep_texts': 2000,
    'ngram': 5,
    'n_out': 1,
    'min_text_length': 100,
    'similarity_threshold': 0.1
}

class Flame:
    def __init__(self, params=None, log_callback=print):
        self.params = params or DEFAULT_PARAMS
        self.args, _ = fargv.fargv(self.params)
        self.log_callback = log_callback
        self.encoder = None
        self.corpus = None
        self.tokenized_corpus = None
        self.dist_mat = None
        self.file_paths = []

    def _tqdm_wrapper(self, iterable, desc="Processing"):
        """A wrapper to provide progress updates"""
        try:
            total = len(iterable)
            self.log_callback(f"{desc}: Starting with {total} items.")
            for i, item in enumerate(iterable):
                # Log progress roughly every 5% or for key milestones
                if total > 20 and i > 0 and i % (total // 20) == 0:
                    self.log_callback(f"... {desc}: {i}/{total} ({(i/total*100):.0f}%) complete.")
                yield item
            self.log_callback(f"{desc}: Finished {total} items.")
        except TypeError: # If iterable has no len()
            self.log_callback(f"{desc}: Starting...")
            for item in iterable:
                yield item
            self.log_callback(f"{desc}: Finished.")


    def find_text_files(self):
        path = pathlib.Path(self.args.input_path)
        if not path.exists():
            raise ValueError(f"Input path {path} does not exist")
        return list(path.rglob(f"*{self.args.file_suffix}"))

    def read_text_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                text = ' '.join(text.split())
                return text
        except Exception as e:
            self.log_callback(f"Warning: Could not read file {file_path}: {str(e)}")
            return None

    def tokenize(self, text_str):
        clean_text = text_str
        for punct in ['.', ',', ';', '[', ']', '(', ')']:
            clean_text = clean_text.replace(punct, '')
        return clean_text.split()

    def get_encoder(self, corpus_for_encoding):
        all_tokens = []
        for text in self._tqdm_wrapper(corpus_for_encoding, desc="Building vocabulary"):
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        unique_tokens = sorted(list(set(all_tokens)))
        return {token: i for i, token in enumerate(unique_tokens)}

    def tokens_to_int(self, tokens):
        return [self.encoder[token] for token in tokens if token in self.encoder]

    def leave_n_out_grams(self, tokens):
        int_tokens = np.array(self.tokens_to_int(tokens))
        if int_tokens.size == 0: return np.array([])
        seq_len = len(int_tokens)
        elements_to_keep = self.args.ngram - self.args.n_out
        if elements_to_keep < 1 or seq_len < self.args.ngram: return np.array([])
        num_ngrams = seq_len - self.args.ngram + 1
        if num_ngrams <= 0: return np.array([])

        sub_ngrams_matrix = np.zeros((self.args.ngram, num_ngrams), dtype=int_tokens.dtype)
        for i in range(self.args.ngram):
            sub_ngrams_matrix[i, :] = int_tokens[i : i + num_ngrams]

        keep_indices_combinations = list(combinations(range(self.args.ngram), elements_to_keep))
        lnout_grams_list = []
        if not self.encoder or not len(self.encoder): return np.array([])
        vocab_size = len(self.encoder)

        for combo_indices in keep_indices_combinations:
            combined_int_values = np.zeros(num_ngrams, dtype=np.int64)
            for i, original_idx in enumerate(combo_indices):
                combined_int_values += sub_ngrams_matrix[original_idx, :] * (vocab_size ** i)
            lnout_grams_list.append(combined_int_values)

        return np.concatenate(lnout_grams_list, axis=0) if lnout_grams_list else np.array([])

    def calculate_distance(self, text1_tokens, text2_tokens):
        ngrams1 = self.leave_n_out_grams(text1_tokens)
        ngrams2 = self.leave_n_out_grams(text2_tokens)
        if ngrams1.size == 0 or ngrams2.size == 0: return 0.0
        intersection_size = np.intersect1d(ngrams1, ngrams2, assume_unique=False).shape[0]
        union_size = np.union1d(ngrams1, ngrams2).shape[0]
        return intersection_size / union_size if union_size > 0 else 0.0

    def load_corpus(self):
        file_paths = self.find_text_files()
        self.log_callback(f"Found {len(file_paths)} files with suffix '{self.args.file_suffix}'.")
        loaded_corpus_original_case, loaded_corpus_for_analysis, valid_file_paths = [], [], []

        for file_path in self._tqdm_wrapper(file_paths, desc="Loading files"):
            text = self.read_text_file(file_path)
            if text and len(text) >= self.args.min_text_length:
                loaded_corpus_original_case.append(text)
                loaded_corpus_for_analysis.append(text.lower())
                valid_file_paths.append(file_path)
                if len(loaded_corpus_original_case) >= self.args.keep_texts:
                    self.log_callback(f"Reached max limit of {self.args.keep_texts} texts.")
                    break

        if not loaded_corpus_original_case:
            self.log_callback("No valid texts loaded. Aborting.")
            self.corpus, self.tokenized_corpus, self.file_paths = [], [], []
            return

        self.log_callback(f"Loaded {len(loaded_corpus_original_case)} valid texts for analysis.")
        self.corpus, self.file_paths = loaded_corpus_original_case, valid_file_paths
        self.encoder = self.get_encoder(loaded_corpus_for_analysis)
        if not self.encoder:
            self.log_callback("Warning: Encoder could not be built (empty vocabulary).")
        self.tokenized_corpus = [self.tokenize(text_lower) for text_lower in loaded_corpus_for_analysis]

    def compute_similarity_matrix(self):
        if not self.tokenized_corpus:
            self.log_callback("Corpus is not tokenized or empty. Skipping similarity matrix.")
            self.dist_mat = np.array([])
            return
        num_texts = len(self.tokenized_corpus)
        self.dist_mat = np.zeros([num_texts, num_texts])

        self.log_callback("Computing similarity matrix...")
        for t1 in range(num_texts):
            if num_texts > 10 and t1 % (num_texts // 10) == 0:
                self.log_callback(f"... analyzing document {t1+1}/{num_texts}")
            for t2 in range(num_texts):
                if t1 == t2: self.dist_mat[t1, t2] = 1.0
                elif t2 < t1: self.dist_mat[t1, t2] = self.dist_mat[t2, t1]
                else: self.dist_mat[t1, t2] = self.calculate_distance(self.tokenized_corpus[t1], self.tokenized_corpus[t2])
        self.log_callback("Similarity matrix computation complete.")
        np.save('dist_mat.npy', self.dist_mat)
        self.log_callback("Saved similarity matrix to dist_mat.npy")

class SimilarityVisualizer:
    @staticmethod
    def highlight_similarities(text1_original_tokens, text2_original_tokens, pair_id):
        text1_lower_tokens = [t.lower() for t in text1_original_tokens]
        text2_lower_tokens = [t.lower() for t in text2_original_tokens]
        matcher = SequenceMatcher(None, text1_lower_tokens, text2_lower_tokens, autojunk=False)
        highlighted_html_text1, highlighted_html_text2, match_details = [], [], {}
        bridge_word_sections = []
        raw_matching_blocks = matcher.get_matching_blocks()

        for idx in range(len(raw_matching_blocks) - 1):
            current_block, next_block = raw_matching_blocks[idx], raw_matching_blocks[idx+1]
            gap1_start_idx, gap1_end_idx = current_block[0] + current_block[2], next_block[0]
            gap2_start_idx, gap2_end_idx = current_block[1] + current_block[2], next_block[1]
            gap1_original_tokens, gap2_original_tokens = text1_original_tokens[gap1_start_idx:gap1_end_idx], text2_original_tokens[gap2_start_idx:gap2_end_idx]
            if (1 <= len(gap1_original_tokens) <= 3 or 1 <= len(gap2_original_tokens) <= 3) and (len(gap1_original_tokens) > 0 or len(gap2_original_tokens) > 0):
                bridge_word_sections.append({'text1_indices': (gap1_start_idx, gap1_end_idx), 'text2_indices': (gap2_start_idx, gap2_end_idx), 'text1_tokens': gap1_original_tokens, 'text2_tokens': gap2_original_tokens})

        current_pos_text1, current_pos_text2, match_id_counter = 0, 0, 0
        for a_start, b_start, length in raw_matching_blocks:
            if length == 0: continue
            if current_pos_text1 < a_start:
                is_bridge = False
                for bridge_idx, bridge in enumerate(bridge_word_sections):
                    if bridge['text1_indices'][0] == current_pos_text1 and bridge['text1_indices'][1] == a_start:
                        highlighted_html_text1.append(f'<span class="bridge-words" data-bridge-pair="{pair_id}-{bridge_idx}">{" ".join(bridge["text1_tokens"])}</span>')
                        is_bridge = True; break
                if not is_bridge: highlighted_html_text1.append(' '.join(text1_original_tokens[current_pos_text1:a_start]))
            if current_pos_text2 < b_start:
                is_bridge = False
                for bridge_idx, bridge in enumerate(bridge_word_sections):
                    if bridge['text2_indices'][0] == current_pos_text2 and bridge['text2_indices'][1] == b_start:
                        highlighted_html_text2.append(f'<span class="bridge-words" data-bridge-pair="{pair_id}-{bridge_idx}">{" ".join(bridge["text2_tokens"])}</span>')
                        is_bridge = True; break
                if not is_bridge: highlighted_html_text2.append(' '.join(text2_original_tokens[current_pos_text2:b_start]))

            match_display_text1 = ' '.join(text1_original_tokens[a_start : a_start + length])
            highlighted_html_text1.append(f'<span class="highlight clickable" data-match-id="{match_id_counter}" data-pair-id="{pair_id}">{match_display_text1}</span>')
            highlighted_html_text2.append(f'<span class="match-text" data-match-id="{match_id_counter}" data-pair-id="{pair_id}">{" ".join(text2_original_tokens[b_start : b_start + length])}</span>')
            match_details[match_id_counter] = {'text1': match_display_text1, 'text2': ' '.join(text2_original_tokens[b_start : b_start + length])}
            match_id_counter +=1
            current_pos_text1, current_pos_text2 = a_start + length, b_start + length

        if current_pos_text1 < len(text1_original_tokens): highlighted_html_text1.append(' '.join(text1_original_tokens[current_pos_text1:]))
        if current_pos_text2 < len(text2_original_tokens): highlighted_html_text2.append(' '.join(text2_original_tokens[current_pos_text2:]))
        return ' '.join(highlighted_html_text1), ' '.join(highlighted_html_text2), match_details

    @staticmethod
    def generate_comparison_html(analyzer, log_callback=print, similarity_threshold=None, max_file_size=20 * 1024 * 1024):
        if similarity_threshold is None: similarity_threshold = analyzer.args.similarity_threshold
        if analyzer.dist_mat is None or analyzer.dist_mat.size == 0 or not analyzer.corpus:
            log_callback("Similarity matrix/corpus empty. Skipping HTML comparison generation.")
            return

        html_template_start = """
        <!DOCTYPE html><html><head><meta charset="UTF-8"><title>Text Similarity Comparison</title><style>body{font-family:Arial,sans-serif;margin:20px;line-height:1.6;background-color:#f5f5f5;color:#333}.comparison-block{margin-bottom:30px;background-color:white;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1)}.comparison-container{display:flex;gap:20px}.text-box{flex:1;padding:15px;border:1px solid #ddd;border-radius:5px;background-color:#fff;height:400px;overflow-y:auto;position:relative;word-wrap:break-word}h2{color:#333;margin-bottom:10px;border-bottom:1px solid #eee;padding-bottom:10px}h3{color:#444;margin-top:0;margin-bottom:10px;font-size:1.2em}.file-info{font-size:.9em;color:#555;margin-bottom:5px}.similarity-score{margin-bottom:15px;font-weight:700;color:#0056b3;padding:5px 10px;background-color:#e7f3ff;border-radius:4px;display:inline-block}.highlight{background-color:#fff3b8;padding:1px 3px;border-radius:3px;transition:background-color .2s ease}.highlight.clickable{cursor:pointer}.highlight.clickable:hover{background-color:#ffe066}.match-text{padding:1px 3px;border-radius:3px;transition:background-color .3s ease;scroll-margin-top:50px}.match-text.active{background-color:#fff3b8}.active-highlight{background-color:#ffe066;box-shadow:0 0 0 2px #ffd700}.bridge-words{padding:1px 3px;border-radius:3px;transition:background-color .3s ease}.bridge-words.highlighted{background-color:#ffcdd2}#controls{margin-bottom:20px;background-color:white;padding:15px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1)}#toggle-all-bridge-words{background-color:#28a745;color:white;padding:8px 15px;border:none;border-radius:4px;cursor:pointer;transition:background-color .3s}#toggle-all-bridge-words:hover{background-color:#218838}#toggle-all-bridge-words.active{background-color:#dc3545}.instructions{font-size:.9em;color:#666;margin-top:10px}</style></head><body><div id="controls"><h2>Interactive Text Similarity Comparison</h2><button id="toggle-all-bridge-words">Show All Bridge Words</button><p class="instructions">Click on a yellow highlighted text segment in the left column to see the corresponding segment in the right column. The right column will scroll to the match. Click again or outside to deselect. Use the toggle button to highlight/hide "bridge words" (short differing segments of 1-3 words between matches).</p></div>
        """

        # FIXED JAVASCRIPT
        html_template_end = """
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const activeHighlights = new Map();

                    const toggleButton = document.getElementById('toggle-all-bridge-words');
                    if (toggleButton) {
                        toggleButton.addEventListener('click', function() {
                            const isActive = this.classList.toggle('active');
                            document.querySelectorAll('.bridge-words').forEach(span => {
                                span.classList.toggle('highlighted', isActive);
                            });
                            this.textContent = isActive ? 'Hide All Bridge Words' : 'Show All Bridge Words';
                        });
                    }

                    document.body.addEventListener('click', function(e) {
                        const clickedElement = e.target;
                        let processedClick = false;

                        if (clickedElement.classList.contains('highlight') && clickedElement.classList.contains('clickable')) {
                            const matchId = clickedElement.dataset.matchId;
                            const pairId = clickedElement.dataset.pairId;
                            processedClick = true;

                            if (activeHighlights.has(pairId) && activeHighlights.get(pairId).matchId === matchId) {
                                resetHighlightVisuals(pairId, activeHighlights.get(pairId).matchId);
                                activeHighlights.delete(pairId);
                            } else {
                                if (activeHighlights.has(pairId)) {
                                    resetHighlightVisuals(pairId, activeHighlights.get(pairId).matchId);
                                }
                                setHighlightVisuals(pairId, matchId, clickedElement);
                                activeHighlights.set(pairId, {matchId: matchId, clickableElement: clickedElement});
                                scrollToMatch(pairId, matchId);
                            }
                        }

                        if (!processedClick && !clickedElement.closest('.highlight.clickable') && !clickedElement.closest('#toggle-all-bridge-words')) {
                            activeHighlights.forEach((val, pairId) => resetHighlightVisuals(pairId, val.matchId));
                            activeHighlights.clear();
                        }
                    });

                    function setHighlightVisuals(pairId, matchId, clickableSpan) {
                        if (clickableSpan) clickableSpan.classList.add('active-highlight');
                        const targetMatchSpan = document.querySelector(`.match-text[data-pair-id="${pairId}"][data-match-id="${matchId}"]`);
                        if (targetMatchSpan) targetMatchSpan.classList.add('active');
                    }

                    function resetHighlightVisuals(pairId, matchId) {
                        const activePairInfo = activeHighlights.get(pairId);
                        let clickableSpan = activePairInfo && activePairInfo.matchId === matchId ? activePairInfo.clickableElement : document.querySelector(`.highlight.clickable[data-pair-id="${pairId}"][data-match-id="${matchId}"]`);
                        if (clickableSpan) clickableSpan.classList.remove('active-highlight');

                        const targetMatchSpan = document.querySelector(`.match-text[data-pair-id="${pairId}"][data-match-id="${matchId}"]`);
                        if (targetMatchSpan) targetMatchSpan.classList.remove('active');
                    }

                    function scrollToMatch(pairId, matchId) {
                        const targetElement = document.querySelector(`.match-text[data-pair-id="${pairId}"][data-match-id="${matchId}"]`);
                        if (targetElement) {
                            const container = targetElement.closest('.text-box');
                            if (container) {
                                const targetRect = targetElement.getBoundingClientRect();
                                const containerRect = container.getBoundingClientRect();
                                const scrollOffset = targetRect.top - containerRect.top - (container.clientHeight / 2) + (targetRect.height / 2);
                                container.scrollTop += scrollOffset;
                            }
                        }
                    }
                });
            </script>
        </body></html>
        """
        file_counter, current_file_content, current_file_size, pair_render_count = 1, html_template_start, len(html_template_start.encode('utf-8')), 0

        def write_html_file(content, counter, end_template):
            filename = f"text_comparisons_{counter:02d}.html"
            with open(filename, "w", encoding='utf-8') as f:
                f.write(content + end_template)
            log_callback(f"Generated {filename}")

        display_token_corpus = [analyzer.tokenize(text) for text in analyzer.corpus]
        log_callback("Generating HTML comparison report(s)...")

        for i in range(len(analyzer.corpus)):
            for j in range(i + 1, len(analyzer.corpus)):
                if analyzer.dist_mat[i, j] >= similarity_threshold:
                    pair_render_count += 1
                    text1_tokens, text2_tokens = display_token_corpus[i], display_token_corpus[j]
                    h_html1, h_html2, _ = SimilarityVisualizer.highlight_similarities(text1_tokens, text2_tokens, pair_render_count)
                    fname1, fname2 = str(analyzer.file_paths[i].name), str(analyzer.file_paths[j].name)

                    comparison_html = f"""<div class="comparison-block" id="pair-{pair_render_count}"><h3>Comparison: {fname1} &harr; {fname2}</h3><div class="similarity-score">Similarity Score (IoU): {analyzer.dist_mat[i, j]:.4f}</div><div class="comparison-container" data-pair-id="{pair_render_count}"><div class="text-box"><p class="file-info"><strong>File 1: {fname1}</strong></p>{h_html1}</div><div class="text-box"><p class="file-info"><strong>File 2: {fname2}</strong></p>{h_html2}</div></div></div>"""
                    encoded_segment = comparison_html.encode('utf-8')

                    if (current_file_size + len(encoded_segment) + len(html_template_end.encode('utf-8'))) > max_file_size and current_file_size > len(html_template_start.encode('utf-8')):
                        write_html_file(current_file_content, file_counter, html_template_end)
                        file_counter += 1
                        current_file_content, current_file_size = html_template_start, len(html_template_start.encode('utf-8'))

                    current_file_content += comparison_html
                    current_file_size += len(encoded_segment)

        if pair_render_count > 0:
            if current_file_size > len(html_template_start.encode('utf-8')):
                write_html_file(current_file_content, file_counter, html_template_end)
        else:
            log_callback("No similar pairs found above the threshold to generate comparison HTML.")

    @staticmethod
    def plot_similarity_heatmap(analyzer, log_callback=print):
        if analyzer.dist_mat is None or analyzer.dist_mat.size == 0 or not analyzer.file_paths:
            log_callback("Similarity matrix/paths empty. Skipping heatmap generation.")
            return
        labels = [str(path.name) for path in analyzer.file_paths]
        fig = go.Figure(data=go.Heatmap(z=analyzer.dist_mat, x=labels, y=labels, colorscale='Blues', colorbar=dict(title='Similarity (IoU)'), zmin=0.0, zmax=1.0))
        fig.update_layout(title='Text Similarity Heatmap', xaxis_title='Files', yaxis_title='Files', xaxis=dict(showgrid=False, tickangle=-45, automargin=True, type='category'), yaxis=dict(showgrid=False, automargin=True, type='category'), plot_bgcolor='white', paper_bgcolor='white', height=max(600, len(labels) * 25 + 150), width=max(700, len(labels) * 25 + 150))
        try:
            fig.write_html("similarity_heatmap.html")
            log_callback("Generated similarity_heatmap.html")
        except Exception as e:
            log_callback(f"Error writing heatmap: {e}")

    @staticmethod
    def generate_similarity_summary_tsv(analyzer, log_callback=print):
        if analyzer.dist_mat is None or analyzer.dist_mat.size == 0 or not analyzer.corpus or not analyzer.file_paths:
            log_callback("Data missing. Skipping TSV summary generation.")
            return
        output_filename = "similarity_summary.tsv"
        header = "DocumentFilename\tSimilarityFrequency\tRelatedDocuments\tLongSimilarities(>4words)\n"
        tsv_rows = [header]
        num_docs = len(analyzer.corpus)
        original_tokenized_corpus = [analyzer.tokenize(text) for text in analyzer.corpus]

        log_callback("Generating TSV summary...")
        for i in range(num_docs):
            source_filename = str(analyzer.file_paths[i].name)
            similarity_count, related_docs_list, all_long_segments = 0, [], []
            tokens_i_original_case = original_tokenized_corpus[i]
            tokens_i_lower = [t.lower() for t in tokens_i_original_case]

            for j in range(num_docs):
                if i == j: continue
                if analyzer.dist_mat[i, j] >= analyzer.args.similarity_threshold:
                    similarity_count += 1
                    related_docs_list.append(str(analyzer.file_paths[j].name))
                    tokens_j_lower = [t.lower() for t in original_tokenized_corpus[j]]
                    sm = SequenceMatcher(None, tokens_i_lower, tokens_j_lower, autojunk=False)
                    for a_start, _, size in sm.get_matching_blocks():
                        if size > 4:
                            all_long_segments.append((size, " ".join(tokens_i_original_case[a_start : a_start + size])))

            all_long_segments.sort(key=lambda x: x[0], reverse=True)
            unique_sorted_texts, seen_segments = [], set()
            for _, text in all_long_segments:
                if text not in seen_segments:
                    unique_sorted_texts.append(f'"{text}"')
                    seen_segments.add(text)
            long_segments_str = " | ".join(unique_sorted_texts) if unique_sorted_texts else "None"
            related_docs_str = ", ".join(sorted(list(set(related_docs_list)))) if related_docs_list else "None"
            tsv_rows.append(f"{source_filename}\t{similarity_count}\t{related_docs_str}\t{long_segments_str}\n")

        try:
            with open(output_filename, "w", encoding='utf-8') as f:
                f.writelines(tsv_rows)
            log_callback(f"Generated {output_filename}")
        except IOError as e:
            log_callback(f"Error writing TSV summary file {output_filename}: {e}")


# GUI Class
class FlameGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("FLAME - Formulaic Language Analysis in Medieval Expressions")
        self.master.geometry("800x750")

        self.params = {}
        self.output_files = {}

        self.create_widgets()

        self.log_queue = queue.Queue()
        self.process_log_queue()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        params_frame = ttk.LabelFrame(main_frame, text="1. Configuration", padding="10")
        params_frame.pack(fill=tk.X, expand=False)
        params_frame.grid_columnconfigure(1, weight=1)

        self.add_param_entry(params_frame, "Input Directory:", 'input_path', row=0, is_dir=True)
        self.add_param_entry(params_frame, "File Suffix:", 'file_suffix', row=1)
        self.add_param_entry(params_frame, "Min Text Length:", 'min_text_length', row=2)
        self.add_param_entry(params_frame, "Similarity Threshold:", 'similarity_threshold', row=3)
        self.add_param_entry(params_frame, "N-Gram Size:", 'ngram', row=4)
        self.add_param_entry(params_frame, "N-Out (from N-Gram):", 'n_out', row=5)
        self.add_param_entry(params_frame, "Max Files to Process:", 'keep_texts', row=6)

        control_frame = ttk.LabelFrame(main_frame, text="2. Execution", padding="10")
        control_frame.pack(fill=tk.X, expand=False, pady=5)
        self.run_button = ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis_thread)
        self.run_button.pack(fill=tk.X)

        log_frame = ttk.LabelFrame(main_frame, text="3. Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, state='disabled', wrap='word', height=10)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = log_scroll.set
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_frame = ttk.LabelFrame(main_frame, text="4. Open Reports", padding="10")
        self.output_frame.pack(fill=tk.X, expand=False, pady=5)

    def add_param_entry(self, parent, label_text, param_key, row, is_dir=False):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        var = tk.StringVar(value=DEFAULT_PARAMS.get(param_key, ''))
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky='ew')
        self.params[param_key] = var
        if is_dir:
            browse_btn = ttk.Button(parent, text="Browse...", command=lambda: self.browse_directory(var))
            browse_btn.grid(row=row, column=2, padx=5)

    def browse_directory(self, var):
        dir_name = filedialog.askdirectory()
        if dir_name:
            var.set(dir_name)

    def log(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if isinstance(message, dict) and message.get("type") == "analysis_complete":
                    self.analysis_finished(message.get("files", {}))
                else:
                    self.log(str(message))
        except queue.Empty:
            pass
        self.master.after(100, self.process_log_queue)

    def start_analysis_thread(self):
        try:
            current_params = {key: var.get() for key, var in self.params.items()}
            for key in ['keep_texts', 'ngram', 'n_out', 'min_text_length']:
                current_params[key] = int(current_params[key])
            current_params['similarity_threshold'] = float(current_params['similarity_threshold'])
        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Please check your input values. Error: {e}")
            return

        self.run_button.config(state='disabled', text="Analysis in Progress...")
        self.log_text.config(state='normal'); self.log_text.delete(1.0, tk.END); self.log_text.config(state='disabled')
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        thread = threading.Thread(target=self.run_analysis_worker, args=(current_params, self.log_queue))
        thread.daemon = True
        thread.start()

    def analysis_finished(self, output_files):
        self.run_button.config(state='normal', text="Start Analysis")
        self.log("ANALYSIS COMPLETE")
        self.output_files = output_files

        btn_info = {"heatmap": "Open Heatmap", "comparison": "Open Comparison Report", "summary": "Open TSV Summary"}
        for key, text in btn_info.items():
            if key in self.output_files:
                path = self.output_files[key]
                btn = ttk.Button(self.output_frame, text=text, command=lambda p=path: self.open_file_in_browser(p))
                btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def open_file_in_browser(self, file_path):
        try:
            abs_path = os.path.realpath(file_path)
            webbrowser.open('file://' + abs_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {file_path}\n{e}")

    def run_analysis_worker(self, params, log_queue):
        try:
            def logger(msg): log_queue.put(msg)

            logger("Starting analysis with the following parameters:")
            for key, val in params.items(): logger(f"  - {key}: {val}")
            logger("-" * 20)

            analyzer = Flame(params=params, log_callback=logger)

            elements_to_keep = analyzer.args.ngram - analyzer.args.n_out
            if elements_to_keep < 1: raise ValueError("ngram - n_out must be >= 1")
            if analyzer.args.n_out < 0: raise ValueError("n_out cannot be negative")
            if analyzer.args.ngram < 1: raise ValueError("ngram must be at least 1")

            analyzer.load_corpus()
            if not analyzer.corpus:
                logger("Exiting: No texts were loaded into the corpus.")
                log_queue.put({"type": "analysis_complete"})
                return

            analyzer.compute_similarity_matrix()

            output_files = {}
            if analyzer.dist_mat is not None and analyzer.dist_mat.size > 0:
                visualizer = SimilarityVisualizer()

                visualizer.plot_similarity_heatmap(analyzer, log_callback=logger)
                output_files["heatmap"] = "similarity_heatmap.html"

                visualizer.generate_comparison_html(analyzer, log_callback=logger)
                output_files["comparison"] = "text_comparisons_01.html"

                visualizer.generate_similarity_summary_tsv(analyzer, log_callback=logger)
                output_files["summary"] = "similarity_summary.tsv"
            else:
                logger("No similarity data computed, skipping visualization.")

            log_queue.put({"type": "analysis_complete", "files": output_files})

        except Exception as e:
            import traceback
            log_queue.put(f"FATAL ERROR")
            log_queue.put(f"An error occurred: {e}")
            log_queue.put(traceback.format_exc())
            log_queue.put({"type": "analysis_complete"})


if __name__ == '__main__':
    root = tk.Tk()
    app = FlameGUI(master=root)
    app.mainloop()
