import numpy as np
from itertools import combinations
import fargv
import tqdm
from difflib import SequenceMatcher
from IPython.display import display # Keep for potential display in notebooks
import re
import plotly.graph_objects as go
import os
import pathlib

# Fargv configuration
DEFAULT_PARAMS = {
    'input_path': './testdir',  # Directory containing text files
    'file_suffix': '.txt',     # File suffix to look for
    'keep_texts': 2000,        # Maximum number of texts to analyze
    'ngram': 4,
    'n_out': 1,
    'min_text_length': 100,    # Minimum text length to consider
    'similarity_threshold': 0.1 # Minimum similarity threshold for comparison
}

class Flame:
    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS
        self.args, _ = fargv.fargv(self.params)
        self.encoder = None
        self.corpus = None # Will store original case text
        self.tokenized_corpus = None # Will store lowercased, tokenized text for analysis
        self.dist_mat = None
        self.file_paths = []

    def find_text_files(self):
        """Recursively find all text files with specified suffix in directory."""
        path = pathlib.Path(self.args.input_path)
        if not path.exists():
            raise ValueError(f"Input path {path} does not exist")
        return list(path.rglob(f"*{self.args.file_suffix}"))

    def read_text_file(self, file_path):
        """Read and clean text from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                text = ' '.join(text.split()) # Basic cleaning
                return text
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {str(e)}")
            return None

    def tokenize(self, text_str):
        """
        Splits text into tokens after removing specified punctuation.
        Does NOT change case. Case handling should be done by the caller if needed.
        """
        clean_text = text_str
        for punct in ['.', ',', ';', '[', ']', '(', ')']: # Consider adding more if needed
            clean_text = clean_text.replace(punct, '')
        return clean_text.split()


    def get_encoder(self, corpus_for_encoding):
        """Token-to-integer encoder from a corpus of texts (assumed to be pre-processed, e.g., lowercased)."""
        all_tokens = []
        for text in corpus_for_encoding: # Expects list of strings
            tokens = self.tokenize(text) # Tokenize the already lowercased text
            all_tokens.extend(tokens)
        # Create a sorted set of unique tokens to ensure consistent encoding
        unique_tokens = sorted(list(set(all_tokens)))
        return {token: i for i, token in enumerate(unique_tokens)}


    def tokens_to_int(self, tokens):
        """Convert tokens to integers using the encoder. Unknown tokens are skipped."""
        return [self.encoder[token] for token in tokens if token in self.encoder]

    def leave_n_out_grams(self, tokens):
        """Generate n-grams leaving out n tokens."""
        int_tokens = np.array(self.tokens_to_int(tokens))

        if int_tokens.size == 0:
            return np.array([])

        seq_len = len(int_tokens)
        # (ngram - n_out) is the number of elements to choose for each combination.
        # This must be at least 1.
        elements_to_keep = self.args.ngram - self.args.n_out
        if elements_to_keep < 1: # Cannot form n-grams if keeping less than 1 element.
            return np.array([])
        if seq_len < self.args.ngram: # Sequence too short to form even a single full n-gram
            return np.array([])

        # Construct the matrix of n-grams (where each column is an n-gram)
        # Number of n-grams that can be formed
        num_ngrams = seq_len - self.args.ngram + 1
        if num_ngrams <= 0:
            return np.array([])

        # sub_ngrams_matrix will have `ngram` rows and `num_ngrams` columns.
        # sub_ngrams_matrix[i, j] is the i-th token of the j-th ngram.
        sub_ngrams_matrix = np.zeros((self.args.ngram, num_ngrams), dtype=int_tokens.dtype)
        for i in range(self.args.ngram):
            sub_ngrams_matrix[i, :] = int_tokens[i : i + num_ngrams]

        keep_indices_combinations = list(combinations(range(self.args.ngram), elements_to_keep))

        lnout_grams_list = []
        if not self.encoder: # Encoder must exist
            return np.array([])
        vocab_size = len(self.encoder)
        if vocab_size == 0: # Vocab size must be > 0 for encoding
            return np.array([])


        for combo_indices in keep_indices_combinations:
            # For each combination of token positions to keep from the n-gram
            combined_int_values = np.zeros(num_ngrams, dtype=np.int64)
            for i, original_idx in enumerate(combo_indices):
                # Effectively create a polynomial hash for the combination
                combined_int_values += sub_ngrams_matrix[original_idx, :] * (vocab_size ** i)
            lnout_grams_list.append(combined_int_values)

        if not lnout_grams_list:
            return np.array([])
        return np.concatenate(lnout_grams_list, axis=0)


    def calculate_distance(self, text1_tokens, text2_tokens):
        """Calculate Intersection over Union (IoU) distance between two token lists."""
        # These tokens are assumed to be pre-processed (lowercased)
        ngrams1 = self.leave_n_out_grams(text1_tokens)
        ngrams2 = self.leave_n_out_grams(text2_tokens)

        if ngrams1.size == 0 or ngrams2.size == 0:
            return 0.0

        # Using assume_unique=False as leave_n_out_grams can produce duplicate feature values
        intersection_size = np.intersect1d(ngrams1, ngrams2, assume_unique=False).shape[0]
        # For union, we need unique elements from both sets combined
        union_size = np.union1d(ngrams1, ngrams2).shape[0]

        return intersection_size / union_size if union_size > 0 else 0.0

    def load_corpus(self):
        """Load and preprocess the corpus from text files."""
        file_paths = self.find_text_files()
        print(f"Found {len(file_paths)} files with suffix {self.args.file_suffix}")

        loaded_corpus_original_case = []
        loaded_corpus_for_analysis = [] # lowercased
        valid_file_paths = []

        for file_path in tqdm.tqdm(file_paths, desc="Loading files"):
            text = self.read_text_file(file_path)
            if text and len(text) >= self.args.min_text_length:
                loaded_corpus_original_case.append(text)
                loaded_corpus_for_analysis.append(text.lower()) # Store lowercased version for analysis
                valid_file_paths.append(file_path)
                if len(loaded_corpus_original_case) >= self.args.keep_texts:
                    break

        if not loaded_corpus_original_case:
            print("No valid texts loaded. Aborting.")
            self.corpus = []
            self.tokenized_corpus = []
            self.file_paths = []
            return

        print(f"Loaded {len(loaded_corpus_original_case)} valid texts for analysis.")
        self.corpus = loaded_corpus_original_case # Original case for display
        self.file_paths = valid_file_paths

        # Build encoder from the lowercased texts
        self.encoder = self.get_encoder(loaded_corpus_for_analysis)
        if not self.encoder:
             print("Warning: Encoder could not be built (empty vocabulary). Similarity scores may all be zero.")

        # Tokenize the lowercased texts for similarity computation
        self.tokenized_corpus = [self.tokenize(text_lower) for text_lower in loaded_corpus_for_analysis]


    def compute_similarity_matrix(self):
        """Compute similarity matrix for all text pairs using pre-tokenized (lowercased) corpus."""
        if not self.tokenized_corpus:
            print("Corpus is not tokenized or empty. Skipping similarity matrix computation.")
            self.dist_mat = np.array([])
            return

        num_texts = len(self.tokenized_corpus)
        self.dist_mat = np.zeros([num_texts, num_texts])

        for t1 in tqdm.tqdm(range(num_texts), desc="Computing similarity matrix"):
            for t2 in range(num_texts):
                if t1 == t2:
                    self.dist_mat[t1, t2] = 1.0
                elif t2 < t1: # Matrix is symmetric
                    self.dist_mat[t1, t2] = self.dist_mat[t2, t1]
                else:
                    # Pass the already tokenized (and lowercased) lists
                    self.dist_mat[t1, t2] = self.calculate_distance(
                        self.tokenized_corpus[t1],
                        self.tokenized_corpus[t2]
                    )
        np.save('dist_mat.npy', self.dist_mat)

class SimilarityVisualizer:
    @staticmethod
    def highlight_similarities(text1_original_tokens, text2_original_tokens, pair_id):
        """
        Highlight similar portions and gap words between two lists of original case tokens.
        Matching is done on lowercased versions of tokens.
        """
        text1_lower_tokens = [t.lower() for t in text1_original_tokens]
        text2_lower_tokens = [t.lower() for t in text2_original_tokens]

        matcher = SequenceMatcher(None, text1_lower_tokens, text2_lower_tokens, autojunk=False)

        highlighted_html_text1 = []
        highlighted_html_text2 = []
        match_details = {} # Stores details of matches for JS

        # --- Logic for finding bridge words (gaps of 1-3 words) ---
        bridge_word_sections = []
        raw_matching_blocks = matcher.get_matching_blocks()

        for idx in range(len(raw_matching_blocks) - 1):
            # Current match block: (a_start, b_start, length)
            current_block = raw_matching_blocks[idx]
            # Next match block
            next_block = raw_matching_blocks[idx+1]

            # Gap in text1: from end of current_block to start of next_block
            gap1_start_idx = current_block[0] + current_block[2]
            gap1_end_idx = next_block[0]
            # Gap in text2
            gap2_start_idx = current_block[1] + current_block[2]
            gap2_end_idx = next_block[1]

            # Original case tokens for the gap
            gap1_original_tokens = text1_original_tokens[gap1_start_idx:gap1_end_idx]
            gap2_original_tokens = text2_original_tokens[gap2_start_idx:gap2_end_idx]

            if (1 <= len(gap1_original_tokens) <= 3) and \
               (1 <= len(gap2_original_tokens) <= 3) and \
               (len(gap1_original_tokens) > 0 or len(gap2_original_tokens) > 0) : # Ensure there is a gap
                bridge_word_sections.append({
                    'text1_indices': (gap1_start_idx, gap1_end_idx),
                    'text2_indices': (gap2_start_idx, gap2_end_idx),
                    'text1_tokens': gap1_original_tokens,
                    'text2_tokens': gap2_original_tokens,
                })
        # --- End of bridge word logic ---

        current_pos_text1 = 0
        current_pos_text2 = 0
        match_id_counter = 0
        # bridge_id_counter = 0 # Unique ID for bridge sections - now using bridge_idx

        for a_start, b_start, length in raw_matching_blocks:
            if length == 0: # Skip zero-length matches (usually only the last block)
                continue

            # 1. Handle text BEFORE the current matching block
            # For text1
            if current_pos_text1 < a_start:
                is_bridge = False
                for bridge_idx, bridge in enumerate(bridge_word_sections):
                    if bridge['text1_indices'][0] == current_pos_text1 and bridge['text1_indices'][1] == a_start:
                        bridge_text_display = ' '.join(bridge['text1_tokens'])
                        highlighted_html_text1.append(f'<span class="bridge-words" data-bridge-pair="{pair_id}-{bridge_idx}">{bridge_text_display}</span>')
                        is_bridge = True
                        break
                if not is_bridge:
                    highlighted_html_text1.append(' '.join(text1_original_tokens[current_pos_text1:a_start]))

            # For text2
            if current_pos_text2 < b_start:
                is_bridge = False
                for bridge_idx, bridge in enumerate(bridge_word_sections):
                    if bridge['text2_indices'][0] == current_pos_text2 and bridge['text2_indices'][1] == b_start:
                        bridge_text_display = ' '.join(bridge['text2_tokens'])
                        highlighted_html_text2.append(f'<span class="bridge-words" data-bridge-pair="{pair_id}-{bridge_idx}">{bridge_text_display}</span>')
                        is_bridge = True
                        break
                if not is_bridge:
                    highlighted_html_text2.append(' '.join(text2_original_tokens[current_pos_text2:b_start]))

            # 2. Handle the matching block itself
            match_display_text1 = ' '.join(text1_original_tokens[a_start : a_start + length])
            match_display_text2 = ' '.join(text2_original_tokens[b_start : b_start + length])

            highlighted_html_text1.append(
                f'<span class="highlight clickable" data-match-id="{match_id_counter}" data-pair-id="{pair_id}">{match_display_text1}</span>'
            )
            highlighted_html_text2.append(
                f'<span class="match-text" data-match-id="{match_id_counter}" data-pair-id="{pair_id}">{match_display_text2}</span>'
            )

            match_details[match_id_counter] = { 'text1': match_display_text1, 'text2': match_display_text2 }
            match_id_counter +=1

            current_pos_text1 = a_start + length
            current_pos_text2 = b_start + length

        # 3. Handle any remaining text AFTER the last matching block
        if current_pos_text1 < len(text1_original_tokens):
            highlighted_html_text1.append(' '.join(text1_original_tokens[current_pos_text1:]))
        if current_pos_text2 < len(text2_original_tokens):
            highlighted_html_text2.append(' '.join(text2_original_tokens[current_pos_text2:]))

        return ' '.join(highlighted_html_text1), ' '.join(highlighted_html_text2), match_details

    @staticmethod
    def generate_comparison_html(analyzer, similarity_threshold=None, max_file_size=20 * 1024 * 1024):
        """Generate interactive HTML comparison of similar text pairs, splitting output if necessary."""
        if similarity_threshold is None:
            similarity_threshold = analyzer.args.similarity_threshold

        if analyzer.dist_mat is None or analyzer.dist_mat.size == 0:
            print("Similarity matrix not computed or empty. Skipping HTML generation.")
            return
        if not analyzer.corpus: # Check if corpus is loaded
            print("Corpus is empty. Skipping HTML generation.")
            return

        html_template_start = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Text Similarity Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; background-color: #f5f5f5; color: #333; }
                .comparison-block { margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .comparison-container { display: flex; gap: 20px; }
                .text-box { flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fff; height: 400px; overflow-y: auto; position: relative; word-wrap: break-word; }
                h2 { color: #333; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom:10px;}
                h3 { color: #444; margin-top: 0; margin-bottom: 10px; font-size: 1.2em;}
                .file-info { font-size: 0.9em; color: #555; margin-bottom:5px; }
                .similarity-score { margin-bottom: 15px; font-weight: bold; color: #0056b3; padding: 5px 10px; background-color: #e7f3ff; border-radius: 4px; display: inline-block; }
                .highlight { background-color: #fff3b8; padding: 1px 3px; border-radius: 3px; transition: background-color 0.2s ease; }
                .highlight.clickable { cursor: pointer; }
                .highlight.clickable:hover { background-color: #ffe066; }
                .match-text { padding: 1px 3px; border-radius: 3px; transition: background-color 0.3s ease; scroll-margin-top: 50px; /* Space for scrolling */ }
                .match-text.active { background-color: #fff3b8; } /* Highlight in target text box */
                .active-highlight { background-color: #ffe066; box-shadow: 0 0 0 2px #ffd700; } /* Clicked highlight */
                .bridge-words { padding: 1px 3px; border-radius: 3px; transition: background-color 0.3s ease; /* Default: no special bg */ }
                .bridge-words.highlighted { background-color: #ffcdd2; /* Light red for bridge words */ }
                #controls { margin-bottom: 20px; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                #toggle-all-bridge-words { background-color: #28a745; color: white; padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
                #toggle-all-bridge-words:hover { background-color: #218838; }
                #toggle-all-bridge-words.active { background-color: #dc3545; /* Red when active (bridges are SHOWN) */ }
                 .instructions { font-size: 0.9em; color: #666; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div id="controls">
                <h2>Interactive Text Similarity Comparison</h2>
                <button id="toggle-all-bridge-words">Show All Bridge Words</button>
                <p class="instructions">
                    Click on a yellow highlighted text segment in the left column to see the corresponding segment in the right column.
                    The right column will scroll to the match. Click again or outside to deselect.
                    Use the toggle button to highlight/hide "bridge words" (short differing segments of 1-3 words between matches).
                </p>
            </div>
        """
        html_template_end = """
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const activeHighlights = new Map(); // {pairId: {matchId: id, clickableElement: element}}

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
                                // Clicked same active highlight: deselect
                                resetHighlightVisuals(pairId, activeHighlights.get(pairId).matchId);
                                activeHighlights.delete(pairId);
                            } else {
                                // New highlight or different highlight in same pair
                                if (activeHighlights.has(pairId)) { // Deselect old in same pair
                                    resetHighlightVisuals(pairId, activeHighlights.get(pairId).matchId);
                                }
                                // Deselect from other pairs if any (optional, for single global active highlight)
                                // activeHighlights.forEach((val, pId) => { if (pId !== pairId) resetHighlightVisuals(pId, val.matchId); });
                                // activeHighlights.clear(); // if only one highlight active globally

                                setHighlightVisuals(pairId, matchId, clickedElement);
                                activeHighlights.set(pairId, {matchId: matchId, clickableElement: clickedElement});
                                scrollToMatch(pairId, matchId);
                            }
                        }

                        // If click was outside any clickable highlight and not on the toggle button
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
                        // Find the originally clicked element if stored, or query generally
                        const activePairInfo = activeHighlights.get(pairId);
                        let clickableSpan;
                        if (activePairInfo && activePairInfo.matchId === matchId && activePairInfo.clickableElement) {
                             clickableSpan = activePairInfo.clickableElement;
                        } else {
                             clickableSpan = document.querySelector(`.highlight.clickable[data-pair-id="${pairId}"][data-match-id="${matchId}"]`);
                        }
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
                                // Try to center the element in the visible part of the container
                                const scrollOffset = targetRect.top - containerRect.top - (container.clientHeight / 2) + (targetRect.height / 2);
                                container.scrollTop += scrollOffset;
                            }
                        }
                    }
                });
            </script>
        </body>
        </html>
        """

        file_counter = 1
        current_file_content = html_template_start
        # Estimate base template size (conservative, actual encoding is what matters)
        base_template_size = len(html_template_start.encode('utf-8')) + len(html_template_end.encode('utf-8'))
        current_file_size = len(html_template_start.encode('utf-8'))
        pair_render_count = 0 # Use a different name for the counter


        def write_html_file(content, counter, end_template):
            filename = f"text_comparisons_{counter:02d}.html"
            with open(filename, "w", encoding='utf-8') as f:
                f.write(content)
                f.write(end_template) # Ensure body and html are closed
            print(f"Generated {filename}")

        # Prepare tokens for display (original case)
        # The Flame.tokenize method is simple enough to be used here on original case texts.
        display_token_corpus = [analyzer.tokenize(text) for text in analyzer.corpus]

        for i in range(len(analyzer.corpus)):
            for j in range(i + 1, len(analyzer.corpus)):
                if analyzer.dist_mat[i, j] >= similarity_threshold:
                    pair_render_count +=1
                    # Use original case tokens for highlighting
                    text1_original_tokens = display_token_corpus[i]
                    text2_original_tokens = display_token_corpus[j]

                    highlighted_html_text1, highlighted_html_text2, _ = SimilarityVisualizer.highlight_similarities(
                        text1_original_tokens, text2_original_tokens, pair_render_count # Use pair_render_count as pair_id
                    )

                    filename1 = str(analyzer.file_paths[i].name)
                    filename2 = str(analyzer.file_paths[j].name)

                    comparison_html_segment = f"""
                    <div class="comparison-block" id="pair-{pair_render_count}">
                        <h3>Comparison: {filename1} &harr; {filename2}</h3>
                        <div class="similarity-score">Similarity Score (IoU): {analyzer.dist_mat[i, j]:.4f}</div>
                        <div class="comparison-container" data-pair-id="{pair_render_count}">
                            <div class="text-box">
                                <p class="file-info"><strong>File 1: {filename1}</strong></p>
                                {highlighted_html_text1}
                            </div>
                            <div class="text-box">
                                <p class="file-info"><strong>File 2: {filename2}</strong></p>
                                {highlighted_html_text2}
                            </div>
                        </div>
                    </div>
                    """
                    encoded_segment = comparison_html_segment.encode('utf-8')
                    # Check if adding this segment (plus end template) would exceed max size
                    if (current_file_size + len(encoded_segment) + len(html_template_end.encode('utf-8'))) > max_file_size \
                       and current_file_size > len(html_template_start.encode('utf-8')): # Check if anything was added besides start_template
                        write_html_file(current_file_content, file_counter, html_template_end)
                        file_counter += 1
                        current_file_content = html_template_start # Start new file
                        current_file_size = len(html_template_start.encode('utf-8'))

                    current_file_content += comparison_html_segment
                    current_file_size += len(encoded_segment)

        # Write the last/current file if it has content beyond the start template
        if pair_render_count > 0: # Only write if comparisons were actually added
            if current_file_size > len(html_template_start.encode('utf-8')): # Ensures content was added
                 write_html_file(current_file_content, file_counter, html_template_end)
        else: # No similar pairs found
            print("No similar pairs found above the threshold to generate comparison HTML.")


    @staticmethod
    def plot_similarity_heatmap(analyzer):
        """Generate interactive heatmap visualization."""
        if analyzer.dist_mat is None or analyzer.dist_mat.size == 0:
            print("Similarity matrix not computed or empty. Skipping heatmap generation.")
            return
        if not analyzer.file_paths: # Check if file_paths is available
            print("No file paths available. Skipping heatmap generation.")
            return

        labels = [str(path.name) for path in analyzer.file_paths]

        fig = go.Figure(data=go.Heatmap(
            z=analyzer.dist_mat,
            x=labels,
            y=labels,
            colorscale='Blues', # Using a standard Plotly colorscale
            colorbar=dict(title='Similarity (IoU)'),
            zmin=0.0,
            zmax=1.0
        ))

        fig.update_layout(
            title='Text Similarity Heatmap',
            xaxis_title='Files',
            yaxis_title='Files',
            xaxis=dict(showgrid=False, tickangle=-45, automargin=True, type='category'), # ensure labels are treated as categories
            yaxis=dict(showgrid=False, automargin=True, type='category'), # ensure labels are treated as categories
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(600, len(labels) * 25 + 150), # Adjust height based on number of labels + padding for titles/margins
            width=max(700, len(labels) * 25 + 150)  # Adjust width based on number of labels + padding
        )

        try:
            fig.write_html("similarity_heatmap.html")
            print("Generated similarity_heatmap.html")
            # Conditionally display if in an environment that supports it (e.g. Jupyter)
            if 'IPython' in __import__('sys').modules and hasattr(display, '__call__'): # Check for IPython and display function
                 display(fig)
        except Exception as e:
            print(f"Error writing or displaying heatmap: {e}")

    @staticmethod
    def generate_similarity_summary_tsv(analyzer):
        """
        Generates a TSV file summarizing similarity frequencies, related documents,
        and the longest shared segments (>4 words) from the perspective of the source document.
        """
        if analyzer.dist_mat is None or analyzer.dist_mat.size == 0:
            print("Similarity matrix not computed or empty. Skipping TSV summary generation.")
            return
        if not analyzer.corpus or not analyzer.file_paths: # Ensure corpus is also available
            print("Corpus or file paths not available. Skipping TSV summary generation.")
            return

        output_filename = "similarity_summary.tsv"
        similarity_threshold = analyzer.args.similarity_threshold

        # New header for the TSV
        header = "DocumentFilename\tSimilarityFrequency\tRelatedDocuments\tLongSimilarities(>4words)\n"
        tsv_rows = [header]

        num_docs = len(analyzer.corpus)
        # Pre-tokenize original corpus for display/SequenceMatcher once, using the class's tokenize method
        original_tokenized_corpus = [analyzer.tokenize(text) for text in analyzer.corpus]

        for i in range(num_docs):
            source_filename = str(analyzer.file_paths[i].name)
            similarity_count = 0
            related_docs_list = []
            all_long_similar_segments_for_doc_i = [] # Stores (size, "segment text from doc i")

            # Tokens for document 'i' (the source document for this row)
            tokens_i_original_case = original_tokenized_corpus[i]
            tokens_i_lower = [t.lower() for t in tokens_i_original_case] # Lowercase for matching

            for j in range(num_docs):
                if i == j: # Skip self-comparison for similarity frequency and segments
                    continue

                if analyzer.dist_mat[i, j] >= similarity_threshold:
                    similarity_count += 1
                    related_docs_list.append(str(analyzer.file_paths[j].name))

                    # --- Extract long similar segments for this pair (i, j) ---
                    tokens_j_original_case = original_tokenized_corpus[j]
                    tokens_j_lower = [t.lower() for t in tokens_j_original_case] # Lowercase for matching

                    # Use SequenceMatcher to find matching blocks
                    sm = SequenceMatcher(None, tokens_i_lower, tokens_j_lower, autojunk=False)
                    matching_blocks = sm.get_matching_blocks()

                    for a_start, b_start, size in matching_blocks:
                        if size > 4: # Only segments longer than 4 words
                            # Extract the segment from document 'i' (original case tokens)
                            segment_text = " ".join(tokens_i_original_case[a_start : a_start + size])
                            all_long_similar_segments_for_doc_i.append((size, segment_text))

            # Sort collected segments by size (descending)
            all_long_similar_segments_for_doc_i.sort(key=lambda x: x[0], reverse=True)

            # Format for TSV output: unique, sorted, quoted, and pipe-separated
            if all_long_similar_segments_for_doc_i:
                # Keep only unique segment texts to avoid redundancy, maintaining order of first appearance for same-length segments
                unique_sorted_segment_texts = []
                seen_segments = set() # To track uniqueness of segment text
                for size, text in all_long_similar_segments_for_doc_i:
                    if text not in seen_segments:
                        unique_sorted_segment_texts.append(f'"{text}"') # Quote segments
                        seen_segments.add(text)
                long_segments_str = " | ".join(unique_sorted_segment_texts)
            else:
                long_segments_str = "None"

            related_docs_str = ", ".join(sorted(list(set(related_docs_list)))) if related_docs_list else "None"

            row_data = f"{source_filename}\t{similarity_count}\t{related_docs_str}\t{long_segments_str}\n"
            tsv_rows.append(row_data)

        try:
            with open(output_filename, "w", encoding='utf-8') as f:
                for r_data in tsv_rows: # Use a different variable name for clarity
                    f.write(r_data)
            print(f"Generated {output_filename}")
        except IOError as e: # More specific exception for file I/O issues
            print(f"Error writing TSV summary file {output_filename}: {e}")


def main():
    analyzer = Flame()

    # Parameter validation for n_out and ngram
    elements_to_keep = analyzer.args.ngram - analyzer.args.n_out
    if elements_to_keep < 1:
        raise ValueError(
            f"The combination of ngram ({analyzer.args.ngram}) and n_out ({analyzer.args.n_out}) "
            f"must result in at least 1 token being kept from the original n-gram context. "
            f"Currently, it results in {elements_to_keep} tokens."
        )
    if analyzer.args.n_out < 0: # n_out cannot be negative
        raise ValueError(f"n_out ({analyzer.args.n_out}) cannot be negative.")
    if analyzer.args.ngram < 1: # ngram must be at least 1
        raise ValueError(f"ngram ({analyzer.args.ngram}) must be at least 1.")


    analyzer.load_corpus()

    # Proceed only if corpus loading was successful and texts were found
    if not analyzer.corpus:
        print("Exiting: No texts were loaded into the corpus.")
        return

    analyzer.compute_similarity_matrix()

    visualizer = SimilarityVisualizer()
    if analyzer.dist_mat is not None and analyzer.dist_mat.size > 0: # Check if dist_mat is valid
        visualizer.plot_similarity_heatmap(analyzer)
        visualizer.generate_comparison_html(analyzer)
        visualizer.generate_similarity_summary_tsv(analyzer) # Call the new TSV generation method
    else:
        print("No similarity data computed, skipping visualization and summary generation.")

if __name__ == '__main__':
    main()
