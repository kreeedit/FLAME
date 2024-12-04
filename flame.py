import numpy as np
from itertools import combinations
import fargv
import tqdm
from difflib import SequenceMatcher
from IPython.display import display
import re
import plotly.graph_objects as go
import os
import pathlib

# Fargv configuration
DEFAULT_PARAMS = {
    'input_path': './testdir',  # Directory containing text files
    'file_suffix': '.txt',    # File suffix to look for
    'keep_texts': 200,        # Maximum number of texts to analyze
    'ngram': 4,
    'n_out': 1,
    'min_text_length': 100,     # Minimum text length to consider
    'similarity_threshold': 0.1 # Minimum similarity threshold for comparison
}

class TextSimilarityAnalyzer:
    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS
        self.args, _ = fargv.fargv(self.params)
        self.encoder = None
        self.corpus = None
        self.tokenized_corpus = None
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
                # Basic cleaning - remove extra whitespace
                text = ' '.join(text.split())
                return text
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {str(e)}")
            return None

    def tokenize(self, text_str):
        """Split text into tokens after removing periods, commas, and semicolons."""
        for punct in ['.', ',', ';', '[', ']', '(', ')']:
            text_str = text_str.replace(punct, '')
        return text_str.split()

    def get_encoder(self, corpus):
        """Token-to-integer encoder."""
        all_tokens = []
        for text in corpus:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        return {token: i for i, token in enumerate(sorted(set(all_tokens)))}

    def tokens_to_int(self, tokens):
        """Convert tokens to integers using the encoder."""
        return [self.encoder[token] for token in tokens]

    def leave_n_out_grams(self, tokens):
        """Generate n-grams leaving out n tokens."""
        int_tokens = np.array(self.tokens_to_int(tokens))
        seq_len = len(tokens)

        sub_ngrams = []
        for n in range(0, self.args.ngram):
            sub_ngrams.append(int_tokens[n:seq_len - (self.args.ngram-n)])

        sub_ngrams = np.stack(sub_ngrams, axis=0)
        keep_indices = list(combinations(range(self.args.ngram), self.args.ngram-self.args.n_out))

        lnout_grams = []
        for indices in keep_indices:
            sub_int_ngrams = np.zeros_like(sub_ngrams[0], dtype=np.int64)
            for n, idx in enumerate(indices):
                sub_int_ngrams += sub_ngrams[idx]*(len(self.encoder)**n)
            lnout_grams.append(sub_int_ngrams)

        return np.concatenate(lnout_grams, axis=0)

    def calculate_distance(self, text1_tokens, text2_tokens):
        """Calculate Intersection over Union (IoU) distance between two texts."""
        text1_int_ngrams = self.leave_n_out_grams(text1_tokens)
        text2_int_ngrams = self.leave_n_out_grams(text2_tokens)

        intersection_size = np.intersect1d(text1_int_ngrams, text2_int_ngrams).shape[0]
        union_size = np.union1d(text1_int_ngrams, text2_int_ngrams).shape[0]

        return intersection_size / union_size if union_size > 0 else 0

    def load_corpus(self):
        """Load and preprocess the corpus from text files."""
        # Find all text files
        file_paths = self.find_text_files()
        print(f"Found {len(file_paths)} files with suffix {self.args.file_suffix}")

        # Read and filter texts
        corpus = []
        valid_file_paths = []

        for file_path in tqdm.tqdm(file_paths, desc="Loading files"):
            text = self.read_text_file(file_path)
            if text and len(text) >= self.args.min_text_length:
                corpus.append(text)
                valid_file_paths.append(file_path)

                if len(corpus) >= self.args.keep_texts:
                    break

        print(f"Loaded {len(corpus)} valid texts")
        self.corpus = corpus
        self.file_paths = valid_file_paths
        self.tokenized_corpus = [self.tokenize(text) for text in self.corpus]
        self.encoder = self.get_encoder(self.corpus)

    def compute_similarity_matrix(self):
        """Compute similarity matrix for all text pairs."""
        self.dist_mat = np.zeros([len(self.corpus), len(self.corpus)])
        for t1, text1 in enumerate(tqdm.tqdm(self.tokenized_corpus)):
            for t2, text2 in enumerate(self.tokenized_corpus):
                self.dist_mat[t1, t2] = self.calculate_distance(text1, text2)
        np.save('dist_mat.npy', self.dist_mat)

class SimilarityVisualizer:
    @staticmethod
    def highlight_similarities(text1, text2, pair_id):
        """Highlight similar portions and gap words between two texts with interactive elements."""
        matcher = SequenceMatcher(None, text1, text2)
        highlighted_text1 = []
        highlighted_text2 = []

        match_id = 0
        matches = {}
        gap_sections = []

        # Get all matching blocks first to analyze potential gaps
        matching_blocks = list(matcher.get_matching_blocks())

        # Find gaps between matches
        for idx in range(len(matching_blocks) - 1):
            curr_match = matching_blocks[idx]
            next_match = matching_blocks[idx + 1]

            # Calculate the gap between matches in both texts
            gap1_start = curr_match[0] + curr_match[2]
            gap1_end = next_match[0]
            gap2_start = curr_match[1] + curr_match[2]
            gap2_end = next_match[1]

            # Get the words in the gaps
            gap1_words = text1[gap1_start:gap1_end]
            gap2_words = text2[gap2_start:gap2_end]

            # If both gaps are 1-3 words, consider them as bridge sections
            if (1 <= len(gap1_words) <= 3 and 1 <= len(gap2_words) <= 3):
                gap_sections.append({
                    'pos1': (gap1_start, gap1_end),
                    'pos2': (gap2_start, gap2_end),
                    'text1': gap1_words,
                    'text2': gap2_words
                })

        pos1 = pos2 = 0
        gap_idx = 0

        for i1, i2, size in matching_blocks:
            if size == 0:
                continue

            # Add non-matching portions before match
            if pos1 < i1:
                gap_found = False
                for gap in gap_sections:
                    if gap['pos1'][0] == pos1 and gap['pos1'][1] == i1:
                        gap_text = ' '.join(text1[gap['pos1'][0]:gap['pos1'][1]])
                        highlighted_text1.append(
                            f'<span class="bridge-words" data-bridge-id="{gap_idx}">{gap_text}</span>'
                        )
                        gap_idx += 1
                        gap_found = True
                        break

                if not gap_found:
                    highlighted_text1.append(' '.join(text1[pos1:i1]))

            if pos2 < i2:
                gap_found = False
                for gap in gap_sections:
                    if gap['pos2'][0] == pos2 and gap['pos2'][1] == i2:
                        gap_text = ' '.join(text2[gap['pos2'][0]:gap['pos2'][1]])
                        highlighted_text2.append(
                            f'<span class="bridge-words" data-bridge-id="{gap_idx-1}">{gap_text}</span>'
                        )
                        gap_found = True
                        break

                if not gap_found:
                    highlighted_text2.append(' '.join(text2[pos2:i2]))

            # Add matching portions
            match_text1 = ' '.join(text1[i1:i1+size])
            match_text2 = ' '.join(text2[i2:i2+size])

            highlighted_text1.append(
                f'<span class="highlight clickable" data-match-id="{match_id}" data-pair-id="{pair_id}">'
                f'{match_text1}</span>'
            )
            highlighted_text2.append(
                f'<span class="match-text" data-match-id="{match_id}" data-pair-id="{pair_id}">'
                f'{match_text2}</span>'
            )

            matches[match_id] = {
                'text1': match_text1,
                'text2': match_text2,
                'pos1': i1,
                'pos2': i2,
                'size': size
            }

            match_id += 1
            pos1, pos2 = i1 + size, i2 + size

        # Add remaining text
        if pos1 < len(text1):
            highlighted_text1.append(' '.join(text1[pos1:]))
        if pos2 < len(text2):
            highlighted_text2.append(' '.join(text2[pos2:]))

        return ' '.join(highlighted_text1), ' '.join(highlighted_text2), matches

    @staticmethod
    def generate_comparison_html(analyzer, similarity_threshold=None):
        """Generate interactive HTML comparison of similar text pairs."""
        if similarity_threshold is None:
            similarity_threshold = analyzer.args.similarity_threshold
        html_template = """
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                    background-color: #f5f5f5;
                }
                .comparison-container {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    position: relative;
                }
                .text-box {
                    flex: 1;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
                    height: 500px;  /* Fixed height */
                    overflow-y: auto;  /* Enable vertical scrolling */
                    position: relative;  /* For scroll positioning */
                }
                h2 { color: #333; margin-bottom: 20px; }
                h3 { color: #444; margin-bottom: 15px; }
                .similarity-score {
                    margin-bottom: 10px;
                    font-weight: bold;
                    color: #666;
                    padding: 5px 10px;
                    background-color: #f0f0f0;
                    border-radius: 4px;
                    display: inline-block;
                }
                .highlight {
                    background-color: #fff3b8;
                    padding: 0 2px;
                    border-radius: 3px;
                    transition: all 0.2s ease;
                }
                .highlight.clickable {
                    cursor: pointer;
                }
                .highlight.clickable:hover {
                    background-color: #ffe066;
                }
                .match-text {
                    padding: 0 2px;
                    border-radius: 3px;
                    transition: background-color 0.3s ease;
                    scroll-margin: 100px;  /* Adds margin for scrolling */
                }
                .match-text.active {
                    background-color: #fff3b8;
                }
                .active-highlight {
                    background-color: #ffe066;
                    box-shadow: 0 0 0 2px #ffd700;
                }
                .bridge-words {
                    transition: background-color 0.3s ease;
                    padding: 0 2px;
                    border-radius: 3px;
                }
                .bridge-words.highlighted {
                    background-color: #ffcdd2;
                }
                #toggle-all-bridge-words {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                    z-index: 1000;
                }
                #toggle-all-bridge-words:hover {
                    background-color: #45a049;
                }
                #toggle-all-bridge-words.active {
                    background-color: #f44336;
                }
            </style>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    // Store active highlights for each text pair
                    const activeHighlights = new Map();

                    // Add single bridge words toggle button
                    const toggleButton = document.getElementById('toggle-all-bridge-words');
                    toggleButton.addEventListener('click', function() {
                        this.classList.toggle('active');
                        const bridgeWords = document.querySelectorAll('.bridge-words');
                        bridgeWords.forEach(word => {
                            word.classList.toggle('highlighted');
                        });
                        this.textContent = this.classList.contains('active') ? 'Hide All Bridge Words' : 'Show All Bridge Words';
                    });

                    document.addEventListener('click', function(e) {
                        const clickedHighlight = e.target.closest('.highlight');

                        // If not clicking a highlight, reset all active highlights
                        if (!clickedHighlight) {
                            activeHighlights.forEach((matchId, pairId) => {
                                resetHighlights(matchId, pairId);
                            });
                            activeHighlights.clear();
                            return;
                        }

                        const matchId = clickedHighlight.dataset.matchId;
                        const pairId = clickedHighlight.dataset.pairId;

                        // If this pair already has an active highlight
                        if (activeHighlights.has(pairId)) {
                            const activeMatchId = activeHighlights.get(pairId);
                            // If clicking the same highlight, reset it
                            if (activeMatchId === matchId) {
                                resetHighlights(matchId, pairId);
                                activeHighlights.delete(pairId);
                                return;
                            }
                            // If clicking a different highlight in the same pair, reset the old one
                            resetHighlights(activeMatchId, pairId);
                        }

                        // Show new highlight and scroll to match
                        showHighlight(matchId, pairId);
                        activeHighlights.set(pairId, matchId);

                        // Scroll the right text box to align with the clicked highlight
                        scrollToMatch(matchId, pairId);
                    });

                    function showHighlight(matchId, pairId) {
                        const clickable = document.querySelector(
                            `.highlight[data-match-id="${matchId}"][data-pair-id="${pairId}"]`
                        );
                        const target = document.querySelector(
                            `.match-text[data-match-id="${matchId}"][data-pair-id="${pairId}"]`
                        );

                        if (clickable && target) {
                            clickable.classList.add('active-highlight');
                            target.classList.add('active');
                        }
                    }

                    function resetHighlights(matchId, pairId) {
                        const clickable = document.querySelector(
                            `.highlight[data-match-id="${matchId}"][data-pair-id="${pairId}"]`
                        );
                        const target = document.querySelector(
                            `.match-text[data-match-id="${matchId}"][data-pair-id="${pairId}"]`
                        );

                        if (clickable && target) {
                            clickable.classList.remove('active-highlight');
                            target.classList.remove('active');
                        }
                    }

                    function scrollToMatch(matchId, pairId) {
                        const sourceElement = document.querySelector(
                            `.highlight[data-match-id="${matchId}"][data-pair-id="${pairId}"]`
                        );
                        const targetElement = document.querySelector(
                            `.match-text[data-match-id="${matchId}"][data-pair-id="${pairId}"]`
                        );

                        if (sourceElement && targetElement) {
                            const container = targetElement.closest('.text-box');
                            const sourceRect = sourceElement.getBoundingClientRect();
                            const containerRect = container.getBoundingClientRect();

                            // Calculate the scroll position to align the target with the source
                            const scrollTop = container.scrollTop + targetElement.getBoundingClientRect().top -
                                            containerRect.top - (sourceRect.top - containerRect.top);

                            // Smooth scroll to the calculated position
                            container.scrollTo({
                                top: scrollTop,
                                behavior: 'smooth'
                            });
                        }
                    }
                });
            </script>
        </head>
        <body>
            <h2>Interactive Text Similarity Comparison</h2>
            <button id="toggle-all-bridge-words">Show All Bridge Words</button>
            <p style="color: #666; margin-bottom: 20px;">
                Click on highlighted text in the left column to see matching sections in the right column.
                The right column will automatically scroll to align with the clicked section.
                Use the toggle button to show/hide bridge word highlighting (1-3 words connecting similar sections).
            </p>
        """

        with open("text_comparisons.html", "w", encoding='utf-8') as f:
            f.write(html_template)
            pair_id = 0

            for i in range(len(analyzer.corpus)):
                for j in range(i + 1, len(analyzer.corpus)):
                    if analyzer.dist_mat[i, j] >= similarity_threshold:
                        text1_tokens = analyzer.tokenize(analyzer.corpus[i])
                        text2_tokens = analyzer.tokenize(analyzer.corpus[j])
                        highlighted_text1, highlighted_text2, matches = SimilarityVisualizer.highlight_similarities(
                            text1_tokens, text2_tokens, pair_id
                        )

                        filename1 = str(analyzer.file_paths[i].name)
                        filename2 = str(analyzer.file_paths[j].name)

                        comparison_html = f"""
                        <h3>Comparing Files:</h3>
                        <div class="similarity-score">Similarity: {analyzer.dist_mat[i, j]:.2f}</div>
                        <p>File 1: {filename1}</p>
                        <p>File 2: {filename2}</p>
                        <div class="comparison-container" data-pair-id="{pair_id}">
                            <div class="text-box">
                                <strong>{filename1}:</strong><br>{highlighted_text1}
                            </div>
                            <div class="text-box">
                                <strong>{filename2}:</strong><br>{highlighted_text2}
                            </div>
                        </div>
                        """
                        f.write(comparison_html)
                        pair_id += 1

            f.write("</body></html>")

    @staticmethod
    def plot_similarity_heatmap(analyzer):
        """Generate interactive heatmap visualization."""
        labels = [str(path.name) for path in analyzer.file_paths]
        fig = go.Figure(data=go.Heatmap(
            z=analyzer.dist_mat,
            x=labels,
            y=labels,
            colorscale=[
                [0, 'rgb(255,255,255)'],
                [0.2, 'rgb(220,230,242)'],
                [0.4, 'rgb(158,202,225)'],
                [0.6, 'rgb(49,130,189)'],
                [1.0, 'rgb(8,48,107)']
        ],
        colorbar=dict(title='Similarity (IoU)'),
        # Show values above a certain threshold
        zmin=0.3,
    ))

        fig.update_layout(
            title='Text Similarity Heatmap',
            xaxis_title='Files',
            yaxis_title='Files',
            xaxis=dict(showgrid=False, tickangle=45),
            yaxis=dict(showgrid=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.write_html("similarity_heatmap.html")
        display(fig)

def main():
    # Initialize analyzer
    analyzer = TextSimilarityAnalyzer()

    # Validate parameters
    if analyzer.args.n_out >= analyzer.args.ngram - 1:
        raise ValueError(
            f"n_out ({analyzer.args.n_out}) must be less than ngram size "
            f"({analyzer.args.ngram})-1. Cannot leave out all or more positions than available."
        )

    # Process corpus and compute similarities
    analyzer.load_corpus()
    analyzer.compute_similarity_matrix()

    # Generate visualizations
    visualizer = SimilarityVisualizer()
    visualizer.plot_similarity_heatmap(analyzer)
    visualizer.generate_comparison_html(analyzer)

if __name__ == '__main__':
    main()
