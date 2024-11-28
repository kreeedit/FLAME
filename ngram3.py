import numpy as np
from itertools import combinations
import fargv
import tqdm
from difflib import SequenceMatcher
from IPython.display import display
import re
import plotly.graph_objects as go

# Configuration
DEFAULT_PARAMS = {
    'corpus_tsv': 'corpus.tsv',
    'keep_texts': 200,
    'ngram': 5,
    'n_out': 1,
}

class TextSimilarityAnalyzer:
    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS
        self.args, _ = fargv.fargv(self.params)
        self.encoder = None
        self.corpus = None
        self.tokenized_corpus = None
        self.dist_mat = None

    def tokenize(self, text_str):
        """Split text into tokens."""
        return text_str.split()  # todo: implement a better tokenizer

    def get_encoder(self, corpus):
        """Create a token-to-integer encoder from corpus."""
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
        """Load and preprocess the corpus."""
        corpus = open(self.args.corpus_tsv).read().split('\n')
        corpus = [line.split('\t')[0] for line in corpus if len(line.split('\t')) > 1]
        corpus = [c for c in corpus if len(c) > 50]
        self.corpus = corpus[:self.args.keep_texts]
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
        """Highlight similar portions between two texts with interactive elements."""
        matcher = SequenceMatcher(None, text1, text2)
        highlighted_text1 = []
        highlighted_text2 = []

        match_id = 0
        matches = {}

        pos1 = pos2 = 0
        for i1, i2, size in matcher.get_matching_blocks():
            if size == 0:
                continue

            # Add non-matching portions before match
            if pos1 < i1:
                highlighted_text1.append(' '.join(text1[pos1:i1]))
            if pos2 < i2:
                highlighted_text2.append(' '.join(text2[pos2:i2]))

            # Add matching portions with pair-specific IDs
            match_text1 = ' '.join(text1[i1:i1+size])
            match_text2 = ' '.join(text2[i2:i2+size])

            # Create clickable highlight in text1
            highlighted_text1.append(
                f'<span class="highlight clickable" data-match-id="{match_id}" data-pair-id="{pair_id}">'
                f'{match_text1}</span>'
            )

            # Create normal text with highlightable span in text2
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
    def generate_comparison_html(corpus, dist_mat, similarity_threshold=0.1):
        """Generate interactive HTML comparison of similar text pairs."""
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
                }
                .text-box {
                    flex: 1;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
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
                }
                .match-text.active {
                    background-color: #fff3b8;
                }
                .active-highlight {
                    background-color: #ffe066;
                    box-shadow: 0 0 0 2px #ffd700;
                }
            </style>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    // Store active highlights for each text pair
                    const activeHighlights = new Map();

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

                        // Show new highlight
                        showHighlight(matchId, pairId);
                        activeHighlights.set(pairId, matchId);
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
                });
            </script>
        </head>
        <body>
            <h2>Interactive Text Similarity Comparison</h2>
            <p style="color: #666; margin-bottom: 20px;">
                Click on highlighted text in the left column to see matching sections in the right column.
                Click again to hide the matching highlight.
            </p>
        """

        with open("text_comparisons.html", "w") as f:
            f.write(html_template)

            analyzer = TextSimilarityAnalyzer()  # Create temporary instance for tokenization
            pair_id = 0  # Add unique ID for each text pair

            for i in range(len(corpus)):
                for j in range(i + 1, len(corpus)):
                    if dist_mat[i, j] >= similarity_threshold:
                        text1_tokens = analyzer.tokenize(corpus[i])
                        text2_tokens = analyzer.tokenize(corpus[j])
                        highlighted_text1, highlighted_text2, matches = SimilarityVisualizer.highlight_similarities(
                            text1_tokens, text2_tokens, pair_id
                        )

                        comparison_html = f"""
                        <h3>Text {i+1} vs Text {j+1}</h3>
                        <div class="similarity-score">Similarity: {dist_mat[i, j]:.2f}</div>
                        <div class="comparison-container" data-pair-id="{pair_id}">
                            <div class="text-box">
                                <strong>Text {i+1}:</strong><br>{highlighted_text1}
                            </div>
                            <div class="text-box">
                                <strong>Text {j+1}:</strong><br>{highlighted_text2}
                            </div>
                        </div>
                        """
                        f.write(comparison_html)
                        pair_id += 1

            f.write("</body></html>")

    @staticmethod
    def plot_similarity_heatmap(dist_mat, corpus):
        """Generate interactive heatmap visualization."""
        fig = go.Figure(data=go.Heatmap(
            z=dist_mat,
            x=[f"Text {i+1}" for i in range(len(corpus))],
            y=[f"Text {i+1}" for i in range(len(corpus))],
            colorscale='YlOrRd',
            colorbar=dict(title='Similarity (IoU)'),
        ))

        fig.update_layout(
            title='Text Similarity Heatmap',
            xaxis_title='Texts',
            yaxis_title='Texts',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
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
    visualizer.plot_similarity_heatmap(analyzer.dist_mat, analyzer.corpus)
    visualizer.generate_comparison_html(analyzer.corpus, analyzer.dist_mat)

if __name__ == '__main__':
    main()
