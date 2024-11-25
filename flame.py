import numpy as np
import fargv
from typing import List, Dict, Union, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass
class Config:
    corpus_tsv: str = 'corpus.tsv'
    keep_texts: int = 100
    mode: str = 'ngram'
    ngram: int = 4
    output_file: str = 'dist_mat.npy'
    min_text_length: int = 50
    visualize: bool = True
    plot_output: str = 'similarity_heatmap.png'
    n_similar_pairs: int = 5

class TextAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.encoder: Dict[str, int] = {}
        self.logger = self._setup_logging()
        self.corpus: List[str] = []
        self.dist_mat: np.ndarray = None

    @staticmethod
    def _setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    @staticmethod
    def tokenize(text_str: str) -> List[str]:
        """Tokenize input text into words."""
        return text_str.strip().lower().split()

    def build_encoder(self, corpus: List[str]) -> Dict[str, int]:
        """Build vocabulary encoder from corpus."""
        all_tokens = []
        for text in corpus:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        self.encoder = {token: i for i, token in enumerate(sorted(set(all_tokens)))}
        self.logger.info(f"Vocabulary size: {len(self.encoder)}")
        return self.encoder

    def tokens_to_int(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to integer representation."""
        try:
            return np.array([self.encoder[token] for token in tokens])
        except KeyError as e:
            self.logger.error(f"Unknown token encountered: {e}")
            raise

    def ngrams_to_int(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to n-gram integer representation."""
        int_tokens = self.tokens_to_int(tokens)
        seq_len = len(tokens)
        vocab_size = len(self.encoder)

        if vocab_size ** self.config.ngram >= np.iinfo(np.int64).max:
            raise ValueError(f"N-gram size too large for vocabulary size {vocab_size}")

        int_ngrams = np.zeros(1 + seq_len - self.config.ngram, dtype=np.int64)
        for n in range(self.config.ngram):
            int_ngrams += int_tokens[n:seq_len + 1 - (self.config.ngram - n)] * (vocab_size ** n)
        return int_ngrams

    def skipngrams_to_int(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to skip-gram integer representation."""
        int_tokens = self.tokens_to_int(tokens)
        seq_len = len(tokens)
        vocab_size = len(self.encoder)

        sub_ngrams = [int_tokens[n:seq_len - (self.config.ngram - n)]
                     for n in range(self.config.ngram)]
        sub_ngrams = np.stack(sub_ngrams, axis=0)

        l1out_grams = []
        for n in range(self.config.ngram):
            sub_gram_idx = list(range(self.config.ngram))
            sub_gram_idx.pop(n)
            sub_int_ngrams = np.zeros_like(sub_ngrams[0], dtype=np.int64)

            for idx, gram_idx in enumerate(sub_gram_idx):
                sub_int_ngrams += sub_ngrams[gram_idx] * (vocab_size ** idx)

            l1out_grams.append(sub_int_ngrams)

        return np.concatenate(l1out_grams, axis=0)

    def calculate_distance(self, text1_tokens: List[str], text2_tokens: List[str]) -> float:
        """Calculate IoU distance between two texts based on configured mode."""
        if self.config.mode == 'skip':
            text1_int_ngrams = self.skipngrams_to_int(text1_tokens)
            text2_int_ngrams = self.skipngrams_to_int(text2_tokens)
        elif self.config.mode == 'ngram':
            text1_int_ngrams = self.ngrams_to_int(text1_tokens)
            text2_int_ngrams = self.ngrams_to_int(text2_tokens)
        else:
            raise ValueError('mode should be either skip or ngram')

        intersection = np.intersect1d(text1_int_ngrams, text2_int_ngrams).shape[0]
        union = np.union1d(text1_int_ngrams, text2_int_ngrams).shape[0]

        return intersection / union if union > 0 else 0.0

    def load_corpus(self) -> List[str]:
        """Load and preprocess corpus from TSV file."""
        try:
            with open(self.config.corpus_tsv, 'r', encoding='utf-8') as f:
                corpus = f.read().strip().split('\n')

            corpus = [line.split('\t')[0] for line in corpus if len(line.split('\t')) > 1]
            corpus = [c for c in corpus if len(c) >= self.config.min_text_length]
            corpus = corpus[:self.config.keep_texts]

            self.logger.info(f"Loaded {len(corpus)} texts from corpus")
            self.corpus = corpus
            return corpus
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}")
            raise

    def process_corpus(self) -> None:
        """Main processing function to calculate distance matrix."""
        corpus = self.load_corpus()
        self.build_encoder(corpus)
        tokenized_corpus = [self.tokenize(text) for text in corpus]

        self.dist_mat = np.zeros([len(corpus), len(corpus)])

        # Calculate upper triangle only (symmetric matrix)
        for i in tqdm(range(len(tokenized_corpus)), desc="Processing texts"):
            for j in range(i + 1, len(tokenized_corpus)):
                dist = self.calculate_distance(tokenized_corpus[i], tokenized_corpus[j])
                self.dist_mat[i, j] = dist
                self.dist_mat[j, i] = dist  # Mirror the result

        output_path = Path(self.config.output_file)
        np.save(output_path, self.dist_mat)
        self.logger.info(f"Distance matrix saved to {output_path}")

    def plot_heatmap(self) -> None:
        """Plot similarity matrix as a heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.dist_mat,
                   cmap='YlOrRd',
                   xticklabels=False,
                   yticklabels=False)
        plt.title('Text Similarity Matrix')
        plt.xlabel('Text Index')
        plt.ylabel('Text Index')

        if self.config.plot_output:
            plt.savefig(self.config.plot_output, dpi=300, bbox_inches='tight')
        plt.show()

    def find_similar_pairs(self, n: int = 5, most_similar: bool = True) -> List[Tuple[int, int, float]]:
        """Find n most similar or dissimilar pairs of texts."""
        mask = np.triu(np.ones_like(self.dist_mat), k=1).astype(bool)
        similarities = self.dist_mat * mask

        if most_similar:
            flat_indices = np.argsort(similarities.flatten())[-n:]
        else:
            flat_indices = np.argsort(similarities.flatten())[:n]

        row_indices, col_indices = np.unravel_index(flat_indices, similarities.shape)

        pairs = [(i, j, similarities[i, j])
                for i, j in zip(row_indices, col_indices)
                if similarities[i, j] > 0]
        return sorted(pairs, key=lambda x: x[2], reverse=most_similar)

    def print_text_pair(self, idx1: int, idx2: int) -> None:
        """Print a pair of texts with their similarity score."""
        similarity = self.dist_mat[idx1, idx2]
        print(f"\nSimilarity Score: {similarity:.3f}")
        print("\nText 1:")
        print("-" * 50)
        print(self.corpus[idx1][:500] + "..." if len(self.corpus[idx1]) > 500 else self.corpus[idx1])
        print("\nText 2:")
        print("-" * 50)
        print(self.corpus[idx2][:500] + "..." if len(self.corpus[idx2]) > 500 else self.corpus[idx2])
        print("-" * 50)

    def analyze_results(self) -> None:
        """Analyze and visualize results."""
        if not self.config.visualize:
            return

        # Plot heatmap
        self.plot_heatmap()

        # Find and display similar pairs
        print("\n=== Most Similar Text Pairs ===")
        similar_pairs = self.find_similar_pairs(self.config.n_similar_pairs, most_similar=True)
        for idx1, idx2, score in similar_pairs:
            self.print_text_pair(idx1, idx2)

        print("\n=== Most Dissimilar Text Pairs ===")
        dissimilar_pairs = self.find_similar_pairs(self.config.n_similar_pairs, most_similar=False)
        for idx1, idx2, score in dissimilar_pairs:
            self.print_text_pair(idx1, idx2)

        # Calculate and display statistics
        stats = self.get_similarity_stats()
        print("\n=== Text Similarity Statistics ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(stats)

    def get_similarity_stats(self) -> pd.DataFrame:
        """Calculate similarity statistics for each text."""
        avg_similarities = np.mean(self.dist_mat, axis=1)
        max_similarities = np.max(self.dist_mat, axis=1)
        min_similarities = np.min(self.dist_mat + np.eye(len(self.dist_mat)) * 999, axis=1)

        stats_df = pd.DataFrame({
            'Text_Index': range(len(self.corpus)),
            'Avg_Similarity': avg_similarities,
            'Max_Similarity': max_similarities,
            'Min_Similarity': min_similarities,
            'Text_Preview': [t[:100] + "..." for t in self.corpus]
        })

        return stats_df.sort_values('Avg_Similarity', ascending=False)

def main():
    parser = argparse.ArgumentParser(description="Text Similarity Analysis and Visualization")
    parser.add_argument('--corpus_tsv', type=str, default='corpus.tsv',
                      help='Path to corpus TSV file')
    parser.add_argument('--keep_texts', type=int, default=100,
                      help='Number of texts to process')
    parser.add_argument('--mode', type=str, choices=['ngram', 'skip'],
                      default='ngram', help='Processing mode')
    parser.add_argument('--ngram', type=int, default=4,
                      help='N-gram size')
    parser.add_argument('--output_file', type=str, default='dist_mat.npy',
                      help='Output file path')
    parser.add_argument('--min_text_length', type=int, default=50,
                      help='Minimum text length to process')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization and analysis')
    parser.add_argument('--plot_output', type=str, default='similarity_heatmap.png',
                      help='Path to save heatmap plot')
    parser.add_argument('--n_similar_pairs', type=int, default=5,
                      help='Number of similar/dissimilar pairs to show')

    args = parser.parse_args()
    config = Config(**vars(args))

    analyzer = TextAnalyzer(config)
    analyzer.process_corpus()
    analyzer.analyze_results()

if __name__ == '__main__':
    main()
