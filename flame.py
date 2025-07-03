import numpy as np
import os
import pathlib
import re
import unicodedata
import fargv
import tqdm
import Levenshtein
from itertools import combinations
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import tempfile

import plotly.graph_objects as go
from scipy.sparse import coo_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import threshold_otsu
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def fast_str_to_numpy(s: str, dtype=np.uint16) -> np.ndarray:
    """Efficiently converts a string to a NumPy array via byte encoding."""
    if dtype == np.uint16:
        return np.frombuffer(s.encode('utf-16le'), dtype=dtype)
    elif dtype == np.uint32:
        return np.frombuffer(s.encode('utf-32le'), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype for fast string conversion: {dtype}")

def fast_numpy_to_str(np_arr: np.ndarray) -> str:
    """Efficiently converts a NumPy array of character codes back to a string."""
    if np_arr.dtype == np.uint16:
        return np_arr.tobytes().decode('utf-16le')
    elif np_arr.dtype == np.uint32:
        return np_arr.tobytes().decode('utf-32le')
    else:
        raise ValueError(f"Unsupported dtype for fast NumPy conversion: {np_arr.dtype}")

def suggest_vocab_size_optimized(
    corpus: list[str],
    min_word_freq: int = 3,
    max_affix_len: int = 6,
    coverage_percentile: float = 0.85
) -> int:
    """Proposes the vocabulary size for the tokenizer based on corpus analysis."""
    print("\n--- Starting Automatic Vocab Size Suggestion (Optimized) ---")

    print(f"Counting word frequencies...")
    word_counts = Counter()
    tokenizer = re.compile(r'\b\w+\b')
    for doc in tqdm.tqdm(corpus, desc="Tokenizing corpus"):
        word_counts.update(token.lower() for token in tokenizer.findall(doc))

    frequent_word_counts = {
        word: count for word, count in word_counts.items()
        if count >= min_word_freq and len(word) > 1
    }
    print(f"Found {len(word_counts)} unique words, keeping {len(frequent_word_counts)} with frequency >= {min_word_freq}.")

    if not frequent_word_counts:
        print("Warning: No frequent words found. Returning a default vocab size.")
        return 2000

    print("Creating NumPy structured array for processing...")
    max_len = max(len(w) for w in frequent_word_counts.keys())
    structured_dtype = [('word', f'U{max_len}'), ('freq', 'i4'), ('rev_word', f'U{max_len}')]

    word_data = np.array([
        (word, freq, word[::-1]) for word, freq in frequent_word_counts.items()
    ], dtype=structured_dtype)

    affix_counts = Counter()

    print("Finding common prefixes with NumPy sort...")
    word_data.sort(order='word')
    for i in tqdm.tqdm(range(len(word_data) - 1), desc="Analyzing prefixes"):
        w1, w2 = word_data[i], word_data[i+1]
        common_prefix = os.path.commonprefix([w1['word'], w2['word']])
        if 1 < len(common_prefix) <= max_affix_len:
            affix_counts[common_prefix] += w1['freq'] + w2['freq']

    print("Finding common suffixes with NumPy sort...")
    word_data.sort(order='rev_word')
    for i in tqdm.tqdm(range(len(word_data) - 1), desc="Analyzing suffixes"):
        rw1, rw2 = word_data[i], word_data[i+1]
        common_rev_suffix = os.path.commonprefix([rw1['rev_word'], rw2['rev_word']])
        if 1 < len(common_rev_suffix) <= max_affix_len:
            affix_counts[common_rev_suffix[::-1]] += rw1['freq'] + rw2['freq']

    print(f"Found {len(affix_counts)} potential affixes (morpheme candidates).")

    if not affix_counts:
        print("Warning: No common affixes found. Returning a default vocab size.")
        return 2000

    print(f"Calculating vocab size for {coverage_percentile:.0%} coverage...")
    sorted_affixes = affix_counts.most_common()
    total_affix_occurrences = sum(count for _, count in sorted_affixes)
    target_coverage_sum = total_affix_occurrences * coverage_percentile

    current_sum = 0
    suggested_size = 0
    for affix, count in sorted_affixes:
        current_sum += count
        suggested_size += 1
        if current_sum >= target_coverage_sum:
            break

    base_size = 256
    suggested_size_with_base = suggested_size + base_size

    print(f"Analysis complete. {suggested_size} affixes are needed to cover {coverage_percentile:.0%} of all affix occurrences.")
    print(f"--- Suggested Vocab Size: {suggested_size_with_base} ---")

    return suggested_size_with_base


class Alphabet(ABC):
    """Abstract Base Class for defining an alphabet handling interface."""
    @property
    @abstractmethod
    def src_alphabet(self) -> str: pass

    @property
    @abstractmethod
    def dst_alphabet(self) -> str: pass

    @property
    @abstractmethod
    def unknown_chr(self) -> str: pass


class AlphabetBMP(Alphabet):
    """Handles character sets within the Unicode Basic Multilingual Plane (BMP)."""
    def __init__(self, sample: Union[str, None] = None, alphabet_str: Union[str, None] = None, unknown_chr: str = ''):
        assert bool(sample is None) != bool(alphabet_str is None), "Either 'sample' or 'alphabet_str' must be provided, but not both."
        if sample is not None:
            self.__src_alphabet_str = ''.join(sorted(set(sample) - set(unknown_chr)))
        else:
            if unknown_chr:
                assert unknown_chr not in alphabet_str, "Alphabet string must not contain the unknown character."
            assert len(alphabet_str) == len(set(alphabet_str)), "Alphabet string must not contain duplicates."
            self.__src_alphabet_str = alphabet_str
        self.__unknown_chr = unknown_chr
        self._chr2chr, self._npint2int = self._create_mappers()

    def _create_mappers(self) -> Tuple[defaultdict, np.ndarray]:
        chr2chr = defaultdict(lambda: self.__unknown_chr)
        chr2chr.update({a: a for a in self.__src_alphabet_str})
        full_str = self.__unknown_chr + self.__src_alphabet_str
        np_int2int = np.zeros(2**16, dtype=np.uint16)
        if self.__unknown_chr:
            np_int2int.fill(ord(self.__unknown_chr))
        for c in full_str:
            np_int2int[ord(c)] = ord(c)
        return chr2chr, np_int2int

    @property
    def src_alphabet(self): return self.__src_alphabet_str

    @property
    def dst_alphabet(self): return self.__src_alphabet_str

    @property
    def unknown_chr(self): return self.__unknown_chr

    def __call__(self, text: str) -> str:
        return fast_numpy_to_str(self._npint2int[fast_str_to_numpy(text)])

    def get_encoding_information_loss(self, text: str) -> float:
        np_text = fast_str_to_numpy(text)
        mapped_np_text = self._npint2int[np_text]
        return np.mean(np_text != mapped_np_text)


class CharacterMapper(AlphabetBMP):
    """Extends AlphabetBMP to support custom, user-defined character-to-character mappings."""
    def __init__(self, src_alphabet: str, mapping_dict: Dict[str, str], unknown_chr: str = ''):
        self.__custom_mapping_dict = mapping_dict
        super().__init__(alphabet_str=src_alphabet, unknown_chr=unknown_chr)
        self.__dst_alphabet_str = ''.join(sorted(set(self.__custom_mapping_dict.values()) - set(unknown_chr)))

    def _create_mappers(self) -> Tuple[defaultdict, np.ndarray]:
        chr2chr, np_int2int = super()._create_mappers()
        for src_char, dst_char in self.__custom_mapping_dict.items():
            np_int2int[ord(src_char)] = ord(dst_char)
        chr2chr.update(self.__custom_mapping_dict)
        return chr2chr, np_int2int

    def _update_mappings(self, new_mappings: Dict[str, str]):
        self.__custom_mapping_dict.update(new_mappings)
        self._chr2chr, self._npint2int = self._create_mappers()


class AdaptiveAlphabet(CharacterMapper):
    """An adaptive normalizer that learns character mappings from a text corpus."""
    def __init__(self, src_alphabet: str, unknown_chr: str = '', initial_mapping_dict: Dict[str, str] = None):
        mapping_dict = initial_mapping_dict if initial_mapping_dict is not None else {}
        super().__init__(src_alphabet=src_alphabet, mapping_dict=mapping_dict, unknown_chr=unknown_chr)

    def analyze_lost_chars(self, text: str) -> defaultdict:
        lost_chars_count = defaultdict(int)
        np_text = fast_str_to_numpy(text)
        mapped_np_text = fast_str_to_numpy(self(text))
        if not self.unknown_chr:
            return lost_chars_count
        unknown_chr_ord = ord(self.unknown_chr)
        lost_indices = np.where(mapped_np_text == unknown_chr_ord)[0]
        original_lost_chars = np_text[lost_indices]
        for char_ord in original_lost_chars:
            if char_ord != unknown_chr_ord:
                lost_chars_count[chr(char_ord)] += 1
        return lost_chars_count

    def learn_mappings(self, text: str, strategy: str = 'normalize', min_freq: int = 2):
        print("\nStarting Autonomous Character Normalization")
        lost_chars = self.analyze_lost_chars(text)
        if not lost_chars:
            print("No characters require normalization. The source alphabet is comprehensive.")
            print("--- Character Normalization Complete ---\n")
            return

        unfound_chars_list = sorted(list(lost_chars.keys()))
        print(f"Found {len(lost_chars)} unique characters not in the source alphabet: [{' '.join(unfound_chars_list)}]")

        new_mappings = {}
        if strategy == 'normalize':
            print(f"Applying '{strategy}' strategy for characters with frequency >= {min_freq}...")
            for char, count in sorted(lost_chars.items(), key=lambda item: item[1], reverse=True):
                if count >= min_freq:
                    normalized_char_seq = unicodedata.normalize('NFKD', char)
                    if normalized_char_seq:
                        normalized_char = normalized_char_seq[0]
                        if normalized_char in self.src_alphabet and normalized_char != char:
                            new_mappings[char] = normalized_char
                            print(f"  + Suggesting mapping: '{char}' -> '{normalized_char}' (found {count} times)")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if new_mappings:
            print(f"Generated {len(new_mappings)} new mapping rules. Updating normalizer.")
            self._update_mappings(new_mappings)
        else:
            print("No new normalization rules were generated based on the current strategy and threshold.")
        print("--- Character Normalization Complete ---\n")

DEFAULT_PARAMS = {
    'input_path': None,
    'input_path2': None,
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 5,
    'n_out': 1,
    'min_text_length': 150,
    'similarity_threshold': 'auto',
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 1,
    'vocab_size': 'auto',
    'vocab_min_word_freq': 3,
    'vocab_coverage': 0.85,
}

class Flame:
    """Text similarity analysis pipeline."""
    def __init__(self, params=None, tmp_dir: str = '.'):
        self.params = params or DEFAULT_PARAMS
        self.args, _ = fargv.fargv(self.params)
        self.is_inter_comparison = bool(self.args.input_path2 and os.path.isdir(self.args.input_path2))
        self.tmp_dir = tmp_dir

        self.corpus: List[str] = []
        self.file_paths: List[pathlib.Path] = []
        self.tokenized_corpus: List[List[str]] = []
        self.corpus2: List[str] = []
        self.file_paths2: List[pathlib.Path] = []
        self.tokenized_corpus2: List[List[str]] = []
        self.encoder: Dict[str, int] = {}
        self.dist_mat = None
        self.tokenizer_model = None

    def _find_text_files(self, input_path: str) -> List[pathlib.Path]:
        path = pathlib.Path(input_path)
        if not path.exists() or not path.is_dir():
            print(f"Warning: Input path {path} does not exist or is not a directory. Skipping.")
            return []
        return list(path.rglob(f"*{self.args.file_suffix}"))

    def _read_text_file(self, file_path: pathlib.Path) -> Union[str, None]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ' '.join(f.read().strip().split())
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            return None

    def _load_corpus_from_path(self, path_str: str) -> Tuple[List[str], List[pathlib.Path]]:
        file_paths = self._find_text_files(path_str)
        print(f"Found {len(file_paths)} files in '{path_str}' with suffix '{self.args.file_suffix}'")
        corpus_data, loaded_paths = [], []
        limit = self.args.keep_texts
        for file_path in tqdm.tqdm(file_paths, desc=f"Loading files from {os.path.basename(path_str)}"):
            text = self._read_text_file(file_path)
            if text and len(text) >= self.args.min_text_length:
                corpus_data.append(text)
                loaded_paths.append(file_path)
                if len(corpus_data) >= limit:
                    print(f"Reached limit of {limit} texts for this directory.")
                    break
        return corpus_data, loaded_paths

    def load_corpus(self):
        self.corpus, self.file_paths = self._load_corpus_from_path(self.args.input_path)
        if self.is_inter_comparison:
            print("\n--- Two-directory comparison mode activated ---")
            self.corpus2, self.file_paths2 = self._load_corpus_from_path(self.args.input_path2)
            if not self.corpus2:
                print("Warning: Second directory is empty or invalid. Reverting to single-directory mode.")
                self.is_inter_comparison = False

        if not self.corpus:
            print("Error: No valid texts loaded from the primary directory. Aborting.")
            return

        corpus_for_learning = self.corpus + self.corpus2
        print(f"\nTotal texts for analysis: {len(corpus_for_learning)}")

        lowercased_corpus = [text.lower() for text in corpus_for_learning]

        mufi_char_mappings = {
            'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ĳ': 'ij', 'ð': 'dh', 'þ': 'th', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬆ': 'st',
            'ſ': 's', 'ꝇ': 'l', 'ꝑ': 'p', 'v': 'u', 'j': 'i', 'ꝛ': 'r', 'ƿ': 'w', 'ᵹ': 'g', 'ꝺ': 'd', 'ꝼ': 'f'
        }
        one_to_many_mappings = {k: v for k, v in mufi_char_mappings.items() if len(v) > 1}
        one_to_one_mappings = {k: v for k, v in mufi_char_mappings.items() if len(v) == 1}

        print("\n--- Applying 1-to-many character replacements (e.g., ligatures) ---")
        pre_processed_corpus = []
        for text in tqdm.tqdm(lowercased_corpus, desc="Pre-processing"):
            for src, dst in one_to_many_mappings.items():
                text = text.replace(src, dst)
            pre_processed_corpus.append(text)

        full_corpus_text = "\n".join(pre_processed_corpus)
        target_alphabet = self.args.char_norm_alphabet.replace(' ', '')

        print("\n--- Initializing Character Normalizer with 1-to-1 MUFI rules ---")
        learner = AdaptiveAlphabet(
            src_alphabet=target_alphabet,
            unknown_chr=' ',
            initial_mapping_dict=one_to_one_mappings
        )
        learner.learn_mappings(
            full_corpus_text,
            strategy=self.args.char_norm_strategy,
            min_freq=self.args.char_norm_min_freq
        )

        print("Applying final normalization rules to the corpus...")
        normalized_corpus_full = [learner(text) for text in tqdm.tqdm(pre_processed_corpus, desc="Normalizing")]

        print("\n--- Training Subword Tokenizer ---")

        corpus_file = os.path.join(self.tmp_dir, 'temp_corpus.txt')
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for line in normalized_corpus_full:
                f.write(line + '\n')

        if str(self.args.vocab_size).lower() == 'auto':
            vocab_size = suggest_vocab_size_optimized(
                normalized_corpus_full,
                min_word_freq=self.args.vocab_min_word_freq,
                coverage_percentile=self.args.vocab_coverage
            )
        else:
            try:
                vocab_size = int(self.args.vocab_size)
                print(f"\n--- Using specified vocab size: {vocab_size} ---")
            except ValueError:
                print(f"Error: Invalid vocab_size '{self.args.vocab_size}'. Please provide a number or 'auto'.")
                return

        print("Counting unique words to determine maximum possible vocab size...")
        all_words = set(word for line in normalized_corpus_full for word in line.split())
        max_possible_size = len(all_words) + 256

        if vocab_size > max_possible_size:
            print(f"Warning: Requested vocab size ({vocab_size}) is larger than the number of unique words found ({len(all_words)}).")
            print(f"Adjusting vocab size to the maximum possible value: {max_possible_size}")
            vocab_size = max_possible_size

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )

        model_path = os.path.join(self.tmp_dir, 'bpe_tokenizer.json')
        print(f"INFO: Training BPE tokenizer with final vocab_size: {vocab_size}")

        tokenizer.train([corpus_file], trainer=trainer)
        tokenizer.save(model_path)
        self.tokenizer_model = Tokenizer.from_file(model_path)

        print("--- Subword Tokenizer Trained and Loaded ---\n")

        print("Tokenizing all documents using subword tokenizer...")
        all_tokenized = [self.tokenize(text) for text in tqdm.tqdm(normalized_corpus_full, desc="Tokenizing")]

        self.encoder = self.get_encoder(all_tokenized)
        if not self.encoder:
            print("Warning: Encoder could not be built (empty vocabulary). Aborting.")
            return

        if self.is_inter_comparison:
            len_corpus1 = len(self.corpus)
            self.tokenized_corpus = all_tokenized[:len_corpus1]
            self.tokenized_corpus2 = all_tokenized[len_corpus1:]
        else:
            self.tokenized_corpus = all_tokenized

    def tokenize(self, text_str: str) -> List[str]:
        """Tokenizes a string using the trained BPE model from the 'tokenizers' library."""
        if not self.tokenizer_model:
            raise RuntimeError("Tokenizer model is not loaded. Run load_corpus first.")
        return self.tokenizer_model.encode(text_str).tokens

    def get_encoder(self, all_tokenized_docs: List[List[str]]) -> Dict[str, int]:
        print("Building vocabulary encoder...")
        all_tokens = {token for doc in all_tokenized_docs for token in doc}
        unique_tokens = sorted(list(all_tokens))
        return {token: i for i, token in enumerate(unique_tokens)}

    def tokens_to_int(self, tokens: List[str]) -> List[int]:
        return [self.encoder[token] for token in tokens if token in self.encoder]

    def _determine_auto_threshold(self, method: str = 'otsu', percentile: int = 99) -> float:
        if self.dist_mat is None or self.dist_mat.nnz == 0:
            print("Warning: Cannot determine auto threshold, similarity matrix is empty.")
            return 0.01
        scores = self.dist_mat.data
        if method == 'otsu':
            try:
                auto_threshold = threshold_otsu(scores)
                print(f"Automatically determined threshold (Otsu's method): {auto_threshold:.4f}")
                return float(auto_threshold)
            except ImportError:
                print("Warning: scikit-image is not installed. Falling back to percentile method.")
                return self._determine_auto_threshold(method='percentile', percentile=percentile)
        elif method == 'percentile':
            if scores.size == 0: return 0.01
            auto_threshold = np.percentile(scores, percentile)
            print(f"Automatically determined threshold ({percentile}th percentile): {auto_threshold:.4f}")
            return float(auto_threshold)
        else:
            raise ValueError(f"Unknown auto-threshold method: {method}")

    def leave_n_out_grams(self, tokens: List[str]) -> np.ndarray:
        MOD = 2**61 - 1
        int_tokens = np.array(self.tokens_to_int(tokens), dtype=np.int64)
        if int_tokens.size == 0: return np.array([])
        seq_len = len(int_tokens)
        elements_to_keep = self.args.ngram - self.args.n_out
        if elements_to_keep < 1 or seq_len < self.args.ngram: return np.array([])
        num_ngrams = seq_len - self.args.ngram + 1
        if num_ngrams <= 0: return np.array([])
        sub_ngrams_matrix = np.array([int_tokens[i:i + num_ngrams] for i in range(self.args.ngram)])
        keep_indices_combinations = list(combinations(range(self.args.ngram), elements_to_keep))
        lnout_grams_list = []
        vocab_size = len(self.encoder)
        if vocab_size == 0: return np.array([])
        for combo_indices in keep_indices_combinations:
            combined_int_values = np.zeros(num_ngrams, dtype=np.int64)
            for i, original_idx in enumerate(combo_indices):
                power_val = pow(vocab_size, i, MOD)
                current_term = (sub_ngrams_matrix[original_idx, :] * power_val) % MOD
                combined_int_values = (combined_int_values + current_term) % MOD
            lnout_grams_list.append(combined_int_values)
        return np.concatenate(lnout_grams_list) if lnout_grams_list else np.array([])

    def compute_similarity_matrix(self):
        if not self.tokenized_corpus:
            print("Corpus is not tokenized. Skipping similarity matrix computation.")
            return

        print("Generating features for all documents...")
        doc_features1 = [self.leave_n_out_grams(tokens) for tokens in tqdm.tqdm(self.tokenized_corpus, desc="Generating features for Corpus 1")]
        all_doc_features_lists = [doc_features1]
        doc_features2 = None

        if self.is_inter_comparison:
            doc_features2 = [self.leave_n_out_grams(tokens) for tokens in tqdm.tqdm(self.tokenized_corpus2, desc="Generating features for Corpus 2")]
            all_doc_features_lists.append(doc_features2)

        print("Building global feature vocabulary iteratively...")
        feature_to_col_idx = {}
        next_col_idx = 0
        for doc_list in all_doc_features_lists:
            for features in tqdm.tqdm(doc_list, desc="Building vocabulary"):
                if features.size > 0:
                    for feature in features:
                        if feature not in feature_to_col_idx:
                            feature_to_col_idx[feature] = next_col_idx
                            next_col_idx += 1

        if not feature_to_col_idx:
            print("No features could be generated from the corpus.")
            shape = (len(self.corpus), len(self.corpus2 if self.is_inter_comparison else self.corpus))
            self.dist_mat = coo_matrix(shape, dtype=np.float32)
            return

        print(f"Vocabulary built. Found {len(feature_to_col_idx)} unique features.")

        def create_sparse_matrix(doc_features_list, num_docs, feature_map):
            rows, cols, data = [], [], []
            for doc_id, features in enumerate(doc_features_list):
                if features.size > 0:
                    unique_doc_features, counts = np.unique(features, return_counts=True)
                    for feature, count in zip(unique_doc_features, counts):
                        if feature in feature_map:
                            rows.append(doc_id)
                            cols.append(feature_map[feature])
                            data.append(count)
            if not rows:
                return coo_matrix((num_docs, len(feature_map)), dtype=np.float32).tocsr()
            return coo_matrix((data, (rows, cols)), shape=(num_docs, len(feature_map)), dtype=np.float32).tocsr()

        print("Creating sparse document-feature matrices...")
        if self.is_inter_comparison:
            matrix1 = create_sparse_matrix(doc_features1, len(self.corpus), feature_to_col_idx)
            matrix2 = create_sparse_matrix(doc_features2, len(self.corpus2), feature_to_col_idx)
            print("Calculating inter-corpus Cosine similarity (sparse output)...")
            self.dist_mat = cosine_similarity(matrix1, matrix2, dense_output=False)
        else:
            matrix = create_sparse_matrix(doc_features1, len(self.corpus), feature_to_col_idx)
            print("Calculating all-pairs Cosine similarity (sparse output)...")
            self.dist_mat = cosine_similarity(matrix, dense_output=False)

        print("Similarity matrix computation complete.")
        save_npz(os.path.join(self.tmp_dir, 'dist_mat.npz'), self.dist_mat)


class SimilarityVisualizer:
    detokenizer = TreebankWordDetokenizer()

    @staticmethod
    def _render_gap_html(gap1_tokens: List[str], gap2_tokens: List[str], is_bridge: bool, similarity_threshold: float = 0.65) -> Tuple[str, str]:
        words1 = [token for token in gap1_tokens if token.isalnum()]
        words2 = [token for token in gap2_tokens if token.isalnum()]
        num_words1 = len(words1)
        num_words2 = len(words2)
        str1 = SimilarityVisualizer.detokenizer.detokenize(gap1_tokens)
        str2 = SimilarityVisualizer.detokenizer.detokenize(gap2_tokens)
        if not str1 and not str2:
            return "", ""
        if (num_words1 <= 3) and (num_words2 <= 3) and (num_words1 + num_words2 > 0) :
            ratio = Levenshtein.ratio(str1.lower(), str2.lower())
            if ratio >= similarity_threshold:
                html1 = f'<span class="bridge-word-similar-static">{str1}</span>'
                html2 = f'<span class="bridge-word-similar-static">{str2}</span>'
            else:
                html1 = f'<span class="bridge-word-dissimilar">{str1}</span>'
                html2 = f'<span class="bridge-word-dissimilar">{str2}</span>'
        else:
            html1 = str1
            html2 = str2
        if (num_words1 <= 3) and (num_words2 <= 3) and (num_words1 + num_words2 > 0) :
            html1 = f'<span class="bridge-words">{html1}</span>' if html1 else ""
            html2 = f'<span class="bridge-words">{html2}</span>' if html2 else ""
        return html1, html2

    @staticmethod
    def highlight_similarities(text1_original_tokens: List[str], text2_original_tokens: List[str], pair_id: int) -> Tuple[str, str]:
        analysis_tokens1 = [t.lower() for t in text1_original_tokens if t.isalnum()]
        analysis_tokens2 = [t.lower() for t in text2_original_tokens if t.isalnum()]
        map_analysis_to_original1 = [i for i, token in enumerate(text1_original_tokens) if token.isalnum()]
        map_analysis_to_original2 = [i for i, token in enumerate(text2_original_tokens) if token.isalnum()]
        if not analysis_tokens1 or not analysis_tokens2:
            return SimilarityVisualizer.detokenizer.detokenize(text1_original_tokens), \
                   SimilarityVisualizer.detokenizer.detokenize(text2_original_tokens)
        matcher = SequenceMatcher(None, analysis_tokens1, analysis_tokens2, autojunk=False)
        raw_matching_blocks = matcher.get_matching_blocks()
        highlighted_html_text1, highlighted_html_text2 = [], []
        bridge_word_sections = []
        for idx in range(len(raw_matching_blocks) - 1):
            curr, next_ = raw_matching_blocks[idx], raw_matching_blocks[idx+1]
            gap1_start_analysis, gap1_end_analysis = curr.a + curr.size, next_.a
            gap2_start_analysis, gap2_end_analysis = curr.b + curr.size, next_.b
            if (1 <= (gap1_end_analysis - gap1_start_analysis) <= 5) and \
               (1 <= (gap2_end_analysis - gap2_start_analysis) <= 5):
                g1s_orig = map_analysis_to_original1[gap1_start_analysis]
                g1e_orig = map_analysis_to_original1[gap1_end_analysis - 1] + 1
                g2s_orig = map_analysis_to_original2[gap2_start_analysis]
                g2e_orig = map_analysis_to_original2[gap2_end_analysis - 1] + 1
                bridge_word_sections.append({'t1i': (g1s_orig, g1e_orig), 't2i': (g2s_orig, g2e_orig)})
        pos1, pos2, m_id = 0, 0, 0
        for a_analysis, b_analysis, size in raw_matching_blocks:
            if size == 0: continue
            a_start_orig = map_analysis_to_original1[a_analysis]
            b_start_orig = map_analysis_to_original2[b_analysis]
            a_end_orig = map_analysis_to_original1[a_analysis + size - 1] + 1
            b_end_orig = map_analysis_to_original2[b_analysis + size - 1] + 1
            if pos1 < a_start_orig or pos2 < b_start_orig:
                gap1_tokens = text1_original_tokens[pos1:a_start_orig]
                gap2_tokens = text2_original_tokens[pos2:b_start_orig]
                is_b = any(b['t1i'] == (pos1, a_start_orig) for b in bridge_word_sections)
                gap1_html, gap2_html = SimilarityVisualizer._render_gap_html(gap1_tokens, gap2_tokens, is_b)
                if gap1_html: highlighted_html_text1.append(gap1_html)
                if gap2_html: highlighted_html_text2.append(gap2_html)
            m_txt1_raw = SimilarityVisualizer.detokenizer.detokenize(text1_original_tokens[a_start_orig:a_end_orig])
            m_txt1_processed = re.sub(r'([^\w\s])', r'<span class="punct-in-match">\1</span>', m_txt1_raw)
            m_txt1_html = f'<span class="highlight clickable" data-match-id="{m_id}" data-pair-id="{pair_id}">{m_txt1_processed}</span>'
            highlighted_html_text1.append(m_txt1_html)
            m_txt2_raw = SimilarityVisualizer.detokenizer.detokenize(text2_original_tokens[b_start_orig:b_end_orig])
            m_txt2_processed = re.sub(r'([^\w\s])', r'<span class="punct-in-match">\1</span>', m_txt2_raw)
            m_txt2_html = f'<span class="match-text" data-match-id="{m_id}" data-pair-id="{pair_id}">{m_txt2_processed}</span>'
            highlighted_html_text2.append(m_txt2_html)
            m_id += 1
            pos1, pos2 = a_end_orig, b_end_orig
        if pos1 < len(text1_original_tokens):
            highlighted_html_text1.append(SimilarityVisualizer.detokenizer.detokenize(text1_original_tokens[pos1:]))
        if pos2 < len(text2_original_tokens):
            highlighted_html_text2.append(SimilarityVisualizer.detokenizer.detokenize(text2_original_tokens[pos2:]))
        return " ".join(filter(None, highlighted_html_text1)), " ".join(filter(None, highlighted_html_text2))

    @staticmethod
    def generate_comparison_html(analyzer, similarity_threshold: float, max_file_size=20 * 1024 * 1024, interactive_range_width: float = 0.3):
        if analyzer.dist_mat is None or not analyzer.corpus:
            print("ERROR: Distance matrix or corpus not found. Cannot generate HTML report.")
            return

        html_template_start = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Text Similarity Comparison</title><style>
        body{font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;margin:20px;line-height:1.6;background-color:#f8f9fa}
        .comparison-block{margin-bottom:2em;background-color:#fff;padding:1.5em;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.05);border:1px solid #dee2e6}
        .comparison-container{display:flex;gap:20px;flex-wrap:wrap}@media(min-width:768px){.comparison-container{flex-wrap:nowrap}}
        .text-box{flex:1 1 100%;min-width:300px;padding:15px;border:1px solid #ced4da;border-radius:5px;background-color:#fff;height:400px;overflow-y:auto;position:relative}
        h2{color:#212529;border-bottom:2px solid #e9ecef;padding-bottom:.5em; margin-top: 0;}
        h3{color:#343a40;margin-top:0}
        .similarity-score{font-weight:700;color:#0056b3}
        .file-info{font-size:.9em;color:#6c757d;margin-bottom:.5em;font-weight:700}
        .highlight{background-color:#fff3b8;border-radius:3px}
        .highlight.clickable{cursor:pointer;transition:background-color .2s}
        .highlight.clickable:hover{background-color:#ffe066}
        .active-highlight{background-color:#ffd700;box-shadow:0 0 0 2px #ffc107}
        .match-text{border-radius:3px;transition:background-color .3s}
        .hover-highlight {background-color: #ffe066 !important; box-shadow: 0 0 0 2px #ffc107;}
        .match-text.active { background-color:#fff3b8; }
        .bridge-word-similar-static.active { background-color: #fff9e0; border-radius: 3px; padding: 0 2px; }
        .bridge-words.show-dissimilar .bridge-word-dissimilar { background-color: #ffcdd2; border-radius: 3px; padding: 0 2px; }
        .bridge-words.highlighted { outline: 1px dotted #e57373; }
        .bridge-words.highlighted .bridge-word-similar-static { background-color: #fff9e0; border-radius: 3px; padding: 0 2px; }
        .bridge-words.highlighted .bridge-word-dissimilar { background-color: #ffcdd2; border-radius: 3px; padding: 0 2px; }
        #controls{margin-bottom:1.5em;background-color:#fff;padding:1em 1.5em;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.05);border:1px solid #dee2e6;}
        .button-container { display: flex; gap: 10px; margin-top: 1em; }
        .control-button { background-color:#28a745;color:#fff;padding:0.4em 0.8em;border:none;border-radius:4px;cursor:pointer;font-weight:700; font-size: 0.9em; }
        .control-button.active { background-color:#dc3545; }
        .filter-container { margin-top: 1.5em; padding-top: 1em; border-top: 1px solid #e9ecef; }
        .slider-wrapper { position: relative; height: 30px; }
        .slider-label { font-weight: 600; color: #495057; margin-bottom: 0.5em; }
        .slider-values { display: flex; justify-content: space-between; font-family: monospace; font-size: 1.1em; color: #0056b3; margin-bottom: -5px; }
        .form-control-range { position: absolute; width: 100%; -webkit-appearance: none; appearance: none; background: transparent; pointer-events: none; }
        .form-control-range:focus { outline: none; }
        .form-control-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; height: 18px; width: 18px; background: #007bff; border-radius: 50%; border: 2px solid #fff; box-shadow: 0 0 5px rgba(0,0,0,0.2); pointer-events: auto; cursor: pointer; }
        .form-control-range::-moz-range-thumb { height: 14px; width: 14px; background: #007bff; border-radius: 50%; border: 2px solid #fff; pointer-events: auto; cursor: pointer; }
        .slider-track { position: absolute; width: 100%; height: 4px; background-color: #ddd; top: 7px; border-radius: 3px; }
        </style></head><body>
        <div id="controls">
            <h2>Interactive Text Similarity Comparison</h2>
            <div class="filter-container">
                <label class="slider-label">Filter by Cosine Similarity</label>
                <div class="slider-values">
                    <span id="min-similarity-val">0.000</span>
                    <span id="max-similarity-val">1.000</span>
                </div>
                <div class="slider-wrapper">
                    <div class="slider-track"></div>
                    <input type="range" min="0" max="1" value="0.75" step="0.001" class="form-control-range" id="min-similarity">
                    <input type="range" min="0" max="1" value="1.0" step="0.001" class="form-control-range" id="max-similarity">
                </div>
            </div>
            <div class="button-container">
                <button id="toggle-all-bridge-words" class="control-button">Show All Bridge Words</button>
                <button id="toggle-all-similarities" class="control-button">Show All Similarities</button>
            </div>
        </div>"""
        html_template_end = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const toggleSimilaritiesBtn = document.getElementById("toggle-all-similarities");
    const comparisonBlocks = document.querySelectorAll(".comparison-block");
    function clearAllActiveHighlights(block) {
        block.querySelectorAll(".active-highlight").forEach(el => el.classList.remove("active-highlight"));
        block.querySelectorAll(".match-text.active").forEach(el => el.classList.remove("active"));
    }
    function highlightPair(block, pairId, matchId) {
        block.querySelector(`.highlight.clickable[data-pair-id="${pairId}"][data-match-id="${matchId}"]`)?.classList.add("active-highlight");
        block.querySelector(`.match-text[data-pair-id="${pairId}"][data-match-id="${matchId}"]`)?.classList.add("active");
    }
    function scrollToPartner(block, pairId, matchId) {
        const partnerEl = block.querySelector(`.match-text[data-pair-id="${pairId}"][data-match-id="${matchId}"], .match-text.active[data-pair-id="${pairId}"][data-match-id="${matchId}"]`);
        const textBox = partnerEl?.closest(".text-box");
        if (textBox && partnerEl) {
            const boxRect = textBox.getBoundingClientRect();
            const partnerRect = partnerEl.getBoundingClientRect();
            const scrollOffset = (partnerRect.top - boxRect.top) - (textBox.clientHeight / 2) + (partnerRect.height / 2);
            textBox.scrollTop += scrollOffset;
        }
    }
    if (toggleSimilaritiesBtn) {
        toggleSimilaritiesBtn.addEventListener("click", function() {
            const isActive = this.classList.toggle("active");
            this.textContent = isActive ? "Hide All Similarities" : "Show All Similarities";
            document.querySelectorAll('.match-text').forEach(el => {
                el.classList.toggle('active', isActive);
            });
            if (!isActive) {
                document.querySelectorAll('.comparison-block').forEach(block => clearAllActiveHighlights(block));
            }
        });
    }
    document.body.addEventListener("click", function(event) {
        const target = event.target;
        if (target.classList.contains("highlight") && target.classList.contains("clickable")) {
            const pairId = target.dataset.pairId;
            const matchId = target.dataset.matchId;
            const comparisonBlock = target.closest('.comparison-block');
            if (!comparisonBlock) return;
            const isGlobalModeActive = toggleSimilaritiesBtn.classList.contains("active");
            if (isGlobalModeActive) {
                scrollToPartner(comparisonBlock, pairId, matchId);
            } else {
                const wasActive = target.classList.contains("active-highlight");
                clearAllActiveHighlights(comparisonBlock);
                if (!wasActive) {
                    highlightPair(comparisonBlock, pairId, matchId);
                    scrollToPartner(comparisonBlock, pairId, matchId);
                }
            }
        }
    });
    document.body.addEventListener('mouseover', function(event) {
        const target = event.target;
        if (target.classList.contains("highlight") || target.classList.contains("match-text")) {
            const pairId = target.dataset.pairId;
            const matchId = target.dataset.matchId;
            const comparisonBlock = target.closest('.comparison-block');
            if (!comparisonBlock || !pairId || !matchId) return;
            comparisonBlock.querySelector(`.highlight.clickable[data-pair-id="${pairId}"][data-match-id="${matchId}"]`)?.classList.add("hover-highlight");
            comparisonBlock.querySelector(`.match-text[data-pair-id="${pairId}"][data-match-id="${matchId}"]`)?.classList.add("hover-highlight");
        }
    });
    document.body.addEventListener('mouseout', function(event) {
        document.querySelectorAll('.hover-highlight').forEach(el => el.classList.remove('hover-highlight'));
    });
    const toggleBridgeBtn = document.getElementById("toggle-all-bridge-words");
    const minSlider = document.getElementById("min-similarity");
    const maxSlider = document.getElementById("max-similarity");
    const minValSpan = document.getElementById("min-similarity-val");
    const maxValSpan = document.getElementById("max-similarity-val");
    if (toggleBridgeBtn) {
        toggleBridgeBtn.addEventListener("click", function() {
            const isActive = this.classList.toggle("active");
            document.querySelectorAll(".bridge-words").forEach(el => el.classList.toggle("highlighted", isActive));
            this.textContent = isActive ? "Hide All Bridge Words" : "Show All Bridge Words";
        });
    }
    function filterDocuments() {
        const minVal = parseFloat(minSlider.value);
        const maxVal = parseFloat(maxSlider.value);
        comparisonBlocks.forEach(block => {
            const score = parseFloat(block.dataset.score);
            if (score >= minVal && score <= maxVal) {
                block.style.display = 'block';
            } else {
                block.style.display = 'none';
            }
        });
    }
    function setupSliders() {
        const minDisplayThreshold = document.body.dataset.minDisplayThreshold || "0.0";
        const initialThreshold = document.body.dataset.initialThreshold || "0.75";
        minSlider.min = minDisplayThreshold;
        maxSlider.min = minDisplayThreshold;
        minSlider.max = "1.0";
        maxSlider.max = "1.0";
        minSlider.value = initialThreshold;
        maxSlider.value = "1.0";
        minValSpan.textContent = parseFloat(minSlider.value).toFixed(3);
        maxValSpan.textContent = parseFloat(maxSlider.value).toFixed(3);
        minSlider.addEventListener("input", () => {
            let minVal = parseFloat(minSlider.value);
            let maxVal = parseFloat(maxSlider.value);
            if (minVal >= maxVal) {
                minSlider.value = maxVal - 0.001 < parseFloat(minSlider.min) ? minSlider.min : maxVal - 0.001;
                minVal = parseFloat(minSlider.value);
            }
            minValSpan.textContent = minVal.toFixed(3);
            filterDocuments();
        });
        maxSlider.addEventListener("input", () => {
            let minVal = parseFloat(minSlider.value);
            let maxVal = parseFloat(maxSlider.value);
            if (maxVal <= minVal) {
                maxSlider.value = minVal + 0.001 > parseFloat(maxSlider.max) ? maxSlider.max : minVal + 0.001;
                maxVal = parseFloat(maxSlider.value);
            }
            maxValSpan.textContent = maxVal.toFixed(3);
            filterDocuments();
        });
        filterDocuments();
    }
    setupSliders();
});
</script>
</body>
</html>"""
        min_display_threshold = max(0.0, similarity_threshold - interactive_range_width)
        print(f"INFO: HTML report will include pairs with similarity >= {min_display_threshold:.4f}")
        print(f"INFO: The default view will be filtered to >= {similarity_threshold:.4f}")
        html_with_thresholds = html_template_start.replace(
            '<body>',
            f'<body data-initial-threshold="{similarity_threshold}" data-min-display-threshold="{min_display_threshold}">'
        )
        file_counter = 1
        current_html = html_with_thresholds
        pair_count = 0
        print("Generating HTML comparison files from sparse matrix...")
        dist_mat_coo = analyzer.dist_mat.tocoo()
        similar_pairs = []
        for i, j, score in zip(dist_mat_coo.row, dist_mat_coo.col, dist_mat_coo.data):
            if score >= min_display_threshold:
                if not analyzer.is_inter_comparison and i >= j:
                    continue
                similar_pairs.append((i, j, score))
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        display_token_corpus1 = [word_tokenize(text) for text in analyzer.corpus]
        display_token_corpus2 = [word_tokenize(text) for text in (analyzer.corpus2 or analyzer.corpus)]

        for pair in tqdm.tqdm(similar_pairs, desc="Generating HTML pairs"):
            i, j, score = pair
            unique_pair_id = f"{i}-{j}"
            h1, h2 = SimilarityVisualizer.highlight_similarities(display_token_corpus1[i], display_token_corpus2[j], unique_pair_id)
            f1 = analyzer.file_paths[i].name
            f2 = analyzer.file_paths2[j].name if analyzer.is_inter_comparison else analyzer.file_paths[j].name
            segment = f'<div class="comparison-block" data-score="{score:.4f}" data-pair-id="{unique_pair_id}">' \
                      f'<h3>Comparison: {f1} &harr; {f2}</h3>' \
                      f'<div class="similarity-score">Cosine Similarity: {score:.4f}</div>' \
                      f'<div class="comparison-container">' \
                      f'<div class="text-box"><p class="file-info">File 1: {f1}</p>{h1}</div>' \
                      f'<div class="text-box"><p class="file-info">File 2: {f2}</p>{h2}</div>' \
                      f'</div></div>'
            current_html_len = len(current_html.encode('utf-8'))
            segment_len = len(segment.encode('utf-8'))
            if pair_count > 0 and current_html_len + segment_len > max_file_size:
                with open(f"text_comparisons_{file_counter:02d}.html", "w", encoding='utf-8') as f:
                    f.write(current_html + html_template_end)
                file_counter += 1
                current_html = html_with_thresholds
                pair_count = 0
            current_html += segment
            pair_count += 1
        if pair_count > 0:
            with open(f"text_comparisons_{file_counter:02d}.html", "w", encoding='utf-8') as f:
                f.write(current_html + html_template_end)
            print(f"Generated {file_counter} HTML comparison file(s).")
        else:
            print("No similar pairs found above the minimum display threshold.")

    @staticmethod
    def plot_similarity_heatmap(analyzer):
        if analyzer.dist_mat is None or not analyzer.file_paths: return
        dense_dist_mat = analyzer.dist_mat.toarray()
        if analyzer.is_inter_comparison:
            y_labels = [p.name for p in analyzer.file_paths]
            x_labels = [p.name for p in analyzer.file_paths2]
            title = 'Inter-Corpus Text Similarity Heatmap'
        else:
            y_labels = x_labels = [p.name for p in analyzer.file_paths]
            title = 'Text Similarity Heatmap (Intra-Corpus)'
        fig = go.Figure(data=go.Heatmap(z=dense_dist_mat, x=x_labels, y=y_labels, colorscale='Blues', zmin=0.0, zmax=1.0, colorbar=dict(title='Cosine Similarity')))
        fig.update_layout(title_text=title, height=max(600, len(y_labels)*20), width=max(700, len(x_labels)*20))
        fig.write_html("similarity_heatmap.html")
        print("Generated similarity_heatmap.html")

    @staticmethod
    def generate_similarity_summary_tsv(analyzer, similarity_threshold: float):
        if analyzer.dist_mat is None or not analyzer.corpus: return
        print(f"Generating TSV summary using threshold: {similarity_threshold:.4f}")
        related_docs_map = defaultdict(list)
        dist_mat_coo = analyzer.dist_mat.tocoo()
        for i, j, score in zip(dist_mat_coo.row, dist_mat_coo.col, dist_mat_coo.data):
            if i != j and score >= similarity_threshold:
                related_docs_map[i].append(j)
        header = "DocumentFilename\tSimilarityFrequency\tRelatedDocuments\tLongSimilarities(>4words)\n"
        rows = [header]
        display_token_corpus1 = [word_tokenize(text) for text in analyzer.corpus]
        if analyzer.is_inter_comparison:
            display_token_corpus2 = [word_tokenize(text) for text in analyzer.corpus2]
        else:
            display_token_corpus2 = display_token_corpus1
        for i in tqdm.tqdm(range(len(analyzer.corpus)), desc="Generating TSV summary"):
            related_docs_indices = related_docs_map.get(i, [])
            long_segments = set()
            for related_idx in related_docs_indices:
                if related_idx >= len(display_token_corpus2): continue
                sm = SequenceMatcher(None, display_token_corpus1[i], display_token_corpus2[related_idx], autojunk=False)
                for a, _, size in sm.get_matching_blocks():
                    if size > 4:
                        segment = display_token_corpus1[i][a:a+size]
                        long_segments.add(SimilarityVisualizer.detokenizer.detokenize(segment))
            if analyzer.is_inter_comparison:
                related_doc_names = sorted([analyzer.file_paths2[j].name for j in related_docs_indices])
            else:
                related_doc_names = sorted([analyzer.file_paths[j].name for j in related_docs_indices])
            long_segments_str = ' | '.join(f'"{s}"' for s in sorted(long_segments, key=len, reverse=True)) or 'None'
            related_docs_str = ', '.join(related_doc_names) or 'None'
            rows.append(f"{analyzer.file_paths[i].name}\t{len(related_doc_names)}\t{related_docs_str}\t{long_segments_str}\n")
        with open("similarity_summary.tsv", "w", encoding='utf-8') as f: f.writelines(rows)
        print("Generated similarity_summary.tsv")

    @staticmethod
    def generate_linguistic_summary_tsv(analyzer, similarity_threshold: float):
        if analyzer.dist_mat is None or not analyzer.corpus: return
        print(f"Generating linguistic variations summary (TSV) using threshold: {similarity_threshold:.4f}")
        levenshtein_threshold = 0.75
        rows = ["File_1\tFile_2\tVariation_Type\tToken_1\tToken_2\n"]
        display_token_corpus1 = [word_tokenize(text) for text in analyzer.corpus]
        display_token_corpus2 = [word_tokenize(text) for text in (analyzer.corpus2 or analyzer.corpus)]
        dist_mat_coo = analyzer.dist_mat.tocoo()
        for i, j, score in tqdm.tqdm(zip(dist_mat_coo.row, dist_mat_coo.col, dist_mat_coo.data), desc="Analyzing linguistic variations", total=dist_mat_coo.nnz):
            if score < similarity_threshold: continue
            if not analyzer.is_inter_comparison and i >= j: continue
            file1_path = analyzer.file_paths[i]
            tokens1 = display_token_corpus1[i]
            file2_path = analyzer.file_paths2[j] if analyzer.is_inter_comparison else analyzer.file_paths[j]
            tokens2 = display_token_corpus2[j]
            analysis_tokens1 = [t.lower() for t in tokens1 if t.isalnum()]
            analysis_tokens2 = [t.lower() for t in tokens2 if t.isalnum()]
            if not analysis_tokens1 or not analysis_tokens2: continue
            matcher = SequenceMatcher(None, analysis_tokens1, analysis_tokens2, autojunk=False)
            pos1_analysis, pos2_analysis = 0, 0
            for a, b, size in matcher.get_matching_blocks():
                if size == 0: continue
                gap_tokens1 = analysis_tokens1[pos1_analysis:a]
                gap_tokens2 = analysis_tokens2[pos2_analysis:b]
                if (1 <= len(gap_tokens1) <= 3) or (1 <= len(gap_tokens2) <= 3):
                    if len(gap_tokens1) == len(gap_tokens2) and len(gap_tokens1) > 0:
                        for t1, t2 in zip(gap_tokens1, gap_tokens2):
                            variation_type = "Similar Bridge Word" if Levenshtein.ratio(t1, t2) >= levenshtein_threshold else "Different Bridge Word"
                            rows.append(f"{file1_path.name}\t{file2_path.name}\t{variation_type}\t{t1}\t{t2}\n")
                    else:
                        for t1 in gap_tokens1: rows.append(f"{file1_path.name}\t{file2_path.name}\tDifferent Bridge Word\t{t1}\t-\n")
                        for t2 in gap_tokens2: rows.append(f"{file1_path.name}\t{file2_path.name}\tDifferent Bridge Word\t-\t{t2}\n")
                pos1_analysis, pos2_analysis = a + size, b + size
        with open("linguistic_variations.tsv", "w", encoding='utf-8') as f:
            f.writelines(rows)
        print("Generated linguistic_variations.tsv")


def main():
    print("--- Formulaic Language Analysis in Medieval Expressions ---")
    print("For command-line options, run with the -h flag.")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Created temporary directory: {tmpdir}")
            analyzer = Flame(tmp_dir=tmpdir)
            if (analyzer.args.ngram - analyzer.args.n_out) < 1:
                raise ValueError(f"N-gram size ({analyzer.args.ngram}) minus n-out ({analyzer.args.n_out}) must be at least 1.")

            analyzer.load_corpus()
            if not analyzer.corpus:
                print("Execution halted because no documents were loaded.")
                return

            analyzer.compute_similarity_matrix()
            if analyzer.dist_mat is None:
                print("Execution halted because the similarity matrix could not be computed.")
                return

            if str(analyzer.args.similarity_threshold).lower() == 'auto':
                final_threshold = analyzer._determine_auto_threshold(method=analyzer.args.auto_threshold_method)
            else:
                final_threshold = float(analyzer.args.similarity_threshold)

            if analyzer.dist_mat.shape[0] < 2000 and analyzer.dist_mat.shape[1] < 2000:
                SimilarityVisualizer.plot_similarity_heatmap(analyzer)
            else:
                print(f"Skipping heatmap generation for large matrix ({analyzer.dist_mat.shape[0]}x{analyzer.dist_mat.shape[1]}).")

            SimilarityVisualizer.generate_comparison_html(analyzer, similarity_threshold=final_threshold)
            SimilarityVisualizer.generate_similarity_summary_tsv(analyzer, similarity_threshold=final_threshold)
            SimilarityVisualizer.generate_linguistic_summary_tsv(analyzer, similarity_threshold=final_threshold)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n--- Execution Finished ---")

if __name__ == '__main__':
    main()
