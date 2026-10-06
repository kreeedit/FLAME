import numpy as np
import glob
import hashlib
import math
import os
import pathlib
import re
import sys
import unicodedata
from datetime import datetime
import fargv
import tqdm
from rapidfuzz import fuzz
from itertools import chain, combinations
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import tempfile
from scipy.sparse import coo_matrix, save_npz, vstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import plotly.graph_objects as go
from skimage.filters import threshold_otsu
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import flame_clustering

# The command line as launched. fargv consumes sys.argv while parsing it, so the
# report's "reproduce with" line has to be copied out before main() runs.
_LAUNCH_ARGV = list(sys.argv)


def fast_str_to_numpy(s: str, dtype=np.uint32) -> np.ndarray:
    """Efficiently converts a string to a NumPy array via byte encoding.

    Args:
        s (str): The input string.
        dtype (type): The target NumPy data type.
            - np.uint32 (default): Safely handles all Unicode characters
              by using the 'utf-32le' encoding.
            - np.uint16: Faster for pure BMP text
              outside the Basic Multilingual Plane (U+0000 to U+FFFF).

    Returns:
        np.ndarray: A NumPy array of character codes.
    """
    if dtype == np.uint32:
        return np.frombuffer(s.encode('utf-32le'), dtype=dtype)
    elif dtype == np.uint16:
        return np.frombuffer(s.encode('utf-16le'), dtype=dtype)
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
    corpus: List[str],
    min_word_freq: int = 3,
    max_affix_len: int = 6,
    coverage_percentile: float = 0.85
) -> int:
    """Proposes an optimal vocabulary size for the BPE tokenizer based on corpus analysis.

    Args:
        corpus (List[str]): List of text documents.
        min_word_freq (int): Minimum frequency for a word to be considered.
        max_affix_len (int): Maximum length of common prefixes/suffixes to analyze.
        coverage_percentile (float): Target percentile of affix occurrences to cover.

    Returns:
        int: Suggested vocabulary size.
    """
    print("\n--- Starting Automatic Vocab Size Suggestion (Optimized) ---")
    print("Counting word frequencies...")

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
        return fast_numpy_to_str(self._npint2int[fast_str_to_numpy(text, dtype=np.uint16)])

    def get_encoding_information_loss(self, text: str) -> float:
        np_text = fast_str_to_numpy(text, dtype=np.uint16)
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
    """An adaptive normalizer that learns character mappings from a text corpus dynamically."""
    def __init__(self, src_alphabet: str, unknown_chr: str = '', initial_mapping_dict: Optional[Dict[str, str]] = None):
        mapping_dict = initial_mapping_dict if initial_mapping_dict is not None else {}
        super().__init__(src_alphabet=src_alphabet, mapping_dict=mapping_dict, unknown_chr=unknown_chr)

    def analyze_lost_chars(self, text: str) -> defaultdict:
        lost_chars_count = defaultdict(int)
        np_text = fast_str_to_numpy(text, dtype=np.uint16)
        mapped_np_text = fast_str_to_numpy(self(text), dtype=np.uint16)
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
    'input_path': '',
    'input_path2': '',
    'file_suffix': '.txt',
    'keep_texts': 10000,
    # Off by default: whether two identical files are two documents or one is a
    # scholarly decision, not a technical one. Corpora assembled from editions and
    # archival copies routinely hold the same charter under several names (a
    # shelfmark, an edition, and a "(1)" re-download), and leaving the flag off
    # reproduces exactly what is on disk.
    'deduplicate': False,
    'ngram': 6,
    'n_out': 1,
    'min_text_length': 150,
    'similarity_threshold': 'auto',
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': 'abcdefghijklmnopqrstuvwxyz',
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 1,
    'phonetic_reduction_enabled': False,
    'phonetic_reduction_alphabet': 'aefiklmnopqrstuwxz',
    'phonetic_reduction_rules': 'b>p,c>k,d>t,g>k,j>i,q>k,v>f,y>i,z>s',
    'bigram_normalization_enabled': False,
    'bigram_normalization_rules': 'ss>s,ff>f,tt>t,ll>l,ie>i,au>u,ei>i,eu>u,oh>o,ah>a,eh>e,uh>u',
    'vocab_size': 'auto',
    'vocab_min_word_freq': 5,
    'vocab_coverage': 0.85,
    # Decides variant vs. bridge for the words between two matches (see classify_gap).
    # Calibrated on a two-copies comparison (Georg_problems0928): genuine divergences
    # scored <= 0.40 once folded, while spelling variants scored >= 0.72 -- so anything
    # in 0.45-0.70 separates cleanly and 0.70 keeps the borderline cases as variants.
    'fuzz_threshold': 0.70,
    'max_gap_words': 5,
    'auto_tune': False,
    'auto_tune_sample_size': 30,
    # Clustering (legal-formula): groups the surviving pairs by the formula
    # core they share. cluster_threshold is the similarity two CORES must reach to
    # merge, on the same 0-1 scale as fuzz_threshold but a different quantity --
    # fuzz_threshold judges the words between two matches, this judges whole
    # formulas against each other, so it is calibrated separately (KONI's default
    # was 0.85). Measured on the MOM corpus at 0.85 vs 0.70: a papal privilege
    # template whose recipients differ (one abbey against one hospital) had been
    # split into four clusters of 24, 12, 6 and 2 charters; at 0.70 those four
    # merge into one 49-charter cluster, and that was the only merge among the
    # clusters of four or more members -- apart from a second papal formula
    # ('militanti ecclesie'). 0.70 finds 40 clusters against 34, adds no
    # near-duplicate (11 either way, the section is judged on coverage, not on
    # this threshold), and 0.65 already over-merges (a 70-charter cluster with
    # 0.44-0.50 cohesion), so 0.70 is the loose end of what still separates.
    'cluster_threshold': 0.70,
    'cluster_min': 2,
    # What a cluster IS. 'louvain' (default) looks for cores that are denser
    # among themselves than with the rest, which is what keeps one formula with
    # variable slots in one piece; 'clique' splits each group into maximal
    # cliques, so every core in a cluster agrees with every other; 'union' keeps
    # KONI's connected components, which also merge A with C when only A~B and
    # B~C hold. A string rather than a boolean because fargv treats a boolean as
    # a presence switch: a True-defaulted flag could never be turned back off
    # from the command line, only from the GUI.
    'cluster_linkage': 'louvain',
    # How far a formula's variable slot may open before the core is cut in two:
    # the gapped alignment (flame_clustering.align_core) chains matching runs
    # across gaps of at most this many tokens on either side. Eight spans the
    # widest slot measured in the corpus (a name, a place, a case ending) without
    # bridging the distance between two different formulas.
    'core_gap_tolerance': 8,
    # Shortest window (its shorter side) that still counts as a formula; below
    # it the pair falls back to the strict contiguous core.
    'core_min_tokens': 12,
    # Where a formula is looked for: the performative verbs that carry a charter's
    # legal act (donamus, contulimus, confirmamus...). Comma-separated *stems*,
    # prefix-matched against the folded tokens, because the ending is what varies
    # (donamus / donauimus / donauerunt) and folding keeps the corpus's u-for-v
    # spelling (contulimus, uendidimus). Without anchoring the alignment maximises
    # matched tokens, and the longest shared run in a mediaeval charter is the
    # protocol -- measured on the MOM corpus, no core came out below 56 tokens and
    # 28 of 40 clusters opened on a protocol phrase -- so the verbs are what tell
    # the aligner where the act is. A pair whose shared text carries none keeps
    # the unanchored window, which the report counts. Empty (or `none`) switches
    # anchoring off and gives back the pre-anchor clustering word for word;
    # `-core_anchors=` does the same from the CLI (the `=` form: fargv crashes on
    # a bare empty argument).
    #
    # There is no built-in list, and that is a measured decision rather than an
    # oversight. A hand-written Latin-and-German list was tried and is gone
    # because it does not hold: on MOM, 11 of its 38 stems (vendidimus, vendimus,
    # emancipauimus, concessimus, promittimus, assignauimus, commutauimus,
    # verleihen, verkaufen, ubergeben, ...) do not occur in the corpus at all,
    # and 153 of the 441 admitted pairs (34.7%) share no word of it, so a third
    # of the corpus fell back to the unanchored window on the very corpus the
    # list was written for.
    #
    # Discovering the verbs from the corpus was then tried and did *not* work,
    # and the measurements are why the file ships without a list rather than
    # with a worse one:
    #   * document frequency, occupancy, context entropy, positional spread, the
    #     IDF of a token's neighbours, and every conjunction of them were ranked
    #     against the verbs the hand list did reach: the best rule selected 8676
    #     tokens to find 45 verbs (precision 0.5%, recall 34.9%). The reason is
    #     that the verbs are 0.3% of the recurring vocabulary and statistically
    #     ordinary: `contulimus` has df 416 beside `iuris` at 1093, in a band
    #     holding 1241 tokens of which 16 are verbs.
    #   * phrase-level rarity (k-gram document frequency) does not locate the act
    #     either: ranked within a pair's shared span it puts the act at relative
    #     rank 0.467 -- chance.
    #   * rarity as the *objective* is the wrong sign: mean unigram IDF, summed
    #     IDF, rarest-word IDF and phrase DF all anti-select the act (median
    #     relative rank 0.57-0.70), because the act formula is shared by its act
    #     family while the average shared passage is a one-off list of names.
    #   * morphology is the signal a reader uses, and it is not separable on the
    #     surface: handed the ending inventory of the known verbs as an oracle,
    #     `-mus/-nt/-it` still selects 2215 tokens to find 50 verbs (2.3%).
    # Identifying the performative verb is a lexical or morphological task, so
    # the resource is per-corpus input, not a constant in this file.
    'core_anchors': '',
    # How far the window may extend past the anchor, in tokens, on either side.
    # Fifteen is the length of the act's habitual surroundings (the operative
    # clause plus the sanctio that follows it) without reaching the arenga.
    'core_anchor_window': 15,
    # Hard ceiling on the reported window, in tokens. The legal act is short --
    # a dispositio runs 15-40 words -- while a protocol panel runs to 200, so
    # without a ceiling a long generic overlap still outvotes a short specific
    # one. 0 leaves the window uncapped (the pre-anchor behaviour).
    'core_max_tokens': 50,
    # A near-duplicate charter is reported separately from a shared formula when
    # its shared window is at least this many words AND covers at least
    # MAX_CORE_FRACTION of the shorter charter. Both are needed: the share alone
    # calls a short charter duplicate when its text is mostly one formula.
    'min_duplicate_tokens': 400,
    # Minimum gap-only identity (2*matched/(len_a+len_b)) for an aligned core to
    # be accepted as a formula; below it the pair falls back to the strict
    # contiguous core. 0.0 accepts every window the aligner returns.
    'core_identity_threshold': 0.0,
    # Drops clusters whose specificity (core length x mean IDF of its words) is
    # below this. 0.0 keeps every cluster: the specificity is reported either way,
    # and whether a widespread panel counts as a finding is a scholarly call.
    'min_core_specificity': 0.0,
    # Optional file of tokens (one per line, '#' comments) whose IDF is forced to
    # zero when specificity is measured -- a hand-curated diplomatic stop list.
    # Empty by default: the corpus's own document frequencies already know that
    # 'salutem' and 'benedictionem' are everywhere.
    'stopwords_file': '',
    'no_reports': False,
    'gen_comparison_html': True,
    'gen_summary_tsv': True,
    'gen_linguistic_tsv': True,
    'gen_heatmap': True,
    'gen_clusters': True,
}

def parse_phonetic_rules(rules_str: str) -> Dict[str, str]:
    """Parse phonetic reduction rules from string format 'b>p,c>k,...' into a mapping dict.

    Args:
        rules_str: Comma-separated rules, each in 'src>dst' format.

    Returns:
        Dict mapping source characters to destination characters.
        Invalid rules are skipped with a warning.
    """
    mapping = {}
    if not rules_str or not rules_str.strip():
        return mapping
    for rule in rules_str.split(','):
        rule = rule.strip()
        if not rule:
            continue
        parts = rule.split('>')
        if len(parts) != 2:
            print(f"  Warning: Invalid phonetic rule '{rule}' — expected 'src>dst' format. Skipping.")
            continue
        src, dst = parts[0].strip(), parts[1].strip()
        if len(src) != 1 or len(dst) != 1:
            print(f"  Warning: Invalid phonetic rule '{rule}' — both src and dst must be single characters. Skipping.")
            continue
        mapping[src] = dst
    return mapping


def parse_bigram_rules(rules_str: str) -> Dict[str, str]:
    """Parse bigram normalization rules from string format 'ss>s,ff>f,...' into a mapping dict.

    Unlike phonetic rules, the source side can be multiple characters (one-to-many).
    The destination must be exactly one character.

    Args:
        rules_str: Comma-separated rules, each in 'src>dst' format.

    Returns:
        Dict mapping source strings to destination characters.
        Invalid rules are skipped with a warning.
    """
    mapping = {}
    if not rules_str or not rules_str.strip():
        return mapping
    for rule in rules_str.split(','):
        rule = rule.strip()
        if not rule:
            continue
        parts = rule.split('>')
        if len(parts) != 2:
            print(f"  Warning: Invalid bigram rule '{rule}' — expected 'src>dst' format. Skipping.")
            continue
        src, dst = parts[0].strip(), parts[1].strip()
        if len(src) < 2:
            print(f"  Warning: Invalid bigram rule '{rule}' — src must be at least 2 characters. Skipping.")
            continue
        if len(dst) != 1:
            print(f"  Warning: Invalid bigram rule '{rule}' — dst must be a single character. Skipping.")
            continue
        mapping[src] = dst
    return mapping


class Flame:
    """Main pipeline execution for medieval formulaic language alignment."""
    def __init__(self, args, tmp_dir: str = '.'):
        self.args = args
        # What loading the corpus skipped, filled in by _load_corpus_from_path.
        # Empty rather than absent so the report can read it even when a corpus
        # was never loaded (a direct caller, a test).
        self.load_stats: Dict[str, int] = {}
        self.load_stats2: Dict[str, int] = {}
        # The second input may be a directory or a glob pattern, so a pattern must not
        # be mistaken for "no second corpus".
        self.is_inter_comparison = bool(self.args.input_path2 and
                                        (os.path.isdir(self.args.input_path2) or
                                         looks_like_pattern(self.args.input_path2)))
        self.tmp_dir = tmp_dir

        self.corpus: List[str] = []
        self.file_paths: List[pathlib.Path] = []
        # Roots the reports name texts relative to (see display_path).
        self.corpus_root: Optional[pathlib.Path] = None
        self.corpus_root2: Optional[pathlib.Path] = None
        self.tokenized_corpus: List[List[str]] = []
        self.corpus2: List[str] = []
        self.file_paths2: List[pathlib.Path] = []
        self.tokenized_corpus2: List[List[str]] = []
        self.encoder: Dict[str, int] = {}
        self.dist_mat = None
        self.tokenizer_model = None

    def _find_text_files(self, input_path: str) -> Tuple[List[pathlib.Path], Optional[pathlib.Path]]:
        """Resolves an input path to a sorted file list plus the root that reported
        paths are relative to.

        The input may be a directory (searched recursively for files ending in
        file_suffix) or a glob pattern such as ./fsdb/DE-LANRWR**/*.htr.txt. For a
        pattern the pattern itself decides which files are read and file_suffix is
        ignored, so a pattern like *.htr is not silently emptied by a .txt filter.
        """
        if looks_like_pattern(input_path):
            base = pattern_base(input_path)
            files = find_pattern_files(input_path)
            if not files:
                print(f"Warning: Pattern '{input_path}' matched no files.")
            return files, base

        path = pathlib.Path(input_path)
        if not path.exists() or not path.is_dir():
            print(f"Warning: Input path {path} does not exist or is not a directory. Skipping.")
            return [], None
        # Sorted so the corpus order -- and with it the distance matrix -- is
        # reproducible instead of depending on filesystem traversal order.
        return sorted(path.rglob(f"*{self.args.file_suffix}")), path

    def _read_text_file(self, file_path: pathlib.Path) -> Union[str, None]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ' '.join(f.read().strip().split())
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            return None

    def _load_corpus_from_path(self, path_str: str
                               ) -> Tuple[List[str], List[pathlib.Path], Optional[pathlib.Path], Dict[str, int]]:
        """Loads texts from a directory or glob pattern; also returns the root that the
        reports name these texts relative to, and what the loading skipped.

        The counts come back rather than only being printed because the cluster
        report has to state its own input: a reader who was handed the HTML and
        nothing else cannot tell a 12 000-charter corpus from a 400-charter one,
        and "how many charters was this computed on" is the first thing they ask.
        """
        file_paths, root = self._find_text_files(path_str)
        if looks_like_pattern(path_str):
            print(f"Found {len(file_paths)} files for pattern '{path_str}'")
        else:
            print(f"Found {len(file_paths)} files in '{path_str}' with suffix '{self.args.file_suffix}'")
        corpus_data, loaded_paths = [], []
        skipped_short = 0
        # Content hash -> the surviving file and its position in corpus_data, so an
        # identical file can be recognised no matter what it is called. Only
        # populated when deduplicating.
        seen_texts: Dict[str, Tuple[pathlib.Path, int]] = {}
        # key -> the files skipped for holding that key's text. A file lands here
        # either because it lost to the current keeper or because it was the keeper
        # and a shorter-named copy displaced it; either way exactly one entry is
        # added per skipped file, so the grouped report stays truthful.
        duplicates: Dict[str, List[pathlib.Path]] = {}
        limit = self.args.keep_texts
        # For a pattern, name the progress bar after the directory it starts in --
        # basename would show the glob itself, e.g. "Loading files from *.htr.txt".
        source = str(pattern_base(path_str)) if looks_like_pattern(path_str) else path_str
        limit_reached = False
        for file_path in tqdm.tqdm(file_paths, desc=f"Loading files from {os.path.basename(source)}"):
            text = self._read_text_file(file_path)
            if not text or len(text) < self.args.min_text_length:
                skipped_short += 1
                continue
            if self.args.deduplicate:
                # Hashed on the text as loaded, not on the raw bytes: two files
                # differing only in whitespace are the same document to every part
                # of the pipeline downstream, since that is the string the engine
                # tokenizes.
                key = hashlib.sha1(text.encode('utf-8')).hexdigest()
                kept = seen_texts.get(key)
                if kept is not None:
                    kept_path, kept_index = kept
                    # Among identical files the shorter name wins. Corpora assemble
                    # the same charter as "X.txt", "X (1).txt" and a long shelfmark,
                    # and the re-download -- the one name nobody wants to keep -- is
                    # never the shortest. The text is identical to the letter, so a
                    # swap changes only the name the reports use: the document keeps
                    # its position, and with it the distance matrix row it had.
                    if len(file_path.name) < len(kept_path.name):
                        duplicates.setdefault(key, []).append(kept_path)
                        loaded_paths[kept_index] = file_path
                        seen_texts[key] = (file_path, kept_index)
                    else:
                        duplicates.setdefault(key, []).append(file_path)
                    continue
                seen_texts[key] = (file_path, len(corpus_data))
            corpus_data.append(text)
            loaded_paths.append(file_path)
            if len(corpus_data) >= limit:
                print(f"Reached limit of {limit} texts for this directory.")
                limit_reached = True
                break
        if self.args.deduplicate:
            self._report_duplicates(duplicates, seen_texts, len(file_paths))
        stats = {
            "found": len(file_paths),
            "loaded": len(corpus_data),
            "short": skipped_short,
            "duplicate": sum(len(paths) for paths in duplicates.values()),
            "limit_reached": limit_reached,
        }
        return corpus_data, loaded_paths, root, stats

    @staticmethod
    def _report_duplicates(duplicates: Dict[str, List[pathlib.Path]],
                           seen_texts: Dict[str, Tuple[pathlib.Path, int]],
                           scanned: int) -> None:
        """Prints what deduplication dropped, naming both sides of every match.

        The full list is printed rather than a summary count on purpose: a corpus
        that silently loses a third of its files to a wrong flag would be a nasty
        surprise, and only the names show whether the dropped ones were true
        repeats (an edition and its archival copy) or something unexpected.

        Grouped by content, not listed pair by pair: the name that survives can
        change while the scan runs (a later, shorter-named copy of the same text
        takes its place), so a per-duplicate "same text as X" line could name a
        file that was itself dropped a moment later. Printing the group with its
        one survivor says the same thing without that trap.
        """
        skipped = sum(len(paths) for paths in duplicates.values())
        print(f"\n--- Deduplication: {skipped} of {scanned} file(s) skipped as repeats ---")
        for key, dropped in duplicates.items():
            kept = seen_texts[key][0]
            print(f"'{kept}' -- {len(dropped) + 1} file(s) hold this text; kept this one, skipped:")
            for path in dropped:
                print(f"    {path}")
        if skipped:
            print(f"Kept {scanned - skipped} file(s) out of {scanned}.")
        else:
            print("No two files held the same text.")

    def load_corpus(self):
        """Loads files, runs character level mapping layers, and builds BPE vocabulary model."""
        self.corpus, self.file_paths, self.corpus_root, self.load_stats = \
            self._load_corpus_from_path(self.args.input_path)
        if self.is_inter_comparison:
            print("\n--- Two-directory comparison mode activated ---")
            self.corpus2, self.file_paths2, self.corpus_root2, self.load_stats2 = \
                self._load_corpus_from_path(self.args.input_path2)
            if not self.corpus2:
                print("Warning: Second directory is empty or invalid. Reverting to single-directory mode.")
                self.is_inter_comparison = False

        if not self.corpus:
            print("Error: No valid texts loaded from the primary directory. Aborting.")
            return

        corpus_for_learning = self.corpus + self.corpus2
        print(f"\nTotal texts for analysis: {len(corpus_for_learning)}")

        lowercased_corpus = [text.lower() for text in corpus_for_learning]

        # Shared with the visualizer (see MUFI_ONE_TO_MANY / MUFI_ONE_TO_ONE below), so
        # that a gap is judged by the same normalization the matcher itself applied.
        one_to_many_mappings = MUFI_ONE_TO_MANY
        one_to_one_mappings = MUFI_ONE_TO_ONE

        print("\n--- Applying 1-to-many character replacements (e.g., ligatures) ---")
        pre_processed_corpus = []
        for text in tqdm.tqdm(lowercased_corpus, desc="Pre-processing"):
            for src, dst in one_to_many_mappings.items():
                text = text.replace(src, dst)
            pre_processed_corpus.append(text)

        # --- BIGRAM NORMALIZATION ---
        if self.args.bigram_normalization_enabled:
            print("\n--- Applying Bigram Normalization ---")
            bigram_rules = parse_bigram_rules(self.args.bigram_normalization_rules)
            if bigram_rules:
                rules_desc = ', '.join(f"'{s}'->'{d}'" for s, d in sorted(bigram_rules.items()))
                print(f"  Active rules: {rules_desc}")
                for i in tqdm.tqdm(range(len(pre_processed_corpus)), desc="Bigram Normalization"):
                    for src, dst in bigram_rules.items():
                        pre_processed_corpus[i] = pre_processed_corpus[i].replace(src, dst)
                print(f"  Bigram normalization applied to {len(pre_processed_corpus)} texts.")
            else:
                print("  No valid bigram rules configured. Skipping.")
            print("--- Bigram Normalization Complete ---\n")

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

        # --- PHONETIC REDUCTION LAYER ---
        if self.args.phonetic_reduction_enabled:
            print("\n--- Applying Phonetic Reduction ---")
            phonetic_alphabet = self.args.phonetic_reduction_alphabet.replace(' ', '')
            if not phonetic_alphabet:
                print("Warning: Phonetic reduction alphabet is empty. Skipping phonetic reduction.")
                print("--- Phonetic Reduction Skipped ---\n")
            else:
                phonetic_rules = parse_phonetic_rules(self.args.phonetic_reduction_rules)
                validated_rules = {}
                for src, dst in phonetic_rules.items():
                    if dst not in phonetic_alphabet:
                        print(f"  Warning: Rule '{src}>{dst}' maps to '{dst}' which is not in the reduced alphabet. Skipping.")
                    else:
                        validated_rules[src] = dst
                if validated_rules:
                    rules_desc = ', '.join(f"'{s}'->'{d}'" for s, d in sorted(validated_rules.items()))
                    print(f"  Active rules: {rules_desc}")
                else:
                    print("  No valid phonetic rules configured. Only alphabet filtering will apply.")
                phonetic_mapper = CharacterMapper(
                    src_alphabet=phonetic_alphabet,
                    mapping_dict=validated_rules,
                    unknown_chr=' '
                )
                normalized_corpus_full = [
                    phonetic_mapper(text)
                    for text in tqdm.tqdm(normalized_corpus_full, desc="Phonetic Reduction")
                ]
                print(f"  Phonetic reduction applied to {len(normalized_corpus_full)} texts.")
                print("--- Phonetic Reduction Complete ---\n")

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

        if all_tokenized:
            print("\n--- Example of Subword Tokenization ---")
            original_text_sample = ' '.join(normalized_corpus_full[0].split()[:25])
            tokenized_sample = all_tokenized[0]
            print(f"Original Text (first 25 words): '{original_text_sample}...'\n")
            print(f"Tokenized Output:\n{tokenized_sample}")
            print("---------------------------------------\n")

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

    def auto_tune_parameters(self):
        """Performs unsupervised parameter digging (auto-tuning) by injecting synthetic noise
        and performing a grid search to optimize the Signal-to-Noise Ratio (SNR) spread.
        """
        print("\n--- Starting Unsupervised Parameter Auto-Tuning (Trial Digging) ---")
        np.random.seed(42)  # Secure structural repeatability across tuning checks
        sample_size = min(int(self.args.auto_tune_sample_size), len(self.tokenized_corpus))
        if sample_size < 2:
            print("Corpus too small to evaluate statistical separation. Skipping tuning.")
            return

        # Prepare a sample and a perturbed twin corpus to emulate transcription / dialect noise
        sample_tokens_list = self.tokenized_corpus[:sample_size]
        perturbed_tokens_list = []

        for tokens in sample_tokens_list:
            perturbed = []
            for t in tokens:
                # 5% probability to simulate subword variation / spelling decay
                if np.random.rand() < 0.05:
                    if np.random.rand() < 0.5 and len(perturbed) > 0:
                        perturbed.pop()  # Simulating character/token dropping
                    continue
                perturbed.append(t)
            perturbed_tokens_list.append(perturbed)

        best_snr = -float('inf')
        best_ngram = self.args.ngram
        best_n_out = self.args.n_out

        # Search parameter space grid
        candidate_grid = [(4, 0), (4, 1), (5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), (7, 1), (7, 2)]
        orig_ngram, orig_n_out = self.args.ngram, self.args.n_out

        print("Evaluating local parameters across structural separation bounds...")
        for ngram, n_out in candidate_grid:
            if ngram - n_out < 1: continue
            self.args.ngram = ngram
            self.args.n_out = n_out

            orig_features = [self.leave_n_out_grams(t) for t in sample_tokens_list]
            pert_features = [self.leave_n_out_grams(t) for t in perturbed_tokens_list]

            local_vocab = {}
            idx = 0
            for feats in orig_features + pert_features:
                for f in feats:
                    if f not in local_vocab:
                        local_vocab[f] = idx
                        idx += 1

            if not local_vocab: continue

            def generate_vector(feats):
                vec = np.zeros(len(local_vocab))
                if feats.size > 0:
                    u, c = np.unique(feats, return_counts=True)
                    for val, count in zip(u, c):
                        if val in local_vocab:
                            vec[local_vocab[val]] = count
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec

            orig_vectors = [generate_vector(f) for f in orig_features]
            pert_vectors = [generate_vector(f) for f in pert_features]

            # Measure Signal (Similarity between original and matched perturbed target)
            signals = [np.dot(orig_vectors[i], pert_vectors[i]) for i in range(sample_size)]
            avg_signal = np.mean(signals)

            # Measure Noise (Cross-similarities with wrong documents)
            noises = []
            for i in range(sample_size):
                for j in range(sample_size):
                    if i != j:
                        noises.append(np.dot(orig_vectors[i], pert_vectors[j]))
            avg_noise = np.mean(noises) if noises else 0.0

            # Compute separation index optimization score
            snr = avg_signal - avg_noise

            if snr > best_snr and avg_signal > 0.05:
                best_snr = snr
                best_ngram = ngram
                best_n_out = n_out

        self.args.ngram = best_ngram
        self.args.n_out = best_n_out
        print(f"--- Auto-Tune Selection Complete ---")
        print(f"Optimal N-Gram set to: {best_ngram}")
        print(f"Optimal N-Out (Gaps) set to: {best_n_out}")
        print(f"Optimized Separation Margin Spread: {best_snr:.4f}\n")

    def tokenize(self, text_str: str) -> List[str]:
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
        """Vectorized rolling hash fingerprinting matching the leave-n-out specification."""
        MOD = 2**61 - 1
        int_tokens = np.array(self.tokens_to_int(tokens), dtype=np.int64)
        seq_len = len(int_tokens)
        elements_to_keep = self.args.ngram - self.args.n_out

        if elements_to_keep < 1 or seq_len < self.args.ngram:
            return np.array([], dtype=np.int64)

        vocab_size = len(self.encoder)
        if vocab_size == 0:
            return np.array([], dtype=np.int64)

        num_ngrams = seq_len - self.args.ngram + 1
        ngram_matrix = np.array([
            int_tokens[i : i + num_ngrams] for i in range(self.args.ngram)
        ], dtype=np.int64)

        indices_to_keep_combinations = list(combinations(range(self.args.ngram), elements_to_keep))
        all_feature_hashes = []

        for combo_indices in indices_to_keep_combinations:
            sub_gram_matrix = ngram_matrix[list(combo_indices), :]
            num_sub_gram_tokens = len(combo_indices)
            powers = np.power(vocab_size, np.arange(num_sub_gram_tokens), dtype=object) % MOD
            hashed_values = np.mod(np.dot(powers, sub_gram_matrix), MOD)
            all_feature_hashes.append(hashed_values)

        return np.concatenate(all_feature_hashes) if all_feature_hashes else np.array([], dtype=np.int64)

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

            print("Applying TF-IDF transformation...")
            combined_matrix = vstack([matrix1, matrix2])
            tfidf = TfidfTransformer().fit(combined_matrix)

            matrix1_tfidf = tfidf.transform(matrix1)
            matrix2_tfidf = tfidf.transform(matrix2)

            print("Calculating inter-corpus Cosine similarity (sparse TF-IDF output)...")
            self.dist_mat = cosine_similarity(matrix1_tfidf, matrix2_tfidf, dense_output=False)
        else:
            matrix = create_sparse_matrix(doc_features1, len(self.corpus), feature_to_col_idx)

            print("Applying TF-IDF transformation...")
            tfidf = TfidfTransformer()
            matrix_tfidf = tfidf.fit_transform(matrix)

            print("Calculating all-pairs Cosine similarity (sparse TF-IDF output)...")
            self.dist_mat = cosine_similarity(matrix_tfidf, dense_output=False)

        print("Similarity matrix computation complete.")
        save_npz(os.path.join(self.tmp_dir, 'dist_mat.npz'), self.dist_mat)


# =============================================================================
# Shared normalization & orthographic-variant machinery
# =============================================================================
# The matching engine (Flame.load_corpus) and the visualizer MUST agree on what
# counts as "the same word". The engine expands ligatures, folds accents and turns
# anything outside a-z into a space, so it happily matches texts that differ in
# spelling. The visualizer used to compare raw lowercased tokens instead, which made
# every orthographic variant look like an unmatched gap -- and every gap was then
# rendered unconditionally as a "bridge word". These helpers give both sides one
# definition to share.

MUFI_ONE_TO_MANY = {
    'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ĳ': 'ij', 'ð': 'dh', 'þ': 'th', 'ﬁ': 'fi',
    'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬆ': 'st'
}
MUFI_ONE_TO_ONE = {
    'ſ': 's', 'ꝇ': 'l', 'ꝑ': 'p', 'ꝛ': 'r', 'ƿ': 'w', 'ᵹ': 'g', 'ꝺ': 'd', 'ꝼ': 'f'
}


def apply_mufi_one_to_many(text: str) -> str:
    """Ligature expansions -- the same table the engine applies in load_corpus()."""
    for src, dst in MUFI_ONE_TO_MANY.items():
        text = text.replace(src, dst)
    return text


def normalize_for_compare(text: str) -> str:
    """Mirrors the engine's AdaptiveAlphabet pass: lowercase, expand ligatures, fold
    combining marks, and turn anything that is not a-z into a space."""
    text = apply_mufi_one_to_many(text.lower())
    for src, dst in MUFI_ONE_TO_ONE.items():
        text = text.replace(src, dst)
    out = []
    for char in text:
        if 'a' <= char <= 'z':
            out.append(char)
            continue
        decomposed = unicodedata.normalize('NFKD', char)
        folded = ''.join(c for c in decomposed if not unicodedata.combining(c))
        out.append(folded if folded and all('a' <= c <= 'z' for c in folded) else ' ')
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def fold_medieval_orthography(text: str) -> str:
    """Collapses the spelling variation medieval scribes actually produced, so that
    vnd/und, deßhalb/deshalb or lohns/lones compare as the same word. Only ever used
    to classify a gap -- never to render text."""
    text = re.sub(r'([aeiou])h', r'\1', text)              # Dehnungs-h: lohns -> lons
    text = text.replace('v', 'u').replace('w', 'u')        # u/v/w merge
    text = text.replace('j', 'i').replace('y', 'i')        # i/j/y merge
    text = text.replace('ck', 'k').replace('tz', 'z').replace('cz', 'z')
    return re.sub(r'(.)\1+', r'\1', text)                  # ss -> s, nn -> n


def fold_for_compare(text: str) -> str:
    return fold_medieval_orthography(normalize_for_compare(text))


def classify_gap(gap1_tokens: List[str], gap2_tokens: List[str], fuzzy_threshold: float) -> Tuple[str, float]:
    """Classifies the words sitting between two matches.

    Returns (kind, ratio), where kind is one of:
      'insertion' -- the words exist in only one of the two texts;
      'variant'   -- both sides carry words and their folded forms agree, i.e. the
                     same wording spelled differently;
      'bridge'    -- both sides carry words but they genuinely diverge.
    The ratio is measured on the folded forms, so punctuation and accent noise can no
    longer drag a variant below the threshold on their own.
    """
    side1 = fold_for_compare(' '.join(gap1_tokens))
    side2 = fold_for_compare(' '.join(gap2_tokens))
    ratio = fuzz.ratio(side1, side2) / 100.0
    if not side1.split() or not side2.split():
        return 'insertion', ratio
    return ('variant' if ratio >= fuzzy_threshold else 'bridge'), ratio


# --- Input path handling: glob patterns and file identity ---------------------
# A corpus directory that nests one folder per document (as fsdb does, where every
# file is called text.htr.txt) makes a bare filename useless as an identifier, so
# every report labels a text by its path relative to the input root instead. When
# flat, that relative path IS the filename, so existing outputs are unchanged.

GLOB_METACHARS = '*?['


def looks_like_pattern(input_path: str) -> bool:
    """True when the input path carries glob metacharacters, e.g.
    ./fsdb/DE-LANRWR**/*.htr.txt -- such a path is a pattern, not a directory."""
    return bool(input_path) and any(char in input_path for char in GLOB_METACHARS)


def split_pattern(pattern: str) -> Tuple[pathlib.Path, str]:
    """Splits a glob pattern at its first metacharacter-bearing component.

    ./fsdb/DE-LANRWR**/*.htr.txt -> (./fsdb, DE-LANRWR**/*.htr.txt)
    *.txt                        -> (., *.txt)

    Splitting per component rather than slicing the string keeps patterns whose very
    first component carries a metacharacter intact.
    """
    fixed, rest = [], []
    for part in pathlib.Path(pattern).parts:
        if rest or any(char in part for char in GLOB_METACHARS):
            rest.append(part)
        else:
            fixed.append(part)
    base = pathlib.Path(*fixed) if fixed else pathlib.Path('.')
    return base, (str(pathlib.Path(*rest)) if rest else '')


def pattern_base(pattern: str) -> pathlib.Path:
    """The leading directory of a pattern that holds no metacharacters -- the root the
    reported paths are relative to, so files come out as DE-LANRWR001/text.htr.txt."""
    return split_pattern(pattern)[0]


def find_pattern_files(pattern: str) -> List[pathlib.Path]:
    """Expands a glob pattern to a SORTED list of files.

    Uses glob.glob rather than pathlib.Path.glob because pathlib rejects a pattern such
    as DE-LANRWR**/*.htr.txt outright ("'**' can only be an entire path component"),
    while glob -- like the shell -- reads X** as X*. So the pattern a user actually
    writes for fsdb works here.

    The pattern alone decides which files are read: file_suffix is deliberately not
    applied on top, or a pattern like *.htr would match nothing once filtered for .txt.
    """
    base, rest = split_pattern(pattern)
    if not rest:
        print(f"Warning: Pattern '{pattern}' has no wildcard component; nothing to expand.")
        return []
    if '**' in pattern and not any(part == '**' for part in pathlib.Path(pattern).parts):
        # Worth saying out loud: the doubled star is not doing what it looks like.
        print(f"Note: '**' is only recursive as a standalone path component, so '{pattern}' "
              f"descends a single level. Use a pattern like '{base}/**/*' for unbounded depth.")
    try:
        return sorted(pathlib.Path(match) for match in glob.glob(pattern, recursive=True)
                      if os.path.isfile(match))
    except (re.error, OSError) as exc:
        print(f"Warning: Could not expand pattern '{pattern}': {exc}")
        return []


def display_path(path, root) -> str:
    """How a text is named in the reports: its path relative to the input root, so
    that nested corpora with identical filenames stay distinguishable. Falls back to
    the bare filename when the path lies outside the root."""
    if root:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def load_stopwords(path: str) -> set:
    """Tokens listed in `path` (one per line, '#' comments) whose IDF counts as zero.

    A hand-written list is a *superset* of what the corpus already knows, and it
    is only useful for words the corpus is too small to have measured as common --
    a two-hundred-charter sample does not yet know that 'benedictionem' is
    everywhere. The list only ever lowers a core's specificity; it never removes a
    word from a core, so two clusters that share a formula stay comparable.
    """
    if not path:
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return {line.split('#')[0].strip().lower() for line in f
                    if line.split('#')[0].strip()}
    except OSError as error:
        print(f"Warning: could not read the stop-word file '{path}' ({error}); "
              f"specificity is measured on the corpus's own document frequencies alone.")
        return set()


def _token_span(positions: List[int], start: int, size: int) -> Tuple[int, int]:
    """Maps a core's position among a document's *kept* tokens to a token range.

    `core_sequence` counts in the filtered token space (punctuation dropped), the
    report highlights in the tokenizer's own space, and this is the conversion
    between the two. An empty run maps to an empty range, which `render_tokens`
    then marks nowhere -- the honest answer for a core that was not found.
    """
    window = positions[start:start + size]
    return (window[0], window[-1] + 1) if window else (0, 0)


def _fold_documents(display_token_corpus: List[List[str]]
                    ) -> Tuple[List[List[int]], List[List[Tuple[str, str]]]]:
    """Folds every document once: (kept-token positions, (display, folded) pairs).

    Split out of `compute_clusters` because the two have to agree token for token
    -- the positions are what put the highlight back on the right words -- and
    deriving one from the other's filtered output is the only way to guarantee
    that.
    """
    positions: List[List[int]] = []
    folded: List[List[Tuple[str, str]]] = []
    for tokens in display_token_corpus:
        entries = list(flame_clustering.iter_token_pairs(tokens, fold_for_compare))
        positions.append([index for index, _display, _folded in entries])
        folded.append([(display, folded_token) for _index, display, folded_token in entries])
    return positions, folded


def _add_span(doc_texts: Dict[str, Dict[str, object]], name: str, tokens: List[str],
              positions: List[int], span: Tuple[int, int]) -> None:
    """Records that `name` carries the core at `span`, for the report's read view.

    Tokens are kept only for the documents that a cluster actually names, and only
    once per document: the same charter can appear in several clusters, but within
    one cluster its text is one string with perhaps several marked runs.
    """
    entry = doc_texts.setdefault(name, {"tokens": tokens, "spans": []})
    entry["spans"].append(_token_span(positions, span[0], span[1]))


def _text_key(text: str) -> str:
    """A short, stable key for a document's text, used to count distinct texts.

    Deliberately the same normalization the loader applies (`_read_text_file`), so
    two files that differ only in whitespace key the same -- they are one document
    to the engine, whatever the reports call them. This is only a *reporting*
    identity: the corpus still holds both files, and both are compared, unless
    deduplication is switched on.
    """
    return hashlib.sha1(' '.join(text.split()).encode('utf-8')).hexdigest()


class SimilarityVisualizer:
    detokenizer = TreebankWordDetokenizer()

    @staticmethod
    def _extract_year_from_filename(filename: str) -> int:
        match = re.search(r'(?<!\d)(1\d{3}|2\d{3})(?!\d)', filename)
        if match:
            return int(match.group(1))
        return 9999

    @staticmethod
    def _render_gap_html(gap1_tokens: List[str], gap2_tokens: List[str], max_gap_words: int, fuzzy_threshold: float) -> Tuple[str, str]:
        """Renders the words between two matches.

        Only a genuine divergence is a bridge: a one-sided gap is an insertion, and a
        gap whose folded forms agree is an orthographic variant of the same wording.
        """
        words1 = [token for token in gap1_tokens if token.isalnum()]
        words2 = [token for token in gap2_tokens if token.isalnum()]
        str1 = SimilarityVisualizer.detokenizer.detokenize(gap1_tokens)
        str2 = SimilarityVisualizer.detokenizer.detokenize(gap2_tokens)

        if not str1 and not str2:
            return "", ""

        if not (len(words1) <= max_gap_words and len(words2) <= max_gap_words and (words1 or words2)):
            return str1, str2

        kind, ratio = classify_gap(gap1_tokens, gap2_tokens, fuzzy_threshold)

        if kind == 'insertion':
            # Wording present in one text only: nothing is bridged here, and the absent
            # side must not turn into an empty highlight.
            html1 = f'<span class="dynamic-bridge-word" data-kind="insertion" data-fuzz="{ratio:.3f}">{str1}</span>' if str1 else ""
            html2 = f'<span class="dynamic-bridge-word" data-kind="insertion" data-fuzz="{ratio:.3f}">{str2}</span>' if str2 else ""
            return html1, html2

        wrapper = "bridge-words" if kind == "bridge" else "variant-words"
        html1 = f'<span class="dynamic-bridge-word" data-kind="{kind}" data-fuzz="{ratio:.3f}">{str1}</span>'
        html2 = f'<span class="dynamic-bridge-word" data-kind="{kind}" data-fuzz="{ratio:.3f}">{str2}</span>'
        return f'<span class="{wrapper}">{html1}</span>', f'<span class="{wrapper}">{html2}</span>'

    @staticmethod
    def highlight_similarities(text1_original_tokens: List[str], text2_original_tokens: List[str], unique_pair_id: str, max_gap_words: int, fuzz_threshold: float) -> Tuple[str, str]:
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
                gap1_html, gap2_html = SimilarityVisualizer._render_gap_html(gap1_tokens, gap2_tokens, max_gap_words, fuzz_threshold)
                if gap1_html: highlighted_html_text1.append(gap1_html)
                if gap2_html: highlighted_html_text2.append(gap2_html)

            m_txt1_raw = SimilarityVisualizer.detokenizer.detokenize(text1_original_tokens[a_start_orig:a_end_orig])
            m_txt1_processed = re.sub(r'([^\w\s])', r'<span class="punct-in-match">\1</span>', m_txt1_raw)
            m_txt1_html = f'<span class="highlight clickable" data-match-id="{m_id}" data-pair-id="{unique_pair_id}">{m_txt1_processed}</span>'
            highlighted_html_text1.append(m_txt1_html)

            m_txt2_raw = SimilarityVisualizer.detokenizer.detokenize(text2_original_tokens[b_start_orig:b_end_orig])
            m_txt2_processed = re.sub(r'([^\w\s])', r'<span class="punct-in-match">\1</span>', m_txt2_raw)
            m_txt2_html = f'<span class="match-text" data-match-id="{m_id}" data-pair-id="{unique_pair_id}">{m_txt2_processed}</span>'
            highlighted_html_text2.append(m_txt2_html)

            m_id += 1
            pos1, pos2 = a_end_orig, b_end_orig

        if pos1 < len(text1_original_tokens):
            highlighted_html_text1.append(SimilarityVisualizer.detokenizer.detokenize(text1_original_tokens[pos1:]))
        if pos2 < len(text2_original_tokens):
            highlighted_html_text2.append(SimilarityVisualizer.detokenizer.detokenize(text2_original_tokens[pos2:]))

        return " ".join(filter(None, highlighted_html_text1)), " ".join(filter(None, highlighted_html_text2))

    @staticmethod
    def generate_comparison_html(analyzer, similarity_threshold: float, max_file_size=20 * 1024 * 1024):
        if analyzer.dist_mat is None or not analyzer.corpus:
            print("ERROR: Distance matrix or corpus not found. Cannot generate HTML report.")
            return

        html_template_start = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Text Similarity Comparison</title><style>
        body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;margin:20px;line-height:1.6;background-color:#f8f9fa}}
        .comparison-block{{margin-bottom:2em;background-color:#fff;padding:1.5em;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.05);border:1px solid #dee2e6}}
        .comparison-container{{display:flex;gap:20px;flex-wrap:wrap}}@media(min-width:768px){{.comparison-container{{flex-wrap:nowrap}}}}
        .text-box{{flex:1 1 100%;min-width:300px;padding:15px;border:1px solid #ced4da;border-radius:5px;background-color:#fff;height:400px;overflow-y:auto;position:relative}}
        h2{{color:#212529;border-bottom:2px solid #e9ecef;padding-bottom:.5em; margin-top: 0;}}
        h3{{color:#343a40;margin-top:0}}
        .similarity-score{{font-weight:700;color:#0056b3}}
        .file-info{{font-size:.9em;color:#6c757d;margin-bottom:.5em;font-weight:700}}
        .highlight{{background-color:#fff3b8;border-radius:3px}}
        .highlight.clickable{{cursor:pointer;transition:background-color .2s}}
        .highlight.clickable:hover{{background-color:#ffe066}}
        .active-highlight{{background-color:#ffd700;box-shadow:0 0 0 2px #ffc107}}
        .match-text{{border-radius:3px;transition:background-color .3s}}
        .hover-highlight {{background-color: #ffe066 !important; box-shadow: 0 0 0 2px #ffc107;}}
        .match-text.active {{ background-color:#fff3b8; }}

        /* Gap classification: a genuine bridge is fuzzy-coloured by the slider, an
           orthographic variant is green, an insertion is blue. */
        .dynamic-bridge-word {{ border-radius: 3px; padding: 0 2px; transition: background-color 0.2s, color 0.2s; }}
        .dynamic-bridge-word[data-kind="bridge"].is-similar {{ background-color: #fff9e0; color: #000; }}
        .dynamic-bridge-word[data-kind="bridge"].is-dissimilar {{ background-color: #ffcdd2; color: #550000; }}
        .dynamic-bridge-word[data-kind="variant"] {{ background-color: #d8f3dc; color: #0b3d1e; }}
        .dynamic-bridge-word[data-kind="insertion"] {{ background-color: #e7f1ff; color: #0b3d6b; }}

        .bridge-words.highlighted .dynamic-bridge-word.is-similar {{ background-color: #fff3b8; box-shadow: 0 0 0 1px #e0a800; }}
        .bridge-words.highlighted .dynamic-bridge-word.is-dissimilar {{ background-color: #ffb3ba; box-shadow: 0 0 0 1px #c62828; }}
        .variant-words.highlighted .dynamic-bridge-word[data-kind="variant"] {{ background-color: #95d5b2; box-shadow: 0 0 0 1px #2d6a4f; }}

        .legend {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 1em; font-size: 0.85em; color: #495057; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; border: 1px solid #ced4da; }}
        .swatch-bridge {{ background-color: #fff9e0; }}
        .swatch-variant {{ background-color: #d8f3dc; }}
        .swatch-insertion {{ background-color: #e7f1ff; }}

        #controls{{margin-bottom:1.5em;background-color:#fff;padding:1em 1.5em;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.05);border:1px solid #dee2e6;}}
        .button-container {{ display: flex; gap: 10px; margin-top: 1em; }}
        .control-button {{ background-color:#28a745;color:#fff;padding:0.4em 0.8em;border:none;border-radius:4px;cursor:pointer;font-weight:700; font-size: 0.9em; }}
        .control-button.active {{ background-color:#dc3545; }}
        .filter-container {{ display: flex; gap: 30px; margin-top: 1.5em; padding-top: 1em; border-top: 1px solid #e9ecef; flex-wrap: wrap; }}
        .slider-block {{ flex: 1; min-width: 250px; }}
        .slider-wrapper {{ position: relative; height: 30px; }}
        .slider-label {{ font-weight: 600; color: #495057; margin-bottom: 0.5em; display: block; }}
        .slider-values {{ display: flex; justify-content: space-between; font-family: monospace; font-size: 1.1em; color: #0056b3; margin-bottom: -5px; }}
        .form-control-range {{ position: absolute; width: 100%; -webkit-appearance: none; appearance: none; background: transparent; pointer-events: none; }}
        .form-control-range:focus {{ outline: none; }}
        .form-control-range::-webkit-slider-thumb {{ -webkit-appearance: none; appearance: none; height: 18px; width: 18px; background: #007bff; border-radius: 50%; border: 2px solid #fff; box-shadow: 0 0 5px rgba(0,0,0,0.2); pointer-events: auto; cursor: pointer; }}
        .form-control-range::-moz-range-thumb {{ height: 14px; width: 14px; background: #007bff; border-radius: 50%; border: 2px solid #fff; pointer-events: auto; cursor: pointer; }}
        .slider-track {{ position: absolute; width: 100%; height: 4px; background-color: #ddd; top: 7px; border-radius: 3px; }}
        </style></head><body>
        <div id="controls">
            <h2>Interactive Text Similarity Comparison</h2>
            <div class="filter-container">
                <div class="slider-block">
                    <label class="slider-label">Filter by Cosine Similarity</label>
                    <div class="slider-values">
                        <span id="min-similarity-val">0.000</span>
                        <span id="max-similarity-val">1.000</span>
                    </div>
                    <div class="slider-wrapper">
                        <div class="slider-track"></div>
                        <input type="range" min="0" max="1" value="{similarity_threshold}" step="0.001" class="form-control-range" id="min-similarity">
                        <input type="range" min="0" max="1" value="1.0" step="0.001" class="form-control-range" id="max-similarity">
                    </div>
                </div>

                <div class="slider-block">
                    <label class="slider-label">Bridge Word Fuzzy Threshold (Yellow Highlight)</label>
                    <div class="slider-values">
                        <span id="fuzz-threshold-val">{analyzer.args.fuzz_threshold:.3f}</span>
                        <span>1.000</span>
                    </div>
                    <div class="slider-wrapper">
                        <div class="slider-track" style="background-color: #ddd;"></div>
                        <input type="range" min="0" max="1" value="{analyzer.args.fuzz_threshold}" step="0.01" class="form-control-range" id="fuzz-threshold" style="pointer-events: auto;">
                    </div>
                </div>
            </div>
            <div class="button-container">
                <button id="toggle-all-bridge-words" class="control-button">Show All Bridge &amp; Variant Words</button>
                <button id="toggle-all-similarities" class="control-button">Show All Similarities</button>
            </div>
            <div class="legend">
                <span class="legend-item"><span class="legend-swatch swatch-bridge"></span>Bridge word &mdash; the two texts genuinely diverge here</span>
                <span class="legend-item"><span class="legend-swatch swatch-variant"></span>Orthographic variant &mdash; same wording, different spelling</span>
                <span class="legend-item"><span class="legend-swatch swatch-insertion"></span>Insertion &mdash; present in one of the two texts only</span>
            </div>
        </div>"""

        html_template_end = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const toggleSimilaritiesBtn = document.getElementById("toggle-all-similarities");
    const comparisonBlocks = document.querySelectorAll(".comparison-block");
    const fuzzSlider = document.getElementById("fuzz-threshold");
    const fuzzValSpan = document.getElementById("fuzz-threshold-val");

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

    // Dynamic Fuzzy Highlighting Processor
    function updateFuzzyBridges() {
        const threshold = parseFloat(fuzzSlider.value);
        fuzzValSpan.textContent = threshold.toFixed(3);

        // Only genuine bridges are fuzzy-coloured; variants and insertions keep the
        // static colours that say what they are.
        document.querySelectorAll('.dynamic-bridge-word[data-kind="bridge"]').forEach(el => {
            const currentFuzzScore = parseFloat(el.dataset.fuzz);
            if (currentFuzzScore >= threshold) {
                el.classList.add("is-similar");
                el.classList.remove("is-dissimilar");
            } else {
                el.classList.add("is-dissimilar");
                el.classList.remove("is-similar");
            }
        });
    }

    if (fuzzSlider) {
        fuzzSlider.addEventListener("input", updateFuzzyBridges);
        updateFuzzyBridges(); // Initialize layout on load
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
            document.querySelectorAll(".bridge-words, .variant-words").forEach(el => el.classList.toggle("highlighted", isActive));
            this.textContent = isActive ? "Hide All Bridge & Variant Words" : "Show All Bridge & Variant Words";
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
        min_display_threshold = similarity_threshold
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

            path1 = analyzer.file_paths[i]
            root1 = analyzer.corpus_root
            tokens1 = display_token_corpus1[i]

            path2 = analyzer.file_paths2[j] if analyzer.is_inter_comparison else analyzer.file_paths[j]
            root2 = analyzer.corpus_root2 if analyzer.is_inter_comparison else analyzer.corpus_root
            tokens2 = display_token_corpus2[j]

            year1 = SimilarityVisualizer._extract_year_from_filename(path1.name)
            year2 = SimilarityVisualizer._extract_year_from_filename(path2.name)

            if year1 > year2:
                # The roots travel with the paths, or a swapped pair would be reported
                # relative to the wrong corpus.
                path1, path2 = path2, path1
                root1, root2 = root2, root1
                tokens1, tokens2 = tokens2, tokens1

            h1, h2 = SimilarityVisualizer.highlight_similarities(
                tokens1, tokens2, unique_pair_id,
                max_gap_words=analyzer.args.max_gap_words,
                fuzz_threshold=analyzer.args.fuzz_threshold
            )
            f1 = display_path(path1, root1)
            f2 = display_path(path2, root2)
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
            y_labels = [display_path(p, analyzer.corpus_root) for p in analyzer.file_paths]
            x_labels = [display_path(p, analyzer.corpus_root2) for p in analyzer.file_paths2]
            title = 'Inter-Corpus Text Similarity Heatmap'
        else:
            y_labels = x_labels = [display_path(p, analyzer.corpus_root) for p in analyzer.file_paths]
            title = 'Text Similarity Heatmap (Intra-Corpus)'
        fig = go.Figure(data=go.Heatmap(z=dense_dist_mat, x=x_labels, y=y_labels, colorscale='Blues', zmin=0.0, zmax=1.0, colorbar=dict(title='Cosine Similarity')))
        fig.update_layout(title_text=title, height=max(600, len(y_labels)*20), width=max(700, len(x_labels)*20))
        fig.write_html("similarity_heatmap.html")
        print("Generated similarity_heatmap.html")

    @staticmethod
    def compute_clusters(analyzer, similarity_threshold: float,
                               cluster_threshold: float, cluster_min: int) -> Optional[dict]:
        """Groups the surviving pairs by the legal formula they share (see flame_clustering).

        Returns a dict the report generator and the TSV generators both read, or
        None when there is nothing to cluster. Kept separate from the report
        generation because the ClusterID column in the summary TSVs has to be
        known *before* those files are written, while the cluster report itself is
        written afterwards.

        The tricky part is naming. In a self-comparison both ends of a pair index
        the same corpus, so the two sides collapse; comparing two corpora, they
        are different documents and must never be merged -- a ClusterID list for
        a query document that quietly included reference-side indices would point
        at the wrong charter.
        """
        # Coerced at the door, not at each use: every caller hands these in from
        # somewhere that may hold a string (the GUI keeps one tk.StringVar per
        # parameter, so an untouched spin box arrives as "0.85"), and a single
        # un-coerced path downstream crashes the whole run at report time --
        # after the clustering has already been computed.
        similarity_threshold = float(similarity_threshold)
        cluster_threshold = float(cluster_threshold)
        cluster_min = int(cluster_min)
        if analyzer.dist_mat is None or not analyzer.corpus:
            return None
        display_token_corpus1 = [word_tokenize(text) for text in analyzer.corpus]
        display_token_corpus2 = [word_tokenize(text) for text in (analyzer.corpus2 or analyzer.corpus)]
        if analyzer.is_inter_comparison:
            doc_names1 = [display_path(p, analyzer.corpus_root) for p in analyzer.file_paths]
            doc_names2 = [display_path(p, analyzer.corpus_root2) for p in analyzer.file_paths2]
            sides1, sides2 = (0,), (1,)
        else:
            doc_names1 = doc_names2 = [display_path(p, analyzer.corpus_root) for p in analyzer.file_paths]
            sides1 = sides2 = (0, 1)

        # Content key per document, so the report can say how many *distinct texts*
        # stand behind a cluster's document list. A corpus that holds one charter
        # under three names (a shelfmark, an edition and a re-download) turns a
        # single comparison into nine pairs, and the pair count alone hides that.
        # Hashed rather than kept as strings: the same key is needed for both sides
        # of every pair, and holding 8000 charter texts twice over is wasteful.
        text_key1 = [_text_key(t) for t in analyzer.corpus]
        text_key2 = [_text_key(t) for t in analyzer.corpus2] if analyzer.is_inter_comparison else text_key1

        # Folded tokens are built once per document, not once per pair: a document
        # sits in many pairs, and folding is the expensive part of the signature.
        # The kept tokens' positions in the tokenizer's own token list travel
        # alongside, because the report marks the core inside the real text and
        # the core is reported in the filtered token space.
        positions1, folded1 = _fold_documents(display_token_corpus1)
        if analyzer.is_inter_comparison:
            positions2, folded2 = _fold_documents(display_token_corpus2)
        else:
            positions2, folded2 = positions1, folded1

        # How distinctive is a core? The corpus's own document frequencies answer
        # it, and they are free here: `folded1`/`folded2` already hold every
        # document's folded tokens, so this is one pass over data that had to be
        # built anyway. A core of forty ordinary charter words scores far below a
        # core of forty rare ones, which is the difference between "this formula is
        # everywhere in this corpus" and "these two charters borrowed from each
        # other". See _record_specificity.
        doc_freq: Counter = Counter()
        corpus_docs = list(folded1) + (list(folded2) if analyzer.is_inter_comparison else [])
        for token_pairs in corpus_docs:
            doc_freq.update({folded for _display, folded in token_pairs})
        stopwords = load_stopwords(getattr(analyzer.args, 'stopwords_file', ''))
        total_docs = max(1, len(corpus_docs))

        def idf(token: str) -> float:
            # Smoothed and non-negative, so a word merely shared by every document
            # scores zero rather than negative. A negative IDF would make a long
            # and utterly generic core score *lower* than a short generic one,
            # which reads as a bug even though the ordering is defensible.
            return 0.0 if token in stopwords else math.log((1 + total_docs) / (1 + doc_freq[token]))

        # The performative verbs, as folded stems, and empty unless the caller
        # supplies them -- the measured reason there is no built-in list is at
        # `core_anchors` in DEFAULT_PARAMS. Empty (or `none`, the only sentinel a
        # comma-separated list can carry) switches anchoring off and gives back
        # the pre-anchor clustering word for word.
        core_anchors = tuple(
            stem.strip() for stem in str(getattr(analyzer.args, 'core_anchors', '') or '').split(',')
            if stem.strip() and stem.strip().lower() != 'none')

        dist_mat_coo = analyzer.dist_mat.tocoo()
        pair_docs: List[Tuple[int, int]] = []
        pair_scores: List[float] = []
        sigs: List[str] = []
        cores: List[Tuple[str, str]] = []
        # The window each pair's core covers in BOTH documents of the pair, so
        # the read-through view can mark the formula where each charter writes
        # it. Both spans come from one alignment (see align_core), rather than
        # from a second search from the other side.
        core_spans: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        # How much the pair shares over the *whole* alignment, before the anchor
        # ceiling narrowed the reported window. The near-duplicate test asks how
        # much of the shorter charter two documents cover, and a 50-token formula
        # window cannot answer that -- measuring it from the window would empty
        # the near-duplicate section (its threshold is 400 shared words).
        core_overlaps: List[int] = []
        core_anchors_used: List[str] = []
        core_identities: List[float] = []
        n_degraded = 0
        n_anchored = 0
        for i, j, score in zip(dist_mat_coo.row, dist_mat_coo.col, dist_mat_coo.data):
            if score < similarity_threshold: continue
            if not analyzer.is_inter_comparison and i >= j: continue
            alignment = flame_clustering.align_core(
                folded1[i], folded2[j],
                gap_tolerance=int(getattr(analyzer.args, 'core_gap_tolerance', 8) or 8),
                min_tokens=int(getattr(analyzer.args, 'core_min_tokens', 12) or 12),
                identity_floor=float(getattr(analyzer.args, 'core_identity_threshold', 0.0) or 0.0),
                anchors=core_anchors,
                anchor_window=int(getattr(analyzer.args, 'core_anchor_window', 15)),
                # Not `or`-guarded: 0 means "no ceiling" here, and `0 or 50` would
                # silently put the ceiling back.
                max_tokens=int(getattr(analyzer.args, 'core_max_tokens', 50)),
                idf=idf)
            core_folded, core_display = alignment.folded, alignment.display
            if not core_folded:
                # Two texts can clear the similarity threshold on scattered short
                # matches yet share no run of words that chains into a formula --
                # there is then nothing to cluster on. Letting them through as an
                # empty signature would put every such pair into one giant
                # "shared nothing" cluster, which is the exact opposite of the
                # answer.
                continue
            pair_docs.append((int(i), int(j)))
            pair_scores.append(float(score))
            sigs.append(core_folded)
            cores.append((core_display, core_folded))
            core_spans.append(((alignment.start, alignment.size),
                               (alignment.start2, alignment.size2)))
            core_overlaps.append(alignment.overlap)
            core_anchors_used.append(alignment.anchor)
            core_identities.append(alignment.identity)
            if alignment.degraded:
                n_degraded += 1
            if alignment.anchor:
                n_anchored += 1

        # The duplicate rule the report already judges *clusters* by, applied to
        # single pairs and handed to the clustering as forced edges. Clustering
        # runs on the cores, and a core narrowed to the legal act matches nothing
        # even between two copies of one charter: measured on the MOM corpus,
        # anchoring dropped six charter documents that shared 400+ words out of
        # the report entirely, their pair left alone below `cluster_min`. The rule
        # reads the pair's whole shared span, which no window narrows, so it -- not
        # the core's similarity -- decides who is a copy of whom.
        forced_groups = flame_clustering.duplicate_groups(
            pair_docs, core_overlaps, [len(f) for f in folded1], [len(f) for f in folded2],
            max_core_fraction=flame_clustering.MAX_CORE_FRACTION,
            min_duplicate_tokens=int(getattr(analyzer.args, 'min_duplicate_tokens', 400) or 400))
        # NB: the per-cluster verdict below re-reads the same two values (see
        # `max_core_fraction` / `min_dup_tokens`); they are parameters, so the two
        # call sites have to agree on where they come from.
        grouped = flame_clustering.cluster_pairs(
            sigs, threshold=cluster_threshold, min_size=max(2, int(cluster_min)),
            linkage=getattr(analyzer.args, 'cluster_linkage', 'louvain'),
            forced_groups=forced_groups)

        # Specificity is measured on the reference core, the same one the report
        # prints, and computed before any filtering so the number shown and the
        # number filtered on can never be two different things.
        clusters = []
        for cluster in grouped["clusters"]:
            core_tokens = cores[cluster["members"][0]][1].split()
            distinct_tokens = sorted(set(core_tokens))
            mean_idf = (sum(idf(t) for t in distinct_tokens) / len(distinct_tokens)) if distinct_tokens else 0.0
            cluster["core_tokens"] = len(core_tokens)
            cluster["mean_idf"] = mean_idf
            cluster["specificity"] = len(core_tokens) * mean_idf
            clusters.append(cluster)

        min_specificity = float(getattr(analyzer.args, 'min_core_specificity', 0.0) or 0.0)
        # The near-duplicate rule, read once: `duplicate_groups` above and the
        # per-cluster verdict below have to be the same rule, or a pair could be
        # wired into the graph as a copy and then have its cluster called a
        # formula.
        max_core_fraction = flame_clustering.MAX_CORE_FRACTION
        min_dup_tokens = int(getattr(analyzer.args, 'min_duplicate_tokens', 400) or 400)
        if min_specificity > 0:
            kept = [c for c in clusters if c["specificity"] >= min_specificity]
            print(f"Specificity filter: {len(clusters) - len(kept)} of {len(clusters)} cluster(s) "
                  f"dropped below {min_specificity:.1f}.")
            # Renumbered and re-indexed before anything downstream reads them: the
            # ClusterID column of both summary TSVs is built from these ids, and a
            # gap in the numbering would make them point at dropped clusters.
            clusters = kept
            remap = {cluster["id"]: cluster_id for cluster_id, cluster in enumerate(clusters)}
            for cluster_id, cluster in enumerate(clusters):
                cluster["id"] = cluster_id
                # `shared_with` holds ids from before the filter, so it has to be
                # translated too: a dropped neighbour must vanish rather than point
                # at whichever cluster inherited its number.
                cluster["shared_with"] = sorted({remap[cid] for cid in cluster.get("shared_with", [])
                                                 if cid in remap})
            grouped["pair_cluster"] = [-1] * len(pair_docs)
            for cluster in clusters:
                for pair_index in cluster["members"]:
                    grouped["pair_cluster"][pair_index] = cluster["id"]
            grouped["clusters"] = clusters
            grouped["stats"]["n_clusters"] = len(clusters)
            grouped["stats"]["n_clustered"] = sum(c["size"] for c in clusters)
            grouped["stats"]["n_singletons"] = len(pair_docs) - grouped["stats"]["n_clustered"]

        records = []
        for cluster in clusters:
            docs = [doc for doc in flame_clustering.cluster_documents([cluster], pair_docs, sides1)[0]]
            # An inter-corpus cluster spans both corpora, so its document list has
            # to name the reference side too; in a self-comparison sides1 already
            # covered both ends and this second pass would repeat itself.
            doc_labels = [doc_names1[d] for d in docs]
            keys = {text_key1[d] for d in docs}
            if analyzer.is_inter_comparison:
                ref_docs = flame_clustering.cluster_documents([cluster], pair_docs, sides2)[0]
                doc_labels += [doc_names2[d] for d in ref_docs]
                keys |= {text_key2[d] for d in ref_docs}
            members = []
            doc_texts: Dict[str, Dict[str, object]] = {}
            for pair_index in cluster["members"]:
                i, j = pair_docs[pair_index]
                name1 = doc_names1[i]
                name2 = doc_names2[j] if analyzer.is_inter_comparison else doc_names1[j]
                members.append((name1, name2, pair_scores[pair_index]))
                # The alignment already carries the formula's window in both
                # documents, so each charter is marked where it writes the
                # formula itself.
                span1, span2 = core_spans[pair_index]
                _add_span(doc_texts, name1, display_token_corpus1[i], positions1[i], span1)
                if span2[1]:
                    _add_span(doc_texts, name2, display_token_corpus2[j], positions2[j], span2)
            reference = cores[cluster["members"][0]]
            # How much of the shorter charter the core's window covers. A formula
            # is a *part* of a charter; when the window swallows most of one, the
            # cluster is not a shared formula but a charter copied out again, and
            # the report says so instead of listing it as a formula find (see
            # flame_clustering.MAX_CORE_FRACTION).
            coverage = 0.0
            shared_tokens = 0
            # Both numbers have to come from ONE pair. Taking the two maxima
            # independently lets a short overlap on a short charter supply the
            # fraction and a long overlap on a long one the word count -- a
            # cluster then reads as a copy although no pair in it is one, and the
            # card's own numbers contradict its `Kind`. `min_duplicate_tokens`
            # and `MAX_CORE_FRACTION` are read from `analyzer.args` so the verdict
            # here and the report's column can never be two different rules.
            dup_coverage = 0.0
            dup_shared = 0
            for pair_index in cluster["members"]:
                i, j = pair_docs[pair_index]
                # The *overlap*, not the reported window: anchoring narrows the
                # window to the formula (at most core_max_tokens), and a copy of
                # a whole charter shares far more than that. Measuring the
                # duplicate test on the window would put every near-duplicate
                # below the 400-word threshold and quietly move the 1721-word
                # copies back into the formula network.
                shared = core_overlaps[pair_index]
                shorter = min(len(folded1[i]), len(folded2[j]))
                if not shared or not shorter:
                    continue
                if shared > shared_tokens:
                    # The widest overlap the cluster has, which is what the card
                    # shows for a formula.
                    coverage, shared_tokens = shared / shorter, shared
                if (shared >= min_dup_tokens and shared / shorter >= max_core_fraction
                        and shared > dup_shared):
                    # ...and the widest overlap that is a copy on its own terms.
                    dup_coverage, dup_shared = shared / shorter, shared
            if dup_shared:
                coverage, shared_tokens = dup_coverage, dup_shared
            records.append({
                "id": cluster["id"], "size": cluster["size"],
                "core_display": reference[0], "core_folded": reference[1],
                # The performative verb the reference pair's window was seeded on:
                # the one-word answer to "which legal act is this cluster?", which
                # is what a diplomatist reads the cluster for. Empty when the
                # pair's shared text carries none (the unanchored fallback).
                "anchor": core_anchors_used[cluster["members"][0]],
                "documents": doc_labels, "distinct_texts": len(keys),
                "members": members, "coverage": coverage, "shared_tokens": shared_tokens,
                # The per-pair core ratio is measured against the cluster's own
                # reference core, which is what the headline CoreFormula shows;
                # below the threshold it means the pair hangs off the cluster
                # through another one -- under the strict linkage it cannot happen.
                "core_ratio": cluster["core_ratio"], "cohesion": cluster["cohesion"],
                "core_tokens": cluster["core_tokens"], "mean_idf": cluster["mean_idf"],
                "specificity": cluster["specificity"], "doc_texts": doc_texts,
                # Under strict linkage a formula can belong to two cliques at once;
                # the card then names the other cluster(s) it also belongs to.
                "shared_with": cluster.get("shared_with", []),
            })

        if core_identities:
            ordered = sorted(core_identities)
            median_identity = ordered[len(ordered) // 2]
        else:
            median_identity = 0.0
        print(f"Clustering: {grouped['stats']['n_pairs']} pair(s) -> "
              f"{grouped['stats']['n_cores']} distinct core(s) -> "
              f"{grouped['stats']['n_clusters']} cluster(s) "
              f"({grouped['stats'].get('linkage', 'louvain')} linkage).")
        print(f"Cores: median aligned identity {median_identity:.3f}; "
              f"{n_degraded} pair(s) fell back to the strict contiguous core.")
        if core_anchors:
            print(f"Anchoring: {n_anchored} of {len(pair_docs)} pair(s) seeded on a performative "
                  f"verb ({', '.join(core_anchors[:6])}...); "
                  f"{len(pair_docs) - n_anchored} kept the unanchored window.")
        else:
            print("Anchoring: off (core_anchors is empty); windows come from the "
                  "matched-token chain alone.")
        if forced_groups:
            n_forced = sum(len(g) for g in forced_groups)
            n_lone = sum(1 for g in forced_groups if len(g) == 1)
            print(f"Duplicate links: {n_forced} pair(s) clear the near-duplicate rule on their "
                  f"shared span and are wired into the graph by that alone, in "
                  f"{len(forced_groups)} charter family/families.")
            if n_lone:
                print(f"  {n_lone} of those families hold a single pair, so cluster_min "
                      f"({cluster_min}) still leaves them out of the cluster report; "
                      f"they are named in the pairs report.")
        # Keyed by (i, j) rather than by position: pairs whose two texts share no
        # single run of words were dropped above, so a positional lookup would
        # silently hand a later pair the cluster of an earlier one.
        pair_cluster_of = {pair_docs[p]: grouped["pair_cluster"][p] for p in range(len(pair_docs))}
        return {
            "grouped": grouped,
            "records": records,
            "pair_cluster_of": pair_cluster_of,
            # Ready-made ClusterID cells for the summary TSVs, one list per side.
            "cluster_ids1": flame_clustering.cluster_id_map(clusters, pair_docs, len(doc_names1), sides1),
            "cluster_ids2": flame_clustering.cluster_id_map(clusters, pair_docs, len(doc_names2), sides2),
            "cluster_threshold": cluster_threshold,
            "cluster_min": cluster_min,
            "similarity_threshold": similarity_threshold,
            # How many pairs the anchor actually served, for the provenance block:
            # the rest kept the unanchored window, and a report that hid that
            # would read as if every cluster had been anchored.
            "n_anchored": n_anchored,
            "n_pairs": len(pair_docs),
        }

    @staticmethod
    def _cluster_input_info(analyzer, clusters: dict) -> Dict[str, object]:
        """What the cluster report has to say about the corpus it was computed on.

        The report is read away from the run that produced it, so it carries its
        own provenance: which corpus, how many charters survived loading and why
        the rest did not, which thresholds were used and where they came from, and
        the command line that reproduces it. All of it is known here and nowhere
        downstream.
        """
        stats1 = analyzer.load_stats or {}
        stats2 = analyzer.load_stats2 or {}
        auto = str(analyzer.args.similarity_threshold).lower() == 'auto'
        return {
            "mode": "two" if analyzer.is_inter_comparison else "single",
            "input": analyzer.args.input_path,
            "input2": analyzer.args.input_path2 or None,
            "pattern": looks_like_pattern(analyzer.args.input_path),
            "suffix": analyzer.args.file_suffix,
            "charters": ([len(analyzer.corpus), len(analyzer.corpus2)] if analyzer.is_inter_comparison
                         else [len(analyzer.corpus)]),
            "files_found": ([stats1.get("found"), stats2.get("found")] if analyzer.is_inter_comparison
                            else [stats1.get("found")]),
            "files_short": ([stats1.get("short", 0), stats2.get("short", 0)] if analyzer.is_inter_comparison
                            else [stats1.get("short", 0)]),
            "files_duplicate": ([stats1.get("duplicate", 0), stats2.get("duplicate", 0)]
                                if analyzer.is_inter_comparison else [stats1.get("duplicate", 0)]),
            "deduplicate": bool(analyzer.args.deduplicate),
            "min_text_length": analyzer.args.min_text_length,
            "keep_texts": analyzer.args.keep_texts,
            "limit_reached": bool(stats1.get("limit_reached") or stats2.get("limit_reached")),
            "threshold_source": (f"{analyzer.args.auto_threshold_method}" if auto else None),
            "linkage": clusters["grouped"]["stats"].get("linkage", "louvain"),
            "alignment": (f"gapped local alignment, gaps up to "
                          f"{int(analyzer.args.core_gap_tolerance)} token(s); a pair whose window "
                          f"falls below {int(analyzer.args.core_min_tokens)} token(s) or under "
                          f"{float(analyzer.args.core_identity_threshold):.2f} identity falls back "
                          f"to the longest contiguous run"),
            "anchors": ([f"seeded on a performative verb (up to +/-"
                         f"{int(analyzer.args.core_anchor_window)} token(s), window capped at "
                         f"{int(analyzer.args.core_max_tokens)} token(s)); a pair whose shared text "
                         f"carries none keeps the unanchored window"]
                        if str(analyzer.args.core_anchors).strip() else
                        ["off: windows come from the matched-token chain alone, so the longest "
                         "shared run (the protocol) wins"]),
            # `.get` because a report can be built from a clustering dict that
            # never went through `compute_clusters` (the tests do exactly that),
            # and a provenance block is not worth an exception.
            "anchor_counts": ([clusters.get("n_anchored", 0),
                               clusters.get("n_pairs",
                                            clusters["grouped"]["stats"].get("n_pairs", 0))]
                              if str(analyzer.args.core_anchors).strip() else None),

            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "command": " ".join([os.path.basename(_LAUNCH_ARGV[0])] + _LAUNCH_ARGV[1:]),
        }

    @staticmethod
    def generate_cluster_report(analyzer, clusters: Optional[dict]) -> None:
        """Writes the standalone cluster report from an already-computed clustering."""
        if not clusters or not clusters["records"]:
            print("No cluster reached the minimum size; skipping the cluster report.")
            return
        flame_clustering.generate_cluster_report(
            clusters["records"], clusters["grouped"]["stats"],
            similarity_threshold=clusters["similarity_threshold"],
            cluster_threshold=clusters["cluster_threshold"],
            cluster_min=clusters["cluster_min"],
            input_info=SimilarityVisualizer._cluster_input_info(analyzer, clusters),
            max_core_fraction=flame_clustering.MAX_CORE_FRACTION,
            min_duplicate_tokens=int(getattr(analyzer.args, 'min_duplicate_tokens', 400) or 400))

    @staticmethod
    def generate_similarity_summary_tsv(analyzer, similarity_threshold: float, clusters: Optional[dict] = None):
        if analyzer.dist_mat is None or not analyzer.corpus: return
        print(f"Generating TSV summary using threshold: {similarity_threshold:.4f}")
        related_docs_map = defaultdict(list)
        dist_mat_coo = analyzer.dist_mat.tocoo()
        for i, j, score in zip(dist_mat_coo.row, dist_mat_coo.col, dist_mat_coo.data):
            if i != j and score >= similarity_threshold:
                related_docs_map[i].append(j)
        # The ClusterID column appears only when clustering actually ran, so a run
        # with -gen_clusters False keeps the older four-column layout and
        # anything parsing this file downstream is unaffected.
        if clusters:
            header = "DocumentFilename\tClusterID\tSimilarityFrequency\tRelatedDocuments\tLongSimilarities(>4words)\n"
        else:
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
                related_doc_names = sorted([display_path(analyzer.file_paths2[j], analyzer.corpus_root2)
                                            for j in related_docs_indices])
            else:
                related_doc_names = sorted([display_path(analyzer.file_paths[j], analyzer.corpus_root)
                                            for j in related_docs_indices])
            long_segments_str = ' | '.join(f'"{s}"' for s in sorted(long_segments, key=len, reverse=True)) or 'None'
            related_docs_str = ', '.join(related_doc_names) or 'None'
            cluster_cell = f"{clusters['cluster_ids1'][i]}\t" if clusters else ""
            rows.append(f"{display_path(analyzer.file_paths[i], analyzer.corpus_root)}\t{cluster_cell}{len(related_doc_names)}\t{related_docs_str}\t{long_segments_str}\n")
        with open("similarity_summary.tsv", "w", encoding='utf-8') as f: f.writelines(rows)
        print("Generated similarity_summary.tsv")

    @staticmethod
    def generate_linguistic_summary_tsv(analyzer, similarity_threshold: float, clusters: Optional[dict] = None):
        if analyzer.dist_mat is None or not analyzer.corpus: return
        print(f"Generating linguistic variations summary (TSV) using threshold: {similarity_threshold:.4f}")
        fuzz_threshold = analyzer.args.fuzz_threshold
        max_gap = analyzer.args.max_gap_words
        # Same contract as in the summary TSV: the column is present exactly when
        # clustering ran, so the older five-column layout survives otherwise. A
        # pair belongs to at most one cluster, so this cell is a single id or None.
        if clusters:
            rows = ["File_1\tFile_2\tClusterID\tVariation_Type\tToken_1\tToken_2\n"]
        else:
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
            # Paths relative to each side's own root, so a nested corpus with repeated
            # filenames stays readable in the report.
            file1_name = display_path(file1_path, analyzer.corpus_root)
            file2_name = display_path(file2_path, analyzer.corpus_root2 if analyzer.is_inter_comparison
                                      else analyzer.corpus_root)
            # Built once here so all the row-building branches below share one
            # prefix and cannot drift apart on the column count. -1 is the
            # internal "this pair's formula is unique" sentinel and must not
            # reach the file as a number a reader would take for a cluster id.
            cluster_cell = ""
            if clusters:
                cid = clusters["pair_cluster_of"].get((int(i), int(j)), -1)
                cluster_cell = f"{cid if cid >= 0 else 'None'}\t"
            pair_prefix = f"{file1_name}\t{file2_name}\t{cluster_cell}"
            analysis_tokens1 = [t.lower() for t in tokens1 if t.isalnum()]
            analysis_tokens2 = [t.lower() for t in tokens2 if t.isalnum()]
            if not analysis_tokens1 or not analysis_tokens2: continue
            matcher = SequenceMatcher(None, analysis_tokens1, analysis_tokens2, autojunk=False)
            pos1_analysis, pos2_analysis = 0, 0
            for a, b, size in matcher.get_matching_blocks():
                if size == 0: continue
                gap_tokens1 = analysis_tokens1[pos1_analysis:a]
                gap_tokens2 = analysis_tokens2[pos2_analysis:b]
                if (1 <= len(gap_tokens1) <= max_gap) or (1 <= len(gap_tokens2) <= max_gap):
                    # Same classification the HTML uses, so the report and the visual no
                    # longer disagree about what is a bridge and what is a spelling variant.
                    kind, _ = classify_gap(gap_tokens1, gap_tokens2, fuzz_threshold)
                    if kind == 'insertion':
                        for t1 in gap_tokens1: rows.append(f"{pair_prefix}Insertion\t{t1}\t-\n")
                        for t2 in gap_tokens2: rows.append(f"{pair_prefix}Insertion\t-\t{t2}\n")
                    elif kind == 'variant' and len(gap_tokens1) == len(gap_tokens2):
                        for t1, t2 in zip(gap_tokens1, gap_tokens2):
                            rows.append(f"{pair_prefix}Orthographic Variant\t{t1}\t{t2}\n")
                    elif kind == 'variant':
                        rows.append(f"{pair_prefix}Orthographic Variant\t{' '.join(gap_tokens1)}\t{' '.join(gap_tokens2)}\n")
                    elif len(gap_tokens1) == len(gap_tokens2) and len(gap_tokens1) > 0:
                        for t1, t2 in zip(gap_tokens1, gap_tokens2):
                            rows.append(f"{pair_prefix}Different Bridge Word\t{t1}\t{t2}\n")
                    else:
                        for t1 in gap_tokens1: rows.append(f"{pair_prefix}Different Bridge Word\t{t1}\t-\n")
                        for t2 in gap_tokens2: rows.append(f"{pair_prefix}Different Bridge Word\t-\t{t2}\n")
                pos1_analysis, pos2_analysis = a + size, b + size
        with open("linguistic_variations.tsv", "w", encoding='utf-8') as f:
            f.writelines(rows)
        print("Generated linguistic_variations.tsv")

def main():
    print("--- Formulaic Language Analysis in Medieval Expressions ---")
    print("For command-line options, run with the -h flag.")

    parsed_args, _ = fargv.fargv(DEFAULT_PARAMS)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Created temporary directory: {tmpdir}")
            analyzer = Flame(args=parsed_args, tmp_dir=tmpdir)

            if not analyzer.args.input_path:
                print("\n\033[91mError: Required argument -input_path is missing.\033[0m")
                print("You must specify the directory containing your text files, or a glob pattern.")
                print("\n\033[92mExample Usage:\033[0m")
                print(f"  python {__file__} -input_path /path/to/your/corpus")
                print(f"  python {__file__} -input_path './corpus/SUBSET**/*.txt'")
                print("\nNote: arguments take a single leading dash (-input_path, not --input_path).")
                print("For a full list of all available options, run:")
                print(f"  python {__file__} -h")
                return

            analyzer.load_corpus()
            if not analyzer.corpus:
                print("Execution halted because no documents were loaded. Please check the input path and file suffix.")
                return

            # Trigger unsupervised learning auto-tune step if selected
            if analyzer.args.auto_tune:
                analyzer.auto_tune_parameters()

            if (analyzer.args.ngram - analyzer.args.n_out) < 1:
                raise ValueError(f"N-gram size ({analyzer.args.ngram}) minus n-out ({analyzer.args.n_out}) must be at least 1.")

            analyzer.compute_similarity_matrix()
            if analyzer.args.no_reports:
                print("\n--- Report generation skipped due to -no_reports flag. ---")
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
                    print("Skipping heatmap generation as per configuration.")

                if analyzer.args.gen_comparison_html:
                    SimilarityVisualizer.generate_comparison_html(analyzer, similarity_threshold=final_threshold)
                else:
                    print("Skipping interactive HTML generation as per configuration.")

                # Clustering has to run before the TSVs, because both of them carry
                # a ClusterID column when it does. Its own report file is written
                # last, once the clustering they consumed is settled.
                clusters = None
                if analyzer.args.gen_clusters:
                    clusters = SimilarityVisualizer.compute_clusters(
                        analyzer, similarity_threshold=final_threshold,
                        cluster_threshold=analyzer.args.cluster_threshold,
                        cluster_min=analyzer.args.cluster_min)
                else:
                    print("Skipping clustering as per configuration.")

                if analyzer.args.gen_summary_tsv:
                    SimilarityVisualizer.generate_similarity_summary_tsv(analyzer, similarity_threshold=final_threshold, clusters=clusters)
                else:
                    print("Skipping summary TSV generation as per configuration.")

                if analyzer.args.gen_linguistic_tsv:
                    SimilarityVisualizer.generate_linguistic_summary_tsv(analyzer, similarity_threshold=final_threshold, clusters=clusters)
                else:
                    print("Skipping linguistic TSV generation as per configuration.")

                if analyzer.args.gen_clusters:
                    SimilarityVisualizer.generate_cluster_report(analyzer, clusters)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n--- Execution Finished ---")

if __name__ == '__main__':
    main()
