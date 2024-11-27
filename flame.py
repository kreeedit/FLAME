import numpy as np
import argparse
from tqdm import tqdm
import os
from pathlib import Path
import re
import unicodedata
import datetime
import html
from typing import Dict, List, Tuple
import json
import difflib

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False

def read_texts_from_directory(corpus_dir, file_suffix=''):
    """
    Reads texts from a directory and its subdirectories.
    Args:
      corpus_dir: The path to the directory.
      file_suffix: Optional suffix filter for files.
    Returns:
      A tuple containing:
        - A list of text strings.
        - A list of filenames.
        - A dictionary mapping filenames to file contents.
    """
    corpus = []
    filenames = []
    file_contents = {}

    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Directory not found: {corpus_dir}")

    # Walk through directory and subdirectories
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(file_suffix) or file_suffix == '':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()  # Add strip() to remove leading/trailing whitespace
                        if len(text) > 50:  # Minimum text length requirement
                            corpus.append(text)
                            filenames.append(Path(file_path).name)
                            file_contents[Path(file_path).name] = text
                except UnicodeDecodeError:
                    print(f"Warning: Skipping {file_path} due to encoding issues")
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read().strip()
                            if len(text) > 50:
                                corpus.append(text)
                                filenames.append(Path(file_path).name)
                                file_contents[Path(file_path).name] = text
                    except Exception as e:
                        print(f"Warning: Failed with latin-1 encoding for {file_path}: {str(e)}")
                except Exception as e:
                    print(f"Warning: Error reading {file_path}: {str(e)}")

    if not corpus:
        raise ValueError(f"No valid texts found in directory: {corpus_dir}")

    return corpus, filenames, file_contents

def read_texts_from_tsv(tsv_path):
    """
    Reads texts from a TSV file.
    Args:
      tsv_path: Path to the TSV file.
    Returns:
      A tuple containing:
        - A list of text strings.
        - A list of line numbers as filenames.
        - A dictionary mapping line numbers to text contents.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    corpus = []
    filenames = []
    file_contents = {}

    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # Use readlines() instead of read().split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line and '\t' in line:  # Check if line is not empty and contains tab
                    text = line.split('\t')[0].strip()
                    if len(text) > 50:
                        corpus.append(text)
                        filename = f"line_{i+1}"
                        filenames.append(filename)
                        file_contents[filename] = text
    except Exception as e:
        print(f"Error reading TSV file: {str(e)}")
        raise

    if not corpus:
        raise ValueError(f"No valid texts found in TSV file: {tsv_path}")

    return corpus, filenames, file_contents

def preprocess_medieval_text(text):
    """
    Preprocesses medieval text before normalization and tokenization.
    Handles hyphenation, line breaks, and cleans up various text artifacts.
    """
    # Skip empty text
    if not text or text.isspace():
        return ""

    # Handle line-break hyphenation with various hyphen types
    text = re.sub(r'(\w+)[‐‑‒–—―][\s\n]+(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Replace multiple newlines and special spaces with single space
    text = re.sub(r'[\n\r\t\f\v]+', ' ', text)

    # Clean up problematic characters and combinations
    text = re.sub(r'[\[\]\{\}»«()!¡¿?†‡•<>]', ' ', text)

    # Handle punctuation with spaces
    text = re.sub(r'([.,;:])(\S)', r'\1 \2', text)  # Add space after punctuation
    text = re.sub(r'(\S)([.,;:])', r'\1 \2', text)  # Add space before punctuation

    # Clean up spaces between words
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove punctuation at word boundaries
    text = re.sub(r'\s[!?,.:;()\[\]{}«»]+\s', ' ', text)  # Remove isolated punctuation
    text = re.sub(r'([!?,.:;])+\s', ' ', text)  # Remove trailing punctuation
    text = re.sub(r'\s([!?,.:;])+', ' ', text)  # Remove leading punctuation

    # Final cleanup
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Ensure single spaces

    # Return empty string if only whitespace remains
    if not text or text.isspace():
        return ""

    return text

def normalize_medieval_text(text):
    """
    Normalizes medieval text by handling various unicode characters and normalizations.
    """
    # First preprocess the text
    text = preprocess_medieval_text(text)

    # Normalize unicode characters (compose or decompose as needed)
    text = unicodedata.normalize('NFKD', text)

    # Convert common medieval abbreviation markers
    replacements = {
        'ꝫ': 'et',    # Tironian et
        'ꝑ': 'per',   # p with bar
        'ꝓ': 'pro',   # p with loop
        'ꝙ': 'que',   # q with stroke
        'ꝯ': 'con',   # reversed comma
        'æ': 'ae',    # ae ligature
        'œ': 'oe',    # oe ligature
        'Æ': 'Ae',    # capital ae ligature
        'Œ': 'Oe',    # capital oe ligature
        'ſ': 's',     # long s
        'ꝛ': 'r',     # r rotunda
        'ꝝ': 'rum',   # r with stroke
        'ꝥ': 'thor',  # thor abbreviation
        'ꝧ': 'con',   # con abbreviation
        'ꝩ': 'ver',   # ver abbreviation
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Standardize various types of quotes and dashes
    text = re.sub(r'[""‟″]', '"', text)
    text = re.sub(r'[''‛′]', "'", text)
    text = re.sub(r'[‒–—―]', '-', text)

    return text

class StanzaProcessor:
    def __init__(self, lang='la', use_lemmatizer=False):
        """
        Initialize Stanza processor for the specified language.
        Args:
            lang: Language code (default: 'la' for Latin)
            use_lemmatizer: Whether to include lemmatization (default: False)
        """
        if not STANZA_AVAILABLE:
            raise ImportError("Stanza is not installed. Please install it with 'pip install stanza'")

        try:
            processors = 'tokenize,lemma' if use_lemmatizer else 'tokenize'
            stanza.download(lang)
            self.nlp = stanza.Pipeline(lang=lang, processors=processors, verbose=False)
            self.use_lemmatizer = use_lemmatizer
        except Exception as e:
            raise Exception(f"Error initializing Stanza for language '{lang}': {str(e)}")

    def process(self, text):
        """
        Process text using Stanza.
        Returns tokens or lemmas based on configuration.
        """
        # Preprocess the text (but skip normalization)
        text = preprocess_medieval_text(text)

        # Process with Stanza
        doc = self.nlp(text)

        # Extract tokens or lemmas
        if self.use_lemmatizer:
            tokens = []
            for sentence in doc.sentences:
                tokens.extend([word.lemma for word in sentence.words])
        else:
            tokens = []
            for sentence in doc.sentences:
                tokens.extend([token.text for token in sentence.tokens])

        return tokens

def default_tokenize(text_str):
    """
    Default tokenizer using built-in medieval text processing.
    """
    # Apply both preprocessing and normalization
    text = normalize_medieval_text(preprocess_medieval_text(text_str))

    # Handle period abbreviations (e.g., "S. Maria")
    text = re.sub(r'(\w)\. (?=\w)', r'\1DOT ', text)

    # Split into tokens using whitespace
    tokens = text.split()

    # Process each token for special cases
    processed_tokens = []
    for token in tokens:
        # Handle punctuation
        token = re.sub(r'[¶†‡•<>]', '', token)  # Remove medieval marks and brackets

        # Skip empty tokens
        if not token or token.isspace():
            continue

        # Handle Roman numerals
        if re.match(r'^[IVXLCDM]+$', token, re.IGNORECASE):
            processed_tokens.append(token)
            continue

        # Restore abbreviated periods
        token = re.sub(r'(\w)DOT', r'\1.', token)

        # Handle punctuation at word boundaries
        token = re.sub(r'^[\[\{(<"\']+', '', token)  # Remove leading punctuation
        token = re.sub(r'[\]\})>"\';:,.!?]+$', '', token)  # Remove trailing punctuation

        if token and not token.isspace():
            processed_tokens.append(token)

    return processed_tokens

def get_encoder(corpus):
    all_tokens = []
    for text in corpus:
        tokens = tokenize(text)
        all_tokens.extend(tokens)
    encoder = {token: i for i, token in enumerate(sorted(set(all_tokens)))}
    return encoder

def tokens_to_int(tokens, encoder):
    return [encoder[token] for token in tokens]

def ngrams_to_int(tokens, ngram, encoder):
    int_tokens = np.array(tokens_to_int(tokens, encoder))
    seq_len = len(tokens)
    int_ngrams = np.zeros(1+seq_len-ngram, dtype=np.int64)
    for n in range(0, ngram):
        int_ngrams += int_tokens[n:seq_len +1 -(ngram-n)]*(len(encoder)**n)
    return int_ngrams

def skipngrams_to_int(tokens, ngram, encoder):
    int_tokens = np.array(tokens_to_int(tokens, encoder))
    seq_len = len(tokens)
    int_ngrams = np.zeros(1+seq_len-ngram, dtype=np.int64)
    skip = ngram-1
    sub_ngrams = []
    for n in range(0, ngram):
        sub_ngrams.append(int_tokens[n:seq_len -(ngram-n)])
    sub_ngrams = np.stack(sub_ngrams, axis=0)
    l1out_grams = []
    for n in range(0, ngram):
        sub_gram_idx = list(range(ngram))
        sub_gram_idx.pop(n)
        sub_int_ngrams = np.zeros_like(sub_ngrams[0], dtype=np.int64)
        for n, idx in enumerate(sub_gram_idx):
            sub_int_ngrams += sub_ngrams[idx]*(len(encoder)**n)
        l1out_grams.append(sub_int_ngrams)
    l1out_grams = np.concatenate(l1out_grams, axis=0)
    return l1out_grams

def distance(text1_tokens, text2_tokens, ngram, encoder):
    text1_int_ngrams = ngrams_to_int(text1_tokens, ngram, encoder)
    text2_int_ngrams = ngrams_to_int(text2_tokens, ngram, encoder)
    IoU = np.intersect1d(text1_int_ngrams, text2_int_ngrams).shape[0] / np.union1d(text1_int_ngrams, text2_int_ngrams).shape[0]
    return IoU

def skip_distance(text1_tokens, text2_tokens, ngram, encoder):
    text1_int_ngrams = skipngrams_to_int(text1_tokens, ngram, encoder)
    text2_int_ngrams = skipngrams_to_int(text2_tokens, ngram, encoder)
    IoU = np.intersect1d(text1_int_ngrams, text2_int_ngrams).shape[0] / np.union1d(text1_int_ngrams, text2_int_ngrams).shape[0]
    return IoU

def highlight_similarities(text1: str, text2: str, min_match_length=3) -> Tuple[str, str]:
    """
    Highlight similar sequences between two texts, only highlighting matches
    longer than min_match_length.
    Returns HTML-formatted versions of both texts with similar parts highlighted.
    """
    # Use SequenceMatcher to find matching blocks
    matcher = difflib.SequenceMatcher(None, text1, text2)

    # Create HTML versions with highlighting
    html1 = []
    html2 = []
    pos1 = 0
    pos2 = 0

    for block in matcher.get_matching_blocks():
        i, j, n = block

        # Add non-matching parts
        html1.append(html.escape(text1[pos1:i]))
        html2.append(html.escape(text2[pos2:j]))

        # Add matching parts with highlighting (if long enough)
        if n >= min_match_length:  # Highlight matches of length >= min_match_length
            html1.append(f'<span class="highlight">{html.escape(text1[i:i+n])}</span>')
            html2.append(f'<span class="highlight">{html.escape(text2[j:j+n])}</span>')

        pos1 = i + n
        pos2 = j + n

    # Add remaining text
    html1.append(html.escape(text1[pos1:]))
    html2.append(html.escape(text2[pos2:]))

    return ''.join(html1), ''.join(html2)

def generate_similarity_report(dist_mat: np.ndarray,
                             filenames: List[str],
                             file_contents: Dict[str, str],
                             output_prefix: str,
                             top_n: int = 3) -> None:
    """
    Generate an HTML report showing text similarities with highlighted common sequences.
    """
    similarity_data = []
    for i, filename in enumerate(filenames):
        similarities = dist_mat[i]

        # Sort similarities in descending order, skipping the self-comparison (file itself)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]  # Exclude the self-comparison

        # Select the top_n most similar files
        most_similar = sorted_indices[:top_n]

        # Now, create the report for the most similar files
        similar_texts = []
        for idx in most_similar:
            similar_file = filenames[idx]
            similarity = similarities[idx] * 100

            # Generate highlighted texts for side-by-side view
            highlighted_original, highlighted_compared = highlight_similarities(
                file_contents[filename],
                file_contents[similar_file]
            )

            # Generate diff view
            diff_html = generate_side_by_side_diff(
                file_contents[filename],
                file_contents[similar_file]
            )

            similar_texts.append({
                'filename': similar_file,
                'similarity': float(f"{similarity:.2f}"),
                'diffHtml': diff_html,
                'originalText': highlighted_original,
                'comparedText': highlighted_compared
            })

        similarity_data.append({
            'filename': filename,
            'similarTexts': similar_texts
        })

    # Generate the HTML content for the report
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Similarity Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .text-panel {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .similar-text {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .similarity-score {{
            color: #007bff;
            font-weight: bold;
        }}
        .comparison-container {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        .text-column {{
            flex: 1;
            min-width: 0;
        }}
        .text-box {{
            background: white;
            padding: 15px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
            line-height: 1.5;
        }}
        .text-header {{
            margin-bottom: 10px;
            font-weight: bold;
            color: #495057;
            padding: 8px;
            background: #e9ecef;
            border-radius: 4px;
        }}
        .highlight {{
            background-color: #fff3cd;
            border-radius: 2px;
            padding: 2px 0;
        }}
        .toggle-view {{
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 10px 0;
            transition: background-color 0.2s;
        }}
        .toggle-view:hover {{
            background: #0056b3;
        }}
        .view-mode {{
            margin-bottom: 10px;
        }}
        .hidden {{
            display: none;
        }}
        .similarity-info {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .stats {{
            font-size: 0.9em;
            color: #666;
        }}
        table.diff {{
            width: 100%;
            border-collapse: collapse;
            font-family: monospace;
            font-size: 14px;
        }}
        .diff_header {{
            background-color: #f8f9fa;
            color: #495057;
            padding: 5px;
            text-align: right;
            width: 40px;
        }}
        .diff_next, .diff_add, .diff_sub, .diff_chg {{
            background-color: #fff3cd !important;
        }}
    </style>
</head>
<body>
    <h1>Text Similarity Report</h1>
    <div id="similarity-container"></div>

    <script>
        const similarityData = {json.dumps(similarity_data)};

        function createTextPanel(textData) {{
            const panel = document.createElement('div');
            panel.className = 'text-panel';

            const heading = document.createElement('h2');
            heading.textContent = textData.filename;
            panel.appendChild(heading);

            textData.similarTexts.forEach((similar, index) => {{
                const similarDiv = document.createElement('div');
                similarDiv.className = 'similar-text';

                const header = document.createElement('div');
                header.className = 'similarity-info';
                header.innerHTML = `
                    <div>
                        <h3 style="margin: 0;">Similar Text #${{index + 1}}: ${{similar.filename}}</h3>
                        <p>Similarity Score: <span class="similarity-score">${{similar.similarity}}%</span></p>
                    </div>
                    <button class="toggle-view" onclick="toggleViewMode(this)">
                        Switch to Diff View
                    </button>
                `;

                const comparisonDiv = document.createElement('div');
                comparisonDiv.className = 'comparison-container';
                comparisonDiv.innerHTML = `
                    <div class="text-column">
                        <div class="text-header">Original Text</div>
                        <div class="text-box">${{similar.originalText}}</div>
                    </div>
                    <div class="text-column">
                        <div class="text-header">Compared Text</div>
                        <div class="text-box">${{similar.comparedText}}</div>
                    </div>
                `;

                const diffView = document.createElement('div');
                diffView.className = 'comparison-container hidden';
                diffView.innerHTML = similar.diffHtml;

                similarDiv.appendChild(header);
                similarDiv.appendChild(comparisonDiv);
                similarDiv.appendChild(diffView);

                panel.appendChild(similarDiv);
            }});

            return panel;
        }}

        function toggleViewMode(button) {{
            const similarDiv = button.closest('.similar-text');
            const comparisonContainer = similarDiv.querySelector('.comparison-container:not(.hidden)');
            const diffView = similarDiv.querySelector('.comparison-container.hidden');

            comparisonContainer.classList.toggle('hidden');
            diffView.classList.toggle('hidden');

            button.textContent = button.textContent.includes('Diff') ?
                'Switch to Side-by-Side View' : 'Switch to Diff View';
        }}

        const container = document.getElementById('similarity-container');
        similarityData.forEach(textData => {{
            container.appendChild(createTextPanel(textData));
        }});
    </script>
</body>
</html>
"""

    report_path = Path(f"{output_prefix}_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report generated: {report_path}")



def generate_side_by_side_diff(text1: str, text2: str) -> str:
    """
    Generate HTML for diff view with improved highlighting of similarities.
    """
    # Use difflib's HtmlDiff with custom styling
    differ = difflib.HtmlDiff(wrapcolumn=80)
    diff_html = differ.make_table(
        text1.splitlines(),
        text2.splitlines(),
        fromdesc="Original Text",
        todesc="Compared Text",
        context=True
    )

    return f'<div style="overflow-x: auto;">{diff_html}</div>'


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate text similarity using n-grams or skip-grams')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--corpus_tsv', type=str,
                          help='Path to the corpus TSV file')
    input_group.add_argument('--corpus_dir', type=str,
                          help='Path to directory containing text files')
    parser.add_argument('--file_suffix', type=str, default='.txt',
                      help='File suffix filter when reading from directory (default: .txt)')
    parser.add_argument('--keep_texts', type=int, default=500,
                      help='Number of texts to keep from corpus')
    parser.add_argument('--mode', type=str, default='ngram', choices=['ngram', 'skip'],
                      help='Mode of operation: ngram or skip')
    parser.add_argument('--ngram', type=int, default=4,
                      help='Size of n-grams')
    parser.add_argument('--output_prefix', type=str, default='similarity',
                      help='Prefix for output files')
    parser.add_argument('--use_stanza', action='store_true',
                      help='Use Stanza for tokenization')
    parser.add_argument('--use_lemmatizer', action='store_true',
                      help='Use Stanza for lemmatization')
    parser.add_argument('--lang', type=str, default='la',
                      help='Language code for Stanza (default: la for Latin)')
    parser.add_argument('--save_rankings', action='store_true',
                      help='Generate detailed similarity rankings')
    parser.add_argument('--top_n', type=int, default=3,
                      help='Number of most similar texts to show in rankings (default: 3)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize text processor based on arguments
    if args.use_stanza or args.use_lemmatizer:
        try:
            processor = StanzaProcessor(args.lang, args.use_lemmatizer)
            process_func = processor.process
            print(f"Using Stanza for {'lemmatization' if args.use_lemmatizer else 'tokenization'}")
            print(f"Language: {args.lang}")
        except Exception as e:
            print(f"Warning: Failed to initialize Stanza ({str(e)}). Falling back to default tokenizer.")
            process_func = default_tokenize
    else:
        process_func = default_tokenize
        print("Using default medieval text tokenizer")

    # Read and process texts
    if args.corpus_tsv:
        corpus, filenames, file_contents = read_texts_from_tsv(args.corpus_tsv)
    else:
        corpus, filenames, file_contents = read_texts_from_directory(args.corpus_dir, args.file_suffix)

    # Apply keep_texts limit
    if args.keep_texts and args.keep_texts < len(corpus):
        corpus = corpus[:args.keep_texts]
        filenames = filenames[:args.keep_texts]
        file_contents = {k: file_contents[k] for k in filenames}

    # Generate encoder and process corpus
    print("Processing texts...")
    processed_corpus = [process_func(text) for text in tqdm(corpus)]

    print("Building encoder...")
    all_tokens = []
    for tokens in processed_corpus:
        all_tokens.extend(tokens)
    encoder = {token: i for i, token in enumerate(sorted(set(all_tokens)))}

    # Calculate distance matrix
    print(f"Calculating {args.mode} distances...")
    dist_mat = np.zeros([len(corpus), len(corpus)])
    dist_func = skip_distance if args.mode == 'skip' else distance

    for t1, text1 in enumerate(tqdm(processed_corpus)):
        for t2, text2 in enumerate(processed_corpus):
            if t2 >= t1:  # Only calculate upper triangle
                dist = dist_func(text1, text2, args.ngram, encoder)
                dist_mat[t1, t2] = dist
                dist_mat[t2, t1] = dist  # Matrix is symmetric

    # Save results
    print("Saving results...")
    output_dir = Path(args.output_prefix).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Save distance matrix
    np.save(f'{args.output_prefix}_matrix.npy', dist_mat)

    # Save filename mapping and additional metadata
    with open(f'{args.output_prefix}_metadata.txt', 'w', encoding='utf-8') as f:
        f.write("# Text Similarity Analysis Metadata\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"N-gram size: {args.ngram}\n")
        f.write(f"Tokenizer: {'Stanza (' + args.lang + ')' if args.use_stanza else 'Default'}\n")
        f.write(f"Lemmatizer: {'Yes' if args.use_lemmatizer else 'No'}\n")
        f.write(f"Number of texts: {len(corpus)}\n")
        f.write(f"Vocabulary size: {len(encoder)}\n")
        f.write("\n# File Index Mapping\n")
        for i, filename in enumerate(filenames):
            f.write(f"{i}\t{filename}\n")

    # Save token list
    with open(f'{args.output_prefix}_vocabulary.txt', 'w', encoding='utf-8') as f:
        f.write("# Vocabulary List\n")
        f.write(f"Total tokens: {len(encoder)}\n\n")
        for token, idx in sorted(encoder.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{token}\n")

    print(f"Results saved with prefix: {args.output_prefix}")
    print(f"Total number of texts processed: {len(corpus)}")
    print(f"Total vocabulary size: {len(encoder)}")

    # Generate text similarity rankings and HTML report
    if args.save_rankings:
        print("Generating similarity rankings...")
        rankings_file = f'{args.output_prefix}_rankings.txt'
        with open(rankings_file, 'w', encoding='utf-8') as f:
            for i, filename in enumerate(filenames):
                # Get similarities for this text
                similarities = dist_mat[i]
                # Sort by similarity (excluding self-similarity)
                most_similar = np.argsort(similarities)[::-1][1:args.top_n + 1]

                f.write(f"\nMost similar texts to {filename}:\n")
                for rank, idx in enumerate(most_similar, 1):
                    similarity = similarities[idx] * 100  # Convert to percentage
                    f.write(f"{rank}. {filenames[idx]} ({similarity:.2f}% similar)\n")

        print(f"Rankings saved to: {rankings_file}")

        # Generate HTML report
        print("Generating HTML similarity report...")
        generate_similarity_report(
            dist_mat=dist_mat,
            filenames=filenames,
            file_contents=file_contents,
            output_prefix=args.output_prefix,
            top_n=args.top_n
        )

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average similarity score: {np.mean(dist_mat):.3f}")
    print(f"Median similarity score: {np.median(dist_mat):.3f}")
    print(f"Maximum similarity score: {np.max(dist_mat[~np.eye(dist_mat.shape[0], dtype=bool)]):.3f}")
    print(f"Minimum similarity score: {np.min(dist_mat):.3f}")

if __name__ == '__main__':
    main()
