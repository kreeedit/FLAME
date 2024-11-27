# FLAME - Formulaic Language Analysis in Medieval Expressions

FLAME is a Python-based tool designed for analyzing textual similarities in (medieval) manuscripts and documents. It specializes in detecting formulaic language patterns using modified n-gram and skip-gram techniques, with support for medieval text preprocessing and normalization.

<img src="flame-little-flame.gif" width="200" />

## Features

- Advanced medieval text preprocessing and normalization
- Support for n-gram and skip-gram analysis
- Multiple input formats (directory of text files or TSV)
- Integration with Stanza NLP for advanced linguistic processing
- Specialized handling of medieval abbreviations and characters
- Interactive HTML report generation with text comparison views
- Detailed similarity rankings and statistics
- Support for multiple languages through Stanza integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kreeedit/flame.git
cd flame
```

2. Install required dependencies:
```bash
pip install numpy tqdm pathlib
```

3. Optional: Install Stanza for advanced linguistic processing:
```bash
pip install stanza
```

## Usage

Basic usage with a directory of text files:
```bash
python flame.py --corpus_dir path/to/texts --file_suffix .txt
```

Using TSV input:
```bash
python flame.py --corpus_tsv path/to/corpus.tsv
```

### Command Line Arguments

- `--corpus_dir`: Directory containing text files
- `--corpus_tsv`: Path to TSV file containing texts
- `--file_suffix`: File suffix filter (default: .txt)
- `--keep_texts`: Number of texts to process (default: 500)
- `--mode`: Analysis mode ('ngram' or 'skip')
- `--ngram`: Size of n-grams (default: 4)
- `--output_prefix`: Prefix for output files
- `--use_stanza`: Enable Stanza NLP processing
- `--use_lemmatizer`: Enable Stanza lemmatization
- `--lang`: Language code for Stanza (default: 'la' for Latin)
- `--save_rankings`: Generate detailed similarity rankings
- `--top_n`: Number of most similar texts to show (default: 3)

### Example

```bash
python flame.py --corpus_dir ./medieval_texts --mode ngram --ngram 4 --output_prefix results/similarity --use_stanza --lang la --save_rankings
```

## Output Files

FLAME generates several output files:

1. `*_matrix.npy`: NumPy array containing the similarity matrix
2. `*_metadata.txt`: Analysis metadata and file mapping
3. `*_vocabulary.txt`: List of unique tokens found in the corpus
4. `*_rankings.txt`: Detailed similarity rankings (if enabled)
5. `*_report.html`: Interactive HTML report with text comparisons

## Medieval Text Processing

FLAME includes specialized processing for medieval texts:

- Handling of medieval abbreviations and special characters
- Unicode normalization
- Smart handling of line breaks and hyphenation
- Processing of common medieval punctuation marks
- Support for Latin abbreviations and ligatures

## HTML Report Features

<img src="html_comparsion.png" width="1200" />

The generated HTML report includes:

- Side-by-side text comparison view
- Highlighted similar passages
- Interactive diff view
- Similarity scores and statistics
- Mobile-responsive design

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0
