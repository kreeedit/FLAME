# FLAME - Formulaic Language Analysis in Medieval Expressions

FLAME is a simple Python script for detecting formulaic languages in medieval manuscripts using improved n-gram and skip-gram based approaches.

<img src="flame-little-flame.gif" width="200" />

## Features
- N-gram and skip-gram based text similarity analysis
- Visualization of similarity matrices
- Statistical analysis of text relationships
- Configurable processing parameters
- Detailed similarity pair analysis

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- fargv

## Usage

```bash
python flame.py --corpus_tsv input.tsv --mode ngram --ngram 4
```

### Key Parameters
- `--corpus_tsv`: Input TSV file containing texts (default: 'corpus.tsv')
- `--mode`: Analysis mode ['ngram', 'skip'] (default: 'ngram')
- `--ngram`: N-gram size (default: 4)
- `--keep_texts`: Number of texts to process (default: 100)
- `--min_text_length`: Minimum text length to analyze (default: 50)
- `--visualize`: Enable visualization (default: False)

## Output
- Distance matrix (numpy array)
- Similarity heatmap visualization
- Most similar/dissimilar text pairs
- Statistical analysis of text relationships

## File Format
Input TSV file should have text content in the first column. Additional columns are ignored.

## Example
```bash
python flame.py --corpus_tsv medieval_texts.tsv --keep_texts 50 --visualize --plot_output heatmap.png
```

## License
MIT License
