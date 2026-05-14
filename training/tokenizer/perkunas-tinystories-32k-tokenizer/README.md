# Perkunas Tokenizer

Tokenizer type: byte-level BPE.

Rationale:

- Byte-level coverage avoids hard OOV failures for multilingual text and code.
- BPE keeps the pipeline compatible with Hugging Face fast tokenizer artifacts.
- Special tokens are fixed at the start of the vocabulary for stable training.

## Configuration

- Vocab size target: `32000`
- Min frequency: `2`
- Special tokens: `<pad>, <s>, </s>, <unk>, <mask>`

## Evaluation

- Actual vocab size: `32000`
- Average chars/token: `4.1354`
- Fertility tokens/word: `1.2256`
- Unknown token rate: `0.00000000`
- Single-char token rate: `0.1917`
