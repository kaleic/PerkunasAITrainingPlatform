# Perkunas Tokenizer

Tokenizer type: byte-level BPE.

Rationale:

- Byte-level coverage avoids hard OOV failures for multilingual text and code.
- BPE keeps the pipeline compatible with Hugging Face fast tokenizer artifacts.
- Special tokens are fixed at the start of the vocabulary for stable training.

## Configuration

- Vocab size target: `8000`
- Min frequency: `2`
- Special tokens: `<pad>, <s>, </s>, <unk>, <mask>`

## Evaluation

- Actual vocab size: `8000`
- Average chars/token: `3.3736`
- Fertility tokens/word: `1.7249`
- Unknown token rate: `0.00000000`
- Single-char token rate: `0.2895`
