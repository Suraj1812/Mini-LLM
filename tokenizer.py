from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


def train_tokenizer(files, output_dir, vocab_size=8000):
    total_chars = sum(len(Path(path).read_text(encoding="utf-8")) for path in files)
    resolved_vocab_size = min(vocab_size, max(256, min(2048, total_chars // 2 or 256)))

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=files,
        vocab_size=resolved_vocab_size,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
        ],
    )

    tokenizer.save_model(str(Path(output_dir)))

    return tokenizer
