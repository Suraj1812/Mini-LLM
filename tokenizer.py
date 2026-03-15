from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


def train_tokenizer(files, output_dir, vocab_size=8000):
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
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
