import logging
import threading


LOGGER = logging.getLogger(__name__)


def _train_model(*args, **kwargs):
    from train import train_model

    return train_model(*args, **kwargs)


def _generate_text(*args, **kwargs):
    from generate import generate_text

    return generate_text(*args, **kwargs)


class MiniLLMService:
    def __init__(self, settings, state):
        self.settings = settings
        self.state = state
        self._training_lock = threading.Lock()

    def start_training(self, text, epochs):
        with self._training_lock:
            if self.state.snapshot()["status"] == "running":
                raise RuntimeError("Training is already running.")

            self.state.mark_running()
            worker = threading.Thread(
                target=self._run_training,
                args=(text, epochs),
                daemon=True,
            )
            worker.start()

    def _run_training(self, text, epochs):
        def on_progress(message, loss):
            self.state.update(
                status="running",
                message=message,
                last_loss=loss,
                last_error=None,
            )

        try:
            result = _train_model(
                text=text,
                epochs=epochs,
                batch_size=self.settings.default_batch_size,
                block_size=self.settings.default_block_size,
                learning_rate=self.settings.learning_rate,
                data_path=self.settings.data_path,
                output_dir=self.settings.artifacts_dir,
                model_path=self.settings.model_path,
                progress_callback=on_progress,
                embed_dim=self.settings.model_embed_dim,
                num_heads=self.settings.model_num_heads,
                num_layers=self.settings.model_num_layers,
                dropout=self.settings.model_dropout,
                weight_decay=self.settings.weight_decay,
                gradient_clip=self.settings.gradient_clip,
                validation_split=self.settings.validation_split,
                tokenizer_vocab_size=self.settings.tokenizer_vocab_size,
            )
            self.state.mark_ready(result["last_loss"])
            LOGGER.info("Training completed with loss %.4f", result["last_loss"])
        except Exception as exc:
            LOGGER.exception("Training failed")
            self.state.mark_error(str(exc))

    def generate(self, prompt, length):
        if self.state.snapshot()["status"] == "running":
            raise RuntimeError("Generation is unavailable while training is running.")

        return _generate_text(
            prompt=prompt,
            length=length,
            model_path=self.settings.model_path,
            vocab_path=self.settings.vocab_path,
            merges_path=self.settings.merges_path,
            block_size=self.settings.default_block_size,
            temperature=self.settings.generation_temperature,
            top_k=self.settings.generation_top_k,
            top_p=self.settings.generation_top_p,
            repetition_penalty=self.settings.repetition_penalty,
        )
