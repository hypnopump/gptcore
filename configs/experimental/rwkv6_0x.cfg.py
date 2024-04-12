import torch
import torch.utils.data
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
import lightning.pytorch.strategies
import transformers
import dataset
import model.hparams
import cli
import dataset.tokenizer
import lit

import model.core
import model.rwkv
import model.experimental.rwkv6_0

BATCH_SIZE = 8
VOCAB_SIZE = 50304
TOKENIZER_FACTORY = lambda: transformers.AutoTokenizer.from_pretrained('gpt2')
MAX_SEQUENCE_LENGTH = 1024

LOG_PROJECT = 'gptcore_memtention'
LOG_NAME = 'RWKV6.0x L8D512H8CM3Adam'

cli.Config(
    seed_everything = 1337,
    compile = False,
    pretest = False,

    model_factory = lambda: model.core.Decoder(
        hparams = model.hparams.HParams(
            vocab_size = VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,

            n_layer=8,
            n_head=8,
            d_model=512,

            feedforward_d_model_ratio=3,

            d_qk_ratio=1,
            d_v_ratio=1,

            n_kv_head_ratio=1,
        ),
        layer_factory=lambda: model.core.TransformerLayer(
            self_attention_sublayer_factory = lambda: model.experimental.rwkv6_0.RWKV6_0_AttentionSubLayer(),
            #feedforward_sublayer_factory = lambda: model.core.RWKVFeedForwardSubLayer(),
            # feedforward_sublayer_factory = lambda: model.rwkv.RWKV_ChannelMixSubLayer(ternary=True),
            feedforward_sublayer_factory = lambda: model.experimental.rwkv6_0.RWKV_ChannelMixSubLayer(),
        ),
    ),

    trainer_factory = lambda: lit.CoreLightningTrainer(
        optimizer_factory = lambda params: torch.optim.Adam(
            params=params,
            lr=6e-4,
            betas=(0.9,0.999),
        ),
        lightning_trainer_factory = lambda: lightning.Trainer(
            enable_progress_bar=False,
            #enable_checkpointing=False,
            max_epochs=-1,
            val_check_interval=1024,
            precision = 'bf16-mixed',
            accumulate_grad_batches=1,
            gradient_clip_val=0.5,
            log_every_n_steps=20,
            logger = [
                #lightning.pytorch.loggers.CSVLogger(save_dir="."),
                lightning.pytorch.loggers.WandbLogger(project=LOG_PROJECT, name=LOG_NAME),
            ],
            #devices=1,
            #strategy='ddp',
            #strategy="deepspeed_stage_2",
            #strategy='ddp_find_unused_parameters_true',
        ),
        datamodule_factory=lambda: dataset.DM(
            dataset_path='dataset/pile.py', 
            tokenizer_factory=TOKENIZER_FACTORY, 
            batch_size=BATCH_SIZE, 
            sequence_length=MAX_SEQUENCE_LENGTH, 
            num_workers=4,
            seed=32,
        ),
    ),
)

