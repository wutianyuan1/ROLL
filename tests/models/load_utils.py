from torch.utils.data import DistributedSampler, DataLoader
from transformers import DataCollatorWithPadding

from roll.configs import ModelArguments, DataArguments
from roll.configs.training_args import TrainingArguments
from roll.datasets.loader import get_dataset
from roll.models.model_providers import default_tokenizer_provider


def get_mock_dataloader(model_args: ModelArguments, data_args: DataArguments, batch_size: int = 4):

    tokenizer = default_tokenizer_provider(model_args=model_args)

    dataset = get_dataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader, tokenizer
