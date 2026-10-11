import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
)
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DATASET_NAME,
    OUTPUT_DIR,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    SEED,
)

from utils import set_seed, get_device


ADDITIONAL_EPOCHS = 2


def select_subset(dataset, n):
    if n is None:
        return dataset

    return dataset.select(
        range(min(n, len(dataset)))
    )


def preprocess_function(examples, tokenizer):

    inputs = [
        "뉴스 요약: " + document
        for document in examples["document"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


@torch.no_grad()
def evaluate_loss(model, dataloader, device):

    model.eval()

    total_loss = 0.0
    steps = 0

    for batch in tqdm(
        dataloader,
        desc="Validation",
        leave=False
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        total_loss += outputs.loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def main():

    set_seed(SEED)

    device = get_device()

    print(f"사용 Device: {device}")

    print("\n[1] Epoch 3 Best Model 불러오기")

    tokenizer = T5TokenizerFast.from_pretrained(
        OUTPUT_DIR
    )

    model = T5ForConditionalGeneration.from_pretrained(
        OUTPUT_DIR
    )

    model.to(device)

    print("\n[2] 데이터셋 불러오기")

    dataset = load_dataset(DATASET_NAME)

    train_dataset = select_subset(
        dataset["train"],
        TRAIN_SAMPLES,
    )

    val_dataset = select_subset(
        dataset["validation"],
        VAL_SAMPLES,
    )

    print(
        f"Train: {len(train_dataset):,}개"
    )

    print(
        f"Validation: {len(val_dataset):,}개"
    )

    print("\n[3] Tokenization")

    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    train_loader = DataLoader(
        train_tokenized,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    val_loader = DataLoader(
        val_tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\n[4] 현재 Epoch 3 모델 Validation 확인")

    best_val_loss = evaluate_loss(
        model,
        val_loader,
        device,
    )

    print(
        f"현재 Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print("\n[5] Epoch 4~5 추가 학습")

    for extra_epoch in range(
        ADDITIONAL_EPOCHS
    ):

        epoch_number = 4 + extra_epoch

        model.train()

        optimizer.zero_grad()

        total_train_loss = 0.0
        train_steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch_number}/5",
        )

        for step, batch in enumerate(
            progress,
            start=1
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = (
                outputs.loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            loss.backward()

            if (
                step
                % GRADIENT_ACCUMULATION_STEPS
                == 0
                or step == len(train_loader)
            ):

                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += (
                loss.item()
                * GRADIENT_ACCUMULATION_STEPS
            )

            train_steps += 1

            progress.set_postfix(
                loss=f"{total_train_loss / train_steps:.4f}"
            )

        train_loss = (
            total_train_loss
            / max(train_steps, 1)
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch {epoch_number} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            model.save_pretrained(
                OUTPUT_DIR
            )

            tokenizer.save_pretrained(
                OUTPUT_DIR
            )

            print(
                f"→ Best Model 저장 "
                f"(Validation Loss: "
                f"{best_val_loss:.4f})"
            )

    print("\n추가 학습 완료")

    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()