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
    MODEL_NAME,
    DATASET_NAME,
    OUTPUT_DIR,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    EPOCHS,
    SEED,
)
from utils import set_seed, get_device, ensure_dir


def select_subset(dataset, n):
    if n is None:
        return dataset
    return dataset.select(range(min(n, len(dataset))))


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

    for batch in tqdm(dataloader, desc="Validation", leave=False):
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
    ensure_dir(OUTPUT_DIR)

    device = get_device()
    print(f"사용 Device: {device}")
    print(f"사용 Model: {MODEL_NAME}")

    print("\n[1] 데이터셋 불러오기")
    dataset = load_dataset(DATASET_NAME)

    train_dataset = select_subset(dataset["train"], TRAIN_SAMPLES)
    val_dataset = select_subset(dataset["validation"], VAL_SAMPLES)

    print(f"Train: {len(train_dataset):,}개")
    print(f"Validation: {len(val_dataset):,}개")

    print("\n[2] Tokenizer / Model 불러오기")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    print("\n[3] Tokenization")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Train tokenization",
    )

    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Validation tokenization",
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

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print("\n[4] Fine-tuning 시작")
    print(
        f"Epoch={EPOCHS}, Batch={BATCH_SIZE}, "
        f"Gradient Accumulation={GRADIENT_ACCUMULATION_STEPS}, "
        f"LR={LEARNING_RATE}"
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        total_train_loss = 0.0
        train_steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for step, batch in enumerate(progress, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()

            if (
                step % GRADIENT_ACCUMULATION_STEPS == 0
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

        train_loss = total_train_loss / max(train_steps, 1)
        val_loss = evaluate_loss(model, val_loader, device)

        print(
            f"\nEpoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            print(
                f"→ Best Model 저장 "
                f"(Validation Loss: {best_val_loss:.4f})"
            )

    print("\n학습 완료")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
