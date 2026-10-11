import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
)
import evaluate
from tqdm import tqdm

from config import (
    DATASET_NAME,
    OUTPUT_DIR,
    RESULT_DIR,
    TEST_SAMPLES,
    MAX_INPUT_LENGTH,
    NUM_BEAMS,
    MIN_NEW_TOKENS,
    MAX_NEW_TOKENS,
)
from utils import get_device, ensure_dir


def main():
    ensure_dir(RESULT_DIR)

    device = get_device()
    print(f"사용 Device: {device}")

    print("\n[1] Best Model 불러오기")
    tokenizer = T5TokenizerFast.from_pretrained(OUTPUT_DIR)
    model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR)
    model.to(device)
    model.eval()

    print("\n[2] Test 데이터 불러오기")
    dataset = load_dataset(DATASET_NAME)

    test_dataset = dataset["test"]

    if TEST_SAMPLES is not None:
        test_dataset = test_dataset.select(
            range(min(TEST_SAMPLES, len(test_dataset)))
        )

    print(f"Test Sample: {len(test_dataset):,}개")

    rouge = evaluate.load("rouge")

    predictions = []
    references = []
    rows = []

    print("\n[3] 요약 생성")

    for sample in tqdm(test_dataset):
        article = sample["document"]
        reference = sample["summary"]

        input_text = "뉴스 요약: " + article

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                min_new_tokens=MIN_NEW_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        prediction = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        predictions.append(prediction)
        references.append(reference)

        rows.append({
            "title": sample["title"],
            "article": article,
            "reference_summary": reference,
            "predicted_summary": prediction,
        })

    print("\n[4] ROUGE 평가")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,
    )

    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")
    print(f"ROUGE-Lsum: {scores['rougeLsum']:.4f}")

    result_path = os.path.join(RESULT_DIR, "test_predictions.csv")

    pd.DataFrame(rows).to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    score_path = os.path.join(RESULT_DIR, "rouge_scores.txt")

    with open(score_path, "w", encoding="utf-8") as f:
        f.write(f"ROUGE-1: {scores['rouge1']:.4f}\n")
        f.write(f"ROUGE-2: {scores['rouge2']:.4f}\n")
        f.write(f"ROUGE-L: {scores['rougeL']:.4f}\n")
        f.write(f"ROUGE-Lsum: {scores['rougeLsum']:.4f}\n")

    print(f"\n예측 결과 저장: {result_path}")
    print(f"ROUGE 결과 저장: {score_path}")

    print("\n[5] 예측 예시 3개")

    for i in range(min(3, len(rows))):
        row = rows[i]
        print("\n" + "=" * 70)
        print(f"[제목]\n{row['title']}")
        print(f"\n[정답 요약]\n{row['reference_summary']}")
        print(f"\n[모델 요약]\n{row['predicted_summary']}")


if __name__ == "__main__":
    main()
