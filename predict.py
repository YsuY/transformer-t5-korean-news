import torch
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
)

from config import (
    OUTPUT_DIR,
    MAX_INPUT_LENGTH,
    NUM_BEAMS,
    MIN_NEW_TOKENS,
    MAX_NEW_TOKENS,
)
from utils import get_device


def summarize(text, tokenizer, model, device):
    input_text = "뉴스 요약: " + text

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

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )


def main():
    device = get_device()
    print(f"사용 Device: {device}")

    tokenizer = T5TokenizerFast.from_pretrained(OUTPUT_DIR)
    model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR)

    model.to(device)
    model.eval()

    print("\n한국어 뉴스 요약 테스트")
    print("여러 줄 입력 후 마지막에 빈 줄을 한 번 입력하세요.\n")

    lines = []

    while True:
        line = input()

        if line.strip() == "":
            break

        lines.append(line)

    article = " ".join(lines).strip()

    if not article:
        print("입력된 기사가 없습니다.")
        return

    summary = summarize(
        article,
        tokenizer,
        model,
        device,
    )

    print("\n[입력 기사]")
    print(article)

    print("\n[모델 요약]")
    print(summary)


if __name__ == "__main__":
    main()
