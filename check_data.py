from datasets import load_dataset
from config import DATASET_NAME


def main():
    print("1. 데이터셋 불러오기")
    dataset = load_dataset(DATASET_NAME)
    print(dataset)

    print("2. Split별 데이터 수")
    for split in dataset.keys():
        print(f"{split}: {len(dataset[split]):,}개")

    print("3. 컬럼 확인")
    print(dataset["train"].column_names)

    print("4. 첫 번째 데이터 확인")
    sample = dataset["train"][0]

    print(f"날짜: {sample['date']}")
    print(f"카테고리: {sample['category']}")
    print(f"언론사: {sample['press']}")
    print(f"제목: {sample['title']}")

    print("\n[기사 본문]")
    print(sample["document"])

    print("\n[정답 요약]")
    print(sample["summary"])

    print("5. 결측치 확인")
    for split in dataset.keys():
        document_missing = sum(
            x is None or str(x).strip() == ""
            for x in dataset[split]["document"]
        )
        summary_missing = sum(
            x is None or str(x).strip() == ""
            for x in dataset[split]["summary"]
        )

        print(
            f"{split}: "
            f"document 결측={document_missing}, "
            f"summary 결측={summary_missing}"
        )


if __name__ == "__main__":
    main()
