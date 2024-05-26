from bert_headlines_data import *
from utils import load_json_dataset

# maybe to be converted to a proper unit test
def test_headlines_sarcasm_dataset():
    data = pd.DataFrame(
        {
            "headline": ["This is a headline", "This is another headline"],
            "is_sarcastic": [1, 0],
        }
    )
    dataset = HeadlinesSarcasmDataset(data)

    assert len(dataset) == 2

    sample = dataset[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
    assert sample["input_ids"].shape == torch.Size([256])
    assert sample["attention_mask"].shape == torch.Size([256])
    assert sample["labels"].shape == torch.Size([])


def dummy_run_headlines_sarcasm_data_module():
    train_data = pd.DataFrame(
        {
            "headline": ["This is a train headline", "This is another train headline"],
            "is_sarcastic": [1, 0],
        }
    )
    val_data = pd.DataFrame(
        {
            "headline": ["This is a val headline", "This is another val headline"],
            "is_sarcastic": [1, 0],
        }
    )
    test_data = pd.DataFrame(
        {
            "headline": ["This is a test headline", "This is another test headline"],
            "is_sarcastic": [1, 0],
        }
    )
    dm = HeadlinesSarcasmDataModule(train_data, val_data, test_data)
    dm.setup()
    next(iter(dm.train_dataloader()))
    next(iter(dm.val_dataloader()))
    next(iter(dm.test_dataloader()))
    # print data
    print(dm.train_data)


def print_headlines_sarcasm_data_module():
    train, val, test = load_json_dataset()
    data_module = HeadlinesSarcasmDataModule(train, val, test)

    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print("Sample Input IDs:", batch["input_ids"].shape)
        print(batch["labels"])
        break


dummy_run_headlines_sarcasm_data_module()
test_headlines_sarcasm_dataset()
print_headlines_sarcasm_data_module()
