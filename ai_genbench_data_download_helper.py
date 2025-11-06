from datasets import load_dataset, Dataset

elsa_data = load_dataset("elsaEU/ELSA_D3", split="train", streaming=False)

#print(elsa_data)

# sampled = elsa_data.shuffle(seed=6050).take(10)

# train_subset = Dataset.from_iterable(sampled)
elsa_data.save_to_disk("testing_data/train")