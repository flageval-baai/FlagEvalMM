import datasets

dataset = datasets.load_dataset("/share/project/benchmarks/Ref-L4/", split="test")

print(dataset[0])

