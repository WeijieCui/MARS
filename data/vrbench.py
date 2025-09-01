from datasets import load_dataset

fw = load_dataset("xiang709/VRSBench", streaming=True)
for sample in fw:
    print(sample)
