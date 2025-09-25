from datasets import load_dataset
import pandas as pd

cache_path = "./dataset_cache"  

ds = load_dataset(
    "sentence-transformers/codesearchnet",
    split="train[:1%]",
    cache_dir=cache_path
)

df = ds.to_pandas()
print(df.head())
