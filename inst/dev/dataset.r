library(devtools)

document()

ds = parquet_dir_dataset("parquet", expected_rows = 100 * 60 * 60)

x = ds$.getitem(10)

ds$.length()
