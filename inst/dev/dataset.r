library(devtools)

document()

#sdf = parquet_3f_spectral_dataset("parquet")
ds = parquet_dir_dataset("parquet", expected_rows = 100 * 60 * 60)

x = ds$.getitem(10)
