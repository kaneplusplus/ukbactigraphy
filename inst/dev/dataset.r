library(devtools)

document()

#ds = parquet_dir_dataset("parquet", expected_rows = 100 * 60 * 60)
ds = ParquetDataDir("parquet", expected_rows = 100 * 60 * 60)

x = ds$.getitem(10)

ds$.length()

# The spectral tensor adaptor takes a data set, calculates the mod of the
# fft coefficients, and returns the result as a torch tensor.
sds = SpectralTensorAdaptor(ds)
sds$.getitem(1)
