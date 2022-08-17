library(devtools)

document()

if (0) {
  #ds = parquet_dir_dataset("parquet", expected_rows = 100 * 60 * 60)
  ds = ParquetDataDirSmall("parquet-small", expected_rows = 100 * 60 * 60)

  x = ds$.getitem(10)

  ds$.length()

  # The spectral tensor adaptor takes a data set, calculates the mod of the
  # fft coefficients, and returns the result as a torch tensor.
  sds = SpectralTensorAdaptor(ds)
  sds$.getitem(1)
} else {
  fns = dir("parquet-subset")
  training_files = file.path("parquet-subset", fns[-1])
  num_train = 10000
  holdout_files = file.path("parquet-subset", fns[1])
  num_holdout = 500
  num_rows = 100 * 60 * 60
  
  library(future)
  plan(mulitcore)
  document()
  ds = ParquetDataDirSample(
    num_rows,
    training_files,
    num_train,
    holdout_files,
    num_holdout
  )
  ds$.length()
  ds$.getitem(10)

  sds = SpectralTensorAdaptor(ds)
  sds$.getitem(1)
}
