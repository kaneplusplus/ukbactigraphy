library(devtools)
library(future)
library(R6)

plan(multicore)

document()

fns = dir("parquet-subset")
training_files = file.path("parquet-subset", fns[-1])
num_train = 10000
holdout_files = file.path("parquet-subset", fns[1])
num_holdout = 500
num_rows = 100 * 60 * 60

sds = ParquetDataDirSample(num_rows, training_files, num_train) |>
  SpectralTensorAdaptor()
sds = SpectralTensorAdaptor(pdd)

ss_model = SelfSupervisedSpectral()
tm = sds |>
  dataloader() |>
  train_ssm(model = ss_model)

