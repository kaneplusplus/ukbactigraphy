library(devtools)
library(future)

#plan(multicore)

document()

fns = dir("parquet-subset")
training_files = file.path("parquet-subset", fns[-1])
num_train = 10000
holdout_files = file.path("parquet-subset", fns[1])
num_rows = 100 * 60 * 60

sds = ParquetDataDirSample(num_rows, training_files, num_train) |>
  SpectralTensorAdaptor()

tm = sds |>
  dataloader(batch_size = 15, shuffle = FALSE)

it = tm$.iter()
s = it$.next()
ss_model = SelfSupervisedSpectral()
ss_model$forward(s)


#tm = sds |>
#  dataloader(batch_size = 15, shuffle = FALSE) |>
#  train_ssm(model = ss_model)

