library(tibble)
library(devtools)
library(torch)

document()

fns = dir("parquet-subset")
training_files = file.path("parquet-subset", fns[-1])
num_train = 10000
testing_files = file.path("parquet-subset", fns[1])
num_rows = 100 * 60 * 60

y = readRDS("outcomes.rds") |>
  model_tensor(index = "eid")

y = readRDS("outcomes.rds") |>
  model_tensor(index = "eid", contrasts.arg = list(snoring = "contr.sum"))

y = readRDS("outcomes.rds") |>
  model_tensor(index = "eid", contrasts.arg = list(snoring = contr.sum))

x = ParquetDataFileSample(num_rows, training_files, num_train) |>
  SpectralTensorAdaptor() 

ds = SummaryOutcomeActgraphyDataSet(y, x)

dl = dataloader(ds)
