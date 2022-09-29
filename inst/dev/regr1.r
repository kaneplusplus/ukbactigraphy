library(tibble)
library(devtools)

document()

y = readRDS("outcomes.rds") |>
  model_tensor(index = "eid")

x = ParquetDataFileSample(num_rows, training_files, num_train) |>
  SpectralTensorAdaptor() 

ds = SummaryOutcomeActigraphyDataSet(y, x)

dl = dataloader(ds)
