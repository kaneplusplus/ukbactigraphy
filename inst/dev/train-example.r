library(devtools)
library(torch)
library(luz)

document()

fns = dir("parquet-subset")
training_files = file.path("parquet-subset", fns[-1])
num_train = 10000
testing_files = file.path("parquet-subset", fns[1])
num_rows = 100 * 60 * 60

train_pdd3w = 
  ParquetDataDirThreeWindowSample(num_rows, training_files, num_train)

test_dl = train_pdd3w |>
  SpectralTensorAdaptor() |>
  ThreeWindowSelfSupervisedDataSet() |>
  dataloader(batch_size = 1)

train_pddws = ParquetDataDirWindowSample(train_pdd3w)

test_pdd3w = 
  ParquetDataDirThreeWindowSample(num_rows, testing_files, num_train)

test_dl = test_pdd3w |>
  SpectralTensorAdaptor() |>
  ThreeWindowSelfSupervisedDataSet() |>
  dataloader()

# May want to switch another loss.

ss_model = SelfSupervisedSpectral() |>
  setup(loss = nnf_mse_loss, optimizer = optim_adam) |>
  fit(train_dl, epochs = 1, valid_data = test_dl)

