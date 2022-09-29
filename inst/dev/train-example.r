library(devtools)
library(torch)
library(luz)

document()

fns = dir("parquet-subset")
training_files = file.path("parquet-subset", fns[-1])
num_train = 10000
testing_files = file.path("parquet-subset", fns[1])
num_rows = 100 * 60 * 60

tf = ParquetDataFileSample(num_rows, training_files, num_train)
tf$.getitem(1)

train_pdf3w = 
  ParquetDataFileThreeWindowSample(num_rows, training_files, num_train)

train_dl = train_pdf3w |>
  SpectralTensorAdaptor() |>
  ThreeWindowSelfSupervisedDataSet() |>
  dataloader(batch_size = 1)

test_pdd3w = 
  ParquetDataFileThreeWindowSample(num_rows, testing_files, num_train)

test_dl = test_pdf3w |>
  SpectralTensorAdaptor() |>
  ThreeWindowSelfSupervisedDataSet() |>
  dataloader()

# May want to switch another loss.

ss_model = SelfSupervisedSpectral() |>
  setup(loss = nnf_mse_loss, optimizer = optim_adam) |>
  fit(train_dl, epochs = 1, valid_data = test_dl)

# Single window.
sw_train = train_pdf3w |>
  SpectralTensorAdaptor()
