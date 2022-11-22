library(devtools)
library(torch)
library(luz)

document()
device = "cpu" # "mps"
Sys.setenv(PYTORCH_ENABLE_MPS_FALLBACK = 1)

fns = dir("parquet-subset")
training_files = file.path("parquet-subset", fns[-1])
num_train = 10000
testing_files = file.path("parquet-subset", fns[1])
num_rows = 100 * 60 * 60

tf = DataFileSample(num_rows, training_files, num_train)
tf$.getitem(1)
tf$getitem(1)

train_pdf3w = 
  DataFileThreeWindowSample(num_rows, training_files, num_train)

train_dl = train_pdf3w |>
  SpectralTensorAdaptor(device = device) |>
  ThreeWindowSelfSupervisedDataSet() |>
  dataloader(batch_size = 1)

test_pdf3w = 
  DataFileThreeWindowSample(num_rows, testing_files, num_train)

train_ta = train_pdf3w |>
  SpectralTensorAdaptor(device = device) |>
  ThreeWindowSelfSupervisedDataSet()
train_ta$.getitem(1)

test_dl = test_pdf3w |>
  SpectralTensorAdaptor(device = device) |>
  ThreeWindowSelfSupervisedDataSet() |>
  dataloader()

# May want to switch another loss.

ss_model = SelfSupervisedSpectral() |>
  setup(loss = nnf_mse_loss, optimizer = optim_adam) |>
  fit(train_dl, epochs = 1, valid_data = test_dl)

# Single window.
sw_train = train_pdf3w |>
  SpectralTensorAdaptor()
