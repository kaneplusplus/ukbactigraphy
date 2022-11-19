library(devtools)
library(torch)
library(arrow)
library(dplyr)
library(lubridate)
library(future)
library(furrr)
plan(multicore)
document()

Sys.setenv(PYTORCH_ENABLE_MPS_FALLBACK = 1)
options(tz="UTC")

x = read_parquet(
    file.path("parquet-subset", dir("parquet-subset")[1]),
    as_data_frame = FALSE
  )

device = "cpu"
pss = ParquetDayHourSpectralSignature(
  file.path("parquet-subset", dir("parquet-subset")[4]),
  device = device
)

pss$.getitem(1) |> system.time()

SpectralSignatureReducer()()(pss$.getitem(1))

future_map_dbl(1:pss$.length(), ~ nrow(pss$.getitem(.x))) |> min()

p = SpectralSignatureTensor(pss)
p$.getitem(1)
