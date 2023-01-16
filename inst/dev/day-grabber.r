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

x = open_dataset(file.path("parquet-subset", dir("parquet-subset")[1:2]))

device = "mps"
pss = DayHourSpectralSignature(
  open_dataset(file.path("parquet-subset", dir("parquet-subset")[4])),
  device = device
)

pss$.getitem(1) |> system.time()

device = "mps"
psc = DayHourSpectralSignatureDataSet(
  open_dataset("parquet-subset"), 
  device = device
)

da = get_day(psc$dsg, 1) |>
  select(time, X:Z) |>
  mutate(hour = hour(time)) |>
  group_by(hour) |>
  group_nest()
da = da$data[[12]]
da = da[seq(1, nrow(da), length.out = 10000),]
da = rename(da, Time = time, Value = value)

dss = psc[1][12,,]$to("cpu", dtype = torch_float32()) |>
  as.matrix() |>
  t()
ds = as_tibble(dss, .name_repair = "minimal")
colnames(ds) = c("X", "Y", "Z")
ds = ds[round(seq(1, nrow(ds), length.out = 10000)),]
ds$Frequency = seq(0, 33.4, length.out = nrow(ds))

library(patchwork)
library(tidyr)
library(ggplot2)
p1 = pivot_longer(da, -Time) |>
  rename(Channel = name, Value = value) |>
  ggplot(aes(x = Time, y = Value)) +
    geom_line() +
    facet_grid(Channel ~ .) +
    theme_bw()

ggsave("accel-example.png", p1, width = 7.5, height = 3)

p2 = pivot_longer(ds, -Frequency) |>
  rename(Channel = name, Value = value) |>
  ggplot(aes(x = Frequency, y = Value)) +
    scale_y_log10() + 
    geom_line() +
    facet_grid(Channel ~ .) +
    theme_bw()

ggsave("spec-sig-example.png", p2, height = 6.73, width = 6.85)



dss = as_tibble(



SpectralSignatureReducer()()(pss$.getitem(1))

future_map_dbl(1:pss$.length(), ~ nrow(pss$.getitem(.x))) |> min()

p = SpectralSignatureTensor(pss)
p$.getitem(1)
