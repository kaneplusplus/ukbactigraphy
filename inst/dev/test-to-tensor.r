library(tibble)
library(devtools)


x = readRDS("outcomes.rds")

document()

mt = x |>
  model_tensor(index = "eid") |>
  to_tensor()

mt2 = x |>
  model_tensor(index = "eid") |>
  to_tensor(x$eid[1:10])
