library(devtools)
library(dplyr)

document()

iris |>
  mutate(index = seq_len(nrow(iris))) |>
  model_tensor("index") |>
  to_tibble()

iris |>
  model_tensor() |>
  to_tibble()
