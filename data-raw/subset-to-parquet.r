library(arrow)
library(tibble)
library(read.cwa)
library(dplyr)
library(purrr)

x = tibble(
  fn = dir("cwa-subset"),
  ofn = gsub(".cwa.bz2", ".parquet", fn),
  user = map_chr(
    fn,
    ~ (gsub(".cwa.bz2", "", .x) |> strsplit("_") |> unlist())[1]
  ),
  info = gsub(".cwa.bz2", "", fn)
)

for (i in seq_len(nrow(x))) {
  read_cwa(file.path("cwa-subset", x$fn[i])) |>
    (\(x) x$data)() |>
    mutate(user = x$user[i], info = x$info[i]) |>
    write_parquet(file.path("parquet-subset", x$ofn[i]))
  
}

