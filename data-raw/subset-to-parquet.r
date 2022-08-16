library(arrow)
library(tibble)
library(read.cwa)

x = tibble(
  fn = dir("cwa-subset"),
  ofn = gsub(".cwa.bz2", ".parquet", fn)
)

for (i in seq_len(nrow(x))) {
  read_cwa(file.path("cwa-subset", x$fn[i])) |>
    (\(x) x$data)() |>
    write_parquet(file.path("parquet-subset", x$ofn[i]))
  
}

