library(arrow)
library(tibble)

x = tibble(
  fn = dir("cwa-subset"),
  ofn = gsub(".cwa.bz2", ".parquet", fn)
)

for (i in seq_along(nrow(x))) {
  read_cwa(file.path("cwa-subset", x$fn[i])) |>
    write_parquet(file.path("parquet-subset", x$ofn[i]))
  
}

