# First argument is the cwa file.
# second argument is the parquet file.
library(read.cwa, verbose = FALSE) |>
  suppressMessages() |>
  suppressWarnings()

library(arrow, verbose = FALSE) |>
  suppressMessages() |>
  suppressWarnings()

args = commandArgs(trailingOnly = TRUE)

if (!file.exists(args[1])) {
  stop("Couldn't find the input file.")
}

user = basename(args[1]) |> 
  strsplit("_") |>
  unlist()

info = basename(args[1]) |>
  strsplit("\\.") |>
  unlist()

x = read_cwa(args[1])$data
x$user = user[1]
x$into = info[1]

write_parquet(x, args[2])
