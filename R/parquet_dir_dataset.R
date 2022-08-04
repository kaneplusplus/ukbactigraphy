#' @importFrom arrow read_parquet
#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @export
#parquet_dir_dataset = dataset(
ParquetDataDir = dataset(
  name = "ParquetDataDir",
  initialize = function(dirname, expected_rows = NULL, verbose = TRUE) {
    self$dirname = file_path_as_absolute(dirname)
    self$expected_rows = expected_rows
    if (!dir.exists(dirname)) {
      stop(paste("Directory", dirname, "does not exist."))
    }
    self$fns = dir(self$dirname)
    if (verbose) {
      msg = sprintf("%d files found in %s\n", length(self$fns), self$dirname)
      cat(msg)
    }
  },
  .getitem = function(index) {
    if (index > length(self$fns) || index < 1) {
      stop("Index out of bounds")
    }
    pqr = 
      read_parquet(
        file.path(self$dirname, self$fns[index]), 
        as_data_frame = FALSE
      )
    if (!is.null(self$expected_rows)) {
      if (nrow(pqr) < self$expected_rows) {
        warning("Data set has fewer lines than number of expected samples.")
      }
      pqr = pqr[seq_len(self$expected_rows),]
    } 
    return(collect(pqr))
  },
  .length = function() {
    length(self$fns)
  }
)

#' @importFrom torch torch_fft_fft
get_spectrum <- function(x) {
  ret = torch_fft_fft(x, dim = 1, norm = "ortho") 
  ret = torch_sqrt(ret$real^2 + ret$imag^2)
  ret[seq_len(ceiling(ret$shape[1] / 2)),]
}

#' @importFrom torch dataset
#' @importFrom dplyr select
#' @importFrom torch torch_tensor torch_float32
#' @export
SpectralTensorAdaptor = dataset(
  name = "SpectralTensorAdaptor",
  initialize = function(dsg, device = NULL, dtype = torch_float32()) {
    self$dsg = dsg
    self$device = device
    self$dtype = dtype
  },
  .getitem = function(index) {
    self$dsg$.getitem(index) |>
      select(X:Z) |>
      as.matrix() |>
      torch_tensor(dtype = self$dtype, device = self$device) |>
      get_spectrum() 
  },
  .length = function() {
    self$dsg$.length()
  }
)

