#' @importFrom arrow read_parquet
#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @export
#parquet_dir_dataset = dataset(
ParquetDataDirSmall = dataset(
  name = "ParquetDataDirSmall",
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


#' @importFrom arrow read_parquet
#' @importFrom purrr map_dbl
#' @importFrom dplyr summarize n
#' @export
sample_file_name_and_offset = function(sfc, num_samples) {
  ret = sfc[sample.int(nrow(sfc), num_samples, replace = TRUE),]
  ret$start = map_dbl(ret$nr, ~ sample.int(.x - num_rows, 1))
  ret
}

#' @importFrom arrow read_parquet
#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @importFrom purrr map_dfr map_chr map_dbl
#' @importFrom furrr future_map_dbl
#' @export
#parquet_dir_dataset = dataset(
ParquetDataDirSample = dataset(
  name = "ParquetDataDirSample",
  initialize = function(num_rows,
                        file_names, 
                        num_samples = 0, 
                        verbose = TRUE) {

    self$num_rows = num_rows
    self$file_names = map_chr(file_names, file_path_as_absolute)
    self$num_samples = num_samples
    if (any(!file.exists(file_names))) {
      stop(paste("Training files missing"))
    }
  
    if (verbose) {
      cat("Calculating number of samples in files.\n")
    }
    nr_pqf = \(x) nrow(read_parquet(x, as_data_frame = FALSE))
    self$file_row_counts = tibble(
      fn = file_names,
      nr = map_dbl(fn, nr_pqf)
    )
    self$samples = sample_file_name_and_offset(
      self$file_row_counts,
      self$num_samples
    )
  },
  .getitem = function(index) {
    if (index > nrow(self$samples) || index < 1) {
      stop("Index out of bounds")
    }
    pqr = read_parquet(self$file_row_counts$fn[index], as_data_frame = FALSE)
    start_row = self$samples$start[index]
    return(pqr[start_row:(start_row + self$num_rows - 1),])
  },
  .length = function() {
    return(self$num_samples)
  }
)

#' @importFrom torch torch_fft_fft
#' @export
get_spectrum <- function(x) {
  ret = torch_fft_fft(x, dim = 1, norm = "ortho") 
  ret = torch_sqrt(ret$real^2 + ret$imag^2)
  return(ret[seq_len(ceiling(ret$shape[1] / 2)),])
}

#' @importFrom torch dataset
#' @importFrom dplyr select collect
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
    tm = self$dsg$.getitem(index) |>
      select(X:Z) |>
      collect() |>
      as.matrix() 

    nr = nrow(tm)
    if (nr %% 3 != 0) {
      warning("Number of rows is not divisible by 3.")
    }

    starts = nr * c(0, 1, 2) / 3 + 1
    ends = nr * c(1, 2, 3) / 3 
    ms = map(
      seq_along(starts), 
      ~ tm[starts[.x]:ends[.x], ] |> 
        torch_tensor(dtype = self$dtype, device = self$device) |>
        get_spectrum()
    )
    return(list(y = ms[[2]], x = ms[c(1, 3)]))
  },
  .length = function() {
    return(self$dsg$.length())
  }
)

