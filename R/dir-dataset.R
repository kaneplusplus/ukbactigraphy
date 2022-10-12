#' @importFrom arrow read_parquet
#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @export
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

ParquetDataFileSample = dataset(
  name = "ParquetDataFileSample",
  initialize = function(num_rows, file_names, num_samples = 0, verbose = TRUE) {
    self$num_rows = num_rows
    self$file_names = map_chr(file_names, file_path_as_absolute)
    self$num_samples = num_samples
    if (any(!file.exists(file_names))) {
      stop(paste("Files could not be found."))
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
  getitem = function(index) {
    if (index > nrow(self$samples) || index < 1) {
      stop("Index out of bounds")
    }
    pqr = read_parquet(self$samples$fn[index], as_data_frame = FALSE)
    return(
      list(
        data = 
          pqr[
            self$samples$start[index]:(self$samples$start[index] + num_rows -1),
          ],
        samples = self$samples[index,]
      )
        
    )
  },
  .getitem = function(index) {
    self$getitem(index)$data
  },
  get_samples = function() {
    self$samples
  },
  get_num_rows = function() {
    self$num_rows
  }
)

#' @importFrom arrow read_parquet
#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @importFrom purrr map_dfr map_chr map_dbl
#' @importFrom furrr future_map_dbl
#' @export
#parquet_dir_dataset = dataset(
ParquetDataFileThreeWindowSample = dataset(
  name = "ParquetDataFileThreeWindowSample",
  inherit = ParquetDataFileSample,
  getitem = function(index) {
    if (index > nrow(self$samples) || index < 1) {
      stop("Index out of bounds")
    }

    # Cast it to the super-class type before calling the getitem method.
    pqrl = super$getitem(index)

    start_row = 1
    starts = self$num_rows * c(0, 1, 2) / 3 + start_row
    ends = self$num_rows * c(1, 2, 3) / 3 + start_row - 1
    ret = map(
      seq_along(starts), 
      ~ pqrl$data[starts[.x]:ends[.x], c("X", "Y", "Z")]
    )
    ss = reduce(map(1:3, ~ pqrl$samples), bind_rows)
    ss$start = ss$start + starts - 1
    ss$nr = self$num_rows
    return(list(data = ret, samples = ss))
#|>
#    )
  },
  .getitem = function(index) { 
    self$getitem(index)$data
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
#' @importFrom torch torch_tensor torch_sqrt
#' @export
SpectralTensorAdaptor = dataset(
  name = "SpectralTensorAdaptor",
  initialize = function(dsg, device = NULL, dtype = torch_float32()) {
    self$dsg = dsg # This should be derived from ParquetDataFileSample
    if (!inherits(self$dsg, "ParquetDataFileSample")) {
      stop("dsg must be derived from ParquetDataFileSample")
    }
    self$device = device
    self$dtype = dtype
  },
  getitem = function(index) {
    items = self$dsg$getitem(index)
    tm = items$data |>
      map(
        ~ collect(.x) |>
          as.matrix() |>
          torch_tensor(dtype = self$dtype, device = self$device) |>
          torch_transpose(2, 1) 
      ) |>
      torch_stack(dim = 1) |>
      torch_fft_fft(norm = "ortho")
    ret = torch_sqrt(tm$real^2 + tm$imag^2)
    return(
      list(
        data = ret[,,seq_len(ceiling(last(ret$shape) / 2))],
        samples = items$samples
      )
    )
  },
  .getitem = function(index) {
    self$getitem(index)$data
  },
  .length = function() {
    return(self$dsg$.length())
  },
  get_sample_data = function() {
    self$dsg$samples
  },
  get_num_rows = function() {
    self$dsg$num_rows
  }
)

#' @importFrom torch dataset
#' @importFrom dplyr select collect
#' @importFrom torch torch_stack
#' @export
ThreeWindowSelfSupervisedDataSet = dataset(
  name = "ThreeWindowSelfSupervisedDataSet",
  initialize = function(dsg) {
    self$dsg = dsg
  },
  getitem = function(index) {
    dsgd = self$dsg$getitem(index)
    return(
      list(
        data = list(
          x = dsgd$data[c(1, 3),,],
          y = dsgd$data[2,,]
        ),
        samples = dsgd$samples
      )
    )
  },
  .getitem = function(index) {
    self$getitem(index)$data
  },
  .length = function() {
    return(self$dsg$.length())
  },
  get_sample_data = function() {
    self$dsg$samples
  },
  get_num_rows = function() {
    self$dsg$num_rows
  }
)

#SummaryOutcomeActgraphyDataSet = dataset(
#  name = "SummaryOutcomeActigraphyDataSet",
#  initialize = function(
#)

OutcomeDataSet = dataset(
  name = "OutcomeDataSet",
  initialize = function(outcome) {
    self$outcome = outcome
  },
)

OutcomeActigraphyDataSet = dataset(
  name = "SummaryActigraphyDataSet",
  initialize = function(outcome_ds, spectral_ds) {
    self$outcome = outcome_ds
    self$spectral_ds = spectral_ds
  },
  .getitem = function(index) {
  },
  .length = function() {
    
  }
)
