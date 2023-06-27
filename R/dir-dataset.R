#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @export
DataDirSmall = dataset(
  name = "DataDirSmall",
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

DataFileSample = dataset(
  name = "DataFileSample",
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
  },
  .length = function() {
    nrow(self$samples)
  }
)

#' @importFrom arrow read_parquet
#' @importFrom torch dataset
#' @importFrom tools file_path_as_absolute
#' @importFrom dplyr collect
#' @importFrom purrr map_dfr map_chr map_dbl reduce
#' @importFrom furrr future_map_dbl
#' @export
#parquet_dir_dataset = dataset(
DataFileThreeWindowSample = dataset(
  name = "DataFileThreeWindowSample",
  inherit = DataFileSample,
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

#' @importFrom torch torch_fft_fft torch_log torch_sqrt
#' @export
get_spectrum <- function(x) {
  ret = torch_fft_fft(x, dim = 1, norm = "ortho") 
  ret = torch_log(torch_sqrt(ret$real^2 + ret$imag^2) + 1.)
  ret = ret[seq_len(ceiling(ret$shape[1] / 2)),]
  gc()
  return(ret)
}

#' @importFrom torch dataset
#' @importFrom dplyr select collect
#' @importFrom torch torch_tensor torch_sqrt torch_transpose torch_log
#' @export
SpectralSignatureTensor = dataset(
  name = "SpectralSignatureTensor",
  initialize = function(dsg, device = NULL, dtype = torch_float32()) {
    self$dsg = dsg 
    self$device = device
    self$dtype = dtype
  },
  getitem = function(index) {
    items = self$dsg$getitem(index)
    if (is.list(items$data)) {
      tm = items$data |>
        map(
          ~ select(.x, X:Z) |>
            collect() |>
            as.matrix() |>
            torch_tensor(dtype = self$dtype, device = self$device) |>
            torch_transpose(2, 1) 
        ) |>
        torch_stack(dim = 1) |>
        tryCatch(
          error = function(e) {
            cat("\nSpectralSignatureTensor::getitem()\n")
            stop(e)
          }
        )
      tl = TRUE
    } else {
      tm = items$data |>
        select(X:Z) |>
        collect() |>
        as.matrix() |>
        torch_tensor(dtype = self$dtype, device = self$device) |>
        torch_transpose(2, 1)
      tl = FALSE
    }
    tm = torch_fft_fft(tm, norm = "ortho")
    ret = torch_log(torch_sqrt(tm$real^2 + tm$imag^2) + 1.)
    ret = 
      list(
        data = 
          if (tl) {
            # TODO: The extra transfer to device isn't needed on Linux.
            ret[,,seq_len(ceiling(last(ret$shape) / 2))]$to(device = device)
          } else {
            ret[,seq_len(ceiling(last(ret$shape) / 2))]$to(device = device)
          },
        samples = items$samples
      )
    gc()
    return(ret)
  },
  .getitem = function(index) {
    if (index < 1 || index > self$dsg$.length()) {
      stop("Index out of range.")
    }
    tm = self$dsg$.getitem(index) |>
      select(X:Z) |>
      collect() |>
      as.matrix() |>
      torch_tensor(dtype = self$dtype, device = self$device) |>
      torch_transpose(2, 1) |>
      torch_fft_fft(norm = "ortho")
    ret = torch_log(torch_sqrt(tm$real^2 + tm$imag^2) + 1.)
    ret = ret[,seq_len(ceiling(last(ret$shape) / 2))]
    gc()
    return(ret)
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
#' @importFrom torch torch_tensor torch_sqrt
#' @export
SpectralTensorAdaptor = dataset(
  name = "SpectralTensorAdaptor",
  initialize = function(dsg, device = NULL, dtype = torch_float32()) {
    self$dsg = dsg # This should be derived from DataFileSample
    if (!inherits(self$dsg, "DataFileSample")) {
      stop("dsg must be derived from DataFileSample")
    }
    self$device = device
    self$dtype = dtype
  },
  getitem = function(index) {
    items = self$dsg$getitem(index)
    if (is.list(items$data)) {
      tm = items$data |>
        map(
          ~ select(.x, X:Z) |>
            collect() |>
            as.matrix() |>
            torch_tensor(dtype = self$dtype, device = self$device) |>
            torch_transpose(2, 1) 
        ) |>
        torch_stack(dim = 1) |>
        tryCatch(
          error = function(e) {
            cat("\nSpectralTensorAdaptor::getitem()\n")
            stop(e)
          }
        )
      tl = TRUE
    } else {
      tm = items$data |>
        select(X:Z) |>
        collect() |>
        as.matrix() |>
        torch_tensor(dtype = self$dtype, device = self$device) |>
        torch_transpose(2, 1)
      tl = FALSE
    }
    tm = torch_fft_fft(tm, norm = "ortho")
    ret = torch_log(torch_sqrt(tm$real^2 + tm$imag^2) + 1.)
    ret = 
      list(
        data = 
          if (tl) {
            ret[,,seq_len(ceiling(last(ret$shape) / 2))]
          } else {
            ret[,seq_len(ceiling(last(ret$shape) / 2))]
          },
        samples = items$samples
      )
    return(ret)
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

#' @importFrom furrr future_map future_map_dbl furrr_options
DayHourSpectralSignatureDataSet = dataset(
  name = "DayHourSpectralSignatureCollection",
  initialize = function(dsg, device = NULL, clip = 117600,
                        dtype = torch_float32()) {
    self$dsg = dsg 
    samples = dsg |> 
      select(info) |>
      distinct() |>
      collect()
    self$spec_sigs = 
      map(
        samples$info, 
        ~ dsg |> 
          filter(info == .x) |> 
          DayHourSpectralSignature(device = device, clip = clip, dtype = dtype)
      )
    self$lengths = future_map_dbl(self$spec_sigs, ~ .x$.length())
    # Create a lookup table mapping index to spectral signature item.
    self$lookup = 
      map_dfr(
        seq_along(self$lengths), 
        ~ tibble(spec_sig = rep(.x, self$lengths[.x]), 
                 item = seq_len(self$lengths[.x]))
      )

  },
  .getitem = function(index) {
    ss = self$lookup[index,]$spec_sig
    item = self$lookup[index,]$item
    return(self$spec_sigs[[ss]]$.getitem(item))
  },
  .length = function() {
    return(nrow(self$lookup))
  }
)

#' @importFrom arrow read_parquet
#' @importFrom torch torch_tensor torch_transpose torch_fft_fft torch_log
#' torch_sqrt
#' @export
DayHourSpectralSignature = dataset(
  name = "DayHourSpectralSignature",
  initialize = function(dsg, device = NULL, clip = 117600,
                        dtype = torch_float32()) {
    self$dsg = dsg
    self$filter_len = 3
    self$clip = clip
    self$device = device
    self$dtype = dtype
  },
  .getitem = function(index) {
    ret = get_day(self$dsg, index) |> 
      select(time, X:Z) |>
      mutate(hour = hour(time)) |>
      group_by(hour) |>
      group_nest() |>
      mutate(
        spec_sig = map(
          data, 
          ~ .x |> 
            select(-time) |>
            as.matrix() |>
            torch_tensor(dtype = self$dtype, device = self$device) |>
            torch_transpose(2, 1) |>
            torch_fft_fft(norm = "ortho", dim = 2) |>
            (\(x) torch_log(torch_sqrt(x$real^2 + x$imag^2)[,1:self$clip] + 1.))()
        )
      ) 
    torch_stack(ret$spec_sig, dim = 1) |>
        tryCatch(
          error = function(e) {
            cat("\nDayHourSpectralSignature::getitem()\n")
            stop(e)
          }
        )
  },
  .length = function() {
    len_full_days(self$dsg)
  }
)

downsample = function(sig, width = 10) {
  ret = torch_conv1d(
    torch_reshape(sig, c(sig$shape[1], 1, sig$shape[2])),
    torch_tensor(rep(1/width, width), device = sig$device) |>
      torch_reshape(c(1, 1, width)),
  )
  ret = torch_reshape(ret, c(ret$shape[1], ret$shape[3]))
  ret = ret[,seq(1, ret$shape[2], by = width)] 
}

#' @importFrom arrow read_parquet
#' @importFrom torch torch_tensor torch_transpose torch_fft_fft torch_log
#  nnf_pad torch_sqrt
#' @export
Day5MinSpectralSignature = dataset(
  name = "Day5MinSpectralSignature",
  initialize = function(dsg, device = NULL, clip = 10000,
                        dtype = torch_float32()) {
    self$dsg = dsg
    self$filter_len = 3
    self$clip = clip
    self$device = device
    self$dtype = dtype
  },
  .getitem = function(index) {
    ret = get_day(self$dsg, index) |> 
      select(time, X:Z) |>
      mutate(hour = hour(time)) |>
      mutate(minute = minute(time)) |>
      mutate(five_min = minute %/% 5) |>
      group_by(hour, five_min) |>
      group_nest() |>
      mutate(
        spec_sig = map(
          data, 
          ~ .x |> 
            select(-time, -minute) |>
            as.matrix() |>
            torch_tensor(dtype = self$dtype, device = self$device) |>
            torch_transpose(2, 1) |>
            torch_fft_fft(norm = "ortho", dim = 2) |>
            (\(x) torch_log(torch_sqrt(x$real^2+x$imag^2)[,1:self$clip]+1.))()|>
            downsample()
        )
      ) 
    length_correct = map_lgl(ret$spec_sig, ~ all(.x$shape == c(3, 1000)))
    if (any(!length_correct)) {
      saveRDS(index, "bad_ss_index.rds")
      for (fix_ind in which(!length_correct)) {
        fix_len = rev(c(ret$spec_sig[fix_ind]$shape - c(3, 1000)))
        log_msg("spec sig shape")
        log_msg(fix_ind)
        log_msg(ret$spec_sig[[fix_ind]]$shape)
        log_msg(fix_len)
        ret$spec_sig[[fix_ind]] = 
          nnf_pad(input = ret$spec_sig[[fix_ind]], pad = fix_len,
                  mode = "constant", value = 0.)
        log_msg("after fix")
        log_msg(ret$spec_sig[[fix_ind]]$shape)
      }
    }
    torch_stack(ret$spec_sig, dim = 1) |>
        tryCatch(
          error = function(e) {
            cat("\nDay5MinSpectralSignature::getitem()\n")
            stop(e)
          }
        )
  },
  .length = function() {
    len_full_days(self$dsg)
  }
)

#' @importFrom torch torch_cat 
#' @importFrom arrow open_dataset
#' @export
Actigraphy24DataSet = dataset(
  name = "Actigraphy24DataSet",
  initialize = function(y, x, user, ss, ss_index, data, 
                        dtype = NULL, device = NULL, requires_grad = FALSE,
                        pin_memory = FALSE) {
    self$y = y
    self$x = x
    self$user = user
    self$ss = ss
    self$ss_index = ss_index
    self$data = data
    self$dtype = dtype
    self$device = device
    self$requires_grad = requires_grad
    self$pin_memory = pin_memory
    if (!is.null(x)) {
      self$x_contr_map = model_tensor(
          self$data[1, c(self$x, self$user)],
          index = self$user,
          dtype = self$dtype,
          device = self$device,
          requires_grad = self$requires_grad,
          pin_memory = self$pin_memory
        )$contr_level_map
    } else {
      self$x_contr_map = NULL
    }
    self$y_contr_map = model_tensor(
      self$data[1, c(self$y, self$user)], 
      index = self$user,
      dtype = self$dtype,
      device = self$device,
      requires_grad = self$requires_grad,
      pin_memory = self$pin_memory
    )$contr_level_map 
  },
  .getitem = function(index) {
    y = model_tensor(
      self$data[index, c(self$y, self$user)], 
      index = self$user,
      dtype = self$dtype,
      device = self$device,
      requires_grad = self$requires_grad,
      pin_memory = self$pin_memory
    ) |> to_tensor()
    if (!is.null(self$x_contr_map)) {
      x = model_tensor(
        self$data[index, c(self$x, self$user)], 
        index = self$user,
        dtype = self$dtype,
        device = self$device,
        requires_grad = self$requires_grad,
        pin_memory = self$pin_memory
      ) |> to_tensor()
    } else {
      x = NULL
    }
    act = DayHourSpectralSignature(
      open_dataset(self$data[[self$ss]][index])
    )$.getitem(self$data[[self$ss_index]][index])
    gc()
    if (is.null(x)) {
      return(
        list(
          x = list(act = act),
          y = y
        )
      )
    } else {
      return(
        list(
          x = list(demo = x, act = act),
          y = y
        )
      )
    }
  },
  .length = function() {
    return(nrow(self$data))
  }
)

log_msg = function(string, fn = "output.log") {
  fh = file(fn, open = "a")
  on.exit(close(fh))
  s = capture.output(print(string))
  writeLines(s, fh)
}

#' @importFrom torch torch_cat nnf_pad
#' @importFrom arrow open_dataset
#' @export
Actigraphy5MinDataSet = dataset(
  name = "Actigraphy5MinDataSet",
  initialize = function(y, x, user, ss, ss_index, data, 
                        dtype = NULL, device = NULL, requires_grad = FALSE,
                        pin_memory = FALSE) {
    self$y = y
    self$x = x
    self$user = user
    self$ss = ss
    self$ss_index = ss_index
    self$data = data
    self$dtype = dtype
    self$device = device
    self$requires_grad = requires_grad
    self$pin_memory = pin_memory
    if (!is.null(x)) {
      self$x_contr_map = model_tensor(
          self$data[1, c(self$x, self$user)],
          index = self$user,
          dtype = self$dtype,
          device = self$device,
          requires_grad = self$requires_grad,
          pin_memory = self$pin_memory
        )$contr_level_map
    } else {
      self$x_contr_map = NULL
    }
    self$y_contr_map = model_tensor(
      self$data[1, c(self$y, self$user)], 
      index = self$user,
      dtype = self$dtype,
      device = self$device,
      requires_grad = self$requires_grad,
      pin_memory = self$pin_memory
    )$contr_level_map 
  },
  .getitem = function(index) {
    y = model_tensor(
      self$data[index, c(self$y, self$user)], 
      index = self$user,
      dtype = self$dtype,
      device = self$device,
      requires_grad = self$requires_grad,
      pin_memory = self$pin_memory
    ) |> to_tensor()
    if (!is.null(self$x_contr_map)) {
      x = model_tensor(
        self$data[index, c(self$x, self$user)], 
        index = self$user,
        dtype = self$dtype,
        device = self$device,
        requires_grad = self$requires_grad,
        pin_memory = self$pin_memory
      ) |> to_tensor()
    } else {
      x = NULL
    }
    act = Day5MinSpectralSignature(
      open_dataset(self$data[[self$ss]][index])
    )$.getitem(self$data[[self$ss_index]][index])
    if (any(tail(act$shape, 3) != c(288, 3, 1000))) {
      saveRDS(index, "badindex.rds")
      # Create padding according to torches goofy standard.
      log_msg("5 min data set")
      pl = rev(c(288, 3, 1000) - tail(act$shape, 3))
      pl = as.vector(t(cbind(rep(0, length(pl)), pl)))
      log_msg(pl)
      log_msg(act$shape)
      act = nnf_pad(input = act, pad = pl, mode = "constant",
                    value = 0.)
      log_msg("after fix")
    }
    gc()
    if (is.null(x)) {
      return(
        list(
          x = list(act = act),
          y = y
        )
      )
    } else {
      return(
        list(
          x = list(demo = x, act = act),
          y = y
        )
      )
    }
  },
  .length = function() {
    return(nrow(self$data))
  }
)


#' @importFrom torch torch_cat
#' @importFrom arrow open_dataset
#' @export
Actigraphy24DataSet = dataset(
  name = "Actigraphy24DataSet",
  initialize = function(y, x, user, ss, ss_index, data, 
                        dtype = NULL, device = NULL, requires_grad = FALSE,
                        pin_memory = FALSE) {
    self$y = y
    self$x = x
    self$user = user
    self$ss = ss
    self$ss_index = ss_index
    self$data = data
    self$dtype = dtype
    self$device = device
    self$requires_grad = requires_grad
    self$pin_memory = pin_memory
    if (!is.null(x)) {
      self$x_contr_map = model_tensor(
          self$data[1, c(self$x, self$user)],
          index = self$user,
          dtype = self$dtype,
          device = self$device,
          requires_grad = self$requires_grad,
          pin_memory = self$pin_memory
        )$contr_level_map
    } else {
      self$x_contr_map = NULL
    }
    self$y_contr_map = model_tensor(
      self$data[1, c(self$y, self$user)], 
      index = self$user,
      dtype = self$dtype,
      device = self$device,
      requires_grad = self$requires_grad,
      pin_memory = self$pin_memory
    )$contr_level_map 
  },
  .getitem = function(index) {
    y = model_tensor(
      self$data[index, c(self$y, self$user)], 
      index = self$user,
      dtype = self$dtype,
      device = self$device,
      requires_grad = self$requires_grad,
      pin_memory = self$pin_memory
    ) |> to_tensor()
    if (!is.null(self$x_contr_map)) {
      x = model_tensor(
        self$data[index, c(self$x, self$user)], 
        index = self$user,
        dtype = self$dtype,
        device = self$device,
        requires_grad = self$requires_grad,
        pin_memory = self$pin_memory
      ) |> to_tensor()
    } else {
      x = NULL
    }
    act = DayHourSpectralSignature(
      open_dataset(self$data[[self$ss]][index])
    )$.getitem(self$data[[self$ss_index]][index])
    gc()
    if (is.null(x)) {
      return(
        list(
          x = list(act = act),
          y = y
        )
      )
    } else {
      return(
        list(
          x = list(demo = x, act = act),
          y = y
        )
      )
    }
  },
  .length = function() {
    return(nrow(self$data))
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

#' @importFrom purrr map_chr
#' @export
SummaryOutcomeActgraphyDataSet = dataset(
  name = "SummaryOutcomeActigraphyDataSet",
  initialize = function(y, x) {
    x_idfu = x$get_sample_data()$fn |>
      basename() |>
      gsub("\\.parquet", "", x = _) |>
      strsplit("_") |>
      map_chr(~ .x[1]) |>
      unique()
    y_idfu = y$idf$eid |>
      unique()
    self$idf_intersect = intersect(x_idfu, y_idfu)
    self$x_idfs = (x$get_sample_data()$fn |>
      basename() |>
      gsub("\\.parquet", "", x = _) |>
      strsplit("_") |>
      map_chr(~ .x[1]) %in% self$idf_intersect) %>% which()
    self$x_int_idfs =  x$get_sample_data()$fn |>
      basename() |>
      gsub("\\.parquet", "", x = _) |>
      strsplit("_") |>
      map_chr(~ .x[1])
    self$x = x
    self$y = y
  },
  .getitem = function(index) {
    y_index = which(self$y$idf$eid == self$x_int_idfs[index])
    list(
      x = self$x$.getitem(self$x_idfs[index]),
      y = self$y$mt[y_index,]
    )
  },
  .length = function() {
    length(self$x_idfs)
  }
)

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
