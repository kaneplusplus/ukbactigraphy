
#' @importFrom dplyr summarize collect
#' @importFrom lubridate hour minute second hour<- minute<- second<-
full_day_range = function(x) {
  tr = x |>
    summarize(min = min(time), max = max(time)) |>
    collect()
  trmin = tr$min
  hour(trmin) = minute(trmin) = second(trmin) = 0
  if (tr$min == trmin) {
    ret_min = trmin
  } else {
    ret_min = trmin + days(1)
  }
  trmax = tr$max
  hour(trmax) = minute(trmax) = second(trmax) = 0
  c(ret_min, trmax)
}

len_full_days = function(x) {
  r = full_day_range(x)
  as.numeric(difftime(r[2], r[1], units = "days"))
}

#' @importFrom dplyr filter
#' @importFrom lubridate days
get_day = function(x, d) {
  if (d > len_full_days(x) || d < 1) {
    stop("Data has ", d, " days.")
  }
  start_dt = full_day_range(x)[1]
  start = (start_dt + days(d - 1)) 
  end = (start_dt + days(d)) 
  filter(x |> collect(), time >= start & time < end)
}

#' @importFrom dplyr select collect mutate across everything filter bind_cols
#' @importFrom stats na.omit
subsample = function(x, filter_len = 3, cols = quote(X:Z)) {
  # grab the column names now so we can maintain column-order-stability.
  cn = names(x)
  ds = x |>
    select(!!cols) |>
    collect() |>
    mutate(
      across(
        everything(),
        ~ stats::filter(.x, rep(1/filter_len, filter_len))
      )
    ) |>
    bind_cols(x |> select(-!!cols) |> collect()) |>
    na.omit()
  ds[
    as.integer(seq(1, nrow(ds), length.out = floor(nrow(ds) / filter_len))),
    cn
  ]
}
