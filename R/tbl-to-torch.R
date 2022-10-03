
#' @param d an object inheriting from type data.frame.
#' @param f a formula.
#' @importFrom torch torch_tensor
#' @importFrom tibble as_tibble
#' @export
model_tensor = 
  function(
    d, 
    index = NULL,
    f = ~ . - 1, 
    dtype = NULL, 
    device = NULL,
    requires_grad = FALSE, 
    pin_memory = FALSE,
    contrasts.arg = NULL,
    to_torch_tensor = TRUE) {

  if (length(index) > 1) {
    stop("Index should be the name of one column in d.")
  }

  d = model.frame(f, d) |>
    as_tibble()

  if (!is.null(index)) {
    idf = d[,index]
    d = select(d, -{index})
  } else {
    idf = NULL
  }
  cnd = colnames(d)

  if (any(map_lgl(d, is.list))) {
    stop("Don't know how to handle list columns.")
  }

  fcl = map_lgl(d, negate(is.numeric))
  fcn = names(fcl)[fcl]
  
  # If you don't supply and contrast.arg, then you get one-hot encoding.
  contr_arg = as.list(rep("contr.onehot", length(setdiff(fcn, contrasts.arg))))
  names(contr_arg) = setdiff(fcn, contr_arg)
  contr_arg = c(contrasts.arg, contr_arg)

  mt = model.matrix(f, d, contrasts.arg = contr_arg)
  contr_map = d |> 
    select_if(is.factor) |>
    map(~contr.onehot(levels(.x)))
  
  cnm = colnames(mt)

  ret =  list(
    mt = torch_tensor(mt),
    contr_map = contr_map, 
    d_col_names = cnd,
    m_col_names = cnm,
    factor_cols = fcn,
    idf = idf
  )
  class(ret) = c("model_tensor")
  return(ret)
}

#' @export
contr.onehot = function(x, sparse = FALSE) {
  contr.treatment(x, contrasts = FALSE, sparse = sparse)
}

#' @export
to_tensor = function(x, indexes) {
  UseMethod("to_tensor")
}

#' @export
to_tensor.default = function(x, indexes) {
  stop("Don't know how to cast x to torch_tensor.")
}

#' @export
to_tensor.model_tensor = function(x, indexes) {
  if (missing(indexes)) {
    return(x$mt)
  }
  x$mt[which(x$idf[[1]] %in% indexes),]
}

#' @export
to_tibble = function(x) {
  UseMethod("to_tibble", x)
}

#' @export
to_tibble.default = function(x) {
  stop(
    paste(
      "Don't know how to cast an object of type",
      paste(class(x), collapse = " "),
      "to tibble."
    )
  )
}

#' @export
to_tibble.model_tensor = function(x) {

  m = as.matrix(x$mt)
  colnames(m) = x$m_col_names
  factor_col_names = 
    map(x$contr_map, colnames)
  for (i in seq_along(factor_col_names)) {
    factor_col_names[[i]] = 
      paste0(names(factor_col_names)[i], factor_col_names[[i]])
  }
  rmf = as_tibble(m)[,setdiff(colnames(m), unlist(factor_col_names))]
 
  for (j in seq_along(factor_col_names)) {
    ms = m[,factor_col_names[[j]]]
    cm = x$contr_map[[names(factor_col_names)[j]]]
    se_cm = apply(cm, 1, paste0, collapse = ";")
    se_ms = apply(ms, 1, paste0, collapse = ";")
    rmf[[names(factor_col_names)[j]]] = 
      factor(names(se_cm)[match(se_ms, se_cm)], levels = names(se_cm))
  } 
  rmf[, x$d_col_names]
}
