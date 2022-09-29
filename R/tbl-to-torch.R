
#' @param d an object inheriting from type data.frame.
#' @param f a formula.
#' @importFrom torch torch_tensor
#' @importFrom tibble as_tibble
#' @export
model_tensor = function(d, index, f = ~ . - 1, dtype = NULL, device = NULL,
                        requires_grad = FALSE, pin_memory = FALSE) {

  if (length(index) != 1) {
    stop("Index should be the name of one column in d.")
  }

  d = model.frame(f, d) |>
    as_tibble()
  idf = d[,index]
  
  d = select(d, -{index})
  cnd = colnames(d)
  contr_map = d |> 
    select_if(is.factor) |>
    map(~colnames(contrasts(.x)))
   
  mt = model.matrix(f, d) 
  cnm = names(mt)

  ret =  list(
    mt = torch_tensor(mt),
    contr_map = contr_map, 
    d_col_names = cnd,
    m_col_names = cnm,
    idf = idf
  )
  class(ret) = c("model_tensor")
  return(ret)
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

