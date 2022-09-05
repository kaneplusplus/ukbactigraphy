
#' @importFrom torch nn_sequential
#' @export
SelfSupervisedSpectral = function() {
  ret = nn_module(
    "SelfSupervisedSpectral",
    initialize = function() {
      self$initialized = FALSE
      if (!self$initialized) {
        self$nn_l1 = nn_sequential(
          nn_linear(60000, 6000),
          nn_linear(6000, 600)
        )
        self$nn_l2 = nn_sequential(
          nn_linear(1800, 600),
          nn_linear(600, 600)
        )
    
        self$nn_l3_c1 = nn_sequential(
          nn_linear(1200, 60000)
        )

        self$nn_l3_c2 = nn_sequential(
          nn_linear(1200, 60000)
        )

        self$nn_l3_c3 = nn_sequential(
          nn_linear(1200, 60000)
        )
        self$initialized = TRUE
      }
    },

    forward = function(x) {
      if (length(x$shape) == 4) {
        out1 = self$nn_l1(x[,1,,]) |>
          torch_flatten(start_dim = 2) |>
          self$nn_l2()
        
        out2 = self$nn_l1(x[,2,,]) |>
          torch_flatten(start_dim = 2) |>
          self$nn_l2()
      } else {
        stop("Only batch fitting is suported so far.")
      }

      tc = torch_cat(list(out1, out2), 2) 
      torch_stack(
        list(self$nn_l3_c1(tc), self$nn_l3_c2(tc), self$nn_l3_c3(tc)),
        dim = 2
      )
    }
  )
  ret 
}

