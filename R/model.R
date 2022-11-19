#' @importFrom torch nn_sequential torch_flatten nn_linear
SpectralSignatureReducer = function() {
  ret = nn_module(
    "SpectralSignatureReducer",
    initialize = function(l2_width = 1000, l3_width = 100) {
      self$initialized = FALSE
      self$l2_width = l2_width
      self$l3_width = l3_width
    },
    forward = function(x) {
      browser()
      if (!self$initialized) {
        ne = x$shape[3]
        self$nn1 = nn_sequential(
          nn_linear(ne, self$l2_width),
          nn_linear(self$l2_width, self$l3_width)
        )
        self$nn2 = nn_sequential(
          nn_linear(self$l3_width * 24 * 3, 100)
        )
      }
      self$nn1(x) |>
        torch_flatten() |>
        self$nn2()
    }
  )
  ret
}

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
        )
    
        self$nn_l3_x = nn_sequential(
          nn_linear(1200, 60000)
        )

        self$nn_l3_y = nn_sequential(
          nn_linear(1200, 60000)
        )

        self$nn_l3_z = nn_sequential(
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
        list(self$nn_l3_x(tc), self$nn_l3_y(tc), self$nn_l3_z(tc)),
        dim = 2
      )
    }
  )
  ret 
}

