#' @importFrom torch nn_sequential torch_flatten nn_linear
SpectralSignatureReducer <- nn_module(
  "SpectralSignatureReducer",
  initialize = function(l2_width = 1000, l3_width = 100) {
    self$initialized = FALSE
    self$l2_width = l2_width
    self$l3_width = l3_width
  },
  forward = function(x) {
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

#' @importFrom torch nn_sequential nn_module torch_cat
#' @export
DemoActigraphyModel <-  nn_module(
  classname = "DemoActigraphyModel",
  initialize = function(y_contr_map, act_reducer) {
    self$initialized <- FALSE
    self$y_contr_map <- y_contr_map
    self$act_reducer <- act_reducer
  },
  forward = function(x) {
    if (!self$initialized) {
      browser()
      xc = torch_cat(list(x$x$demo, self$act_reducer(x$x$act)), dim = 1)
      self$x_width <- xc$shape

      self$cat_layer_1 = nn_linear(self$x_width, self$x_width)
      ycm = self$y_contr_map
      self$outputs = nn_module_list(
        map(
          unique(ycm$var),
          ~ {
            browser()
            xl = ycm |> filter(var == .x)
            if (nrow(xl) == 1) {
              nn_sequential(
                nn_linear(self$x_width, 1)
              )
            } else {
              nn_sequential(
                nn_linear(self$x_width, nrow(xl)),
                nn_softmax(1)
              )
            }
          }
        )
      )
      self$initialized <- TRUE
    }
  }
)

#' @importFrom torch nn_sequential nn_module
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

