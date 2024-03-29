#' @importFrom torch nn_sequential torch_flatten nn_linear
#' @export
SpectralSignatureReducer <- nn_module(
  "SpectralSignatureReducer",
  initialize = function(width, l2_width = 8400, l3_width = 100) {
    self$width = width
    self$initialized = FALSE
    self$l2_width = l2_width
    self$l3_width = l3_width
    self$nn1 = nn_sequential(
      nn_linear(self$width, self$l2_width),
      nn_dropout(0.2),
      nn_linear(self$l2_width, self$l3_width)
    )
    self$nn2 = nn_sequential(
      nn_linear(self$l3_width * 60 / 5 * 24 * 3, 100)
    )
  },
  forward = function(x) {
    if (length(x$shape) == 3) {
      # It's a singleton.
      self$nn1(x) |>
        torch_flatten() |>
        self$nn2()
    } else if (length(x$shape == 4)) {
      # It's batched.
      self$nn1(x) |>
        torch_flatten(start_dim = 2) |>
        self$nn2()
    }
  }
)

#' @importFrom torch torch_tensor torch_float32
#' @export
null_act_reducer = function(x, dtype = torch_float32()) {
  torch_tensor(c(), dtype = dtype)
}

#' @importFrom torch nn_sequential nn_module torch_cat
#' @export
DemoActigraphyModel <-  nn_module(
  "DemoActigraphyModel",
  initialize = function(y_contr_map, act_reducer, x_width, 
                        parent_env = parent.frame()) {
    self$initialized <- FALSE
    self$y_contr_map <- y_contr_map
    self$act_reducer <- act_reducer
    self$x_width <- x_width
    self$cat_layer_0 = nn_linear(self$x_width, self$x_width)
    self$cat_layer_1 = nn_linear(self$x_width, self$x_width)
    ycm = self$y_contr_map
    self$outputs = nn_module_list(
      map(
        unique(ycm$var),
        ~ {
          xl = ycm |> filter(var == .x)
          if (nrow(xl) == 1) {
            nn_sequential(
              nn_linear(self$x_width, self$x_width),
              nn_linear(self$x_width, 1)
            )
          } else {
            nn_sequential(
              nn_linear(self$x_width, self$x_width),
              nn_linear(self$x_width, nrow(xl)),
              nn_softmax(2),
            )
          }
        }
      )
    )
  },
  forward = function(x) {
    shape_lens = map_dbl(x, ~ length(.x$shape))
    is_singleton = 
      reduce(na.omit(c(shape_lens['act'] == 3, shape_lens['demo'] ==2)), `||`)
    if ("demo" %in% names(x)) {
      if (length(x$demo$shape) == 2) {
        # It's a singleton
        xc = torch_cat(list(x$demo$flatten(), self$act_reducer(x$act)), dim = 1)
      } else if (length(x$demo$shape) == 3) {
        act_flatten = self$act_reducer(x$act)
        if (length(act_flatten$shape) > 1) {
          act_flatten = torch_flatten(act_flatten, start_dim = 2)
          xc = torch_cat(
            list(
              torch_flatten(x$demo, start_dim = 2),
              act_flatten
            ),
            dim = 2
          )
        } else {
          xc = torch_flatten(x$demo, start_dim = 2)
        }
      } else {
        stop("Unsupported input shape.")
      }
    } else {
      if (length(x$act$shape) == 3) {
        stop("Singletons not yet supported.")
      } 
      xc = torch_flatten(self$act_reducer(x$act), start_dim = 2)
    }
    xcl1 = xc |> 
      self$cat_layer_0() |>
      self$cat_layer_1()
    if (is_singleton) {
      torch_cat(
        map(
          seq_along(self$outputs),
          ~ self$outputs[[.x]](xcl1)
        ),
        dim = 1
      )
    } else {
      torch_cat(
        map(
          seq_along(self$outputs),
          ~ self$outputs[[.x]](xcl1)
        ),
        dim = 2
      )
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

