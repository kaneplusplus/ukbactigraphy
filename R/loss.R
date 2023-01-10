
#' @title The Multi-task Wakefulness Loss Function
#' @param the estimated value.
#' @param the target value.
#' @export 
ukb_wakeful_loss = function(input, target) {
  target = torch_flatten(target, start_dim = 2)
  # MSE for column 1. (Duration)
  duration_loss = nnf_mse_loss(input[,1], target[,1])

  # cross-entropy for 2 insomnia (4 levels)
  insomnia_loss = 
    torch_mean(-torch_sum(target[,2:5] * log(input[,2:5]), dim = 2))
   
  # cross-entropy for 3 dozing (6 levels)
  dozing_loss = 
    torch_mean(-torch_sum(target[,6:11] * log(input[,6:11]), dim = 2))

  # cross-entropy for 4 snoring (4 levels)
  snoring_loss = 
    torch_mean(-torch_sum(target[,12:15] * log(input[,12:15]), dim = 2))

  return(duration_loss + insomnia_loss + dozing_loss + snoring_loss)  
}
