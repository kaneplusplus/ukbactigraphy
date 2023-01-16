library(torch)
library(dplyr)
library(luz)
library(tibble)
library(future)
library(ukbactigraphy)
library(itertools)
library(foreach)
library(purrr)
library(devtools)


num_workers = 3
plan(multicore, workers = num_workers)
device = "gpu"
batch_size = 32
lr = 1e-4

find_lr = FALSE

x = as_tibble(readRDS(file.path(here::here(), "sample-data.rds"))) |>
  filter(shape1 == 24) |>
  select(-starts_with("afp"), -starts_with("shape"))

#x$afp = file.path(here::here(), x$afp)

set.seed(123)
train_samples = sample(nrow(x), round(0.9 * nrow(x)))
test_samples = setdiff(seq_len(nrow(x)), train_samples)

train_chunks = 
  foreach(it = isplitVector(train_samples, chunkSize = 20000)) %do% it

train_inds = train_samples #train_chunks[[1]]
test_inds = test_samples

train_ads = Demo24DataSet(
  y = c("sleep_duration", "insomnia", "dozing", "snoring"),
  x = c("sex", "yob", "weight", "height"),
  user = "user",
  ss = "afp",
  ss_index = "afp_index",
  data = x[train_inds,],
  device = device
)

test_ads = Demo24DataSet(
  y = c("sleep_duration", "insomnia", "dozing", "snoring"),
  x = c("sex", "yob", "weight", "height"),
  user = "user",
  ss = "afp",
  ss_index = "afp_index",
  data = x[test_inds,],
  device = device
)

if (!dir.exists("checkpoint")) {
  dir.create("checkpoint")
}
if (!dir.exists("logging")) {
  dir.create("logging")
}

make_model_gen = function() {
  model = DemoActigraphyModel |>
    setup(
      loss = ukb_wakeful_loss,
      optimizer = optim_adam,
    ) |>
    set_hparams(
      y_contr_map = train_ads$y_contr_map,
      act_reducer = null_act_reducer,
      x_width = 5
    ) |>
    set_opt_hparams(lr = lr) 
}

fit_model = function(model, chunk) {
  checkpoint_path = 
    sprintf(
      "checkpoint/train-actigraphy-model-%03d-{epoch:03d}.pt", 
      chunk
    )
  
  model |>
    fit(
      data = dataloader(
        train_ads,
        batch_size = 32,
        shuffle = TRUE,
        num_workers = num_workers,
        worker_packages = c("ukbactigraphy", "tibble", "dplyr")
      ),
      epochs = 10,
      valid_data = dataloader(
        test_ads,
        batch_size = 1000,
        num_workers = num_workers,
        worker_packages = c("ukbactigraphy", "tibble", "dplyr"),
      ),
      callbacks = list(
        luz_callback_model_checkpoint(
          path = checkpoint_path,
          save_best_only = TRUE
        ),
        luz_callback_auto_resume(path = "state.pt"),
        luz_callback_csv_logger("logging/training-log.csv")
      )
    )
}

if (find_lr) {
  lr_model = make_model_gen()

  dl = dataloader(
    train_ads,
    batch_size = 32,
    shuffle = TRUE,
    num_workers = num_workers,
    worker_packages = c("ukbactigraphy", "tibble", "dplyr")
  )

  records = lr_finder(lr_model, dl, verbose = TRUE)
  plot(records)
}

print(date())

model = make_model_gen() 
fitted_model =  fit_model(model, chunk = 1)
luz_save(fitted_model, "fitted-model.luz")
print(date())

