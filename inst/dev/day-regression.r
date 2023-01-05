library(devtools)
library(torch)
library(arrow)
library(dplyr)
library(tidyr)
library(lubridate)
library(future)
library(furrr)
library(foreach)
library(luz)
library(doFuture)
library(doRNG)
num_workers = floor(parallel::detectCores() / 2)
plan(multicore, workers = num_workers)
registerDoFuture()
document()

make_data = FALSE
find_lr = FALSE
train_model = TRUE
device = "mps"

#Sys.setenv(PYTORCH_ENABLE_MPS_FALLBACK = 1)
options(tz="UTC")


if (make_data) {
  data_dir = "parquet-subset"
  act = open_dataset("parquet-subset")
  outcomes = readRDS("outcomes.rds")

  users = intersect(
    outcomes$eid, 
    (act |> select(user) |> distinct() |> collect())$user 
  )

  outcomes_train = outcomes |> 
    filter(eid %in% users[-1]) |> 
    arrange(eid) |>
    rename(user = eid)

  outcomes_test = outcomes |> 
    filter(eid %in% users[1]) |> 
    arrange(eid) |>
    rename(user = eid)

  # Make the training data
  train = foreach(iuser = users[-1], .combine = bind_rows) %dorng% {
    dhss_len = DayHourSpectralSignature(act |> filter(user == iuser))$.length()
    gc()
    foreach (i = seq_len(dhss_len), .combine = bind_rows) %do% {
      bind_cols(
        outcomes_train |> filter(user == iuser),
  #      tibble(dhss = list(dhss), dhss_index = i)
        tibble(
          afp = file.path(
            data_dir, 
            paste0(
              (act |> filter(user == iuser) |> head(1) |> collect())$info, 
              ".parquet")
          ),
          afp_index = i
        )
      )
    }
  }
  saveRDS(train, "data-lookup/train.rds")

  # Make the test data
  test = foreach(iuser = users[1], .combine = bind_rows) %dorng% {
    dhss_len = DayHourSpectralSignature(act |> filter(user == iuser))$.length()
    gc()
    foreach (i = seq_len(dhss_len), .combine = bind_rows) %do% {
      bind_cols(
        outcomes_test |> filter(user == iuser),
  #      tibble(dhss = list(dhss), dhss_index = i)
        tibble(
          afp = file.path(
            data_dir, 
            paste0(
              (act |> filter(user == iuser) |> head(1) |> collect())$info, 
              ".parquet")
          ),
          afp_index = i
        )
      )
    }
  }
  saveRDS(test, "data-lookup/test.rds")
} else {
  train = readRDS("data-lookup/train.rds")
  test = readRDS("data-lookup/test.rds")
}

num_norm = function(x) {
  r = range(x)
  (x - r[1]) / (r[2] - r[1])
}
# Normalize the numeric columns
tt = bind_rows(
  train |> mutate(type = "train"),
  test |> mutate(type = "test"),
) 

norm_vars = c("sleep_duration", "yob", "weight", "height")
norm = tt |> 
  select(all_of(norm_vars)) |>
  mutate_all(as.numeric) |>
  pivot_longer(everything()) |>
  group_by(name) |>
  summarize(min = min(value), max = max(value))

saveRDS(norm, "normalization-table.rds")

tt = tt |>
  mutate_at(
    norm_vars, 
    ~ {
      r = range(.x)
      (.x - r[1]) / (r[2] - r[1])
    }
  )

train = tt |> 
  filter(type == "train") |>
  select(-type)

test = tt |> 
  filter(type == "test") |>
  select(-type)

ads_train = Actigraphy24DataSet(
  y = c("sleep_duration", "insomnia", "dozing", "snoring"),
  x = c("sex", "yob", "weight", "height"),
  user = "user",
  ss = "afp",
  ss_index = "afp_index",
  data = train[1:2,],
  device = device
)

ads_test = Actigraphy24DataSet(
  y = c("sleep_duration", "insomnia", "dozing", "snoring"),
  x = c("sex", "yob", "weight", "height"),
  user = "user",
  ss = "afp",
  ss_index = "afp_index",
  data = test[1:2,],
  device = device
)

my_loss = function(input, target) {
  if (length(target$shape == 3)) {
    target = target$flatten(start_dim = 2)
  }
  nnf_mse_loss(input, target)
}


if (find_lr) {
  # see https://mlverse.github.io/luz/reference/lr_finder.html to find the
  # learning rate

  model = DemoActigraphyModel |>
    setup(
      loss = my_loss,
      optimizer = optim_adam,
    ) |>
    set_hparams(
      y_contr_map = ads_train$y_contr_map,
      act_reducer = SpectralSignatureReducer(117600),
      x_width = 105
    ) 

  dl = dataloader(
        ads_train,
        batch_size = 2,
        shuffle = TRUE,
        num_workers = num_workers,
        worker_packages = c("ukbactigraphy", "tibble", "dplyr")
      )

  records <- lr_finder(model, dl, verbose = TRUE)
  plot(records)
}

if (train_model) {
  tm = DemoActigraphyModel |>
    setup(
      loss = my_loss,
      optimizer = optim_adam,
    ) |>
    set_hparams(
      y_contr_map = ads_train$y_contr_map,
      act_reducer = SpectralSignatureReducer(117600),
      x_width = 105
    ) |> 
    set_opt_hparams(lr = 1e-10)
  fm = tm |>
    fit(
      data = dataloader(
        ads_train,
        batch_size = 2,
        shuffle = TRUE,
        num_workers = num_workers, 
        worker_packages = c("ukbactigraphy", "tibble", "dplyr")
      ),
      epochs = 1, 
      valid_data = dataloader(
        ads_test, 
        num_workers = num_workers,
        worker_packages = c("ukbactigraphy", "tibble", "dplyr"),
      ),
      callbacks = list(
        luz_callback_keep_best_model(),
        luz_callback_model_checkpoint(
          path = "checkpoint/demo-actigraphy-model-{epoch:03d}.pt",
          save_best_only = TRUE
        ),
        luz_callback_csv_logger("logging/training-log.csv")
      )
    )  
}

#input = map(ads_train$.getitem(1)$x, ~ .x$to(device = device))
#tm$model(input)

