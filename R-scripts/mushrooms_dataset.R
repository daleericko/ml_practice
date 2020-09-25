## Mushrooms Dataset 
## Erick Cohen 

library(tidyverse)
library(readr)
library(tidymodels)

mushrooms_raw <- 
  read_csv(here::here("data-raw/mushrooms.csv")) %>% 
  janitor::clean_names()


## EDA 
skimr::skim(mushrooms_raw)

mushrooms_raw %>% 
  count(class, sort = TRUE)

mushrooms_raw %>% 
  ggplot(aes(x = class, 
             fill = cap_color)) + 
  geom_bar()


## Building a classification model 

# preprocessing
# sapply(mushrooms_df, levels)

mushrooms_df <- 
  mushrooms_raw %>% 
  mutate_if(is.character, as.factor) %>% 
  select(-c(bruises, gill_attachment, veil_type))

set.seed(2)
mushrooms_split <- initial_split(mushrooms_df, strata = class)
mushrooms_training <- training(mushrooms_split)
mushrooms_testing <- testing(mushrooms_split)

# define a recipie 
mushrooms_recipie <- 
  recipe(class ~., data = mushrooms_training) %>% 
  step_other(all_predictors(), threshold = 0.05) %>% 
  step_dummy(all_nominal(), -all_outcomes()) 
  
mushrooms_prep <- prep(mushrooms_recipie)

# We will tune our perameters 
tune_spec <- 
  rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune() 
  ) %>%
  set_mode("classification") %>% 
  set_engine("ranger")


# define a workflow 
tune_wf <- 
  workflow() %>% 
  add_recipe(mushrooms_recipie) %>% 
  add_model(tune_spec)

# fold cross validation
set.seed(15)
mushroom_folds <-  vfold_cv(mushrooms_training)

doParallel::registerDoParallel()
set.seed(16)

tune_res <- 
  tune_grid(
  tune_wf, 
  resamples = mushroom_folds, 
  grid = 11
)

# Visualize 

tune_res %>% 
  collect_metrics() %>% 
  filter(.metric == "accuracy") %>% 
  select(mean, min_n, mtry) %>% 
  pivot_longer(min_n:mtry, 
               values_to = "value",
               names_to = "parameter") %>% 
  ggplot(aes(value, mean, color = parameter)) + 
  geom_point() +
  facet_wrap(~ parameter, scales = "free_x")


best_acc <- 
  select_best(tune_res, "accuracy")

best_rf <- 
  finalize_model(
  tune_spec, 
  best_acc
)

finalized_wf <- 
  workflow() %>% 
  add_recipe(mushrooms_recipie) %>% 
  add_model(best_rf)

finalized_res <- 
  finalized_wf %>% 
  last_fit(mushrooms_split)

finalized_res %>% 
  collect_metrics()

finalized_res %>% 
  collect_predictions()

###
