### 1.Load Libraries and import dataset

library(tidymodels)
library(xgboost)
library(tidyverse)
library(pROC)
library(ggplot2)
# install.packages(c("devtools"))
devtools::install_github("ldurazo/kaggler", force = TRUE)
library(tidyverse)
library(dplyr)
library(readr)
library(kaggler)

# check file structure
root_dir <- rprojroot::find_rstudio_root_file()
setwd(root_dir)
if (!dir.exists("./data")){dir.create("./data")} # create "./data" is not exist and set up data directory
data_dir <- file.path(root_dir, "data")

# setting up kaggle API
kgl_auth(creds_file = './scripts/kaggle.json')

# download and import
response <- kgl_datasets_download_all(owner_dataset = "dhanushnarayananr/credit-card-fraud/")
download.file(response[["url"]], "./data/temp.zip", mode="wb")
unzip("data/temp.zip", exdir = "./data/", overwrite = TRUE)
data_file <- "./data/card_transdata.csv"
data <- read_csv(data_file)

### 2.Data Cleaning:

# read in raw data
data <- read_csv("data/card_transdata.csv") 
head(data)

trim <- function(x, .at){
  x[abs(x) > .at] <- sign(x[abs(x) > .at])*.at
  return(x)
}

# rename and scale: since some variable are on distance scale/some are ratio
data <- data %>%
  rename(dist_home = distance_from_home,
         dist_last_transact = distance_from_last_transaction,
         ratio_to_med_price = ratio_to_median_purchase_price) %>%
  mutate(across(.cols = c(dist_home, dist_last_transact, ratio_to_med_price), 
                ~ trim(scale((.x))[, 1], .at = 3)))

# One-Hot Encoding(EXPLAIN IN WRITE-UP)
# XGBoost only accept numerical value, so convert categorical to numeric by one-hot encoding
# when doing write up, refer to: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# we don't have to worry about it since all variables in our data set in numeric


### 3.Splitting Data

#set seed for reproducibility
set.seed(197)
#split training/validation/testing
data.split <- data %>% 
  initial_split(prop = 0.6, strata = fraud)

test.split <- initial_split(testing(data.split), prop = 0.5, strata = fraud)
data.val <- training(test.split)
data.test <- testing(test.split)

### 4.Data Preparation

#define predictor and response variables in training set
# NOTE: XGBoost only use matrix data
train.x <-  data.matrix(training(data.split) %>% select(-fraud))
train.y <-  training(data.split) %>% pull(fraud)

#define predictor and response variables in validation set
val.x <-  data.matrix(data.val %>% select(-fraud))
val.y <-  data.val %>% pull(fraud)

#define predictor and response variables in testing set
test.x <-  data.matrix(data.test %>% select(-fraud))
test.y <-  data.test %>% pull(fraud)

#define xgb.DMatirx: This is a specialized data structure that xgboost uses for efficiency
xgb.train <-  xgb.DMatrix(data = train.x, label = train.y)
xgb.val <-  xgb.DMatrix(data = val.x, label = val.y)
xgb.test <-  xgb.DMatrix(data = test.x, label = test.y)

### 5. Model Fitting

# there is a 'nrounds' parameter in the xgboost(), which specifies the number of 
# boosting iterations to be run while training the XGBoost model. In each round,
# a new decision tree is added to the model to correct the mistakes made by trees in the previous rounds
# large dataset usually need more rounds
# we don't want overfitting, so use watchlist to keep an eye on test-loss

# max.depth is how deep to grow the individual decision trees. We typically use small number like 2 or 3, as
# this approach tends to produce more accurate models.

#Define watchlist:
# Using a watchlist and a validation set to select the optimal number of boosting rounds(nrounds), we
# track the performance of the model on both the training and validation datasets during the training process.
# This method helps in determining the point at which the model starts to overfit, and we should stop there
watchlist = list(train=xgb.train, validation=xgb.val)

params <- list(
  objective = "binary:logistic", # For binary classification problem, use this to predict probability
  eta = 0.3 # learning rate
)

#fit XGBoost model and display training and testing data at each round
model <-  xgb.train(params = params, 
                    data = xgb.train, # Training data
                    max.depth = 3, # Size of each individual tree
                    watchlist=watchlist, # Track model performance on train/validation
                    nrounds = 500, # Number of boosting iterations
                    early_stopping_rounds = 50) # Number of iterations we will wait for the next decrease

# Typically, find the number of rounds where test-loss is the lowest and the afterwards test-loss start to increase,
# which means overfitting. Many machine learning models in R have a call back parameter that stops the iterations if the
# amount of accuracy increased is smaller than certain threshold you specified. But XGBoost doesn't have it. Instead,
# it uses a similar approach , which is called 'early_stopping_rounds' to determine when to stop. If the model's performance
# on the validation set hasn't improved for the specified number of consecutive rounds, the training stops.
# We usually use 10% of total rounds as the value of early_stopping_rounds

# From result we noticed, 261 rounds looks good(SET SEED = 197, PLEASE VERIFY)

# Define final model
# The argument verbose = 0 tells R not to display the training and testing error for each round.
final <-  xgboost(params = params, data = xgb.train, max.depth = 3, nrounds = 261, verbose = 0)

### Feature Importance
# WRITE-UP refer to: https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html
importance <- xgb.importance(feature_names = colnames(train.x), model = final)
head(importance)

# plot
# 3 most important features
xgb.plot.importance(importance_matrix = importance, top_n = 5)

### 6. Prediction and Accuracy Measure
# Use model to make predictions on test data
pred.y <- predict(final, test.x)

# Label test data according to the predicted probability
pred.label <- ifelse(pred.y > 0.5, 1, 0)

# Confusion Matrix
confusion.matrix <- table(Predicted = pred.label, Actual = test.y)
print(confusion.matrix)

# AUC-ROC
roc <- roc(test.y, pred.label)
auc <- auc(roc)
print(auc)

# Visualization
ggroc(roc) +
  labs(title = "ROC Curve", x = "False Positive Rate", y = "True Positive Rate") + 
  annotate("text", x = 0.2, y = 0.8, label = paste("AUC =", round(auc, 5)))

####################### This part left for advanced model analysis ####################################################
### One hot coding part ###
# One-hot encoding function using tidyverse
one_hot_encode <- function(data, column_name) {
  data %>%
    separate_rows({{ column_name }}, sep = ";") %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = {{ column_name }}, values_from = value, names_prefix = paste0(column_name, "."), values_fill = 0) %>%
    group_by(response.id) %>%
    summarise(across(starts_with(paste0(column_name, ".")), max, .names = "{.col}"))
}


### k-fold cross-validation

# Number of folds for cross-validation
k <- 5

# Create k-fold cross-validation indices
folds <- vfold_cv(data, v = k, strata = fraud)

# Prepare a list to store xgb.DMatrix data for each fold
xgb_folds <- list()

# Loop
for(i in 1:k){
  # Split data into training and validation sets
  train_fold <- training(folds$splits[[i]])
  val_fold <- testing(folds$splits[[i]])
  
  # Define predictor and response variables
  train.x <- data.matrix(train_fold %>% select(-fraud))
  train.y <- train_fold %>% pull(fraud)
  
  # Define predictor and response variables for validation
  val.x <- data.matrix(val_fold %>% select(-fraud))
  val.y <- val_fold %>% pull(fraud)
  
  # Create xgb.DMatrix for the current fold
  xgb_train_fold <- xgb.DMatrix(data = train.x, label = train.y)
  xgb_val_fold <- xgb.DMatrix(data = val.x, label = val.y)
  
  # Store in the list
  xgb_folds[[i]] <- list(train = xgb_train_fold, val = xgb_val_fold)
}


### Grid Search Part
# Define the parameter grid
param_grid <- expand.grid(
  eta = c(0.01, 0.1, 0.3),
  max_depth = c(1:10),
  gamma = c(0, 1, 5)
)

# Initialize a variable to track the best score
best_score <- -Inf

# Initialize a variable to store the best model
best_model <- NULL

# Initialize a variable to store the best parameters
best_params <- NULL

## Metric Analysis Structure
performance_metric <- function(predictions, actuals) {
  # Replace this with a spectific metric calculation
  # For example, this could be F1 score, precision, recall, etc.
  calculated_metric <- 0
  return(calculated_metric)
}

## Training and Validification
# Loop over all combinations of parameters
for(i in 1:nrow(param_grid)){
  # Extract parameters for this iteration
  params <- param_grid[i, ]
  
  # List to store cross-validation scores
  cv_scores <- numeric(length(xgb_folds))
  
  # Perform k-fold cross-validation
  for(j in 1:length(xgb_folds)){
    # Define the watchlist for the current fold
    watchlist <- list(train = xgb_folds[[j]]$train, validation = xgb_folds[[j]]$val)
    
    # Train the model on the current fold
    model <- xgb.train(params = as.list(params),
                       data = xgb_folds[[j]]$train,
                       nrounds = 100, # Adjust as necessary
                       watchlist = watchlist,
                       early_stopping_rounds = 50)
    
    # Evaluate the model on the validation set
    pred <- predict(model, xgb_folds[[j]]$val)
    
    actual_vals <- getinfo(xgb_folds[[j]]$val, "label")
    score <- roc(actual_vals, pred)$auc
    
    
    # Store the score
    cv_scores[j] <- score
  }
  
  # Compute the average score across all folds
  avg_score <- mean(cv_scores)
  
  # Update best score, model, and parameters if current model is better
  if(avg_score > best_score){
    best_score <- avg_score
    best_model <- model
    best_params <- params
  }
}

# best_model now contains the model with the best parameters
# best_params contains the corresponding parameters


