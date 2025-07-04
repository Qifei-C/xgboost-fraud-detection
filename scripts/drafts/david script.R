### 1.Load Libraries

library(tidymodels)
library(xgboost)
library(tidyverse)
library(pROC)
library(ggplot2)

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
# Using a watchlist and a test set to select the optimal number of boosting rounds(nrounds), we
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
                    watchlist=watchlist, # Track model performance on train/test
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




