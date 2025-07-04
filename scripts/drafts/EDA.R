#Yixin Xue + Amber Wang

library(readr)
library(ggplot2)
library(tidyverse)
library(corrplot)

data <- read_csv("data/card_transdata.csv") 
head(data)

dim(data) # The dataset contains 1000000 observations

# Missing Value
sum(is.na(data)) # no missing data


# distribution and outliers in each feature
par(mfrow = c(3, 3))

for (col in colnames(data)) {
  hist(data[[col]], main = col, xlab = col, col = "skyblue", border = "black", breaks = 20)
}   # plot histograms

for (col in colnames(data)) {
  density_values <- density(data[[col]])
  plot(density_values, main = col, col = "skyblue", lwd = 2)
}  # Plot the density curve

par(mfrow = c(1, 1))

# Fraud Distribution
data %>%
  ggplot(aes(x = fraud)) +
  geom_bar() +
  labs(title = "Fraud Distribution", x = 'fraud', y = 'count') +
  theme(plot.title = element_text(hjust = 0.5))

data %>% 
  group_by(fraud) %>%
  count()



# We noticed that most transactions (over 90%) are not fraud. So accuracy may not be a proper model evaluation metric. 
# Consider AUC-ROC, f1 score as evaluation metric.
# Consider using stratified sampling/oversampling to handle the imbalanced classification problem

# example oversampling
# data %>%
#   smote(form = fraud ~ ., data = ., perc.over = 200, k = 5)


# Correlation
data %>%
  cor(use = 'everything') %>%
  corrplot(type = 'lower')




