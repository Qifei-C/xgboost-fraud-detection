library(readr)
library(ggplot2)
library(tidyverse)
library(ROSE)
library(dplyr)
library(DMwR2)
library(performanceEstimation)

#create subsampled data

data <- read_csv("data/card_transdata.csv")
View(data)

data %>% 
  initial_split(prop = 0.8, strata = fraud)

set.seed(10000)
subdata <- data %>%
  sample_frac(0.1)

dim(subdata)
subdata %>% 
  count(subdata$fraud)

save(list = 'subdata', file = 'data/raw/subdata.RData')



