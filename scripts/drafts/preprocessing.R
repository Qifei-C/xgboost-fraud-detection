#Yiixn Xue

library(readr)
library(tidyverse)

data <- read_csv("data/card_transdata.csv") 
head(data)

trim <- function(x, .at){
  x[abs(x) > .at] <- sign(x[abs(x) > .at])*.at
  return(x)
}

# rename and scale
data <- data %>%
  rename(dist_home = distance_from_home,
         dist_last_transact = distance_from_last_transaction,
         ratio_to_med_price = ratio_to_median_purchase_price) %>%
  mutate(across(.cols = c(dist_home, dist_last_transact, ratio_to_med_price), 
                ~ trim(scale((.x))[, 1], .at = 3)))

head(data)  
dim(data)

save(list = 'data', file = 'data/processed/data.RData')
