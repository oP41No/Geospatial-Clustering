## ----setup, include=FALSE--------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----packages, warning = FALSE, message = FALSE----

library(ggplot2)
library(tidyverse)
library(graphics)
library(mdsr) # install package if not installed
library(discrim) # install package if not installed
library(klaR) # install package if not installed
library(kknn) # install package if not installed
library(utils) # install package if not installed
library(sp) # install package if not installed
library(fs)
library(Shiny)


## ----prepare data----------------------------------
library(tidyverse)
library(mdsr)
url <-
"http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

census <- read_csv(
  url,
  col_names = c(
    "age", "workclass", "fnlwgt", "education", 
    "education_1", "marital_status", "occupation", "relationship", 
    "race", "sex", "capital_gain", "capital_loss", "hours_per_week", 
    "native_country", "income"
  )
) %>%
  mutate(income = factor(income), income_ind = as.numeric(income == ">50K")) # create indicator variable income_ind (0 - low, 1 - high earner)


# look at the structure of the data
glimpse(census)


## ----split data set--------------------------------
library(tidymodels)
set.seed(364)

n <- nrow(census)
census_parts <- census %>%
  initial_split(prop = 0.7)
train <- census_parts %>% training()
test <- census_parts %>% testing()
nrow(test)



## --------------------------------------------------
# if your training set is called `train` this code will produce the correct percentage
pi_bar <- train %>%
  count(income) %>%
  mutate(pct = n / sum(n)) %>%
  filter(income == ">50K") %>%
  pull(pct)

print(c("Percent >50K", pi_bar))


## --------------------------------------------------
library(kknn)

# distance metric only works with quantitative variables, saved them in train_q set
train_q <- train %>% dplyr::select(income, where(is.numeric), -fnlwgt)

# define knn classifier

mod_knn <- nearest_neighbor(neighbors = 5, mode = "classification") %>%
  set_engine("kknn", scale = TRUE) %>%
  fit(income ~ ., data = train_q)

# predict the income using the knn classifier saved in new column called `income_knn`

pred <- train_q %>%
  bind_cols(
    predict(mod_knn, new_data = train_q, type = "class")
  ) %>%
  rename(income_knn = .pred_class)

# print the confusion matrix

pred %>%
  conf_mat(income, income_knn)


## --------------------------------------------------

# Find the Accuracy = (true positive and true negative)/total or use the `accuracy()` function.

pred %>%
  accuracy(income, income_knn)



## ----model form------------------------------------
form <- as.formula(
  "income ~ age + workclass + education + marital_status + 
  occupation + relationship + race + sex + 
  capital_gain + capital_loss + hours_per_week"
)

form


## ----warning = FALSE-------------------------------
library(discrim)

# create naiveBayes classifier

mod_nb <- naive_Bayes(mode = "classification") %>%
  set_engine("klaR") %>%
  fit(form, data = train)


# use the predict method with the mod_nb model

pred <- train %>%  
  bind_cols(
    predict(mod_nb, new_data = train, type = "class")
  ) %>%
  rename(income_nb = .pred_class)


# confusion matrix

pred %>%
  conf_mat(income, income_nb)

# accuracy

pred %>% accuracy(income, income_nb)



## --------------------------------------------------

# create plot of income_ind vs age/education.num/sex/marital status

log_plot <- ggplot(data = census, aes(x = age, y = income_ind)) + 
  geom_jitter(alpha = 0.1, height = 0.05) + 
  geom_smooth(method = "glm", method.args = list(family = "binomial")) + 
  ylab("Earner Status")

log_plot + xlab("Age (in years)")

log_plot + aes(x = education_1) +
   xlab("Education level (1-16, 16 is the highest)")

log_plot + aes(x = sex) +
   xlab("Gender")

log_plot + aes(x = marital_status) +
   xlab("Marital Status")



## --------------------------------------------------
logreg <- glm(income_ind ~ age + sex + education_1, family = "binomial", data = train) 
# 
tidy(logreg)



## ----warning=FALSE---------------------------------
# # use the predict method with the logreg model, below are predicted probability
#
logit_pred_prob <- predict(logreg, newdata = train, type = "response")

# assign 1/0 based on logit_pred_prob > 0.5. This is predicted high-earner status "yes". You can define different cutoff value if preferred.

pred_y <- as.numeric(logit_pred_prob > 0.5)

# confusion matrix
confusion <- table(pred_y, train$income_ind)

confusion

# accuracy
mean(pred_y == train$income_ind, na.rm = TRUE)



## --------------------------------------------------
# use the predict method with the logreg model, below are predicted probability

logit_pred_prob <- predict(logreg, newdata = test, type = "response")

# assign 1/0 based on logit_pred_prob > 0.5. This is predicted high-earner status "yes". You can define different cutoff value if preferred.

pred_y <- as.numeric(logit_pred_prob > 0.5)

# confusion matrix

confusion <- table(pred_y, test$income_ind)

confusion


# accuracy

mean(pred_y == test$income_ind, na.rm = TRUE)


## ----Models----------------------------------------

# load the readxl package to read the xlsx file in R
library(readxl)

filename <- "data/2016 FEGuide.xlsx" # you may need to adjust the path if you opt out to store the data file in different location


# use read_excel function to read the file by using the path stored in the filename 
cars <- read_excel(filename) %>% 
  janitor::clean_names() %>%
  dplyr::select(
    make = mfr_name, 
    model = carline, 
    displacement = eng_displ,
    number_cyl,
    number_gears,
    city_mpg = city_fe_guide_conventional_fuel,
    hwy_mpg = hwy_fe_guide_conventional_fuel
  ) %>%
  distinct(model, .keep_all = TRUE) %>% 
  filter(make == "Toyota") # filter Toyota vehicles only

# have a look at the data
glimpse(cars)



## --------------------------------------------------
car_diffs <- cars %>%
  column_to_rownames(var = "model") %>%
  dist()
str(car_diffs)


## --------------------------------------------------
car_mat <- car_diffs %>% as.matrix() 
car_mat[1:6, 1:6] %>% round(digits = 2)


## ----clustering, warning = FALSE, fig.height = 14, fig.width = 8----
#install if not installed
# install.packages("ape")

library(ape) 
car_diffs %>% 
  hclust() %>% 
  as.phylo() %>% 
  plot(cex = 0.9, label.offset = 1)



## ----cars clustering, warning = FALSE, fig.height = 14, fig.width = 8----

library(ape) 

cars2 <- read_excel(filename) %>% 
  janitor::clean_names() %>%
  dplyr::select(
    make = mfr_name, 
    model = carline, 
    displacement = eng_displ,
    number_cyl,
    number_gears,
    city_mpg = city_fe_guide_conventional_fuel,
    hwy_mpg = hwy_fe_guide_conventional_fuel
  ) %>%
  distinct(model, .keep_all = TRUE) %>% 
  filter(make == "Mercedes-Benz") # filter Mercedes-Benz vehicles only

car_diffs2 <- cars2 %>%
  column_to_rownames(var = "model") %>%
  dist()
str(car_diffs2)

car_mat <- car_diffs %>% as.matrix() 
car_mat[1:6, 1:6] %>% round(digits = 2)


car_diffs2 %>% 
  hclust() %>% 
  as.phylo() %>% 
  plot(cex = 0.9, label.offset = 1)



## ----selectCities----------------------------------
BigCities <- world_cities %>% arrange(desc(population)) %>% 
  head(4000) %>% 
  dplyr::select(longitude, latitude)
glimpse(BigCities)


## --------------------------------------------------

set.seed(15)
# install the package first if not installed
#install.packages("mclust")
library(mclust) 

# form 6 cluster iteratively
city_clusts <- BigCities %>%
kmeans(centers = 6) %>% fitted("classes") %>% as.character()

# form 6 cluster iteratively, by forming initially 10 random sets
km <- kmeans(BigCities, centers = 6, nstart = 10)
# inspect the structure of the kmeans output cluster object
str(km)

# access two important features of cluster, their size and centers
km$size
km$centers

BigCities <- BigCities %>% mutate(cluster = city_clusts) 

# graph the clusters, using the cluster variable to pick the color in standard cartesian coordinate system
BigCities %>% ggplot(aes(x = longitude, y = latitude)) +
geom_point(aes(color = cluster), alpha = 0.5)



## ----Spatial Object--------------------------------

# assign the BigCities data.frame to a working data.frame object d 
d <- BigCities #or BigCities[,c('longitude', 'latitude')]

# create spatial object from d
coordinates(d) <- 1:2

# Set WGS 84 (EPSG:4326) standard for projecting longitude latitude coordinates
proj4string(d) <- CRS("+init=epsg:4326")

# coordinate reference system using the EPSG:4326 standard
CRS.new <- CRS("+init=epsg:4326")

# the d object in the new CRS, you may print out few records to see how it looks in the new CRS
d.new <- spTransform(d, CRS.new)

# just for information review the 
proj4string(d.new) %>% strwrap()


# form 6 cluster iteratively
city_clusts <- as.data.frame(d.new) %>%
kmeans(centers = 6) %>% fitted("classes") %>% as.character()

# add a variable for the newly formed clusters
df.new <- as.data.frame(d.new) %>% mutate(cluster = city_clusts, longitude = coords.x1, latitude = coords.x2)

# graph the clusters, using the cluster variable to pick the color
df.new %>% ggplot(aes(x = coords.x1, y = coords.x2)) +
geom_point(aes(color = cluster), alpha = 0.5) +
scale_color_brewer(palette = "Set3")



## ----Spatial Object 2------------------------------

# assign the BigCities data.frame to a working data.frame object d 
d <- BigCities #or BigCities[,c('longitude', 'latitude')]

# create spatial object from d
coordinates(d) <- 1:2

# Set Mercator projection (EPSG:3857) standard for projecting longitude latitude coordinates
proj4string(d) <- CRS("+init=epsg:3857")

# coordinate reference system using the EPSG:3857 standard
CRS.new <- CRS("+init=epsg:3857")

# the d object in the new CRS, you may print out few records to see how it looks in the new CRS
d.new <- spTransform(d, CRS.new)

# just for information review the 
proj4string(d.new) %>% strwrap()


# form 6 cluster iteratively
city_clusts <- as.data.frame(d.new) %>%
kmeans(centers = 6) %>% fitted("classes") %>% as.character()

# add a variable for the newly formed clusters
df.new <- as.data.frame(d.new) %>% mutate(cluster = city_clusts, longitude = coords.x1, latitude = coords.x2)

# graph the clusters, using the cluster variable to pick the color
df.new %>% ggplot(aes(x = coords.x1, y = coords.x2)) +
geom_point(aes(color = cluster), alpha = 0.5) +
scale_color_brewer(palette = "Set3")


## ----Spatial Object 3------------------------------

# assign the BigCities data.frame to a working data.frame object d 
d <- BigCities #or BigCities[,c('longitude', 'latitude')]

# create spatial object from d
coordinates(d) <- 1:2

# Set NAD83 (EPSG:4269) standard for projecting longitude latitude coordinates
proj4string(d) <- CRS("+init=epsg:4269")

# coordinate reference system using the EPSG:4269 standard
CRS.new <- CRS("+init=epsg:4269")

# the d object in the new CRS, you may print out few records to see how it looks in the new CRS
d.new <- spTransform(d, CRS.new)

# just for information review the 
proj4string(d.new) %>% strwrap()


# form 6 cluster iteratively
city_clusts <- as.data.frame(d.new) %>%
kmeans(centers = 6) %>% fitted("classes") %>% as.character()

# add a variable for the newly formed clusters
df.new <- as.data.frame(d.new) %>% mutate(cluster = city_clusts, longitude = coords.x1, latitude = coords.x2)

# graph the clusters, using the cluster variable to pick the color
df.new %>% ggplot(aes(x = coords.x1, y = coords.x2)) +
geom_point(aes(color = cluster), alpha = 0.5) +
scale_color_brewer(palette = "Set3")

