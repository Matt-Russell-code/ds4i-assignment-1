library(ggplot2)
library(maps)
library(dplyr)
library(tidyr)


data <- read.csv("C:/Users/ninal/OneDrive/Desktop/scotland_avalanche_forecasts_2009_2025.csv")


#separate the date and time so that we can just get the year (for initial analysis)
data$Date <- strptime(data$Date, format = "%d/%m/%Y %H:%M")
data$Year <- as.numeric(format(data$Date, "%Y"))

#separate the categorical number and text for the precipitation
clean_data_2009 <- data %>%
  separate(Precip.Code, into = c("Precip.Num", "Precip.Text"), sep = " - ", remove = FALSE) %>%
  filter(Area == "Creag Meagaidh") %>% #use only this area
  select(c(-Date, -Obs, -OAH, -Precip.Text, -Precip.Code, -Year, -AV.Cat)) %>% #take out weird variables
  #select(c(Alt, Aspect, Incline, FAH, -Area)) %>%
  relocate(FAH, .after = last_col()) #wanted this column last

#make factor levels so that the order makes sense
clean_data_2009$FAH <- factor(clean_data_2009$FAH, levels = c("Low", "Moderate", "Considerable -", "Considerable +", "High"))
#change precip number into numerical variable
clean_data_2009$Precip.Num <- as.numeric(clean_data_2009$Precip.Num)


#check if there are any NA values (newsflash- there are LOTS)
na_data <- sapply(clean_data_2009, function(x) sum(is.na(x)))

#take away the rows that have NA as response variables- these are useless for predictions
clean_data_2009 <- clean_data_2009[!is.na(clean_data_2009$FAH), ]


set.seed(1) #set seed so that the training observations and testing observations are replicable

#split into training 80% and testing 20%
train <- sample(nrow(clean_data_2009), size = 0.8*nrow(clean_data_2009)) #training should be 80%
test <- setdiff(1:nrow(clean_data_2009), train) #assign remainder to test

train_data <- clean_data_2009[train, ] #create training data
test_data <- clean_data_2009[test, ] #create testing data

y_train <- clean_data_2009$FAH[train] #just the response variable
proportion_charges_train <- prop.table(table(y_train)) #proportion of each categorical level in response



#perform (VERY PREMATURE H2O NN)
localH2O = h2o.init()

insurance.h2oTrain <- as.h2o(train_data)
insurance.h2oTest <- as.h2o(test_data)

set.seed(1)

insurance.nn <- h2o.deeplearning(x = 2:27 , #features
                                 y = 28, #response variable
                                 training_frame = insurance.h2oTrain, #data in H2O format
                                 validation_frame = insurance.h2oTest,
                                 activation = "Rectifier", #activation function
                                 hidden = c(128, 64, 32), #one hidden layer with 5 neurons
                                 l1 = 1e-5, #regularisation to avoid overfitting
                                 epochs = 100, #training iterations
                                 variable_importances = TRUE, #will calculate feature importances
                                 standardize = TRUE,
                                 seed = 1, 
                                 reproducible = TRUE,
                                 stopping_rounds = 5,
                                 stopping_metric = "logloss",
                                 stopping_tolerance = 0.001,
                                 rate = 0.0005)

#making predictions on testing sets
predictionsTest <- h2o.predict(insurance.nn, insurance.h2oTest)

#convert predictions to factors so it can be compared to true labels
yhatvanilla <- factor(as.vector(predictionsTest$predict), levels = levels(test_data$FAH))

mean(yhatvanilla == test_data$FAH) #VERY LOW!!!!!!!!


#https://www.r-bloggers.com/2022/10/map-any-region-in-the-world-with-r-part-i-the-basic-map/

#Map of Scotland (cant plot points because weird coordinates)
mapdata <- map_data("world")

scotland_data <- mapdata %>%
  filter(subregion %in% c("Great Britain", "Scotland"))

#not_scotland_data <- mapdata %>%
#  filter(subregion != "Scotland")

#unique <- scotland_data %>%
#  distinct(subregion)

#scotland_map <- subset(mapdata, subregion == "Scotland")

big_scotland_map <- ggplot() +
  geom_polygon(data = mapdata,
               aes(x = long, y = lat, group = group),
               color = "#9c9c9c", fill = "#f3f3f3") +
  
  #geom_polygon(data = scotland_data,
  #             aes(x = long, y = lat, group = group),
  #             color = "red", fill = "pink") +
  
  
  geom_point(data = data, aes(x = longitude, y = latitude)) +
  
  coord_map() +
  coord_fixed(1.3, 
              xlim = c(-8, -0.5),
              ylim = c(55, 61)) +
  
  ggtitle("A map of Scotland Avalanches") +
  theme(panel.background = element_rect(fill = "lightblue"))


small_world_map <- ggplot() +
  geom_rect(aes(xmin = -180, xmax = 180, ymin = -90, ymax = 100),
            fill = "white", color = NA) +
  geom_polygon(data = mapdata,
               aes(x = long, y = lat, group = group),
               fill = "grey90", color = "black") +
  geom_polygon(data = scotland_data,
               aes(x = long, y = lat, group = group),
               fill = "red", color = "red") +
  theme_void()


final_plot <- ggdraw() +
  draw_plot(big_scotland_map) +
  draw_plot(small_world_map, x = 0.25, y = 0.65, width = 0.25, height = 0.25)  # inset

final_plot




#there are 6 unique areas
unique_area <- data %>%
  distinct(Area)
unique_area

unique_precip <- data %>%
  distinct(Precip.Code)
unique_precip

unique_rain <- data %>%
  distinct(Rain.at.900)
unique_rain

unique_boot <- data %>%
  distinct(AV.Cat)
unique_boot

#separate data for each area
Creag_data <- data %>%
  filter(Area == "Creag Meagaidh")

Glencoe_data <- data %>%
  filter(Area == "Glencoe")

Lochaber_data <- data %>%
  filter(Area == "Lochaber")

NorthernCair_data <- data %>%
  filter(Area == "Northern Cairngorms")

SouthernCair_data <- data %>%
  filter(Area == "Southern Cairngorms")

Torridon_data <- data %>%
  filter(Area == "Torridon")




