library(keras)

#lecture example, fitting a NN in keras
rm(list = ls())
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
iris <- as.data.frame(iris)
head(iris)

iris_features <- as.matrix(iris[,1:4])
set.seed(123)

ind <- sample(1:2, nrow(iris), replace = T, prob = c(0.67, 0.33))

#target varibale
iris_target <- as.integer(factor(iris$Species)) - 1
# Split features
x_train <- iris_features[ind==1, ]
x_test <- iris_features[ind==2, ]

# Split target
y_train <- iris_target[ind==1]
y_test <- iris_target[ind==2]

#scaling
x_train <- scale(x_train)
x_test <- scale(x_test, center = attr(x_train, "scaled:center"), 
                scale = attr(x_train, "scaled:scale"))
#^scale based off of training data


#one hot encoding 

y_train <- to_categorical(y_train)
y_test_original <- y_test
y_test <- to_categorical(y_test)


dim(y_train)

input <- layer_input(shape = c(4))

output <- input %>% 
  layer_dense(units = 8, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 3, activation = 'softmax')

model <- keras_model(inputs = input, outputs = output)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 5, 
  validation_split = 0.2, shuffle = TRUE
)

model %>% evaluate(x_test, y_test)


#model to classify a an image

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y



# Dimension required is number of observations ×
# image width in pixels ×
# image height in pixels ×
# number of values per pixel
# 
# number of values per pixel = 1 (grayscale) or 3 (RGB)
# each input image is a 28 x 28 x 1 array (a “tensor”)
# scale pixel values to lie in [0,1] by dividing by 255
x_train <- x_train / 255
x_test <- x_test / 255
dim(x_train) <- c(nrow(x_train), 28, 28, 1) 
dim(x_test) <- c(nrow(x_test), 28, 28, 1)

#preprocessing
y_train <- to_categorical(y_train, 10)
y_test_original <- y_test
y_test <- to_categorical(y_test, 10)


input <- layer_input(shape = c(28, 28, 1))

output <- input %>%
  layer_conv_2d(filters = 16, kernel_size = c(3,3)) %>%
  layer_activation('relu') %>%
  layer_dropout(rate = 0.20) %>%
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = 'softmax')

model <- keras_model(inputs = input, outputs = output)


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c('accuracy'),
)


history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 64, 
  validation_split = 0.2, shuffle = TRUE
)



#Classifcation Time series


library(keras3)

set.seed(123)

# ----- 1) Make very simple temporal data -------------------------------------
n <- 300        # number of sequences
L <- 30         # sequence length (timesteps)
F <- 1          # features per timestep (keep it 1 to stay simple)

# Class 0: pure noise
X0 <- array(rnorm(n/2 * L, sd = 0.8), dim = c(n/2, L, F))

# Class 1: noise + a small "bump" between t=11..15
X1 <- array(rnorm(n/2 * L, sd = 0.8), dim = c(n/2, L, F))
X1[, 11:15, 1] <- X1[, 11:15, 1] + 1.5

# Stack and label
X <- abind::abind(X0, X1, along = 1)        # shape (n, L, 1)
y <- c(rep(0L, n/2), rep(1L, n/2))          # 0/1 labels

# Shuffle
idx <- sample(seq_len(n))
X <- X[idx, , , drop = FALSE]
y <- y[idx]

# ----- 2) Train/test split + simple scaling (train stats only) ---------------
train_frac <- 0.7
n_train <- floor(train_frac * n)
X_train <- X[1:n_train, , , drop = FALSE]
X_test  <- X[(n_train+1):n, , , drop = FALSE]
y_train <- y[1:n_train]
y_test  <- y[(n_train+1):n]

# Standardize using training mean/sd (single feature channel)
mu <- mean(X_train)
sd <- sd(X_train); if (sd == 0) sd <- 1
X_train <- (X_train - mu) / sd
X_test  <- (X_test  - mu) / sd

# ----- 3) Minimal 1-D CNN model ---------------------------------------------
input <- layer_input(shape = c(L, F))
output <- input %>%
  layer_conv_1d(filters = 16, kernel_size = 3, padding = "same", activation = "relu") %>%
  layer_global_average_pooling_1d() %>%     # collapses time dimension
  layer_dense(units = 1, activation = "sigmoid")  # binary prob

model <- keras_model(input, output)
model %>% compile(optimizer = "adam",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")

history <- model %>% fit(
  X_train, y_train,
  epochs = 20, batch_size = 32,
  validation_split = 0.2, verbose = 0
)


print(model %>% evaluate(X_test, y_test, verbose = 0))

# Predicted classes and simple accuracy
p <- model %>% predict(X_test)
pred_class <- as.integer(p >= 0.5)
cat("Test accuracy (threshold 0.5):", mean(pred_class == y_test), "\n")




#MULTICLASS EXAMPLE FOR 30 + FEATURES


# =========================
# 0) Setup
# =========================
library(keras3)
set.seed(123)


# =========================
# 1) Make toy data with MANY features
#    (Replace this section with your real data frame)
# =========================
T_total <- 500                  # number of time rows
F_feats <- 30                   # many features
L_win   <- 30                   # window length (timesteps per sample)
K_cls   <- 3                    # number of classes

feature_names <- paste0("x", 1:F_feats)

# A simple data frame: date + 30 features
df <- data.frame(
  date = seq.Date(as.Date("2020-01-01"), by = "day", length.out = T_total),
  matrix(rnorm(T_total * F_feats), nrow = T_total, ncol = F_feats,
         dimnames = list(NULL, feature_names))
)

# Multi-class target (K = 3) built from a few features + noise
# (Your real target would come from your data)
score <- 0.5*df$x1 - 0.4*df$x7 + 0.3*df$x13 + rnorm(T_total, 0, 0.7)
cuts  <- quantile(score, probs = c(1/3, 2/3))
df$y  <- as.integer(
  ifelse(score <= cuts[1], 0L,
         ifelse(score <= cuts[2], 1L, 2L))
)  # classes: 0,1,2


# =========================
# 2) Turn long series into windows (X: (N, L, F); y: length N)
#    Minimal, straightforward loop — no fancy helpers
# =========================
X_long <- as.matrix(df[, feature_names])  # shape (T_total, F_feats)
y_long <- df$y                            # integer labels 0..K-1

# We predict one step ahead (horizon = 1) using the last L_win rows
horizon <- 1

# Number of samples we can form
N_samples <- T_total - L_win - horizon + 1

# Allocate arrays for Keras
X <- array(NA_real_, dim = c(N_samples, L_win, F_feats))  # (N, L, F)
y <- integer(N_samples)                                   # (N,)

# Fill X and y with sliding windows
for (i in seq_len(N_samples)) {
  idx_win <- i:(i + L_win - 1)
  X[i, , ] <- X_long[idx_win, , drop = FALSE]
  y[i]     <- y_long[i + L_win + horizon - 1]
}


# =========================
# 3) Train / test split (index the FIRST dimension only)
# =========================
set.seed(42)
idx <- sample(seq_len(N_samples))
n_train <- floor(0.7 * N_samples)

tr <- idx[1:n_train]
te <- idx[(n_train + 1):N_samples]

X_train <- X[tr, , , drop = FALSE]
X_test  <- X[te, , , drop = FALSE]
y_train <- y[tr]
y_test  <- y[te]      # keep integer labels for evaluation later


# =========================
# 4) Scale features with TRAIN stats only (per feature channel)
#    This standardizes each feature across all samples and timesteps in TRAIN.
# =========================
mu <- apply(X_train, 3, mean)                      # length F_feats
sd <- pmax(apply(X_train, 3, sd), 1e-8)            # avoid /0

for (k in seq_len(dim(X_train)[3])) {
  X_train[ , , k] <- (X_train[ , , k] - mu[k]) / sd[k]
  X_test [ , , k] <- (X_test  [ , , k] - mu[k]) / sd[k]
}


# =========================
# 5) One-hot encode targets for softmax (K classes)
# =========================
y_train_cat <- to_categorical(y_train, num_classes = K_cls)
y_test_cat  <- to_categorical(y_test,  num_classes = K_cls)


# =========================
# 6) Minimal 1-D CNN (Conv1D -> GAP -> Dense K softmax)
# =========================
input <- layer_input(shape = c(dim(X_train)[2], dim(X_train)[3]))   # (L, F)

output <- input %>%
  layer_conv_1d(filters = 32,
                kernel_size = 3,
                padding = "same",
                activation = "relu") %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = K_cls, activation = "softmax")

model <- keras_model(inputs = input, outputs = output)

model %>% compile(
  optimizer = "adam",
  loss      = "categorical_crossentropy",
  metrics   = "accuracy"
)


# =========================
# 7) Train and evaluate
# =========================
history <- model %>% fit(
  X_train, y_train_cat,
  epochs = 15,
  batch_size = 64,
  validation_split = 0.2,
  verbose = 0
)

test_metrics <- model %>% evaluate(X_test, y_test_cat, verbose = 0)
print(test_metrics)

# Predicted class IDs (0..K-1) and simple accuracy
p_test <- model %>% predict(X_test, verbose = 0)          # probs (N_test, K)
pred_id <- apply(p_test, 1, which.max) - 1L               # back to 0..K-1

mean(pred_id == y_test)
