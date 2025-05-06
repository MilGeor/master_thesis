
#================================================================================#
# Functions
#================================================================================#
# For Upsampling
#------------------------------------------------------
# Function to perform upsampling
upsample_minority_class <- function(X_train, y_train, X_test) {
  
  # Combine features (X_train) and target variable (y_train) into a single data frame
  train_data <- data.frame(X_train, y_train)
  
  # Separate majority (class 0) and minority (class 1) groups
  class_0 <- train_data[train_data$y_train == 0, ]  # Select rows where target == 0
  class_1 <- train_data[train_data$y_train == 1, ]  # Select rows where target == 1
  
  # Perform upsampling: randomly sample with replacement from the minority class
  class_1_upsampled <- class_1[sample(1:nrow(class_1), size = nrow(class_0), replace = TRUE), ]
  
  # Combine the upsampled minority class with the majority class to form a balanced dataset
  balanced_train_data <- rbind(class_0, class_1_upsampled)
  
  # Extract the new resampled X (features) and y (target variable)
  y_upsampled <- balanced_train_data$y_train
  X_upsampled <- balanced_train_data[, colnames(balanced_train_data) != "y_train"]
  
  # Convert X to numeric matrix format for compatibility with ML models
  X_matrix <- as.matrix(data.frame(lapply(X_upsampled, as.numeric)))
  y_vector <- as.numeric(y_upsampled)
  
  # Ensure X_test has matching column names
  colnames(X_matrix) <- colnames(X_test)
  
  # Return resampled data as a list
  return(list(X_upsampled = X_matrix, y_upsampled = y_vector))
}
#------------------------------------------------------
# For Hyperparameter Tuning 
#------------------------------------------------------
# with package caret
tune_params <- function(
    X_train,           # Training features
    y_train,           # Training target
    model_type = c("classification", "regression"), # Type of model
    inner_folds = 3,   # Number of folds for CV
    verbose = TRUE     # Whether to print progress
) {
  # Set random seed
  set.seed(123)
  
  # Validate and process model_type parameter
  model_type <- match.arg(model_type)
  
  # Define hyperparameter grid
  param_grid <- expand.grid(
    max_depth = c(3, 6, 9),
    eta = c(0.01, 0.1),
    nrounds = c(50, 100, 150),
    gamma = 0, # fixed value, not tuned
    colsample_bytree = 1,  # fixed value, not tuned
    min_child_weight = 1,  # fixed value, not tuned
    subsample = 1  # fixed value, not tuned
  )
  
  # Configure settings based on model type
  if (model_type == "classification") {
    # For binary classification
    train_control <- trainControl(
      method = "cv",
      number = inner_folds,
      verboseIter = verbose,
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      allowParallel = TRUE
    )
    
    # Convert target variable to factor for classification 
    y_train <- factor(y_train, 
                      levels = c(0, 1), 
                      labels = c("Zero", "One"))
    # Print the new factor levels
    print(levels(y_train))
    
    metric <- "ROC"
    objective <- "binary:logistic"
    
  } else {
    # For regression
    train_control <- trainControl(
      method = "cv",
      number = inner_folds,
      verboseIter = verbose,
      allowParallel = TRUE
    )
    
    # Ensure target is numeric for regression
    if (!is.numeric(y_train)) {
      stop("Target variable must be numeric for regression")
    }
    
    metric <- "RMSE"
    objective <- "reg:squarederror"
  }
  
  # Train the model using caret's train() function
  model <- train(
    x = as.matrix(X_train),
    y = y_train,
    method = "xgbTree",
    trControl = train_control,
    tuneGrid = param_grid,
    metric = metric,
    objective = objective
  )
  
  # Return the best model and results
  return(list(
    best_model = model,
    best_params = model$bestTune,
    results = model$results
  ))
}


#------------------------------------------------------
# For Training the model
#------------------------------------------------------
train_xgboost_model <- function(X_train, y_train, X_test, y_test,
                                objective, eval_metric ,
                                params = NULL){
  
  # Create DMatrix objects
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgb.DMatrix(data = X_test, label = y_test)
  
  # Default parameters if none provided
  if (is.null(params)) {
    params <- list(
      objective = objective, #binary classification task with logistic regression
      eval_metric = eval_metric, #logarithmic loss for classification, penalizes incorrect predictions
      eta = 0.1, #learning rate, how much the model changes each step
      max_depth = 6, # max depth of a tree, limits N of splits a tree can make
      min_child_weight = 1, #min sum of weights in a child node, how sensitive the model is to outliers
      subsample = 0.8, #fraction of samples for each tree
      colsample_bytree = 0.8 # fraction of features(columns) used when building each tree
    )
  }
  
  # Train the model 
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = params$nrounds, 
    watchlist = list(train = dtrain, test = dtest), 
    early_stopping_rounds = 10, # stops at round 10 if there is no improvement
    verbose = 0
  )
  
  # Store the model and evaluation details
  model_info <- list(
    model = model,
    best_iteration = model$best_iteration,
    best_score = model$best_score,
    best_ntreelimit = model$best_ntreelimit
  )
  
  # Print model summary
  cat("\nPlausible Value:", pv, "- Fold", i, "- Model Summary:\n")
  print(model_info)
  
  return(model_info)
}
#------------------------------------------------------
# For Calculating Evaluation metrics
#------------------------------------------------------
# Functions to calculate dichotomous evaluation metrics:
# Confusion matrix, Accuracy, Precision, Sensitivity, Specificity, F1, AUC

calculate_metrics <- function(true_labels, pred_classes, predictions) {
  
  conf_matrix <- table(Actual = true_labels, Predicted = pred_classes)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  f1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  # Calculate AUC-ROC
  roc_obj <- roc(true_labels, predictions, quiet = TRUE)
  auc_value <- auc(roc_obj)
  
  # Return all metrics as a list
  return(list(
    accuracy = accuracy,
    precision = precision,
    sensitivity = sensitivity,
    specificity = specificity,
    f1 = f1,
    auc = auc_value
  ))
}

# Functions to calculate continuous evaluation metrics:
# RMSE, MAE, R2

calculate_metrics_cont <- function(true_labels, predictions) {
  
  rmse = sqrt(mean((predictions - true_labels)^2))
  mae = mean(abs(predictions - true_labels))
  r_squared = 1 - (sum((true_labels - predictions)^2) /
                     sum((true_labels - mean(true_labels))^2))
  
  # Return all metrics as a list
  return(list(
    rmse = rmse,
    mae = mae, 
    r_squared = r_squared
  ))
}

#------------------------------------------------------
# For Aggregating Results: metrics and feature importance
#------------------------------------------------------
# Function for aggregating and computing final results in dichotomous analysis 
# pools across PVs and computes mean and sd of metrics and feature importance
# returns aggregated metrics, feature importance values and top 10 features plot

compute_and_visualize_results <- function(metrics_list, importance_list) {
  
  # Aggregate results across plausible values
  pooled_metrics <- do.call(rbind, lapply(metrics_list, as.data.frame))
  pooled_importance <- do.call(rbind, importance_list)
  # Rename columns for clarity
  colnames(pooled_metrics) <- c("Fold", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC")
  
  # Compute mean and standard deviation for performance metrics
  final_metrics <- pooled_metrics %>%
    summarise(
      mean_accuracy = mean(Accuracy, na.rm = TRUE),
      sd_accuracy = sd(Accuracy, na.rm = TRUE),
      mean_precision = mean(Precision, na.rm = TRUE),
      sd_precision = sd(Precision, na.rm = TRUE),
      mean_sensitivity = mean(Sensitivity, na.rm = TRUE),
      sd_sensitivity = sd(Sensitivity, na.rm = TRUE),
      mean_specificity = mean(Specificity, na.rm = TRUE),
      sd_specificity = sd(Specificity, na.rm = TRUE),
      mean_f1 = mean(F1, na.rm = TRUE),
      sd_f1 = sd(F1, na.rm = TRUE),
      mean_auc = mean(AUC, na.rm = TRUE),
      sd_auc = sd(AUC, na.rm = TRUE)
    )
  
  # Compute mean and standard deviation for feature importance
  final_importance <- pooled_importance %>%
    group_by(Feature) %>%
    summarise(
      mean_gain = mean(Gain),
      sd_gain = sd(Gain),
      mean_cover = mean(Cover),
      sd_cover = sd(Cover),
      mean_frequency = mean(Frequency),
      sd_frequency = sd(Frequency)
    ) %>%
    arrange(desc(mean_gain)) # Sort by mean_gain for feature ranking
  
  # Print final results
  print(final_metrics)
  print(head(final_importance, 10))
  
  # Create visualization of feature importance
  top_features <- head(final_importance, 10)
  plot <- ggplot(top_features, aes(x = reorder(Feature, mean_gain), y = mean_gain)) +
    geom_bar(stat = "identity", fill = "gray", width = 0.8) +
    geom_text(aes(label = sprintf("%.2f", mean_gain)), hjust = 1.2, size = 3, color = "black") + # Add mean gain values in the bars 
    coord_flip() +
    scale_y_discrete(expand = c(0.1, 0)) +
    theme_minimal() +
    theme(
      text = element_text(family = "Times New Roman"),
      axis.text.x = element_blank(), # Remove x-axis tick labels
      axis.text.y = element_text(size=11,hjust=0), # y-axis text size 
      panel.grid.major = element_blank(), # Remove major grid lines 
      panel.grid.minor = element_blank(), # Remove minor grid lines 
      panel.background = element_blank() # Remove background color 
    ) + 
    labs(title = "Top 10 Most Important Features",
         x = "Features",
         y = "Mean Gain"
    )
  # Return both metrics, feature importances and plots 
  return(list(
    metrics = final_metrics,
    importance = final_importance,
    plot = plot ))
}

# Function for aggregating and computing final results in continuous analysis 
# pools across PVs and computes mean and sd of overall metrics, high-resilience metrics and feature importance
# returns aggregated metrics, high metrics, feature importance values and top 10 features plot

compute_and_visualize_results_cont <- function(high_metrics_list, metrics_list_cont, importance_list_cont) {
  
  # Aggregate results across plausible values
  pooled_metrics_cont <- do.call(rbind,lapply(metrics_list_cont, as.data.frame))
  pooled_high_metrics <- do.call(rbind, lapply(high_metrics_list,as.data.frame))
  # Rename columns for clarity
  colnames(pooled_metrics_cont) <- c("Fold", "RMSE", "MAE", "R_squared")
  colnames(pooled_high_metrics) <- c("Fold", "High_RMSE", "High_MAE", "High_R_squared")
  
  pooled_importance_cont <- do.call(rbind, importance_list_cont)
  
  # Compute mean and standard deviation for overall performance metrics
  final_metrics_cont <- pooled_metrics_cont %>%
    summarise(
      mean_rmse = mean(RMSE),
      sd_rmse = sd(RMSE),
      mean_mae = mean(MAE),
      sd_mae = sd(MAE),
      mean_r_squared = mean(R_squared),
      sd_r_squared = sd(R_squared)
    )
  
  # Compute mean and standard deviation for high-resilience performance metrics
  final_high_metrics <- pooled_high_metrics %>%
    summarise(
      mean_high_rmse = mean(High_RMSE),
      sd_high_rmse = sd(High_RMSE),
      mean_high_mae = mean(High_MAE),
      sd_high_mae = sd(High_MAE),
      mean_high_r_squared = mean(High_R_squared),
      sd_high_r_squared = sd(High_R_squared)
    )
  
  # Compute mean and standard deviation for feature importance
  final_importance_cont <- pooled_importance_cont %>%
    group_by(Feature) %>%
    summarise(
      mean_gain_cont = mean(Gain, na.rm = TRUE),
      sd_gain_cont = sd(Gain, na.rm = TRUE),
      mean_cover_cont = mean(Cover, na.rm = TRUE),
      sd_cover_cont = sd(Cover, na.rm = TRUE),
      mean_frequency_cont = mean(Frequency, na.rm = TRUE),
      sd_frequency_cont = sd(Frequency, na.rm = TRUE)
    ) %>%
    arrange(desc(mean_gain_cont))  # Sort by most important features  
  
  # Print final results
  
  cat("\nAverage Overall Performance Metrics Across Plausible Values:\n")
  print(final_metrics_cont)
  
  cat("\nAverage High-resilience Performance Metrics Across Plasuible Values:\n")
  print(final_high_metrics)
  
  cat("\nTop 10 Most Important Features:\n")
  print(head(final_importance_cont, 10))
  
  
  # Create visualization of feature importance
  top_features_cont <- head(final_importance_cont, 10)
  plot_cont <- ggplot(top_features_cont, aes(x = reorder(Feature, mean_gain_cont), y = mean_gain_cont)) +
    geom_bar(stat = "identity", fill = "gray", width = 0.8) +
    geom_text(aes(label = sprintf("%.2f", mean_gain_cont)), hjust = 1.2, size = 3, color = "black") + # Add mean gain values in the bars 
    coord_flip() +
    scale_y_discrete(expand = c(0.1, 0)) +
    theme_minimal() +
    theme(
      text = element_text(family = "Times New Roman"),
      axis.text.x = element_blank(), # Remove x-axis tick labels
      axis.text.y = element_text(size=11,hjust=0), # y-axis text size 
      panel.grid.major = element_blank(), # Remove major grid lines 
      panel.grid.minor = element_blank(), # Remove minor grid lines 
      panel.background = element_blank() # Remove background color 
    ) + 
    labs(title = "Top 10 Most Important Features",
         x = "Features",
         y = "Mean Gain"
    )
  # Return both metrics, feature importances and plots 
  return(list(
    metrics = final_metrics_cont,
    high_metrics = final_high_metrics,
    importance = final_importance_cont,
    plot = plot_cont ))
}
