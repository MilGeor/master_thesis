#================================================================================#
# Data Analysis
#================================================================================#
#################################################################
######### COMPOSITE SCALE: HOME RESOURCES FOR LEARNING (HRL) ####
#################################################################
#Data Analysis with Dichotomous AR (whole HRL scale)
#-------------------------------------------------------
# Initialize storage for results across PVs

HRLfinal_metrics <- list() # To store performance metrics
HRLfinal_importance <- list() # To store feature importances
HRLproportions <- list() # To store proportions of resilient/non-resilient stundents

# Outer loop over normalized plausible values  
for (pv in plausible_values_norm) {
  
  # Define cut points using quantiles
  HRLcutpoint <- quantile(data$HRL_norm, probs=0.30, na.rm=TRUE)
  HRL_PVcutpoint <- quantile(data[[pv]], probs=0.60, na.rm=TRUE)
  
  # Classify students as resilient-1 or non-resilient-0 for the current PV
  data$AR1 <- ifelse(data$HRL_norm < HRLcutpoint & 
                       data[[pv]] > HRL_PVcutpoint, 1,0)
  
  # Store counts and proportions for the current PV
  HRLproportions [[pv]] <- list(
    Counts = table(data$AR1),
    Proportions = round(prop.table(table(data$AR1)),3)
  )
  # Print the results 
  cat("\nPlausible value", pv, "- Proportion of Resilient(=1) and Non-resilient students:\n" )
  print(HRLproportions[[pv]])
  
  # Define the columns to exclude from the dataset 
  HRLcols_exclude <- c("AR1",
                       "Home Resources for Learning", "HRL_norm",
                       "Books", "Books_norm",
                       "N of Home Study Supports", "HSS_norm",
                       "Parents' Highest Education", "ED_norm",
                       "Student's Age",
                       "Teacher ID and Link", "Teacher Link Number", 
                       "Class ID", "Student ID", "School ID",
                       "PV1", "PV2", "PV3", "PV4", "PV5", 
                       "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                       "Teacher ID.x", "Subject Code.x",
                       "Math Teacher Link", "Grade ID", "Subject ID", 
                       "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
  #and exclude cases which have NA for AR
  HRLfeatures <- as.matrix(data[!is.na(data$AR1), !names(data) %in% HRLcols_exclude])
  
  #Create vector for AR without NAs
  sum(is.na(data$AR1))
  AR1 <- na.omit(data$AR1)
  
  # Create 4 folds for cross-validation
  set.seed(123)
  HRLfolds <- createFolds(c(1:nrow(HRLfeatures)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  HRLimportance_list <<- list() #to store feature importance for each fold
  HRLmodel_list <<- list() #to store models for each fold
  HRLmetrics_list <- matrix(NA, nrow = 4, ncol = 7) # 4 folds, 6 metrics + fold index
  colnames(HRLmetrics_list) <- c("Fold", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC")

  # Inner loop performing cross-validation for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    HRLtest_indices <- HRLfolds[[i]]
    HRLtrain_indices <- unlist(HRLfolds[-i])
    
    HRL_X_train <- HRLfeatures[HRLtrain_indices,]
    HRL_y_train <- AR1[HRLtrain_indices]
    HRL_X_test <- HRLfeatures[HRLtest_indices,]
    HRL_y_test <- AR1[HRLtest_indices]
    
    # Call function to tune parameters
    HRLtuning <- tune_params(
      X_train = HRL_X_train,
      y_train = HRL_y_train,
      model_type = "classification",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    HRLbest_params <- HRLtuning$best_params

    # Call function to upsample the training data and save the resulting upsampled data
    HRLupsampled_data <- upsample_minority_class(HRL_X_train, HRL_y_train, HRL_X_test)
    
    # Extract the upsampled X and y 
    HRL_X_matrix <- HRLupsampled_data$X_upsampled
    HRL_y_vector <- HRLupsampled_data$y_upsampled
    
    # Call function to train the model with the best parameters from the tuning and store the results as HRLmodel
    HRLmodel <- train_xgboost_model(
      X_train = HRL_X_matrix, 
      y_train = HRL_y_vector, 
      X_test = HRL_X_test, 
      y_test = HRL_y_test, 
      objective = "binary:logistic", 
      eval_metric = "logloss",
      params = list(
        max_depth = HRLbest_params$max_depth,
        eta = HRLbest_params$eta,
        nrounds = HRLbest_params$nrounds
      )
    )
    
    # Also store it in the list of models 
    HRLmodel_list[[i]] <- HRLmodel
    
    # Get feature importance and store it
    HRLimportance_list[[i]] <- xgb.importance(feature_names = colnames(HRLfeatures), model = HRLmodel$model)
    
    # Print feature importance for the 10 most important features in the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Feature Importance:\n")
    print(head(HRLimportance_list[[i]], 10))
    
    # Make predictions
    HRLpredictions <- predict(HRLmodel$model, HRL_X_test) # Predict probabilities for each observation in X_test using the trained model 
    HRLpred_classes <- ifelse(HRLpredictions > 0.5, 1, 0) # Classify predictions into two classes 1 and 0, depending if they are > or < 0.5

    # Call function to calculate eval. metrics and store them for the current fold
    #Eval metrics include sensitivity,specificity,accuracy, precision, f1, auc
    HRLmetrics <- calculate_metrics(HRL_y_test, HRLpred_classes, HRLpredictions)
    HRLmetrics_list[i, ] <- c(i, unlist(HRLmetrics))
    
    # Print performance metrics for the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Performance Metrics:\n")
    print(HRLmetrics_list[i, ])
  } # end of inner loop
  
  # Store results for the current plausible value
  HRLfinal_metrics[[pv]] <- HRLmetrics_list
  HRLfinal_importance[[pv]] <- do.call(rbind, HRLimportance_list)

  # Print results for the current Plausible Value
  cat("\nPlausible value", pv, "- Performance Metrics:\n" )
  print(HRLfinal_metrics[[pv]])
  cat("\nPlausible value", pv, "- Variable Importance:\n" )
  print(HRLfinal_importance[[pv]])

} #end of outer loop
#-------------------------------------------------------
# Results Dichotomous Analysis (whole HRL scale)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
HRL_results <- compute_and_visualize_results(HRLfinal_metrics, HRLfinal_importance)

#-------------------------------------------------------
#Data Analysis with Continuous AR (whole HRL scale)
#-------------------------------------------------------
# Initialize storage for results across PVs

HRLfinal_metrics_cont <- list() # To store performance metrics
HRLfinal_importance_cont <- list() # To store feature importances
HRLfinal_high_metrics <- list() # To store high resilience performance metrics 

# Outer loop over normalized plausible values
for (pv in plausible_values_norm)  {
  
  #Create continuous indicator with normalized PV and HRL(SES)
  data$AR1_cont <- data[[pv]] * (1 - data$HRL_norm)
  
  # Histogram of AR1 - showing distribution of resilience 
  hist(data$AR1_cont)
  
  # Define the columns to exclude
  HRLcols_exclude_cont <- c("AR1","AR1_cont", 
                            "Home Resources for Learning", "HRL_norm",
                            "Books", "Books_norm",
                            "N of Home Study Supports", "HSS_norm",
                            "Parents' Highest Education", "ED_norm",
                            "Student's Age",
                            "Teacher ID and Link", "Teacher Link Number",
                            "Class ID", "Student ID", "School ID",
                            "PV1", "PV2", "PV3", "PV4", "PV5",
                            "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                            "Teacher ID.x", "Subject Code.x",
                            "Math Teacher Link", "Grade ID", "Subject ID", 
                            "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
  #and exclude cases which have NA for AR_cont
  HRLfeatures_cont <- as.matrix(data[!is.na(data$AR1_cont), !names(data) %in% HRLcols_exclude_cont])
  
  #Create vector for AR_cont without NAs
  AR1_cont <- na.omit(data$AR1_cont)
  sum(is.na(AR1_cont))
  
  # Create 4 folds for cross-validation
  set.seed(123)
  HRLfolds_cont <- createFolds(c(1:nrow(HRLfeatures_cont)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  HRLimportance_list_cont <- list() #to store feature importance for each fold
  HRLmodel_list_cont <- list() #to store models for each fold
  
  HRLmetrics_list_cont <- matrix(NA, nrow=4, ncol=4) # to store overall evaluation metrics
  colnames(HRLmetrics_list_cont) <- c("Fold", "RMSE", "MAE", "R-squared")
  
  HRLhigh_metrics_list <- matrix(NA, nrow=4, ncol=4) # to store high-resilience evaluation metrics
  colnames(HRLhigh_metrics_list) <- c("Fold", "High RMSE", "High MAE", "High R-squared")
  
  # Inner loop performing CV for folds
  
  for(i in 1:4) {
    
    # Create training and testing sets
    HRLtest_indices_cont <- HRLfolds_cont[[i]]
    HRLtrain_indices_cont <- unlist(HRLfolds_cont[-i])
    
    HRL_X_train_cont <- HRLfeatures_cont[HRLtrain_indices_cont,]
    HRL_y_train_cont <- AR1_cont[HRLtrain_indices_cont]
    HRL_X_test_cont <- HRLfeatures_cont[HRLtest_indices_cont,]
    HRL_y_test_cont <- AR1_cont[HRLtest_indices_cont]
    
    # Call function to tune parameters
    HRLtuning_cont <- tune_params(
      X_train = HRL_X_train_cont,
      y_train = HRL_y_train_cont,
      model_type = "regression",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    HRLbest_params_cont <- HRLtuning_cont$best_params

    # Call function to train the model with best_params_cont and store the results as HRLmodel_cont
    HRLmodel_cont <- train_xgboost_model(
      X_train = HRL_X_train_cont, 
      y_train = HRL_y_train_cont, 
      X_test = HRL_X_test_cont, 
      y_test = HRL_y_test_cont, 
      objective = "reg:squarederror", 
      eval_metric = "rmse",
      params = list(
        max_depth = HRLbest_params_cont$max_depth,
        eta = HRLbest_params_cont$eta,
        nrounds = HRLbest_params_cont$nrounds
      )
    )
    
    # Also store it in the list of models 
    HRLmodel_list_cont[[i]] <- HRLmodel_cont
    
    # Get feature importance and store it
    HRLimportance_list_cont[[i]] <- xgb.importance(feature_names = colnames(HRLfeatures_cont), model = HRLmodel_cont$model)
    
    # Print feature importance for the 10 most important features
    cat("\nPlausible Value:", pv, "- Fold", i, "- Feature Importance:\n")
    print(head(HRLimportance_list_cont[[i]], 10))
    
    # Make predictions
    HRLpredictions_cont <- predict(HRLmodel_cont$model, HRL_X_test_cont) # continuous predictions
    
    # Call function to calculate overall performance metrics
    HRLmetrics_cont <- calculate_metrics_cont (HRL_y_test_cont, HRLpredictions_cont)
    
    # Store overall performance metrics
    HRLmetrics_list_cont[i,] <- c(i, unlist(HRLmetrics_cont))
    
    # Print overall performance metrics for each fold
    cat("\nPlausible Value:", pv, "- Fold", i, "- Overall Performance Metrics:\n")
    print(HRLmetrics_list_cont[i, ])
    
    # High-resilience subgroup analysis
    #---------------------------------------
    # Identify the upper quartile in AR_cont
    HRLupper_quantile_cutoff <- quantile(data$AR1_cont, 0.75, na.rm = TRUE)
    
    # Filter high-resilient students
    HRLhigh_resilience_indices <- which(HRL_y_test_cont >= HRLupper_quantile_cutoff)
    
    # Extract high resilience data for both actual values(y_test) and predicted values(predictions_cont)
    HRLhigh_y_test <- HRL_y_test_cont[HRLhigh_resilience_indices]
    HRLhigh_predictions <- HRLpredictions_cont[HRLhigh_resilience_indices]
    
    # Call function to calculate metrics for high-resilience students
    HRLhigh_metrics <- calculate_metrics_cont (HRLhigh_y_test, HRLhigh_predictions)
    # Store high-resilience metrics
    HRLhigh_metrics_list[i,] <- c(i, unlist(HRLhigh_metrics))
    
    # Print high-resilience performance metrics for each fold 
    cat("\nPlausible Value:",pv, "- Fold", i, "- High-Resilience Performance Metrics:\n")
    print(HRLhigh_metrics_list[i, ])
    
  } #end of inner loop
  
  # Store results for the current plausible value
  HRLfinal_metrics_cont[[pv]] <- HRLmetrics_list_cont
  HRLfinal_importance_cont[[pv]] <- do.call(rbind, HRLimportance_list_cont)
  HRLfinal_high_metrics[[pv]] <- HRLhigh_metrics_list
  
  # Print results for the current Plausible Value
  cat("Plausible value", pv, "- Overall Performance Metrics:\n")
  print(HRLfinal_metrics_cont[[pv]])
  cat("Plausible value", pv, "Variable Importance:\n")
  print(HRLfinal_importance_cont[[pv]])
  cat("Plausible value", pv, "- High-resilience Performance Metrics:\n")
  print(HRLfinal_high_metrics[[pv]])

} #end of outer loop

#-------------------------------------------------------  
# Results Continuous Analysis (whole HRL scale)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
HRL_results_cont <- compute_and_visualize_results_cont (HRLfinal_high_metrics,HRLfinal_metrics_cont, HRLfinal_importance_cont)

#################################################################
######### SUBDIMENSION: HOME STUDY SUPPORT (HSS) ###############
################################################################
#---------------------------------------------------------
#Data Analysis with Dichotomous AR (HSS subdimension)
#-------------------------------------------------------
# Initialize storage for results across PVs
HSSfinal_metrics <- list() # To store performance metrics
HSSfinal_importance <- list() # To store feature importances
HSSproportions <- list() # To store proportions of resilient/non-resilient stundents

# Outer loop over plausible values  
for (pv in plausible_values_norm) {
  
  # Define cut points 
  HSScutpoint <- quantile(data$HRL_norm, probs=0.30, na.rm=TRUE)
  HSS_PVcutpoint <- quantile(data[[pv]], probs=0.60, na.rm=TRUE)
  
  # Classify students as resilient-1 or non-resilient-0 for the current PV
  data$AR2 <- ifelse(data$"HSS_norm"<= HSScutpoint & 
                       data[[pv]] > HSS_PVcutpoint, 1,0)
  # Store counts and proportions for the current PV
  HSSproportions [[pv]] <- list(
    Counts = table(data$AR2),
    Proportions = round(prop.table(table(data$AR2)),3)
  )
  # Print the results 
  cat("\nPlausible value", pv, "- Proportion of Resilient(=1) and Non-resilient students:\n" )
  print(HSSproportions[[pv]])
  
  # Define the columns to exclude from the dataset 
  HSScols_exclude <- c("AR1","AR1_cont","AR2",
                       "Home Resources for Learning", "HRL_norm",
                       "Books", "Books_norm",
                       "N of Home Study Supports", "HSS_norm",
                       "Parents' Highest Education", "ED_norm",
                       "Student's Age",
                       "Teacher ID and Link", "Teacher Link Number", 
                       "Class ID", "Student ID", "School ID",
                       "PV1", "PV2", "PV3", "PV4", "PV5", 
                       "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                       "Teacher ID.x", "Subject Code.x",
                       "Math Teacher Link", "Grade ID", "Subject ID", 
                       "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 18 predictors 
  #and exclude cases which have NA for AR
  HSSfeatures <- as.matrix(data[!is.na(data$AR2), !names(data) %in% HSScols_exclude])
  
  #Create vector for AR without NAs
  AR2 <- na.omit(data$AR2)
  
  # Create 4 folds for cross-validation
  set.seed(123)
  HSSfolds <- createFolds(c(1:nrow(HSSfeatures)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  HSSimportance_list <- list() #to store feature importance for each fold
  HSSmodel_list <- list() #to store models for each fold
  HSSmetrics_list <- matrix(NA, nrow = 4, ncol = 7) # to store performance metrics: 4 folds, 6 metrics + fold index
  colnames(HSSmetrics_list) <- c("Fold", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC")

  # Inner loop performing cross-validation for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    HSStest_indices <- HSSfolds[[i]]
    HSStrain_indices <- unlist(HSSfolds[-i])
    
    HSS_X_train <- HSSfeatures[HSStrain_indices,]
    HSS_y_train <- AR2[HSStrain_indices]
    HSS_X_test <- HSSfeatures[HSStest_indices,]
    HSS_y_test <- AR2[HSStest_indices]
    
    # Call function to tune parameters
    HSStuning <- tune_params(
      X_train = HSS_X_train,
      y_train = HSS_y_train,
      model_type = "classification",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    HSSbest_params <- HSStuning$best_params

    # Call function to upsample the training data
    HSSupsampled_data <- upsample_minority_class(HSS_X_train, HSS_y_train, HSS_X_test)
    
    # Extract the upsampled X and y 
    HSS_X_matrix <- HSSupsampled_data$X_upsampled
    HSS_y_vector <- HSSupsampled_data$y_upsampled
    
    # Call function to train the model with the best parameters and store the results as HSSmodel
    HSSmodel <- train_xgboost_model(
      X_train = HSS_X_matrix, 
      y_train = HSS_y_vector, 
      X_test = HSS_X_test, 
      y_test = HSS_y_test, 
      objective = "binary:logistic", 
      eval_metric = "logloss",
      params = list(
        max_depth = HSSbest_params$max_depth,
        eta = HSSbest_params$eta,
        nrounds = HSSbest_params$nrounds
      )
    )
    
    # Also store it in the list of models 
    HSSmodel_list[[i]] <- HSSmodel
    
    # Get feature importance and store it
    HSSimportance_list[[i]] <- xgb.importance(feature_names = colnames(HSSfeatures), model = HSSmodel$model)
    
    # Print feature importance for the 10 most important features in the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Feature Importance:\n")
    print(head(HSSimportance_list[[i]], 10))
    
    # Make predictions
    HSSpredictions <- predict(HSSmodel$model, HSS_X_test) # Predict probabilities for each observation in X_test using the trained model 
    HSSpred_classes <- ifelse(HSSpredictions > 0.5, 1, 0) # Classify predictions into two classes 1 and 0, depending if they are > or < 0.5

    # Call function to calculate performance metrics and store them for the current fold
    #Eval metrics include sensitivity,specificity,accuracy, precision, f1, auc
    HSSmetrics <- calculate_metrics(HSS_y_test, HSSpred_classes, HSSpredictions)
    HSSmetrics_list[i, ] <- c(i, unlist(HSSmetrics))
    
    # Print performance metrics for the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Performance Metrics:\n")
    print(HSSmetrics_list[i, ])
  } # end of inner loop
  
  # Store results for the current plausible value
  HSSfinal_metrics[[pv]] <- HSSmetrics_list
  HSSfinal_importance[[pv]] <- do.call(rbind, HSSimportance_list)

  # Print results for the current Plausible Value
  cat("\nPlausible value", pv, "- Performance Metrics:\n" )
  print(HSSfinal_metrics[[pv]])
  cat("\nPlausible value", pv, "- Variable Importance:\n" )
  print(HSSfinal_importance[[pv]])

} #end of outer loop
#-------------------------------------------------------
# Results Dichotomous Analysis (HSS subdimension)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
HSS_results <- compute_and_visualize_results(HSSfinal_metrics, HSSfinal_importance)

#-------------------------------------------------------
#Data Analysis with Continuous AR (HSS subdimension)
#-------------------------------------------------------
# Initialize storage for results across PVs

HSSfinal_metrics_cont <- list() # To store performance metrics
HSSfinal_importance_cont <- list() # To store feature importances
HSSfinal_high_metrics <- list() # To store high resilience performance metrics 

# Outer loop over plausible values
for (pv in plausible_values_norm)  {
  
  #Create continuous indicator with normalized PV and HSS(SES)
  data$AR2_cont <- data[[pv]] * (1 - data$HSS_norm)
  
  # Histogram of AR2_cont - showing distribution of continuous resilience 
  hist(data$AR2_cont)
  
  # Define the columns to exclude
  HSScols_exclude_cont <- c("AR1","AR1_cont","AR2","AR2_cont",
                            "Home Resources for Learning", "HRL_norm",
                            "Books", "Books_norm",
                            "N of Home Study Supports", "HSS_norm",
                            "Parents' Highest Education", "ED_norm",
                            "Student's Age",
                            "Teacher ID and Link", "Teacher Link Number",
                            "Class ID", "Student ID", "School ID",
                            "PV1", "PV2", "PV3", "PV4", "PV5",
                            "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                            "Teacher ID.x", "Subject Code.x",
                            "Math Teacher Link", "Grade ID", "Subject ID", 
                            "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
  #and exclude cases which have NA for AR_cont
  HSSfeatures_cont <- as.matrix(data[!is.na(data$AR2_cont), !names(data) %in% HSScols_exclude_cont])
  
  #Create vector for AR_cont without NAs
  AR2_cont <- na.omit(data$AR2_cont)
  
  # Create 4 folds for cross-validation
  #each fold has roughly the same proportion of classes(reislient, non-resilient)
  set.seed(123)
  HSSfolds_cont <- createFolds(c(1:nrow(HSSfeatures_cont)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  HSSimportance_list_cont <- list() #to store feature importance for each fold
  HSSmodel_list_cont <- list() #to store models for each fold
 
  HSSmetrics_list_cont <- matrix(NA, nrow=4, ncol=4) # to store overall evaluation metrics
  colnames(HSSmetrics_list_cont) <- c("Fold", "RMSE", "MAE", "R-squared")
  
  HSShigh_metrics_list <- matrix(NA, nrow=4, ncol=4) # to store high-resilience evaluation metrics
  colnames(HSShigh_metrics_list) <- c("Fold", "High RMSE", "High MAE", "High R-squared")
  

  # Inner loop performing CV for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    HSStest_indices_cont <- HSSfolds_cont[[i]]
    HSStrain_indices_cont <- unlist(HSSfolds_cont[-i])
    
    HSS_X_train_cont <- HSSfeatures_cont[HSStrain_indices_cont,]
    HSS_y_train_cont <- AR2_cont[HSStrain_indices_cont]
    HSS_X_test_cont <- HSSfeatures_cont[HSStest_indices_cont,]
    HSS_y_test_cont <- AR2_cont[HSStest_indices_cont]
    
    # Call function to tune parameters
    HSStuning_cont <- tune_params(
      X_train = HSS_X_train_cont,
      y_train = HSS_y_train_cont,
      model_type = "regression",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    HSSbest_params_cont <- HSStuning_cont$best_params

    # Call function to train the model and store the results as HSSmodel_cont
    HSSmodel_cont <- train_xgboost_model(
      X_train = HSS_X_train_cont, 
      y_train = HSS_y_train_cont, 
      X_test = HSS_X_test_cont, 
      y_test = HSS_y_test_cont, 
      objective = "reg:squarederror", 
      eval_metric = "rmse",
      params = list(
        max_depth = HSSbest_params_cont$max_depth,
        eta = HSSbest_params_cont$eta,
        nrounds = HSSbest_params_cont$nrounds
      )
    )
    
    # Also store it in the list of models 
    HSSmodel_list_cont[[i]] <- HSSmodel_cont
    
    # Get feature importance and store it
    HSSimportance_list_cont[[i]] <- xgb.importance(feature_names = colnames(HSSfeatures_cont), model = HSSmodel_cont$model)
    
    # Print feature importance for the 10 most important features
    cat("\nPlausible Value:", pv, "- Fold", i, "- Feature Importance:\n")
    print(head(HSSimportance_list_cont[[i]], 10))
    
    # Make predictions
    HSSpredictions_cont <- predict(HSSmodel_cont$model, HSS_X_test_cont) # continuous predictions
    
    # Call function to calculate overall performance metrics
    HSSmetrics_cont <- calculate_metrics_cont(HSS_y_test_cont, HSSpredictions_cont)
    
    # Store overall performance metrics
    HSSmetrics_list_cont[i,] <- c(i, unlist(HSSmetrics_cont))
    
    # Print overall performance metrics for current fold
    cat("\nPlausible Value:", pv, "- Fold", i, "- Overall Performance Metrics:\n")
    print(HSSmetrics_list_cont[i, ])
    
    # High-resilience subgroup analysis
    #---------------------------------------
    # Identify the upper quartile in AR2_cont
    HSSupper_quantile_cutoff <- quantile(data$AR2_cont, 0.75, na.rm = TRUE)
    
    # Filter high-resilience students based on the test set
    HSShigh_resilience_indices <- which(HSS_y_test_cont >= HSSupper_quantile_cutoff)
    
    # Extracting high resilience data for both the actual values(y_test) and predicted values (predictions_cont)
    
    HSShigh_y_test <- HSS_y_test_cont[HSShigh_resilience_indices]
    HSShigh_predictions <- HSSpredictions_cont[HSShigh_resilience_indices]
    
    # Call function to calculate metrics for high-resilience students
    HSShigh_metrics <- calculate_metrics_cont (HSShigh_y_test, HSShigh_predictions)
    # Store high-resilience metrics
    HSShigh_metrics_list[i,] <- c(i, unlist(HSShigh_metrics))
    
    # Print high-resilience performance metrics for each fold 
    cat("\nPlausible Value:",pv, "- Fold", i, "- High-Resilience Performance Metrics:\n")
    print(HSShigh_metrics_list[i, ])
    
  } #end of inner loop
  
  # Store results for the current plausible value
  HSSfinal_metrics_cont[[pv]] <- HSSmetrics_list_cont
  HSSfinal_importance_cont[[pv]] <- do.call(rbind, HSSimportance_list_cont)
  HSSfinal_high_metrics[[pv]] <- HSShigh_metrics_list
  
  # Print results for the current Plausible Value
  cat("Plausible value", pv, "- Overall Performance Metrics:\n")
  print(HSSfinal_metrics_cont[[pv]])
  cat("Plausible value", pv, "Variable Importance:\n")
  print(HSSfinal_importance_cont[[pv]])
  cat("Plausible value", pv, "- High-resilience Performance Metrics:\n")
  print(HSSfinal_high_metrics[[pv]])

} #end of outer loop

#-------------------------------------------------------  
# Results Continuous Analysis (HSS subdimension)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
HSS_results_cont <- compute_and_visualize_results_cont (HSSfinal_high_metrics,HSSfinal_metrics_cont, HSSfinal_importance_cont)

##########################################################
######### SUBDIMENSION: N OF BOOKS #######################
########################################################## 
#Data Analysis with Dichotomous AR (Books subdimension)
#-------------------------------------------------------
# Initialize storage for results across PVs
Bfinal_metrics <- list() # To store performance metrics
Bfinal_importance <- list() # To store feature importances
Bproportions <- list() # To store proportions of resilient/non-resilient stundents

# Outer loop over normalised plausible values  
for (pv in plausible_values_norm) {
  
  # Define cut points using quantiles 
  Bcutpoint <- quantile(data$"Books_norm", probs=0.30, na.rm=TRUE)
  B_PVcutpoint <- quantile(data[[pv]], probs=0.60, na.rm=TRUE)
  
  # Classify students as resilient-1 or non-resilient-0 for the current PV
  data$AR3 <- ifelse(data$"Books_norm"< Bcutpoint & 
                       data[[pv]] > B_PVcutpoint, 1,0)
  # Store counts and proportions for the current PV
  Bproportions [[pv]] <- list(
    Counts = table(data$AR3),
    Proportions = round(prop.table(table(data$AR3)),3)
  )
  # Print the results 
  cat("\nPlausible value", pv, "- Proportion of Resilient(=1) and Non-resilient students:\n" )
  print(Bproportions[[pv]])
  
  # Define the columns to exclude from the dataset 
  Bcols_exclude <- c("AR1","AR1_cont","AR2","AR2_cont", "AR3", 
                     "Home Resources for Learning", "HRL_norm",
                     "Books", "Books_norm",
                     "N of Home Study Supports", "HSS_norm",
                     "Parents' Highest Education", "ED_norm",
                     "Student's Age",
                     "Teacher ID and Link", "Teacher Link Number", 
                     "Class ID", "Student ID", "School ID",
                     "PV1", "PV2", "PV3", "PV4", "PV5", 
                     "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                     "Teacher ID.x", "Subject Code.x",
                     "Math Teacher Link", "Grade ID", "Subject ID", 
                     "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
  #and exclude cases which have NA for AR
  Bfeatures <- as.matrix(data[!is.na(data$AR3), !names(data) %in% Bcols_exclude])
  
  #Create vector for AR without NAs
  AR3 <- na.omit(data$AR3)
  
  # Create 4 folds for cross-validation
  set.seed(123)
  Bfolds <- createFolds(c(1:nrow(Bfeatures)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  Bimportance_list <<- list() #to store feature importance for each fold
  Bmodel_list <<- list() #to store models for each fold
  Bmetrics_list <- matrix(NA, nrow = 4, ncol = 7) # 4 folds, 6 metrics + fold index
    colnames(Bmetrics_list) <- c("Fold", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC")

  # Inner loop performing cross-validation for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    Btest_indices <- Bfolds[[i]]
    Btrain_indices <- unlist(Bfolds[-i])
    
    B_X_train <- Bfeatures[Btrain_indices,]
    B_y_train <- AR3[Btrain_indices]
    B_X_test <- Bfeatures[Btest_indices,]
    B_y_test <- AR3[Btest_indices]
    
    # Call function to tune parameters
    Btuning <- tune_params(
      X_train = B_X_train,
      y_train = B_y_train,
      model_type = "classification",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    Bbest_params <- Btuning$best_params

    # Call function to upsample the training data and save the resulting upsampled data
    Bupsampled_data <- upsample_minority_class(B_X_train, B_y_train, B_X_test)
    
    # Extract the upsampled X and y 
    B_X_matrix <- Bupsampled_data$X_upsampled
    B_y_vector <- Bupsampled_data$y_upsampled
    
    # Call function to train the model with the best parameters and store the results as Bmodel
    Bmodel <- train_xgboost_model(
      X_train = B_X_matrix, 
      y_train = B_y_vector, 
      X_test = B_X_test, 
      y_test = B_y_test, 
      objective = "binary:logistic", 
      eval_metric = "logloss",
      params = list(
        max_depth = Bbest_params$max_depth,
        eta = Bbest_params$eta,
        nrounds = Bbest_params$nrounds
      )
    )
    
    # Also store it in the list of models 
    Bmodel_list[[i]] <- Bmodel
    
    # Get feature importance and store it
    Bimportance_list[[i]] <- xgb.importance(feature_names = colnames(Bfeatures), model = Bmodel$model)
    
    # Print feature importance for the 10 most important features in the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Feature Importance:\n")
    print(head(Bimportance_list[[i]], 10))
    
    # Make predictions
    Bpredictions <- predict(Bmodel$model, B_X_test) # Predict probabilities for each observation in X_test using the trained model 
    Bpred_classes <- ifelse(Bpredictions > 0.5, 1, 0) # Classify predictions into two classes 1 and 0, depending if they are > or < 0.5

    # Call function to calculate eval. metrics and store them for the current fold
       #Eval metrics include sensitivity,specificity,accuracy, precision, f1, auc
    Bmetrics <- calculate_metrics(B_y_test, Bpred_classes, Bpredictions)
    Bmetrics_list[i, ] <- c(i, unlist(Bmetrics))
    
    # Print performance metrics for the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Performance Metrics:\n")
    print(Bmetrics_list[i, ])
    
  } # end of inner loop
  
  # Store results for the current plausible value
  Bfinal_metrics[[pv]] <- Bmetrics_list
  Bfinal_importance[[pv]] <- do.call(rbind, Bimportance_list)

  # Print results for the current Plausible Value
  cat("\nPlausible value", pv, "- Performance Metrics:\n" )
  print(Bfinal_metrics[[pv]])
  cat("\nPlausible value", pv, "- Variable Importance:\n" )
  print(Bfinal_importance[[pv]])
} #end of outer loop
#-------------------------------------------------------
# Results Dichotomous Analysis (Books subdimension)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
B_results <- compute_and_visualize_results(Bfinal_metrics, Bfinal_importance)

#-------------------------------------------------------
#Data Analysis with Continuous AR (Books subdimension)
#-------------------------------------------------------
# Initialize storage for results across PVs
Bfinal_metrics_cont <- list() # To store performance metrics
Bfinal_importance_cont <- list() # To store feature importances
Bfinal_high_metrics <- list() # To store high resilience performance metrics 

# Outer loop over normalized plausible values

for (pv in plausible_values_norm)  {
  
  #Create continuous indicator with normalized PV and Books(SES)
  data$AR3_cont <- data[[pv]] * (1 - data$Books_norm)
  
  # Histogram of AR1 - showing distribution of resilience 
  hist(data$AR3_cont)
  
  # Define the columns to exclude
  Bcols_exclude_cont <- c("AR1","AR1_cont","AR2","AR2_cont", "AR3", "AR3_cont",
                          "Home Resources for Learning", "HRL_norm",
                          "Books", "Books_norm",
                          "N of Home Study Supports", "HSS_norm",
                          "Parents' Highest Education", "ED_norm",
                          "Student's Age",
                          "Teacher ID and Link", "Teacher Link Number",
                          "Class ID", "Student ID", "School ID",
                          "PV1", "PV2", "PV3", "PV4", "PV5",
                          "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                          "Teacher ID.x", "Subject Code.x",
                          "Math Teacher Link", "Grade ID", "Subject ID", 
                          "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
  #and exclude cases which have NA for AR_cont
  Bfeatures_cont <- as.matrix(data[!is.na(data$AR3_cont), !names(data) %in% Bcols_exclude_cont])
  
  #Create vector for AR_cont without NAs
  AR3_cont <- na.omit(data$AR3_cont)
  
  # Create 4 folds for cross-validation
  set.seed(123)
  Bfolds_cont <- createFolds(c(1:nrow(Bfeatures_cont)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  Bimportance_list_cont <- list() #to store feature importance for each fold
  Bmodel_list_cont <- list() #to store models for each fold
  
  Bmetrics_list_cont <- matrix(NA, nrow=4, ncol=4) # to store overall evaluation metrics
    colnames(Bmetrics_list_cont) <- c("Fold", "RMSE", "MAE", "R-squared")
  
  Bhigh_metrics_list <- matrix(NA, nrow=4, ncol=4) # to store high-resilience evaluation metrics
    colnames(Bhigh_metrics_list) <- c("Fold", "High RMSE", "High MAE", "High R-squared")
  
  # Inner loop performing CV for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    Btest_indices_cont <- Bfolds_cont[[i]]
    Btrain_indices_cont <- unlist(Bfolds_cont[-i])
    
    B_X_train_cont <- Bfeatures_cont[Btrain_indices_cont,]
    B_y_train_cont <- AR3_cont[Btrain_indices_cont]
    B_X_test_cont <- Bfeatures_cont[Btest_indices_cont,]
    B_y_test_cont <- AR3_cont[Btest_indices_cont]
    
    # Call function to tune parameters
    Btuning_cont <- tune_params(
      X_train = B_X_train_cont,
      y_train = B_y_train_cont,
      model_type = "regression",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    Bbest_params_cont <- Btuning_cont$best_params

    # Call function to train the model with the best parameters and store the results as HRLmodel_cont
    Bmodel_cont <- train_xgboost_model(
      X_train = B_X_train_cont, 
      y_train = B_y_train_cont, 
      X_test = B_X_test_cont, 
      y_test = B_y_test_cont, 
      objective = "reg:squarederror", 
      eval_metric = "rmse",
      params = list(
        max_depth = Bbest_params_cont$max_depth,
        eta = Bbest_params_cont$eta,
        nrounds = Bbest_params_cont$nrounds
      )
    )
    # Also store it in the list of models 
    Bmodel_list_cont[[i]] <- Bmodel_cont
    
    # Get feature importance and store it
    Bimportance_list_cont[[i]] <- xgb.importance(feature_names = colnames(Bfeatures_cont), model = Bmodel_cont$model)
    
    # Print feature importance for the 10 most important features
    cat("\nPlausible Value:", pv, "- Fold", i, "- Feature Importance:\n")
    print(head(Bimportance_list_cont[[i]], 10))
    
    # Make predictions
    Bpredictions_cont <- predict(Bmodel_cont$model, B_X_test_cont) # continuous predictions
    
    # Call function to calculate overall performance metrics
    Bmetrics_cont <- calculate_metrics_cont (B_y_test_cont, Bpredictions_cont)
    
    # Store overall performance metrics
    Bmetrics_list_cont[i,] <- c(i, unlist(Bmetrics_cont))
    
    # Print overall performance metrics for the current fold
    cat("\nPlausible Value:", pv, "- Fold", i, "- Overall Performance Metrics:\n")
    print(Bmetrics_list_cont[i, ])

    # High-resilience subgroup analysis
    #---------------------------------------
    # Identify the upper quartile in AR_cont
    Bupper_quantile_cutoff <- quantile(data$AR3_cont, 0.75, na.rm = TRUE)
    
    # Filter high-resilient students
    Bhigh_resilience_indices <- which(B_y_test_cont >= Bupper_quantile_cutoff)

    # Extract high resilience data for both actual values(y_test) and predicted values(predictions_cont)
    Bhigh_y_test <- B_y_test_cont[Bhigh_resilience_indices]
    Bhigh_predictions <- Bpredictions_cont[Bhigh_resilience_indices]
    
    # Call function to calculate metrics for high-resilience students
    Bhigh_metrics <- calculate_metrics_cont (Bhigh_y_test, Bhigh_predictions)
    # Store high-resilience metrics
    Bhigh_metrics_list[i,] <- c(i, unlist(Bhigh_metrics))
    
    # Print high-resilience performance metrics for each fold 
    cat("\nPlausible Value:",pv, "- Fold", i, "- High-Resilience Performance Metrics:\n")
    print(Bhigh_metrics_list[i, ])
    
  } #end of inner loop
  
  # Store results for the current plausible value
  Bfinal_metrics_cont[[pv]] <- Bmetrics_list_cont
  Bfinal_importance_cont[[pv]] <- do.call(rbind, Bimportance_list_cont)
  Bfinal_high_metrics[[pv]] <- Bhigh_metrics_list

  # Print results for the current Plausible Value
  cat("Plausible value", pv, "- Overall Performance Metrics:\n")
  print(Bfinal_metrics_cont[[pv]])
  cat("Plausible value", pv, "Variable Importance:\n")
  print(Bfinal_importance_cont[[pv]])
  cat("Plausible value", pv, "- High-resilience Performance Metrics:\n")
  print(Bfinal_high_metrics[[pv]])
} #end of outer loop

#-------------------------------------------------------  
# Results Continuous Analysis (Books subdimension)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
B_results_cont <- compute_and_visualize_results_cont (Bfinal_high_metrics,Bfinal_metrics_cont, Bfinal_importance_cont)

###############################################################
######## SUBDIMENSION: PARENTAL EDUCATION (ED) ################
###############################################################
#-------------------------------------------------------
#Data Analysis with Dichotomous AR (Parental Education subdimension)
#-------------------------------------------------------
# Initialize storage for results across PVs

EDfinal_metrics <- list() # To store performance metrics
EDfinal_importance <- list() # To store feature importances
EDproportions <- list() # To store proportions of resilient/non-resilient stundents

# Outer loop over normalised plausible values  
for (pv in plausible_values_norm) {
  
  # Define cut points using quantiles 
  EDcutpoint <- quantile(data$"ED_norm", probs=0.30, na.rm=TRUE)
  ED_PVcutpoint <- quantile(data[[pv]], probs=0.60, na.rm=TRUE)
  
  # Classify students as resilient-1 or non-resilient-0 for the current PV
  data$AR4 <- ifelse(data$"ED_norm"< EDcutpoint & 
                       data[[pv]] > ED_PVcutpoint, 1,0)
  # Store counts and proportions for the current PV
  EDproportions [[pv]] <- list(
    Counts = table(data$AR4),
    Proportions = round(prop.table(table(data$AR4)),3)
  )
  # Print the results 
  cat("\nPlausible value", pv, "- Proportion of Resilient(=1) and Non-resilient students:\n" )
  print(EDproportions[[pv]])
  
  # Define the columns to exclude from the dataset 
  EDcols_exclude <- c("AR1","AR1_cont","AR2","AR2_cont","AR3", "AR3_cont", "AR4",
                      "Home Resources for Learning", "HRL_norm",
                      "Books", "Books_norm",
                      "N of Home Study Supports", "HSS_norm",
                      "Parents' Highest Education", "ED_norm",
                      "Student's Age",
                      "Teacher ID and Link", "Teacher Link Number", 
                      "Class ID", "Student ID", "School ID",
                      "PV1", "PV2", "PV3", "PV4", "PV5", 
                      "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                      "Teacher ID.x", "Subject Code.x",
                      "Math Teacher Link", "Grade ID", "Subject ID", 
                      "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
  #and exclude cases which have NA for AR
  EDfeatures <- as.matrix(data[!is.na(data$AR4), !names(data) %in% EDcols_exclude])
  
  #Create vector for AR without NAs
  AR4 <- na.omit(data$AR4)
  
  # Create 4 folds for cross-validation
  set.seed(123)
  EDfolds <- createFolds(c(1:nrow(EDfeatures)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  EDimportance_list <<- list() #to store feature importance for each fold
  EDmodel_list <<- list() #to store models for each fold
  EDmetrics_list <- matrix(NA, nrow = 4, ncol = 7) # 4 folds, 6 metrics + fold index
    colnames(EDmetrics_list) <- c("Fold", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "AUC")

  # Inner loop performing cross-validation for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    EDtest_indices <- EDfolds[[i]]
    EDtrain_indices <- unlist(EDfolds[-i])
    
    ED_X_train <- EDfeatures[EDtrain_indices,]
    ED_y_train <- AR4[EDtrain_indices]
    ED_X_test <- EDfeatures[EDtest_indices,]
    ED_y_test <- AR4[EDtest_indices]
    
    # Call function to tune parameters
    EDtuning <- tune_params(
      X_train = ED_X_train,
      y_train = ED_y_train,
      model_type = "classification",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    EDbest_params <- EDtuning$best_params

    # Call function to upsample the training data and save the resulting upsampled data
    EDupsampled_data <- upsample_minority_class(ED_X_train, ED_y_train, ED_X_test)
    
    # Extract the upsampled X and y 
    ED_X_matrix <- EDupsampled_data$X_upsampled
    ED_y_vector <- EDupsampled_data$y_upsampled
    
    # Call function to train the model with the best params and store the results as EDmodel
    EDmodel <- train_xgboost_model(
      X_train = ED_X_matrix, 
      y_train = ED_y_vector, 
      X_test = ED_X_test, 
      y_test = ED_y_test, 
      objective = "binary:logistic", 
      eval_metric = "logloss",
      params = list(
        max_depth = EDbest_params$max_depth,
        eta = EDbest_params$eta,
        nrounds = EDbest_params$nrounds
      )
    )
    
    # Also store it in the list of models 
    EDmodel_list[[i]] <- EDmodel
    
    # Get feature importance and store it
    EDimportance_list[[i]] <- xgb.importance(feature_names = colnames(EDfeatures), model = EDmodel$model)
    
    # Print feature importance for the 10 most important features in the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Feature Importance:\n")
    print(head(EDimportance_list[[i]], 10))
    
    # Make predictions
    EDpredictions <- predict(EDmodel$model, ED_X_test) # Predict probabilities for each observation in X_test using the trained model 
    EDpred_classes <- ifelse(EDpredictions > 0.5, 1, 0) # Classify predictions into two classes 1 and 0, depending if they are > or < 0.5
    
    # Call function to calculate eval. metrics and store them for the current fold
         #Eval metrics include sensitivity,specificity,accuracy, precision, f1, auc
    EDmetrics <- calculate_metrics(ED_y_test, EDpred_classes, EDpredictions)
    EDmetrics_list[i, ] <- c(i, unlist(EDmetrics))
    
    # Print performance metrics for the current fold
    cat("\nPlausible Value:", pv, "-Fold", i, "- Performance Metrics:\n")
    print(EDmetrics_list[i, ])
  } # end of inner loop
  
  # Store results for the current plausible value
  EDfinal_metrics[[pv]] <- EDmetrics_list
  EDfinal_importance[[pv]] <- do.call(rbind, EDimportance_list)

  # Print results for the current Plausible Value
  cat("\nPlausible value", pv, "- Performance Metrics:\n" )
  print(EDfinal_metrics[[pv]])
  cat("\nPlausible value", pv, "- Variable Importance:\n" )
  print(EDfinal_importance[[pv]])

} #end of outer loop
#-------------------------------------------------------
# Results Dichotomous analysis (Parents' Education subdimension)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
ED_results <- compute_and_visualize_results(EDfinal_metrics, EDfinal_importance)

#-------------------------------------------------------
#Data Analysis with Continuous AR (Parents' Education subdimension)
#-------------------------------------------------------
# Initialize storage for results across PVs

EDfinal_metrics_cont <- list() # To store performance metrics
EDfinal_importance_cont <- list() # To store feature importances
EDfinal_high_metrics <- list() # To store high resilience performance metrics 

# Outer loop over normalised plausible values

for (pv in plausible_values_norm)  {
  
  # Create continuous indicator with normalized PV and EDSES)
  data$AR4_cont <- data[[pv]] * (1 - data$ED_norm)
  
  # Histogram of continuous AR4 - showing distribution of resilience 
  hist(data$AR4_cont)
  
  # Define the columns to exclude
  EDcols_exclude_cont <- c("AR1","AR1_cont","AR2","AR2_cont",
                           "AR3", "AR3_cont","AR4", "AR4_cont", 
                           "Home Resources for Learning", "HRL_norm",
                           "Books", "Books_norm",
                           "N of Home Study Supports", "HSS_norm",
                           "Parents' Highest Education", "ED_norm",
                           "Student's Age",
                           "Teacher ID and Link", "Teacher Link Number",
                           "Class ID", "Student ID", "School ID",
                           "PV1", "PV2", "PV3", "PV4", "PV5",
                           "PV1_norm", "PV2_norm", "PV3_norm", "PV4_norm", "PV5_norm", 
                           "Teacher ID.x", "Subject Code.x",
                           "Math Teacher Link", "Grade ID", "Subject ID", 
                           "School ID.1", "Teacher ID.y", "Subject Code.y")
  
  # Create features by selecting all columns except those in cols_to_exclude, essentially all 19 predictors 
    #and exclude cases which have NA for AR_cont
  EDfeatures_cont <- as.matrix(data[!is.na(data$AR4_cont), !names(data) %in% EDcols_exclude_cont])
  
  #Create vector for AR_cont without NAs
  AR4_cont <- na.omit(data$AR4_cont)
  
  # Create 4 folds for cross-validation
  set.seed(123)
  EDfolds_cont <- createFolds(c(1:nrow(EDfeatures_cont)), k = 4, list = TRUE)
  
  # Initialize storage across folds
  EDimportance_list_cont <- list() #to store feature importance for each fold
  EDmodel_list_cont <- list() #to store models for each fold
  
  EDmetrics_list_cont <- matrix(NA, nrow=4, ncol=4) # to store overall evaluation metrics
    colnames(EDmetrics_list_cont) <- c("Fold", "RMSE", "MAE", "R-squared")
  
  EDhigh_metrics_list <- matrix(NA, nrow=4, ncol=4) # to store high-resilience evaluation metrics
    colnames(EDhigh_metrics_list) <- c("Fold", "High RMSE", "High MAE", "High R-squared")
  

  # Inner loop performing CV for folds
  for(i in 1:4) {
    
    # Create training and testing sets
    EDtest_indices_cont <- EDfolds_cont[[i]]
    EDtrain_indices_cont <- unlist(EDfolds_cont[-i])
    
    ED_X_train_cont <- EDfeatures_cont[EDtrain_indices_cont,]
    ED_y_train_cont <- AR4_cont[EDtrain_indices_cont]
    ED_X_test_cont <- EDfeatures_cont[EDtest_indices_cont,]
    ED_y_test_cont <- AR4_cont[EDtest_indices_cont]
    
    # Call function to tune parameters
    EDtuning_cont <- tune_params(
      X_train = ED_X_train_cont,
      y_train = ED_y_train_cont,
      model_type = "regression",
      inner_folds = 3,
    )
    # Extract the best parameters found by the tuning 
    EDbest_params_cont <- EDtuning_cont$best_params

    # Call function to train the model with the best params and store the results as EDmodel_cont
    EDmodel_cont <- train_xgboost_model(
      X_train = ED_X_train_cont, 
      y_train = ED_y_train_cont, 
      X_test = ED_X_test_cont, 
      y_test = ED_y_test_cont, 
      objective = "reg:squarederror", 
      eval_metric = "rmse",
      params = list(
        max_depth = EDbest_params_cont$max_depth,
        eta = EDbest_params_cont$eta,
        nrounds = EDbest_params_cont$nrounds
      )
    )
    
    # Also store it in the list of models 
    EDmodel_list_cont[[i]] <- EDmodel_cont
    
    # Get feature importance and store it
    EDimportance_list_cont[[i]] <- xgb.importance(feature_names = colnames(EDfeatures_cont), model = EDmodel_cont$model)
    
    # Print feature importance for the 10 most important features
    cat("\nPlausible Value:", pv, "- Fold", i, "- Feature Importance:\n")
    print(head(EDimportance_list_cont[[i]], 10))
    
    # Make predictions
    EDpredictions_cont <- predict(EDmodel_cont$model, ED_X_test_cont) # continuous predictions
    
    # Call function to calculate overall performance metrics
    EDmetrics_cont <- calculate_metrics_cont (ED_y_test_cont, EDpredictions_cont)
    
    # Store overall performance metrics
    EDmetrics_list_cont[i,] <- c(i, unlist(EDmetrics_cont))
    
    # Print overall performance metrics for each fold
    cat("\nPlausible Value:", pv, "- Fold", i, "- Overall Performance Metrics:\n")
    print(EDmetrics_list_cont[i, ])

    # High-resilience subgroup analysis
    #---------------------------------------
    # Identify the upper quartile in AR_cont
    EDupper_quantile_cutoff <- quantile(data$AR4_cont, 0.75, na.rm = TRUE)
    
    # Filter high-resilient students
    EDhigh_resilience_indices <- which(ED_y_test_cont >= EDupper_quantile_cutoff)

    # Extract high resilience data for both actual values(y_test) and predicted values(predictions_cont)
    EDhigh_y_test <- ED_y_test_cont[EDhigh_resilience_indices]
    EDhigh_predictions <- EDpredictions_cont[EDhigh_resilience_indices]
    
    # Call function to calculate metrics for high-resilience students
    EDhigh_metrics <- calculate_metrics_cont (EDhigh_y_test, EDhigh_predictions)
    # Store high-resilience metrics
    EDhigh_metrics_list[i,] <- c(i, unlist(EDhigh_metrics))
    
    # Print high-resilience performance metrics for each fold 
    cat("\nPlausible Value:",pv, "- Fold", i, "- High-Resilience Performance Metrics:\n")
    print(EDhigh_metrics_list[i, ])
    
  } #end of inner loop
  
  # Store results for the current plausible value 
  EDfinal_metrics_cont[[pv]] <- EDmetrics_list_cont
  EDfinal_importance_cont[[pv]] <- do.call(rbind, EDimportance_list_cont)
  EDfinal_high_metrics[[pv]] <- EDhigh_metrics_list
  
  # Print results for the current Plausible Value
  cat("Plausible value", pv, "- Overall Performance Metrics:\n")
  print(EDfinal_metrics_cont[[pv]])
  cat("Plausible value", pv, "Variable Importance:\n")
  print(EDfinal_importance_cont[[pv]])
  cat("Plausible value", pv, "- High-resilience Performance Metrics:\n")
  print(EDfinal_high_metrics[[pv]])

} #end of outer loop

#-------------------------------------------------------  
# Results Continuous Analysis (Parents' Education subdimension)
#-------------------------------------------------------
# Call function to aggregate, compute and print results across PVs
ED_results_cont <- compute_and_visualize_results_cont (EDfinal_high_metrics,EDfinal_metrics_cont, EDfinal_importance_cont)
