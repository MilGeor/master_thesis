#==============================================================
# Overall Results 
#==============================================================
# Descriptives of Gender
#-------------------------------------------------------  
gender_counts <- table(data$"Gender")
gender_percentages <- (gender_counts / sum(gender_counts)) * 100
#-------------------------------------------------------  
# Descriptives table - means, SDs, correlations (Appendix III, Table A2)
#-------------------------------------------------------  
# Define resilience variables
resilience_vars <- c("AR1","AR1_cont","AR2","AR2_cont",
                     "AR3", "AR3_cont","AR4", "AR4_cont")

# Calculate means and SDs for resilience variables
resilience_means <- sapply(data[resilience_vars], mean, na.rm = TRUE)
resilience_sds <- sapply(data[resilience_vars], sd, na.rm = TRUE)

# Create descr stats for resilience variables
resilience_descr <- data.frame(
  Feature = names(resilience_means),
  Mean = round(resilience_means, 2),
  SD = round(resilience_sds, 2)
)

# Combine descr stats for predictors and resilience vars 
all_descr<- rbind(predictors_descr, resilience_descr)

# Add correlation columns for each resilience variable
all_features <- c(predictors, resilience_vars)

# Calculate all correlations at once and round them
corr_matrix <- cor(data[all_features], 
                   data[resilience_vars], 
                   use = "pairwise.complete.obs")

corr_matrix <- round(corr_matrix, 2)

# Create the final descriptive table
descr_table <- cbind(all_descr, corr_matrix)
print(descr_table)
# Save the table as an Excel file
#write_xlsx(descr_table, path = "descr_table2.xlsx")
#-------------------------------------------------------  
# Distribution plots for the SES measures - Figure 3 
#-------------------------------------------------------  
# Descriptives of the HRL composite scale 
summary(data$`Home Resources for Learning`)
range(data$`Home Resources for Learning`, na.rm = T)
sd(data$`Home Resources for Learning`, na.rm = TRUE)
mean(data$`Home Resources for Learning`, na.rm = TRUE)

# HRL plot 
HRL_plot <- ggplot(data %>% filter(!is.na(`Home Resources for Learning`)),
                   aes(x = as.numeric(`Home Resources for Learning`))) + 
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  labs(title = "Home Resources for Learning",
       x = "HRL Score",
       y = "Frequency") +
  scale_y_continuous(breaks = seq(0, 1500, 500), limits = c(0, 1500)) +
  theme_minimal() +
  theme( 
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5), # Consistent title size and alignment 
    axis.title = element_text(size = 12), # Axis title size 
    axis.text = element_text(size = 10) # Axis text size
  )

# Descriptives of Books subdimension
summary(data$`Books`)
range(data$`Books`, na.rm = T)
sd(data$`Books`, na.rm = TRUE)
table(data$`Books`)

# Books plot
B_plot <- ggplot(data %>% filter(!is.na(`Books`)), 
                 aes(x = factor(Books, levels = 1:5, labels = c("0–10", "11–25", "26–100", "101–200", "200+")))) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Books",
       x = "Number of Books",
       y = "Frequency") +
  scale_y_continuous(breaks = seq(0, 1500, 500), limits = c(0, 1500)) +
  theme_minimal() +
  theme( 
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5), # Consistent title size and alignment 
    axis.title = element_text(size = 12), # Axis title size 
    axis.text = element_text(size = 10) # Axis text size
  )
print(B_plot)

# Descriptives of the Number of HSS subdimension
summary(data$`N of Home Study Supports`)
range(data$`N of Home Study Supports`, na.rm = T)
sd(data$`N of Home Study Supportsooks`, na.rm = TRUE)
table(data$`N of Home Study Supports`)

# HSS plot
HSS_plot <- ggplot(data %>% filter(!is.na(`N of Home Study Supports`)),
                   aes(x = `N of Home Study Supports`)) + 
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  labs(title = "Home Study Supports",
       x = "Number of Home Study Supports",
       y = "Frequency") +
  scale_y_continuous(breaks = seq(0, 3000, 500)) +
  theme_minimal() +
  theme( 
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5), # Consistent title size and alignment 
    axis.title = element_text(size = 12), # Axis title size 
    axis.text = element_text(size = 10) # Axis text size
  )
print(HSS_plot)

# Descriptives of Parental education

summary(data$`Parents' Highest Education`)
range(data$`Parents' Highest Education`, na.rm = T)
sd(data$`Parents' Highest Education`, na.rm = TRUE)
table(data$`Parents' Highest Education`)
sum(is.na(data$`Parents' Highest Education` ))

# ED plot without NA values
ED_plot <- ggplot(data %>% filter(!is.na(`Parents' Highest Education`)), 
                  aes(x = factor(`Parents' Highest Education`))) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Parental Education",
       x = "Education Level",
       y = "Frequency") +
  scale_y_continuous(breaks = seq(0, 2000, 500)) +
  theme_minimal() +
  theme( 
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5), # Consistent title size and alignment 
    axis.title = element_text(size = 12), # Axis title size 
    axis.text = element_text(size = 10) # Axis text size
  )
print(ED_plot)

# Combine the distribution plots together 
SES_plots <- HRL_plot + B_plot + ED_plot + HSS_plot +
  plot_layout(ncol = 4)

print(SES_plots)
#-------------------------------------------------------  
# Proportions (only in dichotomous analyses)
#------------------------------------------------------- 
HRLproportions
HSSproportions
Bproportions
EDproportions
#------------------------------------------------------- 
# Academic Resilience Distributions - Figure 4 (only in continuous analyses)
#-------------------------------------------------------  
# Define a theme style to use in all histograms
theme_AR_hist <- function() {
  theme_minimal(base_family = "Helvetica") +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14), # Centered, bold title
      axis.title = element_text(size = 12), # Axis title size
      axis.text = element_text(size = 10, color = "black"), # Axis text size and color
      panel.grid.major.y = element_line(color = "grey90"), # Light grey horizontal grid lines
      panel.grid.major.x = element_line(color = "grey90"), # Light grey vertical grid lines 
      panel.grid.minor = element_line(color = "grey90"), # Light grey minor grid lines
      panel.background = element_blank(), # White background
      axis.line.x = element_line(color = "black"), # Add x-axis line
      axis.ticks = element_line(color = "black") # Add axis ticks
    )
}

# Academic Resilience distribution in HRL analysis 
AR1_cont_hist <- ggplot(data %>% filter(!is.na(AR1_cont)), aes(x = AR1_cont)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "black") + # Adjust bins as needed
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) + # Ensure y-axis starts near 0
  coord_cartesian(xlim = c(0, 0.8)) + # Set consistent x-axis limits
  labs(
    title = "HRL analysis",
    x = "Continuous Academic Resilience",
    y = "Frequency"
  ) +
  theme_AR_hist () # Apply theme

# Academic Resilience distribution in HSS analysis
AR2_cont_hist <- ggplot(data %>% filter(!is.na(AR2_cont)), aes(x = AR2_cont)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "black") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(xlim = c(0, 0.8)) +
  labs(
    title = "HSS analysis",
    x = "Continuous Academic Resilience",
    y = "Frequency"
  ) +
  theme_AR_hist () 

# Academic Resilience distribution in Books analysis
AR3_cont_hist <- ggplot(data %>% filter(!is.na(AR3_cont)), aes(x = AR3_cont)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "black") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(xlim = c(0, 0.8)) +
  labs(
    title = "Books analysis",
    x = "Continuous Academic Resilience",
    y = "Frequency"
  ) +
  theme_AR_hist () 

# Academic Resilience distribution in ED analysis
AR4_cont_hist <- ggplot(data %>% filter(!is.na(AR4_cont)), aes(x = AR4_cont)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "black") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(xlim = c(0, 0.8)) +
  labs(
    title = "ED analysis",
    x = "Continuous Academic Resilience",
    y = "Frequency"
  ) +
  theme_AR_hist () 

# Combined AR histograms plot
AR_histograms <- AR1_cont_hist + AR2_cont_hist + AR3_cont_hist + AR4_cont_hist +
  plot_layout(ncol = 4, guides = 'collect') 

print(AR_histograms)
#-------------------------------------------------------  
# Evaluation metrics tables 
#-------------------------------------------------------  
# Dichotomous (Table 1)
#--------------#
# Combine dichotomous results in a single list 
all_results <- list(HRL_results, HSS_results, B_results, ED_results)
# Function for extracting & formatting metrics with both mean and sd
format_metric <- function(results, mean_name, sd_name) {
  means <- sapply(results, function(res) res$metrics[[mean_name]])
  sds <- sapply(results, function(res) res$metrics[[sd_name]])
  # Format as "mean (sd)" with 2 decimal places
  return(paste0(format(round(means, 2), nsmall = 2), " (", format(round(sds, 2), nsmall = 2), ")"))
}
# Create formatted metrics with mean (sd)
accuracy <- format_metric(all_results, "mean_accuracy", "sd_accuracy")
precision <- format_metric(all_results, "mean_precision", "sd_precision")
sensitivity <- format_metric(all_results, "mean_sensitivity", "sd_sensitivity")
specificity <- format_metric(all_results, "mean_specificity", "sd_specificity")
f1 <- format_metric(all_results, "mean_f1", "sd_f1")
auc <- format_metric(all_results, "mean_auc", "sd_auc")

# Combine all metrics into a single table
eval_metrics_table <- data.frame(
  Measure = c("HRL", "HSS", "Books", "ED"),
  Accuracy = accuracy,
  Precision = precision,
  Sensitivity = sensitivity,
  Specificity = specificity,
  F1 = f1,
  AUC = auc
)
# Print the final table
print(eval_metrics_table)
#write_xlsx(eval_metrics_table, path = "eval_metrics_table.xlsx")

# Continuous (Table 2)
#--------------#
# Combine all results into a single list
all_results_cont <- list(HRL_results_cont, HSS_results_cont, B_results_cont, ED_results_cont)

# Helper function for extracting &formatting the high resilient subgroup metrics
format_high_metric <- function(results, mean_name, sd_name) {
  means <- sapply(results, function(res) res$high_metrics[[mean_name]])
  sds <- sapply(results, function(res) res$high_metrics[[sd_name]])
  return(paste0(format(round(means, 2), nsmall = 2), " (", format(round(sds, 2), nsmall = 2), ")"))
}

# Format overall metrics with mean (sd)
rmse_overall <- format_metric(all_results_cont, "mean_rmse", "sd_rmse")
mae_overall <- format_metric(all_results_cont, "mean_mae", "sd_mae")
r2_overall <- format_metric(all_results_cont, "mean_r_squared", "sd_r_squared")

# Format high resilient subgroup metrics with mean (sd)
rmse_high <- format_high_metric(all_results_cont, "mean_high_rmse", "sd_high_rmse")
mae_high <- format_high_metric(all_results_cont, "mean_high_mae", "sd_high_mae")
r2_high <- format_high_metric(all_results_cont, "mean_high_r_squared", "sd_high_r_squared")

# Combine all metrics into a single table
eval_metrics_table_cont <- data.frame(
  Measure = c("HRL", "HSS", "Books", "ED"),
  RMSE = rmse_overall,
  MAE = mae_overall,
  R2 = r2_overall,
  High_RMSE = rmse_high,
  High_MAE = mae_high,
  High_R2 = r2_high)

# Print the final table
print(eval_metrics_table_cont)
#write_xlsx(eval_metrics_table, path = "eval_metrics_table_cont.xlsx")

#-------------------------------------------------------  
# Feature Importance figures
#-------------------------------------------------------  
# Dichotomous (Figure 5)
#--------------#
# Convert tibbles to data frames and add indicator columns
HRL_imp <- as.data.frame(HRL_results$importance)
HRL_imp$indicator <- "HRL"

HSS_imp <- as.data.frame(HSS_results$importance)
HSS_imp$indicator <- "HSS"

B_imp <- as.data.frame(B_results$importance)
B_imp$indicator <- "Books"

ED_imp <- as.data.frame(ED_results$importance)
ED_imp$indicator <- "ED"

# Combine all data frames
FI <- rbind(HRL_imp, HSS_imp, B_imp, ED_imp)

# Sort features based on HRL's importance
sorted_features <- HRL_imp$Feature[order(HRL_imp$mean_gain, decreasing = FALSE)]

# Convert Feature to factor with HRL-based order
FI$Feature <- factor(FI$Feature, levels = sorted_features)
FI$indicator <- factor(FI$indicator, levels = c("HRL", "HSS", "Books", "ED"))

# Create FI figure for dichotomous models
FI_figure <- ggplot(FI, aes(mean_gain, Feature, color = indicator)) +
  geom_point(size = 3) +
  labs(title = "",
       x = "Mean gain", 
       y = "Feature",
       color = "SES Measure") +
  theme_minimal() +
  scale_x_continuous(limits = c(0, 0.30))
print (FI_figure)
# Continuous (Figure 6)
#--------------#
# Convert tibbles to data frames and add indicator columns  
HRL_imp_cont <- as.data.frame(HRL_results_cont$importance)
HRL_imp_cont$indicator <- "HRL"

HSS_imp_cont <- as.data.frame(HSS_results_cont$importance)
HSS_imp_cont$indicator <- "HSS"

B_imp_cont <- as.data.frame(B_results_cont$importance)
B_imp_cont$indicator <- "Books"

ED_imp_cont <- as.data.frame(ED_results_cont$importance)
ED_imp_cont$indicator <- "ED"

# Combine all data frames
FI_cont <- rbind(HRL_imp_cont, HSS_imp_cont, B_imp_cont, ED_imp_cont)

# Sort features based on HRL's importance
sorted_features_cont <- HRL_imp_cont$Feature[order(HRL_imp_cont$mean_gain_cont)]

# Convert Feature to factor with HRL-based order
FI_cont$Feature <- factor(FI_cont$Feature, levels = sorted_features_cont)
FI_cont$indicator <- factor(FI_cont$indicator, levels = c("HRL", "HSS", "Books", "ED"))

# Create comparison plot for continuous models
FI_figure_cont <- ggplot(FI_cont, aes(mean_gain_cont, Feature, color = indicator)) +
  geom_point(size = 3) +
  labs(title = "",
       x = "Mean gain", 
       y = "Feature",
       color ="SES Measure") +
  theme_minimal()
print (FI_figure_cont)
#==============================================================


