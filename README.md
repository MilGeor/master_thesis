Appendix II: Analysis Code

Data should be obtained at https://timss2019.org/international-database/. Download TIMSS 2019 Data from Grade 4 Bulgaria (T19_G4_BGR_SAS). 

The repository contains several R script files: component scripts (data_preprocessing.R, functions.R, data analysis.R, results.R) are structured to be sourced and executed by the main.R script. It is required to download or clone the entire repository in order to run the code in main.R. The data file should be inside the same folder as the Rscripts, and make that your working directory. 

•	main.R – Conducts the entire analysis by sourcing the following component scripts in the specific order that they are listed.
•	data_preprocessing.R – Installs and loads necessary packages, loads TIMSS data files, subsets variables of interests and merges them to create the final dataset, calculates descriptive statistics of predictors before standardization, performs data preparation: reverse coding, Z-standardization and min-max normalization.
•	functions.R – Creates custom functions used in analysis for: upsampling, hyperparameter tuning, training the XGBoost model, calculating evaluation metrics, and aggregating results (metrics and feature importance).
•	data_analysis.R – Performs eight separate data analysis, one for each combination of SES measure (HRL, HSS, Books, ED) and construction method (dichotomous continuous).
•	results.R – Gathers lists with proportions across dichotomous analysis, creates descriptive statistics table, SES distribution plots, academic resilience distribution plots, evaluation metrics tables, and feature importance figures.
