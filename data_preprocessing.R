#================================================================================#
# Installing and Loading Packages
#================================================================================#
#------------------------------------------------------
#install.packages("haven")
#install.packages("sas7bdat")
#install.packages("xgboost")
#install.packages("caret")
#install.packages("pROC")
#install.packages("writexl") # for saving as an excel file 
#install.packages ("ggplot2")
#install.packages ("patchwork") # for combining plots

library(haven)
library("sas7bdat")
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)
library(pROC)
library(writexl)
library (patchwork)
#------------------------------------------------------
#================================================================================#
# Load data for Grade 4, Bulgaria, TIMSS 2019, and convert them to data frames 
#================================================================================#
#------------------------------------------------------
# Data available at https://timss2019.org/international-database/. 
# Download TIMSS 2019 Data from Grade 4 Bulgaria (T19_G4_BGR_SAS).

#student context file ASG 

st_cont <- read_sas ("asgbgrm7.sas7bdat")
st_cont <- as.data.frame(st_cont)

#school context file ACG: contains

sch_cont <- read_sas ("acgbgrm7.sas7bdat")
sch_cont <- as.data.frame(sch_cont)

#home context file ASH: 

home_cont <- read_sas ("ashbgrm7.sas7bdat")
home_cont <- as.data.frame(home_cont)

#teacher context file ATG 

teach_cont <- read_sas ("atgbgrm7.sas7bdat")
teach_cont <- as.data.frame(teach_cont)

#student-teacher linkage file BST
stu_teach_link <- read_sas ("astbgrm7.sas7bdat")
stu_teach_link <- as.data.frame(stu_teach_link)
#------------------------------------------------------
#================================================================================#
# Creating the dataset 
#================================================================================#
####### Subsetting data #####
#subsetting variables from teacher context ATG file 

teach_var <- teach_cont[,c("IDSCHOOL","IDTEALIN","IDTEACH","IDLINK","ITCOURSE",
                           "ATBM01","ATDMNUM", "ATDMGEO","ATDMDAT","ATBGLSN",
                           "ATBM02B", "ATBM02D", "ATBM02E", "ATBM02F")]

# Reverse-code the individual items: 1 -> 4, 2 -> 3, 3 -> 2, 4 -> 1
teach_var$ATBM02B <- 5 - teach_var$ATBM02B
teach_var$ATBM02D <- 5 - teach_var$ATBM02D
teach_var$ATBM02E <- 5 - teach_var$ATBM02E
teach_var$ATBM02F <- 5 - teach_var$ATBM02F

# Create a new variable Teacher Emphasis on Math Investigation scale by summing up those individual items
teach_var$"Emphasis on Math Investigation" <- rowSums(teach_var[, c("ATBM02B", "ATBM02D", "ATBM02E", "ATBM02F")])

# remove the individual items as we no longer need them
teach_var <- teach_var[, !colnames(teach_var) %in% c("ATBM02B", "ATBM02D", "ATBM02E", "ATBM02F")]

#subsetting variables from school context ACG file 

sch_var <- sch_cont[,c("IDSCHOOL","ACBGMRS","ACBGEAS","ACBGDAS","ACDGSBC","ACBG04")]

#subsetting variables from student context ASG file

st_var <- st_cont[,c("IDSCHOOL","IDCLASS","IDSTUD","ASBG01","ASBG03", "ASBG04",
                     "ASDG05S",
                     "ASBGICM", "ASBM01","ASBGSLM","ASBGSCM","ASBGSSB","ASBG08","ASBGHRL",
                     "ASMMAT01","ASMMAT02","ASMMAT03","ASMMAT04","ASMMAT05")]

#subsetting variables from home context ASH file

home_var <- home_cont[,c("IDSCHOOL","IDCLASS","IDSTUD",
                         "ASDAGE", "ASDHEDUP")]


#subsetting variables from teacher student link AST file

link_var <- stu_teach_link[,c("IDSCHOOL","IDTEALIN","IDTEACH","IDLINK","ITCOURSE","MATSUBJ",
                              "IDGRADE","IDSUBJ","IDSCHOOL","IDCLASS","IDSTUD")]

####### Merging data #####

# Step 1: Merge student_var with link_var using common identifiers (IDSCHOOL, IDCLASS, IDSTUD)
st_link <- merge(st_var, link_var, by = c("IDSCHOOL", "IDCLASS", "IDSTUD"), all = TRUE)

# Step 2: Merge the result with teacher_var using identifiers (IDSCHOOL, IDTEALIN, IDLINK)
st_teach <- merge(st_link, teach_var, by = c("IDSCHOOL", "IDTEALIN", "IDLINK"), all = TRUE)

# Step 3: Merge the combined dataset with school_var using IDSCHOOL
st_teach_sch_data <- merge(st_teach, sch_var, by = "IDSCHOOL", all = TRUE)

# Step 4: Finally, merge the combined dataset with home _var using IDSCHOOL, IDCLASS, IDSTUD
final_data <- merge(st_teach_sch_data, home_var, by = c("IDSCHOOL", "IDCLASS", "IDSTUD"), all = TRUE)

# View the resulting merged dataset
head(final_data)
str(final_data)

data <- final_data
str(data)

####### Renaming variables ########
names(data) <- c("School ID", "Class ID", "Student ID", "Teacher ID and Link", "Teacher Link Number", 
                 "Gender", "Test Language",
                 "Books", "N of Home Study Supports",
                 "Instr.Clarity", "Work independently",
                 "Like learning Maths","Confidence Maths",
                 "Sch. Belonging", "Absenteeism","Home Resources for Learning",
                 "PV1","PV2","PV3","PV4","PV5",
                 "Teacher ID.x","Subject Code.x","Math Teacher Link",
                 "Grade ID","Subject ID","School ID.1","Teacher ID.y","Subject Code.y",
                 "Instr. Time",
                 "Number Topics","Meas and Geo Topics","Data Topics",
                 "Teachig Limited", "Math Investigation",
                 "Resources Shortage",
                 "Emphasis Ac.Success","Sch. Discipline",
                 "School SES","Test Lang. is Native",
                 "Student's Age","Parents' Highest Education")

str(data)
dim(data)
#------------------------------------------------------
#================================================================================#
# Descriptive Statistics of Predictors (before standardization)
#================================================================================#
#------------------------------------------------------
# Predictor variables to be included in the descriptives table later
predictors <- c("Gender", "Test Language",
                "Instr.Clarity", "Work independently",
                "Like learning Maths","Confidence Maths",
                "Sch. Belonging", "Absenteeism",
                "Instr. Time",
                "Number Topics","Meas and Geo Topics","Data Topics",
                "Teachig Limited", "Math Investigation",
                "Resources Shortage",
                "Emphasis Ac.Success","Sch. Discipline",
                "School SES","Test Lang. is Native")

# Calculate their means and SDs before standardizing them later
predictors_means <- sapply(data[predictors], mean, na.rm = TRUE)
predictors_sds <- sapply(data[predictors], sd, na.rm = TRUE)

# Store in a data frame
predictors_descr <- data.frame(
  Feature = names(predictors_means),
  Mean = round(predictors_means, 2),
  SD = round(predictors_sds, 2)
)

#------------------------------------------------------
#================================================================================#
# Data Preparation
#================================================================================#
#------------------------------------------------------
# Reverse-coding 
# Gender: so it is 0:Girl and 1:Boy
data$"Gender" <- ifelse(data$"Gender" == 1, 0, 1)

# Parental Education, used as an SES
  # Replace "Not Applicable" with NA
  data$`Parents' Highest Education` [data$`Parents' Highest Education`  == 6] <- NA

  # Reverse code the variable so that: 1->5, 2->4, 3->3, 4->2, 5->1
  data$`Parents' Highest Education` <- 6 - data$`Parents' Highest Education`

# Z-standardization of all variables used as predictors except for Gender
vars_to_standardize <- c("Test Language",
                         "Instr.Clarity", "Work independently",
                         "Like learning Maths","Confidence Maths",
                         "Sch. Belonging", "Absenteeism",
                         "Instr. Time",
                         "Number Topics","Meas and Geo Topics","Data Topics",
                         "Teachig Limited", "Math Investigation",
                         "Resources Shortage",
                         "Emphasis Ac.Success","Sch. Discipline",
                         "School SES","Test Lang. is Native"
)


data[vars_to_standardize] <- as.data.frame(scale(data[vars_to_standardize]))

# Min-max normalization for all 4 SES measures
data$HRL_norm <- (data$"Home Resources for Learning" - min(data$"Home Resources for Learning", na.rm = TRUE)) /
  (max(data$"Home Resources for Learning", na.rm = TRUE) - min(data$"Home Resources for Learning", na.rm = TRUE))

data$HSS_norm <- (data$"N of Home Study Supports" - min(data$"N of Home Study Supports", na.rm = TRUE)) /
  (max(data$"N of Home Study Supports", na.rm = TRUE) - min(data$"N of Home Study Supports", na.rm = TRUE))

data$Books_norm <- (data$"Books" - min(data$"Books", na.rm = TRUE)) /
  (max(data$"Books", na.rm = TRUE) - min(data$"Books", na.rm = TRUE))

data$ED_norm <- (data$"Parents' Highest Education" - min(data$"Parents' Highest Education", na.rm = TRUE)) /
  (max(data$"Parents' Highest Education", na.rm = TRUE) - min(data$"Parents' Highest Education", na.rm = TRUE))

# Min-max normalization for all 5 PV
  plausible_values <- c("PV1","PV2","PV3","PV4","PV5")

  # Find overall min and max, because we need to normalize PVs using the same scale 
  all_PVs <- unlist(data[plausible_values])
  overall_min <- min(all_PVs, na.rm = TRUE)
  overall_max <- max(all_PVs, na.rm = TRUE)

  # Create normalized versions using a loop
  for(pv in plausible_values) {
  norm_name <- paste0(pv, "_norm")
  data[[norm_name]] <- (data[[pv]] - overall_min) / (overall_max - overall_min)
  }

# Create a list with normalized plausible values to use later in analyses
plausible_values_norm <- c("PV1_norm","PV2_norm","PV3_norm","PV4_norm","PV5_norm")

# Check for missing values in the whole datasets and their proportions
colSums(is.na(data))
colMeans(is.na(data))
#------------------------------------------------------


