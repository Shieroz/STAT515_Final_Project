library('tidyverse')
library(ggplot2)
source('./hw.R')
library(dplyr)
library(randomForest)

load('./loans_full_schema.rda')

df <- loans_full_schema
df[df == ""] <- NA

(colSums(is.na(df))/nrow(df))*100

unique(df$verification_income_joint)

## Data Cleaning

# Replace the annual income with joint one if verified
df$annual_income <- ifelse(df$verification_income_joint %in% c("Verified", "Source Verified"), df$annual_income_joint, df$annual_income)
df$debt_to_income <- ifelse(df$verification_income_joint %in% c("Verified", "Source Verified"), df$debt_to_income_joint, df$debt_to_income)

df_cleaned = df[ , !names(df) %in% c('emp_title', 'annual_income_joint', 'months_since_last_delinq', 'months_since_90d_late', 'debt_to_income_joint', 'current_accounts_delinq', 'verification_income_joint', 'months_since_last_credit_inquiry')]
str(df_cleaned)

replace_emp_length <- median(na.omit(df_cleaned$emp_length)) # mean: 5.9, median: 6
df_cleaned$emp_length[is.na(df_cleaned$emp_length)] <- replace_emp_length

(colSums(is.na(df_cleaned))/nrow(df_cleaned))*100
ggplot(df_cleaned, aes(x = debt_to_income)) +
  geom_histogram(fill="cornsilk", color = "black")
 
# Replace the NA with mean
replace_debt_income <- mean(na.omit(df_cleaned$debt_to_income)) # mean: 17.43, median: 18.32
df_cleaned$debt_to_income[is.na(df_cleaned$debt_to_income)] <- replace_debt_income
df_cleaned <- na.omit(df_cleaned)


## Question 1: With different home-ownership is the income have significant different? 
### Using One-way ANOVA

anova_result <- aov(annual_income ~ homeownership, data = df_cleaned)
summary(anova_result)
tukey_result <- TukeyHSD(anova_result)
plot(tukey_result, las = 1)
abline(v = 0, col = "red", lwd = 2, lty = "dashed") # horizontal line is a confidence interval for the difference in means between two groups.

# Interpretation: ANOVA result indicated that there are significant difference between the mean of the homeownership group. We use tukey to see which pairs of group are different. 
#   The result showed that the mean of annual_income in Mortgage have significant difference compared to others two group.
#   However, people who rent the houses and own houses didn't have significant difference in the confidence level of mean in annual_income cause it cross the 0 in xline. 


## Question 2: Did the people with verified income have lower interest_rate
anova_result <- aov(interest_rate ~ verified_income, data = df_cleaned)
summary(anova_result)
tukey_result <- TukeyHSD(anova_result)
par(mar = c(4, 9.5, 2, 2)) # bottom, left, top, right
plot(tukey_result, las = 1, cex.axis = 0.8)
abline(v = 0, col = "red", lwd = 2, lty = "dashed")

# Interpretation: ANOVA result 



## Question 3: Predict Grade by Random Forest (To be continue)
df_cleaned$grade <- droplevels(df_cleaned$grade) # Remove the unused target level
rf_data <- df_cleaned
# removing columns
rf_data <- rf_data %>% select(-sub_grade, -issue_month)

rf_data$state <- as.factor(rf_data$state)
rf_data$application_type <- as.factor(rf_data$application_type)
rf_data$homeownership <- as.factor(rf_data$homeownership)
rf_data$loan_purpose <- as.factor(rf_data$loan_purpose)
rf_data$loan_status <- as.factor(rf_data$loan_status)
rf_data$disbursement_method <- as.factor(rf_data$disbursement_method)
rf_data$initial_listing_status <- as.factor(rf_data$initial_listing_status)
rf_data$verified_income <- as.factor(rf_data$verified_income)

str(rf_data)

rf_model <- randomForest(grade ~ ., data = rf_data, importance = TRUE)
importance(rf_model)
top_vars <- names(sort(importance(rf_model)[,1], decreasing = TRUE))[1:5]
df_subset <- df[, c(top_vars, "target")]
rf_subset <- randomForest(target ~ ., data = df_subset)

