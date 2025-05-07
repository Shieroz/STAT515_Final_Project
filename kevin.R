library('tidyverse')
library(ggplot2)
library(dplyr)
library(randomForest)
library(tidyr)
library(car)
library(caret)

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

df_cleaned$grade <- factor(df_cleaned$grade, levels = c("A", "B", "C", "D", "E", "F", "G"), ordered=TRUE)
# See the outlier
num_cols <- sapply(df_cleaned, is.numeric)
df_num   <- df_cleaned[ , num_cols]
boxplot(df_num,
        main = "Boxplots of Numeric Columns",
        las  = 2,             # rotate x-labels
        col  = "lightgray")
outliers_list <- lapply(df_num, function(x) boxplot.stats(x)$out)



## Question 1: With different home-ownership is the income have significant different? 

### Check whether variables are normal distribution
df_cleaned <- df_cleaned[df_cleaned$homeownership != "ANY", ]
anova_result <- aov(annual_income ~ homeownership, data = df_cleaned)
qqnorm(residuals(anova_result)); qqline(residuals(anova_result))


resid_df <- data.frame(
  resid = residuals(anova_result),
  group = df_cleaned$homeownership
)

boxplot(resid ~ group, data = resid_df,
        main = "Residuals by Group",
        ylab = "Residuals")


# Try transform the data and try again

df_cleaned <- df_cleaned %>%
  filter(homeownership != "ANY") %>%
  droplevels()

anova_trans <- df_cleaned
anova_trans$annual_income_log <- log(anova_trans$annual_income + 1)
anova_trans$annual_income_sqrt <- sqrt(anova_trans$annual_income)

anova_result <- aov(annual_income_log ~ homeownership, data = anova_trans)
qqnorm(residuals(anova_result)); qqline(residuals(anova_result))

anova_result <- aov(annual_income_sqrt ~ homeownership, data = anova_trans)
qqnorm(residuals(anova_result)); qqline(residuals(anova_result))

### Using Kruskal–Wallis test (nonparametric): Compares group medians by ranking all data, then testing whether rank-sums differ more than expected by chance.

table(df_cleaned$homeownership)
# Since no values for 'ANY' we remove it.
df_cleaned <- df_cleaned[df_cleaned$homeownership != "ANY", ]

kw <- kruskal.test(annual_income ~ homeownership, data = df_cleaned)
print(kw)


boxplot(annual_income ~ homeownership, data = df_cleaned,
        main = "Boxplot of Response by Group",
        xlab = "Group", ylab = "Response")

ggplot(df, aes(x = annual_income, color = homeownership, fill = homeownership)) +
  geom_density(alpha = 0.3) +
  labs(title = "Density of annual_income by homeownership",
       x = "annual_income", y = "Density")

# Interpretation: ANOVA result indicated that there are significant difference between the mean of the homeownership group. We use tukey to see which pairs of group are different. 
#   The result showed that the mean of annual_income in Mortgage have significant difference compared to others two group.
#   However, people who rent the houses and own houses didn't have significant difference in the confidence level of mean in annual_income cause it cross the 0 in xline. 





## Question 2: Do different loan_purpose categories have significantly different average interest_rates, even after controlling for borrower risk profiles (e.g., grade)?
# H0: All purpose‐group means are equal after adjusting for grade.
# H1: All purpose‐group means are not equal after adjusting for grade.

# ANCOVA assumes the relationship between the covariate (grade) and outcome is the same in every group.
mod_int <- lm(interest_rate ~ loan_purpose * grade, data = df_cleaned)
anova(mod_int)
# Since the loan_purpose:grade interaction isn’t significant the homogeneity of slopes assumption holds

# 2. Fit ANCOVA without interaction
mod_ancova <- lm(interest_rate ~ grade + loan_purpose, data = df_cleaned)
Anova(mod_ancova, type = "III")
summary(mod_ancova)


par(mfrow = c(2,2))
plot(mod_ancova)  







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


ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
# sizes = the candidate subset sizes you want to try
rfe_res <- rfe(
  x          = rf_data[, setdiff(names(rf_data), "grade")],
  y          = rf_data$grade,
  sizes      = c(5, 10, 15, 20),
  rfeControl = ctrl
)
# best variables
predictors(rfe_res)
# re-fit on those
rf_rfe <- randomForest(grade ~ ., data = rf_data[, c(predictors(rfe_res), "grade")])

# Compare original model and variable select model



