library('tidyverse')
library(ggplot2)
library(dplyr)
library(randomForest)
library(tidyr)
library(tree)
library(WRS2)
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





## Question 2: What affect people loan_status the most?
loan_status_df <- df_cleaned
loan_status_df$loan_status <- factor(rf_loan_status_df, levels = c("Charged Off", "Late(31-120 days)", "Late (16-30 days)", "In Grace Period", "Current", "Fully Paid"), ordered = TRUE)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)




## Question 2: Did the people in different state have significant differences in average interest_rate
mod_int <- lm(interest_rate ~ state*loan_amount   # income:state, grade:state, if you want
              + state*annual_income 
              + state*grade,
              data = df_cleaned)
anova(mod_int)

mod33_simple <- lm(
  interest_rate ~ state 
  + grade 
  + loan_amount 
  + annual_income,
  data = df_cleaned
)
car::Anova(mod33_simple, type="III")


# ANOVA
mod_state <- aov(interest_rate ~ state, data = df_cleaned)
summary(mod_state)

# Check Variance differ
plot(mod_state, which = 3, main = "Scale–Location") 
leveneTest(interest_rate ~ state, data = df)

# Check normality
# Extract residuals
res <- residuals(mod_state)

# QQ–plot (visual)
qqnorm(res, main="Q–Q Plot of Residuals")
qqline(res, col="red")

hist(res, breaks=30, prob=TRUE,
     main="Histogram of Residuals",
     xlab="Residuals")
lines(density(res), lwd=2)



## Question 3: Predict Grade by Random Forest (To be continue)
df_cleaned$grade <- droplevels(df_cleaned$grade) # Remove the unused target level
rf_data <- df_cleaned
# removing columns
rf_data$sub_grade   <- NULL
rf_data$issue_month <- NULL

rf_data$state <- as.factor(rf_data$state)
rf_data$application_type <- as.factor(rf_data$application_type)
rf_data$homeownership <- as.factor(rf_data$homeownership)
rf_data$loan_purpose <- as.factor(rf_data$loan_purpose)
rf_data$loan_status <- as.factor(rf_data$loan_status)
rf_data$disbursement_method <- as.factor(rf_data$disbursement_method)
rf_data$initial_listing_status <- as.factor(rf_data$initial_listing_status)
rf_data$verified_income <- as.factor(rf_data$verified_income)

# Split data
set.seed(42)
train_index <- sample(seq_len(nrow(rf_data)), size = 0.8 * nrow(rf_data))
train_data <- rf_data[train_index, ]
test_data  <- rf_data[-train_index, ]

rf_model <- randomForest(grade ~ ., data = train_data, importance = TRUE)
importance(rf_model)
top_vars <- names(sort(importance(rf_model)[,1], decreasing = TRUE))[1:5]


ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

# We first test 4, 10, 15, 20. 5 perform best then we narrow the range to 5-9
rfe_res <- rfe(
  x          = train_data[, setdiff(names(train_data), "grade")],
  y          = train_data$grade,
  sizes      = c(5, 6, 7, 8, 9),
  rfeControl = ctrl
)

predictors(rfe_res)

# re-fit on those
rf_rfe <- randomForest(grade ~ ., data = train_data[, c(predictors(rfe_res), "grade")])

# Evaluate on test data
pred <- predict(rf_rfe, newdata = test_data[, predictors(rfe_res)])
confusionMatrix(pred, test_data$grade)



