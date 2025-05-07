library('tidyverse')
library(ggplot2)
library(dplyr)
library(randomForest)
library(FSA)
library(tidyr)
library(tree)
library(glmnet)
library(WRS2)
library(car)
library(caret)

load("~/Downloads/STAT515_Final_Project/loans_full_schema.rda")

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
print(kw)      # if needed

dunnTest(annual_income ~ homeownership, data = df_cleaned,
         method = "bonferroni")

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



# Question 2:
df_cleaned$num_open_cc_accounts <- NULL
df_cleaned$public_record_bankrupt <- NULL
df_cleaned$term <- NULL
df_cleaned$num_collections_last_12m <- NULL
numeric_cols <- sapply(df_cleaned, is.numeric)
# Exclude the response variable from predictors
predictors <- names(df_cleaned)[numeric_cols & names(df_cleaned) != "interest_rate"]


# Create the feature matrix and response vector
X <- as.matrix(df_cleaned[, predictors])
y <- df_cleaned$interest_rate

# Set up k-fold cross-validation
set.seed(123)  # For reproducibility
k <- 3  # Number of folds

# Create folds for cross-validation
folds <- createFolds(y, k = k, returnTrain = FALSE)

# Initialize vectors to store cross-validation results
cv_rmse <- numeric(k)
cv_mae <- numeric(k)
cv_r_squared <- numeric(k)

# Perform k-fold cross-validation
for (i in 1:k) {
  # Split data into training and test sets
  test_indices <- folds[[i]]
  X_train <- X[-test_indices, ]
  y_train <- y[-test_indices]
  X_test <- X[test_indices, ]
  y_test <- y[test_indices]
  
  # Fit LASSO model with cross-validation to select the optimal lambda
  cv_model <- cv.glmnet(X_train, y_train, alpha = 1)
  
  # Use the best lambda (lambda.min) for prediction
  best_lambda <- cv_model$lambda.min
  
  # Make predictions on the test set
  predictions <- predict(cv_model, s = best_lambda, newx = X_test)
  
  # Calculate performance metrics
  rmse <- sqrt(mean((predictions - y_test)^2))
  mae <- mean(abs(predictions - y_test))
  ss_total <- sum((y_test - mean(y_test))^2)
  ss_residual <- sum((y_test - predictions)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Store the metrics
  cv_rmse[i] <- rmse
  cv_mae[i] <- mae
  cv_r_squared[i] <- r_squared
  
  cat("Fold", i, "- RMSE:", round(rmse, 4), "MAE:", round(mae, 4), "R²:", round(r_squared, 4), "\n")
}

# Calculate and print average performance metrics
avg_rmse <- mean(cv_rmse)
avg_mae <- mean(cv_mae)
avg_r_squared <- mean(cv_r_squared)

cat("\nAverage Cross-Validation Metrics:\n")
cat("RMSE:", round(avg_rmse, 4), "\n")
cat("MAE:", round(avg_mae, 4), "\n")
cat("R²:", round(avg_r_squared, 4), "\n")

# Train the final model on the entire dataset
final_cv_model <- cv.glmnet(X, y, alpha = 1)
final_lambda <- final_cv_model$lambda.min

# Plot the cross-validation curve
plot(final_cv_model)
title("LASSO Cross-Validation")

# Fit the final model with the optimal lambda
final_model <- glmnet(X, y, alpha = 1, lambda = final_lambda)

# Extract and display the coefficients
coefs <- coef(final_model)
important_vars <- coefs[which(coefs != 0), ]
cat("\nSelected Variables and Their Coefficients:\n")
print(important_vars)

# Plot the coefficients
coef_df <- data.frame(
  Variable = row.names(as.matrix(coefs)),
  Coefficient = as.vector(as.matrix(coefs))
) %>% filter(Coefficient != 0)

# Skip the intercept for better visualization
coef_plot_data <- coef_df[-1, ]

ggplot(coef_plot_data, aes(x = reorder(Variable, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "Variables", y = "Coefficient Value", 
       title = "LASSO Regression Coefficients") +
  theme_minimal()

# Function to make predictions on new data
predict_interest_rate <- function(new_data) {
  # Convert to matrix format
  new_matrix <- as.matrix(new_data[, predictors])
  # Make predictions
  predictions <- predict(final_model, newx = new_matrix)
  return(predictions)
}

# Example usage of prediction function
# If you have new data:
# new_predictions <- predict_interest_rate(new_data)
# print(new_predictions)

# Optional: Save the model for future use
saveRDS(list(model = final_model, predictors = predictors), "lasso_interest_rate_model.rds")

cat("\nModel saved as 'lasso_interest_rate_model.rds'\n")
cat("To load the model: model_data <- readRDS('lasso_interest_rate_model.rds')\n")




















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



