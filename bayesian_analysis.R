
cat("Loading clean data from DuckDB...\n")
data <- read.csv("bayesian_input_data.csv")
print(data)

total_users <- sum(data$total_users)
fraud_cases <- data$total_users[data$is_fraud_risk == "True"]

cat("\nTotal Users:", total_users, "\n")
cat("Observed Fraud Cases:", fraud_cases, "\n")


# Prior: We expect a 1% fraud rate baseline (Alpha = 1, Beta = 99)
prior_alpha <- 1
prior_beta <- 99


successes <- fraud_cases          # 26
failures <- total_users - successes # 974


posterior_alpha <- prior_alpha + successes
posterior_beta <- prior_beta + failures

expected_prior_rate <- prior_alpha / (prior_alpha + prior_beta)
expected_posterior_rate <- posterior_alpha / (posterior_alpha + posterior_beta)

credible_interval <- qbeta(c(0.025, 0.975), posterior_alpha, posterior_beta)

cat("\n======================================================\n")
cat("BAYESIAN POSTERIOR ANALYSIS RESULTS\n")
cat("======================================================\n")
cat(sprintf("Baseline Prior Fraud Rate:  %.2f%%\n", expected_prior_rate * 100))
cat(sprintf("Updated Posterior Fraud Rate: %.2f%%\n", expected_posterior_rate * 100))
cat(sprintf("95%% Credible Interval:       [%.2f%% to %.2f%%]\n", 
            credible_interval[1] * 100, credible_interval[2] * 100))
cat("======================================================\n\n")

cat("INTERPRETATION:\n")
cat(sprintf("We are 95%% confident that the true, underlying fraud risk rate\n"))
cat(sprintf("in this system lies between %.2f%% and %.2f%%, given our prior\n", 
            credible_interval[1] * 100, credible_interval[2] * 100))
cat("beliefs and the newly analyzed dataset.\n")