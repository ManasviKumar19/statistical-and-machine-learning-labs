#!/usr/bin/env python
# coding: utf-8

# In[1]:


library("MASS")
data(Boston)


# In[2]:


set.seed(3938425)
test <- sample(1:nrow(Boston), size = 100)
train <- which(!(1:nrow(Boston)) %in% test)


# In[3]:


install.packages("e1071")
library(e1071)


# In[6]:


pred_vals <- colnames(Boston)[-which(colnames(Boston) == "medv")]
result <- "medv"


# In[7]:


tuning <- tune(svm,
                     as.formula(paste(result, "~", paste(pred_vals, collapse = "+"))),
                     data = Boston[train, ],
                     ranges = list(epsilon = seq(0, 1, by = 0.1)),
                     kernel = "radial",
                     scale = TRUE)


# In[8]:


best_epsilon <- tuning$best.parameters$epsilon


# In[9]:


best_epsilon


# In[10]:


svm_model <- svm(as.formula(paste(result, "~", paste(pred_vals, collapse = "+"))),
                 data = Boston[train, ],
                 kernel = "radial",
                 scale = TRUE,
                 epsilon = best_epsilon)


# In[11]:


svm_model


# In[12]:


summary(svm_model)


# In[ ]:





# # a) SVR with linear kernel

# In[13]:


library(ggplot2)


# In[14]:


cost <- c(0.001, 0.01, 0.1, 1, 10)
epsilon <- c(0.001, 0.01, 0.1, 1)


# In[15]:


set.seed(42)
tune.out <- tune(svm, medv ~ ., data = Boston[train, ], kernel = "linear",
                 ranges = list(cost = cost, epsilon = epsilon))


# In[16]:


perf <- tune.out$performances


# In[17]:


perf$cost <- factor(perf$cost)
ggplot(perf) +
  geom_line(aes(epsilon, error, group = cost, color = cost)) +
  scale_x_log10() +
  labs(x = "Epsilon", y = "Mean Squared Error") +
  theme_minimal()


# In[ ]:





# In[18]:


tune.out$best.parameters


# In[19]:


cost <- c(0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 10)
epsilon <- c(0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.5, 0.75, 0.9, 1)


# In[20]:


set.seed(42)
tune.out <- tune(svm, medv ~ ., data = Boston[train, ], kernel = "linear",
                 ranges = list(cost = cost, epsilon = epsilon))


# In[21]:


tune.out$best.parameters


# In[22]:


perf <- tune.out$performances
perf$cost <- factor(perf$cost)
ggplot(perf) +
  geom_line(aes(epsilon, error, group = cost, color = cost)) +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Epsilon", y = "Mean Squared Error") +
  theme_minimal()


# ### Fitting the SVR model

# In[23]:


optimal_params <- tune.out$best.parameters
svm_model <- svm(medv ~ ., data = Boston[train, ], kernel = "linear",
                 cost = optimal_params$cost, epsilon = optimal_params$epsilon)


# In[24]:


predictions <- predict(svm_model, newdata = Boston[test, ])
mse <- mean((predictions - Boston$medv[test])^2)
mae <- mean(abs(predictions - Boston$medv[test]))


# In[25]:


cat("Mean Squared Error (MSE) on test data:", mse, "\n")
cat("Mean Absolute Error (MAE) on test data:", mae, "\n")


# In[ ]:





# # b) SVR with radial basis kernel

# In[26]:


cost <- c(0.001, 0.01, 0.1, 1)
gamma <- c(0.001, 0.01, 0.1, 1)
epsilon <- c(0.001, 0.01, 0.1, 1)


# In[27]:


set.seed(42)
tune.out <- tune(svm, medv ~ ., data = Boston[train, ], kernel = "radial",
                 ranges = list(cost = cost, gamma = gamma, epsilon = epsilon))


# In[34]:


graph_perf <- tune.out$performances


# ### Finding the optimal values

# In[35]:


ggplot(graph_perf) +
  geom_line(aes(cost, error, group = gamma, color = as.factor(gamma))) +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Cost", y = "Mean Squared Error", color = "Gamma") +
  theme_minimal()


# In[30]:


tune.out$best.parameters


# ## Fitting the model

# In[36]:


optimal_params <- tune.out$best.parameters
radial_svm <- svm(medv ~ ., data = Boston[train, ], kernel = "radial",
                        cost = optimal_params$cost, gamma = optimal_params$gamma,
                        epsilon = optimal_params$epsilon)


# 

# In[37]:


SVM_radial_pred <- predict(radial_svm, newdata = Boston[test, ])
mse_radial <- mean((SVM_radial_pred - Boston$medv[test])^2)
mae_radial <- mean(abs(SVM_radial_pred - Boston$medv[test]))


# ## Comparing accuracy of Radial SVR and Linear SVR

# In[38]:


cat("Mean Squared Error (MSE) on test data (Radial SVR):", mse_radial, "\n")
cat("Mean Absolute Error (MAE) on test data (Radial SVR):", mae_radial, "\n")


# In[39]:


cat("\nComparison with Linear SVR:\n")
cat("Mean Squared Error (MSE) on test data (Linear SVR):", mse, "\n")
cat("Mean Absolute Error (MAE) on test data (Linear SVR):", mae, "\n")


# In[ ]:





# In[ ]:





# In[ ]:




