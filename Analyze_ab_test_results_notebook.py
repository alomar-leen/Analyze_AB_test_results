#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


AB = pd.read_csv('ab_data.csv')
AB.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


num_rows= len(AB)
print(num_rows)


# c. The number of unique users in the dataset.

# In[4]:


unique_users= AB['user_id'].unique()
print(len(unique_users))


# d. The proportion of users converted.

# In[5]:


(len(AB[AB['converted']==1]))/AB.shape[0]


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


treat_old = AB.query("group == 'treatment' and landing_page == 'old_page'").shape[0]
control_new = AB.query("group == 'control' and landing_page == 'new_page'").shape[0]

treat_old + control_new


# f. Do any of the rows have missing values?

# In[7]:


AB.info()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df2 = AB.query("group == 'treatment' and landing_page == 'new_page'")
df2 = df2.append(AB.query("group == 'control' and landing_page == 'old_page'"))


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


unique_users= df2['user_id'].unique()
print(len(unique_users))


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


df2[df2['user_id'].duplicated()]


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2[df2['user_id'] == 773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


df2 = df2.drop(2893)


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


df2.query('converted == 1').shape[0]/df2.shape[0]


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


cont_groub = df2.query("group == 'control'")['converted'].mean()
cont_groub


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treat_groub = df2.query("group == 'treatment'")['converted'].mean()
treat_groub


# d. What is the probability that an individual received the new page?

# In[17]:


df2.query('landing_page == "new_page"').shape[0]/df2.shape[0]


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# there is no sufficient evidence that treatment page leads to more conversions, because the probability that an individual was in the treatment group and converted is lowwer than old control page

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# H0:𝑝𝑜𝑙𝑑>=𝑝𝑛𝑒𝑤
# 
# H1:𝑝𝑛𝑒𝑤<𝑝𝑜𝑙𝑑

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[18]:


p_new = df2['converted'].mean()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[19]:


p_old = df2['converted'].mean()
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[20]:


n_new = df2.query("group == 'treatment'").shape[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[21]:


n_old = df2.query("group == 'control'").shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[22]:


new_page_converted = np.random.choice([0, 1], size=n_new, p=[1-p_new, p_new])
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[23]:


old_page_converted = np.random.choice([0, 1], size=n_old, p=[1-p_old, p_old])
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[24]:


new_page_converted.mean() - old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[25]:


p_diffs = []
for x in range(1000):
    new_page_converted = np.random.choice([0, 1], size=n_new, p=[1-p_new, p_new])
    old_page_converted = np.random.choice([0, 1], size=n_old, p=[1-p_old, p_old])
    p_diffs.append(new_page_converted.mean() - old_page_converted.mean())


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[26]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[27]:


actual_diffs = df2[df2['group'] == 'treatment']['converted'].mean() -  df2[df2['group'] == 'control']['converted'].mean()
accumulator = 0
for x in p_diffs:
    if x> actual_diffs:
        accumulator = accumulator+1        
print (accumulator/(len(p_diffs)))


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# in scince this value called the p value, it is used to make a decision wether to reject the null hypothesis or not. 
# the null hypothesis is rejected if the p value is bellow 0.05, so in this case we have approximitly 0.9 p value which indicate that we shoulde not reject the null hypothesis that signify that the old page is better than or the same as the New page.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[28]:


import statsmodels.api as sm
convert_old = df2.query("landing_page == 'old_page'")['converted'].sum()
convert_new = df2.query("landing_page == 'new_page'")['converted'].sum()
n_old = len(df2.query('landing_page=="old_page"'))
n_new = len(df2.query('landing_page=="old_page"'))


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[29]:


z_score, p_value = sm.stats.proportions_ztest([convert_old,convert_new], [n_old, n_new],alternative='smaller') 
print(z_score,p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# In[30]:


from scipy.stats import norm
norm.ppf(1-(0.05))


# since the z-score is 1.28632078586 which is lower than the critical value 1.6448536269514722 (rejection region) ,and the p value is 0.900834434411 which is higher than 0.05 ,we should not reject the null hypothesis (the conversion rate of the old page is better than or the same as the conversion rate of the new page). and this finding agree with the findings in parts j. and k.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# The Logistic Regression, since we want to predict a categorical response with only two possible outcome

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[31]:


df2['intercept'] = 1
df2[['control','treatment']] = pd.get_dummies(df2['group'])
df2 = df2.drop('control', axis = 1)
df2.rename(columns={"treatment": "ab_page"}, inplace=True)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[32]:


logit = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
results = logit.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[33]:


results.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# The p-value associated with ab_page is 0.1899 and it is differ because the null and alternative hypotheses were:
# 
# H0:𝑝𝑜𝑙𝑑>=𝑝𝑛𝑒𝑤
# 
# H1:𝑝𝑛𝑒𝑤<𝑝𝑜𝑙𝑑
# 
# and the null and alternative hypotheses associated with the regression model are:
# 
# H0:𝑝𝑜𝑙𝑑-𝑝𝑛𝑒𝑤=0
# 
# H1:𝑝𝑛𝑒𝑤-𝑝𝑜𝑙𝑑!=0
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# Considering additional factors in the regression modele may give more information about what may influence conversions .
# however, considering additional factors may drive some confusion espacily when we have a bunch of factor to look at.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[34]:


countries = pd.read_csv('countries.csv')
countries.head()


# In[35]:


countries['country'].unique()


# In[36]:


countries.head()


# In[37]:


df2.head()


# In[38]:


df3 = countries.set_index('user_id').join(df2.set_index('user_id'), how = 'inner')


# In[39]:


df3.head()


# In[50]:


df3[['UK', 'US', 'CA']] = pd.get_dummies(df3['country'])[['UK', 'US', 'CA']]
df3 = df3.drop('UK', axis =1 )
df3.head()


# In[51]:


logit = sm.Logit(df3['converted'],df3[['intercept','US','CA']])
results = logit.fit()
results.summary2()


# There is no significant p value for any of the variables. Therefore, we should stick with the null hypothises and conclude that it does not appear that the country has a significant impact on conversion.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[55]:


df3['US_page'] = df3['US']*df3['ab_page']


# In[57]:


df3['CA_page'] = df3['CA']*df3['ab_page']
df3.head()


# In[58]:


df3['CA_page'] = df3['CA']*df3['ab_page']
df3['US_page'] = df3['US']*df3['ab_page']
logit_mod = sm.Logit(df3['converted'], df3[['intercept', 'CA_page', 'US_page']])
results = logit_mod.fit()
results.summary2()


# the p value of the CA and ab_page is under the alpha (0.05) which is significant.

# Conclusion:
# Based on the analysis done in this report , we do not have sufficient evidence that the new page drive more conversions than the old page. hence we will go with the null hypothesis.

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

