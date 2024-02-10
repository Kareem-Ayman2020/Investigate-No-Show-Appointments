#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate No_Show_Appointments Dataset.
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# >This dataset is based on data from 100k Brazilian medical visits and focuses on whether or not patients show up for their appointments. Each row contains a number of characteristics about the subject.
# 
# 
# ### Question(s) for Analysis
# >1-What factors are important for us to know in order to predict if a patient will show up for their scheduled appointment?
# 
# > 2- What is the correlation between each feature?

# In[1]:


# import statements for all of the packages that you

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > loading dataset and handling it
# 
# 
# 

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv("noshowappointments-kagglev2-may-2016.csv")


# In[3]:


#show head of dataset
df.head(3)


# In[4]:


#show number of columns and rows
df.shape


# The data has 110527 rows and 14 columns

# In[5]:


#some information about dataset
df.info()


# In[6]:


df.describe()


# average age is : 37 years

# minimum age : -1 and this isn't rational so that i will drop it in the next step
# 

# maximum age : 115 years , it's so old but it is possible

# 
# ### Data Cleaning
# >Clean data and remove data that leads to missing value

# In[7]:


# change columns from upper to lower case
df.columns= df.columns.str.lower()


# In[8]:


df.head()


# In[9]:


#drop patient that has -1 year!
df = df[df.age>=0]


# In[10]:


df.shape


# In[11]:


#show missing value
df.isnull().sum()


# Their is no missing value

# In[12]:


#show duplicated row
df.duplicated().sum()


# Their is no duplicated rows

# In[13]:


#rename col name to correct name
df.rename(columns = {'patientid': 'patient_id'}, inplace = True)
df.rename(columns = {'appointmentid': 'appointment_id'}, inplace = True)
df.rename(columns = {'scheduledday': 'scheduled_day'}, inplace = True)
df.rename(columns = {'appointmentday': 'appointment_day'}, inplace = True)
df.rename(columns = {'hipertension': 'hypertension'}, inplace = True)
df.rename(columns = {'handcap': 'handicap'}, inplace = True)
df.rename(columns = {'no-show': 'no_show'}, inplace = True)


# In[14]:


df.head()


# In[15]:


#now i drop columns that isn't important
df.drop(['patient_id', 'appointment_id', 'scheduled_day', 'appointment_day'], axis = 1 , inplace = True)


# In[16]:


df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > After I cleaned up my data, I'm ready to move on to exploring
# 
# 
# ### Research Question 
# 1-What factors are important for us to know in order to predict if a patient will show up for their scheduled appointment?

# In[17]:


Show = df.no_show == 'No'
NoShow = df.no_show == 'Yes'


# In[18]:


df.gender[Show].hist(figsize=(7,5));
plt.title("Comparison between gender and number of patient who didn't show" )
plt.xlabel('gender')
plt.ylabel('patients number no show')


# in this figure we show that The number of female who  is twice the number of male which didn't come to the clinic

# In[19]:


df.gender[NoShow].hist(figsize=(7,5));
plt.title("Comparison between gender and number of patient who show" )
plt.xlabel('gender')
plt.ylabel('patients number show')


# in this figuer we show that The number of female who  is twice the number of male which come to the clinic

# In[20]:


gender_no = df[NoShow].gender.count()
gender_sh = df[Show].gender.count()
print(int((gender_no/gender_sh)*100),'%')


# The percentage of patients attending the clinic is 25%, which is a small percentage.
# 
# gender is unimportant feature

# In[21]:


df.age[Show].hist(figsize=(7,5));
plt.title("Comparison between age and number of patient who didn't show" )
plt.xlabel('age')
plt.ylabel('patients number no show')


# through this figure We conclude that:
# 
# patients from 0 to 10 is the most precentage which didn't come 
# 
# patients from 90 to 115 is the least precentage which didn't come 

# In[22]:


df.age[NoShow].hist(figsize=(7,5));
plt.title("Comparison between age and number of patient who show" )
plt.xlabel('age')
plt.ylabel('patients number show')


# through this figure We conclude that:
# 
# patients from 0 to 10 and from 20 to 30 is the most precentage which come
# 
# patients from 90 to 115 is the least precentage which come

# In[23]:


age_no = df[NoShow].age.sum()
age_sh = df[Show].age.sum()
print(int((age_no/age_sh)*100),'%')


# patients between 0-10 showed more than other, and group from 35-70 are showed less than 0-10 , and While the show rate decreases from 70 to 115
# 
# The percentage of patients attending the clinic is 22%, which is a small percentage.
# 
# age is an unimportant feature

# In[24]:


df.scholarship[Show].hist(figsize=(7,5));
plt.title("Comparison between scholarship and number of patient who didn't show" )
plt.xlabel('scholarship')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients non scholarship holder and slightly percentage who enrolled
# 
# 

# In[25]:


df.scholarship[NoShow].hist(figsize=(7,5));
plt.title("Comparison between scholarship and number of patient who show" )
plt.xlabel('scholarship')
plt.ylabel('patients number show')


# through this figure We conclude that:
# 
# most patients who come to the clinic non scholarship holder and slightly percentage who enrolled

# In[26]:


scholarship_no = df[NoShow].scholarship.sum()
scholarship_sh = df[Show].scholarship.sum()
print(int((scholarship_no/scholarship_sh)*100),'%')


# The percentage of patients attending the clinic is 31%, which is a small percentage.
# 
# about 10% are enrolled in the Brasillan welfare program
# 
# scholarship is an unimportant feature

# In[27]:


df.hypertension[Show].hist(figsize=(7,5));
plt.title("Comparison between hypertension and number of patient who didn't show" )
plt.xlabel('hypertension')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients which didn't come to the clinic  do not suffer from hypertension and slightly percentage who suffer from it

# In[28]:


df.hypertension[NoShow].hist(figsize=(7,5));
plt.title("Comparison between hypertension and number of patient who show" )
plt.xlabel('hypertension')
plt.ylabel('patients number show')


# through this figure We conclude that:
# 
# most patients  which come to the clinic do not suffer from hypertension and slightly percentage who suffer from it

# In[29]:


hypertension_no = df[NoShow].hypertension.sum()
hypertension_sh = df[Show].hypertension.sum()
print(int((hypertension_no/hypertension_sh)*100),'%')


# In[30]:


df.diabetes[Show].hist(figsize=(7,5));
plt.title("Comparison between diabetes and number of patient who didn't show" )
plt.xlabel('diabetes')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients do not suffer from diabetes and slightly percentage who suffer from it

# In[31]:


df.diabetes[NoShow].hist(figsize=(7,5));
plt.title("Comparison between diabetes and number of patient who show" )
plt.xlabel('diabetes')
plt.ylabel('patients number show')


# through this figure We conclude that:
# 
# most patients whose come do not suffer from diabetes and slightly percentage who suffer from it

# In[32]:


df.alcoholism[Show].hist(figsize=(7,5));
plt.title("Comparison between alcoholism and number of patient who didn't show" )
plt.xlabel('alcoholism')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients do not drink alcohol and slightly percentage drink it

# In[33]:


df.alcoholism[NoShow].hist(figsize=(7,5));
plt.title("Comparison between alcoholism and number of patient who show" )
plt.xlabel('alcoholism')
plt.ylabel('patients number show')


# through this figure We conclude that:
# 
# most patients who show the clinic do not drink alcohol and slightly percentage  drink it

# In[34]:


df.handicap[Show].hist(figsize=(7,5));
plt.title("Comparison between handicap and number of patient who didn't show" )
plt.xlabel('handicap')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients who didn't show the clinic are undisabled and slightly percentage are handicap 

# In[35]:


df.handicap[NoShow].hist(figsize=(7,5));
plt.title("Comparison between handicap and number of patient show" )
plt.xlabel('handicap')
plt.ylabel('patients number show')


# through this figure We conclude that:
# 
# most patients who show the clinic are undisabled and slightly percentage are handicap

# In[36]:


df.handicap[Show].hist(figsize=(7,5));
plt.title("Comparison between handicap and number of patient who didn't show" )
plt.xlabel('handicap')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients who didn't show the clinic are undisabled and slightly percentage are handicap

# In[37]:


df.sms_received[Show].hist(figsize=(7,5));
plt.title("Comparison between sms_received and number of patient who didn't show" )
plt.xlabel('sms_received')
plt.ylabel('patients number noshow')


# through this figure We conclude that:
# 
# most patients who didn't show They did not receive messages and slightly percentage  received messages

# In[38]:


df.sms_received[NoShow].hist(figsize=(7,5));
plt.title("Comparison between sms_received and number of patient show" )
plt.xlabel('sms_received')
plt.ylabel('patients number show')

through this figure We conclude that:

most patients who show They did not receive messages and a large percentage received messages
# In[39]:


df_copy = df.copy()


# In[40]:


df_copy['no_show'].replace(['Yes','No'],[0,1], inplace=True)


# In[41]:


df_copy.head()


# ### Research Question 2  (Replace this header name!)

# In[42]:


#Comparison between who show and didn't according to Neighbourhood
plt.figure(figsize=[15,10])
df_copy.neighbourhood[Show].value_counts().plot(kind='bar', alpha = 1, color = 'green', label = 'Show')
df_copy.neighbourhood[NoShow].value_counts().plot(kind='bar', alpha = 1, color = 'red', label = 'NoShow')
plt.legend()
plt.title("Comparison between who show and didn't according to Neighbourhood" )
plt.xlabel('Neighbourhood')
plt.ylabel('patients number')


# The neighborhood is strongly affecting the showing patients at the clinic

# <a id='conclusions'></a>
# ## Conclusions
# 
# >  Finally,
# 
# > I would like to say that their is no missing value ,but in my opinion 'patient_id', 'appointment_id', 'scheduled_day'and 'appointment_day' columns didn't affect the results and is 
# expected to mislead the data so that i drop it.
# 
# > In the age column thier is a person has -1 year i expect this is a data entry error so that i drop it.
# 
# >I would like to say that the neighborhood is the strongest element affected to attend the clinic.
# 
# > The number of patients who Show are four times NoShow.
# 
# > Most people came to the clinic without receiving an SMS
# 
# > Most of those registered do not have chronic diseases
# 
# > about 10% are enrolled in the Brasillan welfare program
# ## Limitations: 
# >There is no a direct correlation between attendance and not attendance and other characteristics such as handicap, diabetes, gender,  hypertension, and alcoholism. and neighborhood is the strongest element affected to attend the clinic.
# 
# 
# ## Submitting your Project 
# 
# >  Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# >  Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# >  Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[43]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

