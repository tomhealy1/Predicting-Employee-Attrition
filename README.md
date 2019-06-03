# Using the IBM Dataset to predict churn. 

##We will be folloing Susan Li's epic tutorial. I have added a few bit out of necessity but have tried to stay as close Susan's code as possible. The next steps I would like carry out are:
1. The use of the Random Forest Model as a pickle file so it can be packaged in Flask and called as an api
2. Drawing up HR initiatives to target these drivers of attrition. We have 10 and the most important appear to be:

* promotion_last_5years-0.20%
* department_management-0.22%
* department_hr-0.29%
* department_RandD-0.34%

We would take a look at the main blocker for people within promotion cycles. Having sufficient mechanisms to support development is a key pillar of retain people. This is a fictional dataset so we can make things easy for ourselves and say that they did have the framewsork or guidlelines in place to facilitate the development of people. The departure of talent from the three departments is worrying. The use of exit interviews and targeted survey to better understand is required.  

