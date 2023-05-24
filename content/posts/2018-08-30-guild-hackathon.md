---
title: "Deloitte 2018 ML Guild Hackathon"
date: 2018-08-30T13:25:11-04:00
draft: false
author: "Kaushik Moudgalya"
showtoc: false
# cover:
#     image: img/ash_lynx.jpg
#     alt: 'image of me'
#     caption: 'Me'
tags: ["Machine Learning", "Plotly", "sklearn", "XGBoost", "Random Forests", "Neural Networks", "Deloitte"]
---

##### [View this project on Github](https://github.com/mmehta0943/ML-grocery-gap/tree/master)

### Background
Every year the Deloitte Machine Learning Guild hosts an their annual hackathon, which is the largest Data Science / ML hackathon of its kind within Deloitte. Interested folks form teams of 2-4 people and work on the assigned hackathon theme for a duration of 2-4 weeks, after which the teams are judged based on a variety of criteria. The judges are Deloite ML Guild Masters who are subject matter experts in ML and DS, as well as partners who are interested in the field. The winning team gets a cash prize (USD 500 each) and a chance to present their work to the entire ML Guild.

### Team
I was part of a team of 5 people, including myself. The other members were [Rohit Shah](https://github.com/rshah1990), [Swapna Batta](https://github.com/swapnaveer), [Milonee Mehta](https://github.com/mmehta0943) and [Austin Lasseter](https://github.com/austinlasseter). We participated and placed `First` in the 2018 version of the Annual Deloitte ML Guild Hackathon.

### Idea
Access to fresh, healthy food plays a big role in determining whether the population of a given area is healthy or not. In the US, there are many areas where access to fresh food is limited, and the population is forced to rely on fast food and other unhealthy options. This leads to a variety of health issues, including obesity, diabetes, heart disease, etc. 

We wanted to use ML to help local governments decide where to provide benefits and subsidies to open new grocery stores. Governments could also decide to increase SNAP (food stamps) benefits for the people in these areas, or undertake some other measure to address the problem.

Our work is based on research conducted by [Alana Rhone](https://www.ers.usda.gov/authors/ers-staff-directory/alana-rhone/), who I believe is also the one who graciously made this data public. 

### Data
We used publicly available data, the [USDA Food Environment Atlas](https://www.ers.usda.gov/data-products/food-environment-atlas/data-access-and-documentation-downloads/#Current%20Version) to predict which counties in the US are most in need of a grocery store, and then use this information to help local governments decide where to provide benefits and subsidies to open new grocery stores.

### Approach
Based on health based factors such as BMI and the incidence of diabetes, we created a predictive health dashboard that predicts which counties in the US are most in need of a grocery store. We used a variety of ML models, including Random Forests, XGBoost, and Neural Networks to predict the health of a county based on the food environment in that county. We also used Plotly to create an interactive dashboard that allows users to explore the data and see the predictions for each county.

We came up with a small matrix based the predictions of our models and the actual health of the county, and used this matrix to rank the counties in the US based on their need for a grocery store. This matrix is based on two different models that we trained, a model which predicts whether the number of grocery stores in the a certain country will increase or decrease and another model predicts whether the diabetes rate in a county will increase or decrease.

Based on these two models, we create a matrix as follows:
| Grocery Stores | Diabetes Rate | Recommendation |
| ----------- | ----------- | ----------- |
| Decrease | Decrease | Monitor concentration of stores & population served|
| Decrease | Increase | Open more stores / provide more SNAP benefits|
| Increase | Decrease | Ideal|
| Increase | Increase | Monitor concentration of stores & population served|

We then used this ranking to create a map of the US, where the counties are colored based on their need for a grocery store. This map can be used by local governments to decide where to provide benefits and subsidies to open new grocery stores.  

### Future Work
- I want to check whether I can somehow add to the analysis performed by Alana and her colleagues in their work.
- I want to redo this project from scratch, and check how my current skills compare to my skills when I first did this project.
- Presentation is everything. I want to redo the dash app where this was deployed and figure out if there's a way to keep it running without having to pay for it. Maybe another github.io page?
- Are there any drawbacks to combining the predictions of the two models like this? Does it matter whether they are trained on the same data or different data?

### Ulterior Motives
Everytime I have to prepare for an interview, I need to go through the repo that we made for this project, which is a headache. This page helps me ease that process. Additionally, I can use this as a resource to present in case I talk about this project during my interviews, especially if I make the app live again.
