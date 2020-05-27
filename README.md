# IGAudit
 #FakersGonnaFake: Using Simple Statistical Tools and Machine Learning to Audit Instagram Accounts for Authenticity

**Motivation:** During lockdown, businesses have started increasing the use of social media influencers to market their products while their physical outlets are temporary closed. However, it is sad that there are some that will try and game the system for their own good. But in a world where a single influencer's post is worth as much as an average 9-5 Joe's annual salary, influencer marketing fake followers and fake engagement is a price that brands shouldn't have to pay for.

*Inspired by igaudit.io, a very userful tool that was unfortunately taken down by Facebook recently.*

## Got your attention? Great!
- If you want to read through the code and the outputs quickly, head over to the [PDF](https://github.com/athiyadeviyani/IGAudit/blob/master/IGAudit.pdf), [Markdown](https://github.com/athiyadeviyani/IGAudit/blob/master/IGAudit_mdfiles/IGAudit.md), or the [HTML](https://igaudit-by-tia.glitch.me/) version of the Jupyter Notebook
- If you want to get your hands dirty and understand the inner workings of the code, make sure you install the following requirements and run the [Jupyter Notebook](https://github.com/athiyadeviyani/IGAudit/blob/master/IGAudit.ipynb)!

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``$ pip install numpy pandas seaborn matplotlib sklearn``
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``$ pip install git+https://git@github.com/ping/instagram_private_api.git@1.6.0``

## Contents
- Short Introduction
- Part 1: Understanding and Splitting the Data
- Part 2: Comparing Classification Models 
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Trees
  - Random Forest
- Part 3: Obtaining Instagram Data
  - Getter methods for the API
- Part 4: Preparing the Data
  - Understanding the data obtained
  - Filtering and extracting relevant information
- Part 5: Make the prediction!
- Part 6: Extension - Fake Likes
- Part 7: Comparison with Another User
- Part 8: Thoughts
  - Making sense of the result
  - So is this influencer worth investing or not?
- Conclusion
