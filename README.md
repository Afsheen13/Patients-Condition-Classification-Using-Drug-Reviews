# Patients-Condition-Classification-Using-Drug-Reviews

### Business Objective:
	
This is a sample dataset which consists of 161297 drug name, condition reviews and ratings from different patients and our goal is to examine how patients are feeling using the drugs their positive and negative experiences so that we can recommend him a suitable drug. By analyzing the reviews, we can understand the drug effectiveness and its side effects. 

The dataset provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction.
So in this dataset, we can see many patients conditions but we will focus only on the below, classify the below conditions from the patients reviews 

a. Depression

b. High Blood Pressure

c. Diabetes, Type 2

![image](https://user-images.githubusercontent.com/118348424/235468233-b5a58b2c-ab16-420f-84de-2db39522b805.png)

Attribute Information:

1. DrugName (categorical): name of drug

2. condition (categorical): name of condition

3. review (text): patient review

4. rating (numerical): 10 star patient rating

5. date (date): date of review entry

6. usefulCount (numerical): number of users who found review useful

### Manual EDA using bar charts and wordclouds.

![image](https://user-images.githubusercontent.com/118348424/235468669-7ed37847-74dc-4205-85fa-6f3716d39e42.png)

![image](https://user-images.githubusercontent.com/118348424/235468698-31289fd9-a8e6-41a2-a721-038fa5a7f916.png)

![image](https://user-images.githubusercontent.com/118348424/235468810-adee9ad9-54d0-4264-8334-d15ffd2f1040.png)

![image](https://user-images.githubusercontent.com/118348424/235468825-d3799de9-f7c1-43de-87e1-126a6e9c83bf.png)

![image](https://user-images.githubusercontent.com/118348424/235468888-0b27c038-136f-445c-906e-48e909e88c87.png)

![image](https://user-images.githubusercontent.com/118348424/235468926-b42de5a6-c158-44a9-bcdf-e2d5d314c5cb.png)

### Word clouds of reviews for 3 conditions!

![image](https://user-images.githubusercontent.com/118348424/235469057-03d93210-2628-4425-a206-773c582fb0d7.png)

![image](https://user-images.githubusercontent.com/118348424/235469074-082d8aae-2098-4fdb-b913-a466ce4be157.png)

![image](https://user-images.githubusercontent.com/118348424/235469091-623a33b3-0c59-4556-8830-c38732d4ac04.png)

### AUTO EDA (sweetviz)!

![image](https://user-images.githubusercontent.com/118348424/235469217-69e049ac-ffef-4257-bc84-6a74fe9cd84c.png)

### Auto eda (pandas profiling)!

![image](https://user-images.githubusercontent.com/118348424/235469775-2baa65a6-34c7-41fc-9f4a-f0ade1e96034.png)

![image](https://user-images.githubusercontent.com/118348424/235469987-bf82ed39-d52e-4c90-8a09-5c851c064084.png)

### Stop words and example data without stopwords!

![image](https://user-images.githubusercontent.com/118348424/235470103-1b60c617-30bd-4db7-9c47-1fdc8254afa9.png)

![image](https://user-images.githubusercontent.com/118348424/235470237-d327f392-9763-47f9-ad0d-50321c7d251e.png)

### TFIDF Random forest classifier!

Model was able to correctly predict 1811 cases of depression, 475 cases of Diabetes type-2 and 405 cases of High blood pressure with 96.5% accuracy

![image](https://user-images.githubusercontent.com/118348424/235470582-314b224a-24bf-4db6-b6ca-be9ebe1b217f.png)

### Deployment using Streamlit!

![image](https://user-images.githubusercontent.com/118348424/235487853-9a07503e-81fd-413e-a634-9b7cf693ecd1.png)

![image](https://user-images.githubusercontent.com/118348424/235487886-1a6a3093-53bc-4544-bd9e-80a0bc310975.png)

![image](https://user-images.githubusercontent.com/118348424/235487922-8475cfc9-c5f2-4bd0-8424-f3d30cb9a9b3.png)










