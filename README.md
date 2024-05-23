# **TELECOMMUNICATIONS CHURN PREDICTION PROJECT**

---
---

## **Overview**

Within the dynamic telecommunications industry, constant evolution is imperative to meet consumer demands. Technological advancements and shifting user behaviors present ongoing challenges for telecom operators, influencing their business trajectory. To remain competitive and responsive to customer needs, diligent data analysis is paramount.

In this project, my objective is to delve into telecom data to uncover pertinent challenges encountered by operators. Through an in-depth exploratory data analysis (EDA) of extensive telecom datasets, I aim to extract valuable insights encompassing user behavior, network efficacy, customer demographics, and beyond. By discerning patterns and anomalies, I endeavor to pinpoint areas ripe for enhancement, such as bolstering customer satisfaction, optimizing network functionality, and driving revenue growth via targeted marketing initiatives.

The findings derived from this endeavor hold the potential to empower telecom operators in shaping informed business strategies. By aligning with customer expectations and market demands, operators can fortify their competitive stance and elevate service provision standards.

---


## **Business Impact**

### **Data Analysis**

Data analysis plays a crucial role in uncovering actionable insights that drive informed decision-making within the telecom industry. By conducting comprehensive exploratory data analysis (EDA), telecom companies can gain valuable insights into customer behavior, network performance, and market trends. These insights enable businesses to:


* **Enhance Customer Satisfaction:** By understanding customer preferences and pain points, telecom companies can tailor their services to meet customer needs more effectively, thereby increasing overall satisfaction and loyalty.


* **Optimize Network Functionality:** Analyzing network data allows telecom operators to identify areas for improvement in network performance and reliability. By optimizing network functionality, companies can provide better service quality to customers, reducing complaints and churn rates.


* **Drive Revenue Growth:** Data analysis helps telecom companies identify opportunities for revenue growth through targeted marketing initiatives and service innovations. By leveraging insights from EDA, companies can develop and launch new products or services that resonate with customers, leading to increased revenue streams.



### **Customer Lifetime Value (CLTV)**


Understanding Customer Lifetime Value (CLTV) is instrumental in devising effective customer acquisition and retention strategies within the telecom industry. The business impact of CLTV analysis includes:


* **Optimized Marketing Strategies:** By identifying high-value customers with the potential for long-term revenue generation, telecom companies can tailor marketing campaigns to target these segments more effectively. This targeted approach maximizes return on investment (ROI) and reduces marketing spend on low-value customers.


* **Improved Customer Retention:** CLTV analysis enables telecom operators to prioritize efforts to retain high-value customers by offering personalized incentives, loyalty programs, and enhanced service experiences. By focusing on retaining valuable customers, companies can reduce churn rates and increase overall customer lifetime value.


* **Revenue Maximization:** By understanding the lifetime value of each customer segment, telecom companies can optimize pricing strategies and service offerings to maximize revenue generation. This may involve upselling or cross-selling higher-value services to existing customers or introducing premium tiers tailored to high-value segments.



### **Multi-level Churn Classification**


Implementing multi-level churn classification techniques enables telecom companies to proactively address customer attrition, thereby minimizing revenue loss and maximizing customer retention. The business impact of multi-level churn classification includes:


* **Improved Customer Retention:** By accurately predicting churn behavior and reasons, telecom operators can implement targeted retention strategies to retain at-risk customers. This proactive approach helps reduce churn rates and increase customer lifetime value.


* **Enhanced Operational Efficiency:** Multi-level churn classification allows telecom companies to allocate resources more efficiently by focusing retention efforts on high-value customers with the highest likelihood of churning. This optimization of resources leads to cost savings and improved profitability.


* **Personalized Customer Interactions** Understanding the reasons behind churn enables telecom companies to personalize customer interactions and offer tailored solutions to address specific issues or concerns. By demonstrating a commitment to addressing customer needs, companies can foster stronger customer relationships and loyalty.


* **Competitive Advantage:** By leveraging advanced analytics for churn prediction, telecom companies can differentiate themselves in the market by providing superior customer experiences and service quality. This competitive advantage helps attract new customers and retain existing ones, driving long-term business growth.



### **Churn Prediction**


Churn prediction empowers telecom companies to mitigate customer attrition and preserve revenue streams by identifying customers at risk of churn. The business impact of churn prediction includes:


* **Reduced Churn Rates:** By accurately predicting churn behavior, telecom operators can implement targeted retention strategies to intervene before customers defect to competitors. This proactive approach reduces churn rates, preserves customer relationships, and maintains revenue stability.


* **Enhanced Customer Retention:** Churn prediction enables telecom companies to understand the underlying reasons for customer churn and address them proactively. By resolving issues related to service quality, pricing, or customer satisfaction, companies can improve overall customer retention and loyalty.


* **Cost Savings:** By focusing retention efforts on customers with the highest propensity to churn, telecom companies can optimize resource allocation and minimize the cost of customer acquisition. This efficient allocation of resources reduces marketing expenses and enhances profitability.



### **Offer Recommendation**


Offer recommendation systems leverage advanced analytics and machine learning algorithms to personalize offers and promotions for telecom customers. The business impact of offer recommendation includes:


* **Increased Revenue:** By recommending personalized offers based on customer preferences, usage patterns, and demographics, telecom companies can stimulate additional purchases and increase customer spending. This targeted approach drives incremental revenue growth and enhances overall profitability.


* **Improved Customer Satisfaction:** Offer recommendation systems enable telecom operators to deliver relevant and timely offers that align with customer needs and preferences. By enhancing the customer experience and providing added value, companies can strengthen customer satisfaction and loyalty.


* **Enhanced Marketing Effectiveness:** By leveraging insights from offer recommendation systems, telecom companies can optimize marketing campaigns and promotions to target specific customer segments more effectively. This targeted marketing approach increases campaign success rates and improves ROI on marketing investments.


---


## Project Parts

### Part 1: Exploratory Data Analysis (EDA)

The EDA notebook explores the telecom dataset to gain insights into the data. It analyzes the distribution of features, examines correlations, and identifies patterns that could be useful for building predictive models.

### Part 2: Data Preprocessing and Feature Engineering

This notebook focuses on preprocessing the data and engineering relevant features for the churn prediction model. It handles missing values, encodes categorical variables, and scales numerical features to prepare the dataset for model training.

### Part 3: Model Training and Evaluation

In this part, several machine learning models, including logistic regression, random forest, and XGBoost, are trained and evaluated using the preprocessed dataset. The performance of each model is assessed using metrics such as accuracy, precision, recall, and F1 score.

### Part 4: Deployment and Inference Pipeline

The final part of the project involves deploying the trained model into an inference pipeline to make predictions on new data. It discusses monitoring data drift, handling missing labels, and updating the model based on changing business requirements.



---


## **Execution Order for IPYNB Files**

To ensure proper execution and understanding of the project, follow the order below when running the IPYNB files:

1. **01_data_analysis.ipynb**: This notebook contains exploratory data analysis (EDA) to gain insights into the telecom dataset.

2. **02_customer-lifetime-value.ipynb**: Proceed to analyze Customer Lifetime Value (CLTV) to identify high-value customers and optimize business strategies accordingly.

3. **03_multilevel_churn_classification.ipynb**: Explore churn prediction by categorizing churn behavior and reasons using multi-class and multi-label classification techniques.

4. **04_churn_modeling.ipynb**: Focus on building machine learning models to predict churn and understand the importance of monitoring data changes.

5. **05_offer_recommendation_system.ipynb**: Lastly, delve into creating a recommendation system to provide personalized offers based on customer behavior and preferences.

Following this sequence ensures a structured and comprehensive understanding of the project's objectives and outcomes.



---


## **Execution Instructions**


### **Installation and Setup for IPYNB**

For the best experience, please stay connected to the internet while executing this Project

#### **Running an IPYNB on Google Colab**:

* Open the [Google Colab website](https://colab.research.google.com/).
* Click on the "New Notebook" button.
* Click the "File" menu in the new notebook and choose "Upload notebook."
* Select the IPYNB file(notebooks/churn_modeling.ipynb) you want to upload.
* Once the file is uploaded, click on the "Runtime" menu and choose "Run all" to execute all the cells in the notebook.
* Alternatively, you can execute each cell individually by clicking the "Play" button next to the cell OR by pressing `"shift” + "enter"`.
* The Default version of Python that Colab uses currently is 3.8

#### **Python setup steps for Local Machine**:
* If you're using a local machine and do not have Python installed, follow these steps to set up Python:
* Download and install the latest version of Python from the official Python website: https://www.python.org/downloads/.
* Once the installation is complete, open a command prompt/terminal and type the following command to check if Python is installed `python –-version`
* If Python is installed, the version number will be displayed.
* This Project has been created using **Python version 3.8.10**

#### **Setting up a Python Virtual Environment on Windows**

* Open a command prompt by pressing `Windows Key + R`, type `cmd`, and press `Enter`.
* Navigate to the directory where you want to create the virtual environment.
* Install virtualenv by running the command in the command prompt `pip install virtualenv`
* Create a new virtual environment by running the command `virtualenv env`
* This will create a new directory called `env`, containing the virtual environment.
* Activate the virtual environment by running the command `env\Scripts\activate`
* You can now install packages and work on your project within this virtual environment.
* To deactivate the virtual environment, simply run the command `deactivate`

#### **Setting up a Python Virtual Environment on Mac**

* Open the Terminal by pressing `Command + Spacebar`, type `Terminal`, and press `Enter`.
* Navigate to the directory where you want to create the virtual environment.
* Install virtualenv by running the following command in the terminal `pip install virtualenv`
* Create a new virtual environment by running the following command `virtualenv env`
* This will create a new directory called `env`, containing the virtual environment.
* Activate the virtual environment by running the following command `source env/bin/activate`
* You can now install packages and work on your project within this virtual environment.
* To deactivate the virtual environment, simply run the following command `deactivate`

#### **Setting up a Python Virtual Environment on Linux**

* Open the Terminal by pressing `Ctrl + Alt + T`.
* Navigate to the directory where you want to create the virtual environment.
* Install `virtualenv` by running the following command in the terminal `sudo apt-get install python3-virtualenv`
* Create a new virtual environment by running the following command `virtualenv -p python3 env`
* This will create a new directory called `env`, containing the virtual environment.
* Activate the virtual environment by running the following command `source env/bin/activate`
* You can now install packages and work on your project within this virtual environment.
* To deactivate the virtual environment, simply run the following command `deactivate`

#### **Installing Jupyter with pip**
If pip is installed on your local machine, you can install Jupyter. 

Here are the steps:
* Open the command prompt (Windows) or terminal (Mac/Linux).
* Install Jupyter with pip by running the following command `pip install jupyter`
* Launch Jupyter Notebook by running the following command `jupyter notebook`

#### **Installing Jupyter with Conda**

* Download and install Anaconda from the official website: https://www.anaconda.com/products/individual.
* Open the Anaconda Navigator App and launch Jupyter Notebook 
* Running IPYNB in Jupyter Notebook
* Open Terminal / Command Prompt and Navigate to the notebooks directory using cd
* Launch Jupyter Notebook by running the following command jupyter notebook
* This will open a browser window displaying the Jupyter interface.
* Click on the IPYNB file you want to open.
* To execute all the cells in the notebook, click on the "Cell" menu and choose "Run all".
* Alternatively, you can execute each cell individually by clicking on the "Play" button next to the cell.



# Executing the project via Modular Code
* Install the dependencies using the command, navigate to the **Telecom Machine Learning Project to Predict Customer Churn** directory where the requirements.txt file exists, and run ***pip install -r requirements.txt*** in the terminal/CMD.

* Navigate to the src folder in the project directory using the cd command 
* Run the command python Engine.py in Terminal or run the Engine.py file in either VScode or PyCharm.

![churn_project_structure.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABRAAAAL1CAYAAAC/ssg7AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAI0jSURBVHhe7f09axvL/wBuz/duXP3ehSAEQRp1fgWGHFA6dwF37twEly5SijTu3AnSuYvhBPQK3KkJGBNQ/b/v/l+pOvfO7upZa83KK0u2rwv2nEhe7+PM7M7H8/C//9//9//5LwDAQVh+JP2v/H+qur+/af26PwfYk03FUd3ilHr+8zzgGf63IYNuSl91f7/p/TWd/utuv+nzf2+WL6/LQ4X/jcfjhnM7AAAAAPBW/H/K/wMAAAAArBBABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJV2FkAcnB+Fo/NB+QkAAAAAeI20QAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiGEQzo+OwvH1qPwMAAAAAEwIID6bACQAAAAAb5cAIgAAAABQSQARAAAAAKj0DgOIo3B9fBSOjiZLN/TLnywYXYfj6TrFcj4of5YZXR8v/P7wsr2w7kqX5g3bAwAAAIBD9L4CiHkQrx1uTx/CeDwul7twVv54ZhDOv4bwc7rOODz0OqHfPQ6TuGDr4n7h9zu9+W2Ow/1Fq1gxt3l7AAAAAHCI3lUAcfDjMgw7vfBzIbi3zkm4ub8I82u1Lq7CWRiG29/bRPya3h4AAAAAvIzdBhD73YUuu0dH/4R/Fj5XLanrnYf0XsCD8KsfQuf080IgDwAAAACottsA4tndQrfe8fjf8O/C56oldb2bcFLuqmmzMQ4nS8VYiYma3h4AAAAAvIR3OInKZjHY174chrO7+WDlurES0zS9PQAAAAB4Ke8ogNgKHzshDB8XxxwcXX9fagk4Cr9vh3FWlPAtqXnj+u3O1N0eAAAAAByOdxVA/HzaCaH/fTrzcd4y8PZTOMu+nmmFD5+y/w1vw3R+k3z25qoux+X6c9tdVHd7AAAAAHA43lUX5tbFfbg7G4bLdjEOYfvxKozvv4WP5c8nTm4eQq8zW++ofRtOH+J35QpLVtbPluO5aGLd7QEAAADAofjfeDz+r/x3owbnR6Eb7sL4Rr9dAAAAAHitTKICAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgErvNoAYJ3k5Oh+UnwAAAACAdbRABAAAAAAqCSACAAAAAJUEEAEAAACASgKIwCs2COdHR+H4elR+rmF0HY6z3z2aLMfXYYutvB7v7XzXekZ6eS2m9/k8O1sAAIBmCCC+efuqML+DijqvW+si3I/HYZwtd2fld29ZrfOVfwEAAJgRQAR2oAhArbSC0joKdmsaKL4JJ+VXAAAAzyWACO/Q6Pr4hYJ4/fB9rhXb6PdtGJb/BgAAAF4HAcSDM2m5NVvOl6I864M/5e+VKxfrxN/vhn72eXjZnm4vLvNdE/N18/HQFve93H2x6f2yH4Pzo9C+HIZO79uOWyh1Qq93Foa3v8ux9kbh920Ivbte9pM/4e9CUtic7gujcH08v16RzlYtr3ccdp708taVxX5m+aBq36nHt3xdVtdLyZep6uffzcdXy/I4jdmytvxbU16tP9fU9JKqxvkOzufWy5a1x7e8vTXpPjVdLV+7ijEua12/NfejWJ55nwEAgFdHAPGQ5JW1bvjTe8jHKYvLQ68T+t2qYEq11sV9uY27EIc768xtMy73F61ixYnhZWhn+w535ToPvRAu27vfLy+oCKZ0+yGcZfd5p/di9Df8if///CWcDW/D7xhsGP0Ot+E0fF7ebWq6z9drh9vT+TRVpLNFMTDSDpef7ua2F5Nz/XxU3zDfT/vxqtz3Q+h1su++zgdzUo8vrtcN/bPZeuPxVXhc2Faz6uXfpo8v297XEH7O7a9IB2uCVcvlVRzUsd/dMr2kSj/fPEiXZbSYz6brfvm15vhSy/uEdFVnjMvU69e+DGF6fHGf8Qed7Kv7oCgHAID3RQDxYIzC9dfLMMwqp/OV9Fihz+t233cXNJiIld2bSZO0rDJ69UL75SWUQathrPzP3eedOwnfeiHc/h6FwY/L8OnqIizGHdLTffz9YacXfm6IXIyuv4d+OAt3cyfZuviZBz/6v1YiM82LAaZZRgqfT7MdDx+n55F8fGUQ9uzL/M06CTf3y9dwTxo/vtXfbV1cZVdqmKefZQvl1cmXPDD4Z65Za2p6SZZ6vqPr8DVv4fuwmM9ObuY+b1Heb0hXdW26fsVwA2fhanp8rXDxM7YeHobHbXcKAAC8Wu87gNjvLnXL+if8s/C5akldr8YYc7F11jCEzsfVyu7Jl6xqN2nFtTNnYaFenGl9fF4FlQNRtnTqd3rhYfzyLYdan09DuP0avvfn01gZhEhO94Pwq5+td/p5Q3AqdpPONnj2JSwm51b48Cn735+/O0/PiwGmbM95q77JhBY1jq/1IcSvtmmB/CL2enyr5VU0nEa2UtNLDYnnuxp4W2OL8v7pdFXXpusHAACw6H0HEGOLjrLLV7H8G/5d+Fy1pK5Xv3L36UNj1V3I5S2x4j8SgsFxfMTlQPhW41aOHrN9fgp5cm59DqfZp+FK0GymuXQ/Co/xZFf+OFB03d5/QLzO8Z2Em7LbaAxaFesd0thzzR/f4hh/cXnumIVNSjvfUX6D0xxyeR8D/53s6s8mQSpbTVYEHwEAgLdNF+YDM9+FbGL0Nx9N7sXlFeHOx+Za8LAXJzcxmB3HfuuH7oYAT7Hu4vL8sRJb4eI+29akv2TZkmtec+m+FWLD2dU/DkyWbVtsNaXu8ZXXLv9ZvIfFWHiH0yKxueOLwcM4uc/CuIHPGrNwFzafb95yO9Ehlfcr8j8ChLmJdNrhMsRWzPvOQwAAwD4IIB6K2Eorq3eu60JWBPLWTD4xbzJpxYoiYFG/a9oo5PXYTx+yLTyh8f2yG5PWU4cTgMqDJ8npfn16KsYTnPdyXZW385zji/ewCKitCzwtqMyXqbbJvzWOb0XZtbvTC98aiU6lppfnWH++rfwG98P8cJYrnlve79woXH/PrtRyoPtQxt8EAABenADiwWiFi3zWku5Cl9HYKqfb74Tez1nFbdK1bFpBLWfLjK1FVpUBi/73Wl0LB+dxwo2liR5eYL/sUtF6qphwdcuuyYnSW1Glpvty0oi59JS3WLv9FM6yr+edfOuFTpxl9nCa6S1IPr7B+co9KgJgnXA6F12qly9TJeTfxONLU+5vfuy/ydid5cd60tNLstTzPflWdnNeau2b/f7slqeX9/sxuf+/0sfxBQAA3jQBxENychPGd2dzXcaOQvvyU7hbnviidRF+ZjXU6Thc7cdwVY7Ntc7Jzazl2WS7yxXhrKYYuuXP4tL9s6ar2k72y0uL3ZQfsps2vPxxGMGBxHRfzFA7S0/tx6swvv8WPpY/n8rS6X1sGbZmnMHlYM3k+2L8wcvQXrNe4xKPL16Xn+Hrws/blyH0HrYrD+qe78b8m3p8iVb2174Npw/V5csmyeklVfL5TgL1i9fu6NeXlVmZk8r7RE2n55Ob2Lpy8bkwWQ40Ng8AAOzQ/8bj8X/lvxsVJ2PohrvZuGcH5tCP7yXlLXPyiquxrQAYhPPJzO1L3ZbzZ2dsJblloBgAAHidtEAEAGbKMTQ7p59XulIXk8QMQ63hMQEAgFdPABEAmGkVM6UPb38vTfYzCD8um5zsBgAAeC0EEAGAOeWs7WE2jmKxdEM/zsxsNmYAAHh3BBDJJxsYG/8QgKliMpjxeGkxbjAAALxLAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSAyBs3COdHR+H4elR+5tCMro/D0dF5dqf2qUgns9lmj8L58gGNrsNx/rMXOtbp/srl+Do0n4rlj8OQkP7epGekv7r546XzLwAAvDECiMA7F4MY3dA/u1uYbXbvk822LsJ9eSx3Z+V3B+etBCD3eR4Hmv4O3avIHwAA8HYIIALNW24dtLIch4OJOQ1+hX7ohN63DRGbacDiJojt0JjU9MfzyL8AAPAsAojwDu282/Bc66Dx+CH0Otl3nV54mH53Hy5axar7Nvr7J/vvp/DhQI6H90X6AwAAXgMBxAOSB3XycZyK7nST1lor3ery1l1FC64iEPRUq67FbcWlcmytwfnCekdrVxyF6+O5dSpbki3v97nrpe53eb1u6Jc/oTA4Pwrty2Ho9L4dWEucxXu3vjtpajpoUOJYa1X5d20+WtNC82XGvEvMHxuOb1buFL8/vGwvrLu+zJr9PC7PO99myo3a55G836Yt73fN9av1XNhguq3ZfuP+pttdyQMJx5dLLZ8bzOfLaW9H+bdYXio9AADAyxNAPDTDy9DOKlXhrmyp9dALIavUrtZhhtnXR6H9eDXXyiv77utc5Siv5HTDn95Duc44PPQ6od9drdzlladuP5xN9huXL7+W1osVq3a4/DQbq6s4vOXtxfWyiuHCmF5X4XH+2HJ11kvYb36+7XB7Ojvf8fguGB5roqiUZ7c5v8/3h9IEMNcP3ezePV6V9+3uLA/mbJf+EswFy2Mwtdj/XCBgPshQZ6y15fwbf6HfXT2PryH8LLdZnEfMlzsOPiTnj83H17q4X/j9zlwZE5fFtNX0+cZ00Ey5Ufs8kvaboE76q1GOJz0Xarj9+j18fChaEMf95dvNLmInS+c/JvtOPb466a+pfB41nX/jebQvQ5ieb9nCOnZDfzicltUAANA0AcQDFIM70wH0s8rPVazDfF9TAYwV2dmK4fNpVosZPpbrjcL118swzNaZrwTHCnNeJ5rfXlYh+pq3SHtYHLj/5Gbh8+j6e1bNPQt3c1+2Ln4WlctfczWs0d8QO+WdfVnYWLi5v8iOck7ieqn7HfzIzrfTCz/V4NYoK+XDWMk9zAkaFtL9yZc8sPDn7yzVJ6e/FFnangUnYu0/2275OV+W02oNm85jXRpvXVxl6w3D7e+VXN6Y9PzR9PE1vL2Gy41kqeVaiuT0V6Mcn3jyuVDHMAw/XYWLVit8+BQ/rxunMf34UtNf4/etpo3l0O/b7MqchavpebTCxc9ednWG4XGL5AwAAK/F+w4g9ruzFh/58k/4Z+Fz1ZK63jZjzJ2FhfpppvVxfQVwsSJbVNqmA8SPfofbYVbl+7haWTv5klWJhrdhUndfrRCtMwq/4wbPvhTbnyorl3/+zo6v9SHEr9a3kJmTtF7qfgfhVz8739PPZcWbqbKFUD8fg/BQW8ispvtoOK2R10h/e7XpPPblDeWPRsuNGlLLtSbVKMcnnnwu1LSwrc5p+Lx8GMnHl5r+9p3PDzX/AgDA/r3vAGJsqTHf6mP8b/h34XPVkrre/md7/JQwMv/oMauwbTQK+WorQdeiS+xigPMk3JTdumJlu1hvXXfFlPXq7Jd18pY/8R8J1yqOj7h8ndePRfjS3lY6WByjLi7dEE/jUDR9fM1ub1/lRmq51ryUcnyfmju+w8/nrc+noZOl3u/TG1+2wqwIPgIAwFuhC/MrkAf4Oh+3ajm02HWyUMz6OZO3cNyoFfLVVoKuVcHSVri4n/wsjnNVjM212nJn03p198uyk5t4neK1jeOsPR3wKNZdXBbHgduXt5MOYjAtjnsXu0rOjn/dWHD70fTx7eZ891VupJZrzUopx/epueN7Bfl89Jjd9TA34U47XIbYutuzCACAt00A8eCNQl4P+/Qhq1rV0PociqGv1lTs8oDkrDtaK+8b1g9PDy/1nC5kseVOETBYV9GcWbde6n6Liufy+RbjaTFrPfUyAY/mHVpX5W2VXTQ7vbAynNxOpeaPuse3frszL3G+zyk3Jjadxzqp5doz1CjH9yL5+FLT36Hn81G4/p4d8XKAc5txMAEA4JURQDxwg/M48cXigPJpWuEin32lu9AFNbYG6vY7ofdzrsJz8q3slrfUOm1wvhBoOvlWzL7Z3hR9yn5vudtrUVHshNP52m7iemn7LScL6H+fnkPe8un2UzjLviYqWk8VE4seStfkdMnp76CVAZL5sesmY1SWH3cjNX/UPb5y/bntLtrB+TZabkxsOo9MarnWqBrl+F6kHl96+XzY+XySTn5tMb4xAAC8bgKIByd2M50b9+nPM7pGxZk+787mulodhfblp3C3MpHGJLBUtE6brHv068vibL2ti3AfW9ysGZ9qoWKd7fdn+Lrw8/ZlCL2Hpf2mrpe432Lmz9k5tB+vwvj+W/hY/pxC7KYcZ34dXv54XZXg1PTXsBjomOynGIftMrSfsd+Tm1lL0Hy77dtw+lCMrTev6f2m5o/U45tYWX/p+Opub6OGy42JTeeRvN+mJZfje5J4fMnlc+J9S80fzeff7NiWntOT5VX/bQMAADb433g8/q/8d6PiZAzdcBfGtVvOvYxDPL68RUZe8TKWEgAclkE4j61n44z2S92W83eK2Opy1wFlAADYEy0QAQA2Gf0NcUjizunnheBhVExGNgy1htEEAIBXRAARAGCT1odQDOn5e2mSl0H4cbmPyZEAAODlCCACAGxUzmYfZuMoFks39OPMzGZjBgDgDRNAPCBxkPmx8Q8B4EAVk46Nx0vLgY73DAAATRFABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAURgr0bXx+Ho6DwMys+v3ug6HOczszZ0Tk1vb5Pp/srl+DqMyh+9iJc+3z1JTvf7vh8LBuE8O4bj6/0dQTPeynkAAMDLEUAEYKZ1Ee7LmWXvzsrvKO0h8PSm7ofAHQAAvFYCiEDzlltNrSzH4c3GEKYBn5twUn71LE1v79C9t/MFAAB4BQQQ4R3aebfhuVZT4/FD6HWy7zq98DD97j5ctIpVAQAAgMMmgHhI8lZbRcusIsDzRGutNS28zueiQYPz7LvKsbJ0I3vPYtpoXw5Dp/ftwFp4jcL18Sw9r0+fi+s8vyVjkRdm21vMR7nUfFlnrLrKFppz20zcXn5M+c+WzmXlRDIbyo29SjnffJ14jWbnGo9/el8mv5O6Xr5q/G45mF7+XnlxZve9G/rZ5+Flu/xcLCtptdZ1XkrTz7ohTeeP5e0V579iw/nu9vql2HweKemgOK4a6WpNvvTcBQDgtRJAPDjDcNk+Cu3Hq7nWW9l3X+cr01mF5GsIP6etucbhodcJ/e6sstj62Mk29Tj3O1BUortZzfnsbhzuD6oJYD90j9rh8apM03dneZBhMWgQK+LtcPnpbi7dhzy/bBVcyIMB3fCn97CUj9ZtLyFfpo5VF/fbvgxhut+yhWboZF/NtcxM3V40vAzt7FxCdl/zbcZf6HdXr9+GcmOvapzv7dfv4eNDcd3i/crvS5YYOtl1+DF3zqnrbdK6uC+v2V2Ih9aZSzNxWcxLda7zUrqPx5bdt+2CTLvIH+1wezp/rsX5L0p4Hu3s+iVIPo90yelqOV8WN2S7+wEAAHsmgHiIzrIK4M2kbVgrfD5dDgaehJv7i+wnM62Lq6xCNAy3v4u1Wh8+Zf/9E/6Wv5S3SJzUWkZ/s5+E8OnDIQWQ2K0yuDCMQapxmCavAxKDmtPjOvmSV/D/TBJwZnT9PfSzb+/mDr518bOoxP+qWyMfheuvl2GY5bX54EUMdOSxt+/zAfvSxnyZZvT7NsupZ+FqFikMFz97oZN9+1h3Y3M2Xb+UcuN1GIbhp6tw0WqFvJiLgddv6xJ06npNq3edF+5b6yJcZTduePu7frpqNH9kJcaPLH90euHnQnBvnabTVbPbSz+PVPXS1br7u7Z8AQCAA/e+A4j97rRbUbH8E/5Z+Fy1pK633RhzZ18WKyNF642aEwq0PmbVmolB+BX7a/V/zR1PJ3xsqj7FYStb2vXzMQgPdezBs7CU7HPDaURtFH7fDmPmWMoHZSX+z996FfLR7xA311mTCU6+5BGcsByraCRf7sym6/e2LNyLzmn4XJGmU9fbn9X7tl3r8YbzR/nM6Jx+XgjkvT67OY/0dNXU/QUAgP17twHEk5tJV6b55d/w78p365bU9XYXXJiNKTVZlsZ0an0In0LZomnwK/TPeqHX6Ye8IcroMfvJp6AB4vuQt8CJ/0iotOYtVRfS1aGM2TUKj/EkVoL+RZfsbSvk+2iF2/p8GjpZbv0+va5la8iKIGCTNpYbNOLlr/Nu8keqps9XOgUAgMOjC/MrFCtXcRKM2DVqFqxcHtOpFWJDh9iFcfT3T+h8/Jx3uYxd2eLn7ItX3rKEVEWwPKaPON7a0+OIrQusH8ZYiUV6zrsRLx1fsWwXrF/s4lvI88cu5QH8MDeRRDtchtg6dLetGdPKDZ5rP9d5N/kjRdPn+9bT6ShGej1/AQB4hQQQX52yq1qnF54ezqvoujZ8/J2tH8Lp51bR8mnSle3TBxWYd+Uk3Ewm/th2UoW92rYrZoXW51AMYbgmgJhX8HfV3XUUrr/3VwM9S2O+NS+13NhGHF8zBkK3G7LhIJXjxK4qAnXVXcOfe52ruiJv0nD+qDjPYpzFeXXPd9fXb1nqeVSoTAfbGoX87xOevwAAvEICiK9OWVGcH6NtMsZd+XEiH2upfxkuQxkMid2ah5ehe5lV0HiHWuHivpjhNs4cehhdk9OdfCtmOW03Ev3MrkU+m8HijLex9VO33wm9n7sK6JX5d2E80peQXm7UFodIyP8x3y379Zh0KZ/OMxKvS7vs9r9icv++V7Tkfd51Hl1/LSY62iJ61nT+yCcJmjvPvGXg7adwln09U/d8d3v9VqWeR7ZmrXSwncF5nMhqcaIbAAB4LQQQX6GTm1lLsrwLZPs2nD7E78oVlk1bO5yEOD9EtG7yCN6H2E35IUssw8sfr6vFWOsi3MeujGvGedsqGHpyE8Z3Z3NdiY9C+/JTuNtyopkYmJhspxh37jK0y8/zx3dyE7tjxu7ks+OfLPOxn9TtpUotN2rv9+TbdBvVLcs2a/p8k2Xp6md2AjGonu+//Riu8ta65c+XrFzHpeNLvc6TgNV8Omjfnq5MdJR8XRrOH8WM5LPzaD9ehfH9t/Cx/PlE3edRU9cvVep51E0HaRbvb/fP7ocqAACAXfnfeDz+r/x3o+JkDN1wF8b+0g5Qit19u8WM2EvdlvMyM7Z+fDjUmbKrxSDXZNw6RT5M8kT8g4SAIQAAb4MWiAAvpRxTrXP6eSF4GOVDDoRy5vRXZRB+xGERdjK+IgAAAIdAABHgpcRxSLP/DW9/L0128RqDcKNwfRy7ZnbDn97DC0wEAwAAwL4IIAK8mHI27DAbx65YuqEfZ2Z+VUG4YlKeOIv0/Wvrcw0AAEAtxkAEAAAAACppgQgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSA2bhDOj47C8fWo/FzD6DocZ797NFmOr8MWWwGiaX46z3LlvjyjPHgRh358T2jq/tYtd6WrBJ6DAADw1gggHpLWRbgfj8M4W+7Oyu8A2B3l7mFxPwAA4CAJIALNW25FtLIch503npoGIm7CSfkVb8i+7q90BQAAvEMCiPAOja6Pw067YM61IhqPH0Kvk33X6YWH6Xf34aJVrAoAAAActvcTQMxbRMVWT8XYTLEV1PlgEkjJPq+MszRbb7LE9VeNwvXx/Hrd0C9/smh5vee2wFo+vhdo0cWbMDg/Cu3LYej0vh1GC6pp3pzLj1Vpek3LxpV8mTiGWr6v/GdLeWltRk/Nv6nlQaKE833eeTzz+JLLoYTrl5oOEu9vLiW9pKqTrubXmy7rrs0Lp6vpNZ7dt3g9pse8ck7L99dzEAAA3qt31wLx9uv38PGhaBHV7x6F9uNVGD/0Qmd4GX5MKkZ5Jasb/vQeytZS4/CQ/UJcf6HylK/XDrens/XG47uwOmxTrOS0w+Wnu7nthXDZ3rYyG7eXVdDOZtsbj6/C49cnKtJQVt67Wc3+7G4c7g+qCeAwzw95fszTc8yj2XcLaTpL919D+DlN85N8uRQ0qDOGWpbv21leCtn1yLcZf6HfXcqXifk3uTxIlXi+Ucp57OL4ksqhxOuXS0gHyfe3xvVLkbjf1sX9dH/FUjxvOr2fS61u95WuPAcBAID63lkAcRiGn66ySlwrfPgUP3dC79tyG6xRuP56GYZZpWQ+wBIrhXmd/PuscjL4ka3X6YWfGwIxo+vvoZ9Vp+5uZvtqXfwsKm+/tqg5jf6GP9n/zr7MH/tJuLm/CHqFsl5ZeR9maf5hHOaS4uGIgYDpgbXC59Msgwwf54IBq2m8dXGV5axhuP29fcggBlOnuz35kgc+/vydbS81/6aWB+nqne+m82j8+BLLodrl38Z0kGo36aWu0fXXcBlWr/v+0pXnIAAAUN9uA4j97lzXorj8E/5Z+Fy1pK5Xfwy3hcpG5zR8Xq5pjH6H22H2o4+rVZCTL1nNaXgbirrnIPzqx0183lBZGYXfcYNnX7Kqzbyy8vbnb/2KcetDiL+60hIE1ilbEvWzSv7DAY89uBgIKIIVu5+o4iws7TY3fJzkytT8m1oe7Mqm89jB8SWVQ/XLv/2kgx3J8t7Xy2E4u1oOau03XXkOAgAAde02gLjQtSgu/4Z/Fz5XLanr7a5S+elDU9W1UXjM6k2rwdSiK+nWLWvKbnGx8lRsz9hPrJe3EIr/SEhrcXzE5XR6fEAJa3V8uaqx1pqyi/yb7uXPt46Ucug9X79ZK765Rnel/V6XVJ6DAADAxLsbAzHVfNe/idHf2GGqrlb4mFVwVoOpzw2CtsLF/WQbcbypYuwwLTFYdnIzSSP90N1QwS7WXVwOZazEGAyKk7/Errqz43veWHCb7Sr/braf861rUzn0fq9f3nV5uNhld2Z/16UOz0EAAGBCAHFZ63MohtxaU3GKTSim3b2KCtHyesU4T/Oe0UUrWWyJUVSM11X4YNZa57VWsMsukJ1eWBmubadS829qeZCq6fNt+vjWWVcOvUT5t86+0ktp0nX5riowtq90lchzEAAAWCKAuKIVLq6yKki/u9B1M7Zm6fY7ofdzMpZVObh///u0RVfe4uX2UzjLvp538q2Y3bLdVNRmcL7SrbSosHXC6cpgVjBRtNYpJug9rK7Jm5UBiOnYa5nJ2I7lx11Jy7/p5UGaps+36ePLJJZDjZd/SfaXXrIdPdF1eWY/6SqV5yAAALBIAHGdk5swvjsLw8v2dJym9uWncLc0AUUxI2XRoitf5/EqjO+/hY/lz6daF+E+toxYM/7TcuVs8n0xLlRW2VqzXjy+n+HrdN24tC9D6D0c7gQZHI7YTfmh18mS14/wkiGd5zq5mbWgzNN9+zacPhRjoM1LzkepEvNvcnmQKPV8UzV9fMnlUOL1S5V6f5tOL+nrxa7L2T82ne+e0lUyz0EAAGDO/8bj8X/lvxsVJ2PohrswfqoJBgAAAABw0LRABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQDxgIyuj8PR0XkYlJ9fzOA82+9xuB6Vn4Ha9pZ/6xpdh+Ojo4M/1uJ6Zsd5fvBXlASebwAA8LoJID7bIJxnldxjtROAN+69lfeebwAAQEEAEditvAVQbPE2WV5BK723rHUR7sfjMB7fhJPyq0PUurjPjjE7zptDPkoAAID3QQAR3qGX6U44CtfHR+Go+yf0HmLAarJ8Cb+Or7OfAgAAAK/Buwog5kGTPHBRdMuatohaO8bW0jrZMr/adHyuo27oZ5+Hl+2FdVe7fJXBlOk6T43JtLTuFsc3k7rekmmrseUg0/L2jC312gzOj0L7chg6vW87bYE2uv4ast2Es7v7cNEqv8ydhJv7i7Dw1aZ0mo/bF9PabL3482k+nAtIVuXzlTw53ebcdvJlXZrePv+u7/65eXvPK6/WHN907MNyWRfELddZd8zFNVrebp3r8rTFezBblo+lWK+iXJq7NlXXb2F7ZTm36Xxnx9ZgeV8r/W1Wdb6eb2t4vgEAQG3vrwXi8DK0s0pRuCtbQ92dhdDvrglWdMOf3sO01dRDr5OtNqucTLvXje9CtoXQmVs3LvcLEZNYKWmHy093058/9EK4bK+r7PRDN1v38arcVrZiJzu+hQpbwvHlUtdbFitX3azaeBaPd76bYzyPrEKZf18e3/gqPH5dE4jgABUV9+LWLqfRpo3C79thzBjh26YoZY10evv1e/j48BCyH+c/bz9eFXkky9c/5tddzudFhluT7od5Psy3k+87bjv7biFNPyP/ZuVLDL4srldjeynlVWq+nHZdHoe4mbVan8Npdm2Ht7+X8vTkfp6Gz9NkU+e6bDYrUydLUbY+y6Z0cPItT0ur5zsIP/Lo91Ue/N5deZ+S/mrwfFtdb5nnGwAAbOVddmGOwZPpsFonX/IK0p+/kyrCKFx/vQzDrBIxX0mKFaq8Lva9fmVidP09qzadhbu5sbxaFz+LIMiv1ZrOwvFllf6rbL+zCm7q8W15HmXlKq8wLo89Nvob/mT/O/sy//261mQcnrKSP+zk3Yl3Pqzc6HeI8abw6cOGtFEnnQ7D8FMM6LTCh0/xc3YuT0Qn1+Wjtek+BgxmK4bPeQTtcbres/LvSvnS/PaazZetcFEUOOH3/IUq7+fZ1Wybdc9jX55OB5P7vXS+g1/ZuS1f0zS1r8uG9FfX0+nF883zDQAAtrPbAGK/O9cVKC7/hH8WPlctqettM4bbWVhXJxw+ltWNsqLc+bhaZTj5ktd0FiuaG5Utd86+zLV0iMogyJ+/SxWd1eNrfZyrUKYe3zbn8XtWuZqvlE21PoR4yBtbeHBYypY6/U4vPIyXuxPv1rr0t6BmOl2o3C+0hlu2IR/NWQ4SxSDErGXS8/NvNC1fGt9epul8mQedhuF27sKPft9m38wfS93z2JfN6aB1cbVyvoNf/Sx9JbSeXVH/ujyd/urakF483zzfAABgS7sNIMaWFdOuQHH5N/y78LlqSV1vd7OIfvpQGZmoaRQes4rOajC16Eo6X5GtI/X40s9jGC4v4wE99Tsn4SbvYldUsorzMEbUoRv8uMzubiYhrcXxEZfT6brx4VItBLqe0Fx+a1rT+XcX5UHT+fIkFDGYWauw1SDRbsq1/Vg+30HI44enn0P9VPk6rovn2zqebwAA8JR32YU5xUIXwdLob+zgVFcrxAYWq8HU5wVBU48v/TyKrq15968nW2C0wsX95Njj+FjFGF5abByuk5vJvYrjjz1dIS7WXVzWttbZpFW05lnX4mqd5vJbtVGMdHQ+1gwKNZ1/d1MeNJ0vT77FcSXnW3stdxff1Xns3rp0sNBqLe++fBautmqq+zqui+db+fUKzzcAAKgigLhsMonAmpZTRcVzudtkUYGqbmn13C59S61/Uo+v9nkUTm4mLTBSWl7EFhvFIPvrKnIckknrmpeqEBetutZ2JZy3ZTqtbxTyuMLGMRmXNd0l9yW6+DaQL/P7UnTrzbsvryn3dn8eNZTj121WkQ7yyVSyvPFjUHRfXumSO7Hr8n7HPN883wAAYEsCiCvKSQT6izNDjq6PQ7ffCb2fywOqlxWo/vfKCknRmucytLeI2oyuvxYTX0xb/6QeX93zmIgtMCaBpqVK1uB8pTtrMYB+J5w2E+Vhp4rWNZNWOM/pmpzi5GbSgme5sj4I58ezSSy2S6f1DM7jBDKLEz2kek7+Xafp7e0mXxaTeQxvf4Qft8Owrjtv4+eRqPX5NDuzfvg12W0c47NddtPfoDodlJOXZOkwdr2tnjxlt+X97tXNb55vnm8AAFAQQFzn5CaM787C8LI9Hc+pffkp3FVMQFG0aihadk3WX6iItC7CfWzJkFV2Jj9ft96kYtyd+3n79nR14ovU46t5HjOxkjUJ/sy1Vsu29zN8nW6r2F4IvYeXnZiD54ndlB96nSxd/NhiEqI6YgueGLBczBtHR7/Cl/mZTbdOp09ZzEfdP3ECmS27jybm32RNby8xX8bgyuTnxfh0l6Fdfl6337w8GvZDf1jRnbfp80iV7fdnln6n49S1H8NVOXbdqvR0UEymEq2fiGSiqfJ+b2rmN8+3TdsDAID34X/j8fi/8t+NipMxdMNdGG/R4gdgWzFQVgQSDnccPhKULQtD1ay5G9RPB4NwftQNf7bcHwAAwFumBSIAh2f0mHdLfqkZuovuqttOngIAAPC2CSACsGejcH0+GRczGoTz2M+60wsLkz/vyuA8tC+H2e6+abUKAACwhgAiAHvWChffQvg6HX+uG/pnd2E8P1bmDkzHhez2w9ndWNdlAACACsZABAAAAAAqaYEIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgNmYQzo+OwtHccj4ofwS8nNF1OJ7Pi8fXYVT+CCpN0815Vpo/wwukv8H5UTi+bnKrxfOrepu7er5t2i/vXlP5ck9G18cHdOwHmN823d8XeZ6/wnLoRa7LC3oV+XxXz0EOSpkWX/7ePqMcemvlwVbWX7/8Gawe2DgBxEbERNsN/bO7MB6Pp8vNSflj4OW0LsJ9mQfvzsrvDs4rrLC8iDdwXXac/mLwsNvvhNPPrfKbXfN82723Uh68t3LN+e7cq3ie74H3nBcWz8Vz8DDsNl2Nft+GYacXvr2me6ucrNT6fBo6w8vQFu1vlABiEwa/Qj90Qu9VlTawY8t/EZtblOMcnOkL2E042JJ8cB66/RDO7u7DxYvFDz3f2KPXkC/ZnvtLdOjpwHPwnRiEH5fD0Dn9HF7qFYsdi2VLjKr2u+qeDRJAbMDo75/sv5/CB6UNr8RLdqs6uyv+KuavtvAcg3BeRA9fNP94vgHwnnkOvhNloPjlenjwIk5u8paZ/e7rHArlEL2vAGLeIuo4xFbPRQBl0iKq+G7RKFwfz7eaWrdOHQnba/j4Zv3+i+be03WrQvCD87ntVa2Xel2W9lm5Hi8tdoFsx7+w9b4dwF96J+lkOX2sfl+Vnp/VjWFNK8nlZF8rH23Y3ixfd7OXlBCGl+2FdVfPZTkfrWm9mVJulHl73bUqfqdm/pzuc3Z88bim+18eb6Tx69Jgebp8bE+NlbLhPOpbvr/V92F0/T27NmfhriJ6mJ9jcv5Yvn7Fdd+91P0mpPuJDc+t4t4vvzSW25+sWzc9r5zH6n2ruh/zxzdLl6npvoak5/nm67yT80jJR4n5MuX4ptbst1iq89062923xTSTli/rHdeq1Pz29H73eX/rSb1+qdcl1ab9TtLm5u+r0vOzyoKE+/HcfDS/Wq30UisdLB1btjzrPBq3+fiK8y3u9+w6xaUqrSZ6wfJ+QUP1xqr9zqeVWukqt3y+m65xdpzfsy2fXS328Jjes9n24mlOj2clzW6+zoXUcmjz9aun7nWpVlyDDe9XU6n7TT3f1OtXOPnWC51sje/Pu3iU3mELxGG4bB+F9uNV2RrqIfQ62Xdf5wuAmMjb4fLTbKyLh17If2+aH+YKzRiMCVmi7E4TcbYsFCgJ25tq6PgmYr//LFOFSSuwima8eSHQzaqn863FvvxaWi91v3G9LCMvjBVyFR4XzoGXVxS2RSOmcbh/sT6QTzkJN1ki6sR0P5c+BufxQdAJvYelrprL6blIgGvyUYosnX4N4ec0jcb0nD1eumseVkn5aPP2Whf35c/uQhympNN7mK4bl4V7kr+0dMOfuXWK7W1Rbpx8yz5na93+XsqDRXeNlRemRLdfv4ePD3Ff8XKU+4/3M7teP6bH2PB1ycuXBsvTadepTWPH1EgvSeJ5pJaTo/D7Nt6nL08H/VPyR56u2uH2dP4aF9d9qtbzLVHKfqMa6T7tuZUuOT039Pytl+7TJV2XOuVLo+eRmI+S82Vmw/Hl4vm2L0P2UCn3W9zn7GhXnzMb1L9vMf+0w+NVuU52fLHyu3id65RrCVLzW8J+935/kyRev+Trkiplv95zJuvGZSG9pKaDBsurZHWeg3WOL6mel+6ly/uJZuuNmQ3pvnY5lG2rVj109DsUr1nr37KS3g9Sr3NyOVTj+iXZ4ro0InW/ieebfP3mZGXNVbbCaj2IbbzPLswxAU9bcbTC59NYs36cJqh1LT1aFz+LQuNXmYJPbuYSd3wLzdafJuJsub+Yjp+QtL15TRzfnFi4T1c9+ZJnsD9/57JPlhG/5i3SHha7xmXnOP85eb+jvyE29l8shLMXqLlrwksrC+VhfFk9sC7E8QUyvqBMHsLZS1sMcnZ6P9dW6hbSc/lA6H/f5uG3miZbF1dZCh+G29+rW9uYj2pu72mjcP31MgyzsmD+pSi+QOXvcuvO98lyY/L5NiwcSt5dYzmvphqG4acYeGyFD5/i56rxgZq8LvXLv03labpmz6NWOVm+2HY+rvxkxab8MfiRpatOL/xcl7kmajzfUiXtt066T3xupUtLz40/f5uWdF3qly/NnUfD+ai06fjygfGzb6+m59sKFz+LoM7jDm/HxMbjq1uubZCW35rf767u7yap55F6XVIlXz/vOc+wp/Iq+TnY9PtaDfsq75uuN5YaS/db1EOfnjwl5f0g/TrvrXzeV/08cb+7LsdbH2M+W6oHsZXdBhD73dlfavLln/DPwueqJXW97fqyL1eWY+aeDdxb1dKjLDT+/K1ZkNXfXrPHdxaWNpcbzr0xr75Yr1Njv60PIX61/i9vvLjsQR//ItbPCtuH8QtOvlCK6WAx3675y3f20pE/YOO68a166QE8s5qeiwfCNgGhOjbno0Y9ETA6+ZJdqDUPwKfLjfh59SV/8CuvwWw929zCPjunYffDxjRdnu5RnXJy9JjduRA+bRyAaVP+GIT8lr/4AOGJ+62R7tOeW/VsTs/NP3+blnRdapcvL38e9bz246tfrj0tNZ83vd99ST2Ppsu/mtfPe852Dr282sH7Wqp9lfeN1xtzDab72vXQzZOnbHw/SL7Oeyyf91U/T9rv7svxVr6hl/mj4Vu32wBi/AvL/F9rxv+Gfxc+Vy2p6+2iEjgKj1n6zVL5UtCj6PpZvyA79O1lW8w3uEmd/Z6Em7J7UP6ilK+3JmjEi8j/UhP/kZA24viIy/f3WWPvZOJfFBfz7fog5snNpPl5VUu25uVdMBbON3Yp2l7T29scMKrjJBTvMJPm+01Xpqo1d12aL//qaPb+1i0nO2HNe+mblJLu055bTdtv+ktR57o0W76ka7qcTNH6fJrloPnxj8rWIhWV5Ze1r3S1m/2+/P19PdfPe8729lVepdrH8e2rvG++3ti0mu9XeW+cZv4g2dx13sX121f9PGW/L5BeWh+zkpcmvM8uzE9qhfgHj9Xg52SpG7Q89O3F/JSSnerutxUu7ic/iy9MxZgfWiS+vJObyT2I47c8/aAo1l1c1v+FvHn5eECds3BWcyyY/EWm8zFLcfXEl+A4rs1igHPDGBpPaHp70bouN8VsgNtZ+Ctogy9MT2n2ujRf/qXaxf2tV05u91fTbfPHPqWk+7TnVtP2l/5S1bkuTZcvKXaTjxKUrXhnA++3w2WIrfL3f8/2l66a3+9+7u/ruX7ec7a3j/Kqjn0c377K+93UG1c97/0l9f1qVE6estzybTvNXeddlWv7qp9v2u8LlOP5e8D7+WP8LgkgrnhG0+C1Dn172RbzDfbD08MpPGe/8S8PxQvGuoKVlzD5689LPSjqiS+l3X4n9H7ehJs4LtXwMnxN+pPYKOTP5U8fshRaR9lU/hnddxfV3V7xoKzsUtP6HIohcVZ/XrxQbdldOJ9MJUsDPwZF9+WGXpiqNXxddlD+pWk6vazzRDlZ/tW0fvm5nD/WX99i3JldStxvjXSf9tyqUI7HU9+u0t+mdJ8u6brsqnzZeB4vkY/WmVQQlyomzx73qan71vx7Ylo+r7vfQ72/qeeRel1S1bt+3nO2tLPyqiF7PL59lfe7rzdGVel+m3T1xPtV2fV4uVt5bcnXeVfl8zZ2UD9Per9at9/dl+OH9AeH104AcY18qu/swd5uKMpy6NubzNC6Mivb4Hwh0JS83+z3lru9Fhm7E073+pR/74q//kzG4Hlu1+TGlIMxT2cCbl2En1mCXJ2lctXgPE4MszjgbpryQTU/9stkrMjyYz11t1eu3/9e0SI0u1f56NHdhfs0q4BsW/EtB+vOthu7BDz7hWmjpq/LDsq/JE2nl0ydcvKJF9OnrOaPyf2fXd+8Rcntp3CWfb07qfutke4Tn1uTrqu/Jt/F+9Yuh3XYwm7S3+Z0nyzpuuyufNlUrjWej5JMjutXaLbUaO6+NZuu0vN5vf0e6v1NPY/my7/k65ddB+8529pVedWUPR7fvsr7puuNa1Sn+4R0VeP9ajIhx/OD7KnXeVflc4Ia1yVF8vtV4n53XY4fxB8c3orxePzfLpa7s/BfOLtb+7O9LQ+9/7K09d/Z3ZqfrSx3/2XFwH/ZJVpYOr2HlXUfep3sZ2f/3S19v7gkbK/h41t/XOXvrbk3+T2b397a+5d2XYp9z6/T+a/3ML8dyz6XtDT7zKVMz4vpoFimaXy6TkU6nVt3NU1lS6f338PC71WsVy6L6fThv+wFaO7nMY0W322Xj9K2V73+mvXuzhZ+vvae1So34jK5ts+4/0v7zMuOyb1YOZ4dXJeUcijxuuwmvcyvN1uWz2N13epyslh308/nt5Uta/JHXBbK+jwNV9+PYrvNlBXJ+01J9+WS8txavDZxW+W9nKxbKz3HZXP6W3/d1pUbk2U5fa2/H6lLynVJuc7Nn8fyz5+Xj9KPb/09i8umMuLp5enzrXf91h/jtukgPZ/X2e8+7u/m9Yol7TzSr0vqsmG/ZRlSmQ6yZZIG157vG37PqXV/d1JepS3rt7u07OR9LW3ZV3mftN9N+SNb1qaDiveXYnk6XcVldZvr3p+KY1ufdsul7vtB4vtLk+Xz2utXLttdl/RlcXvxXMt7s5QW0ve7+XzjUrscL+9V5c8ttZb/xf9kF79xcTKGbribm6YeYHv5X5guP4W7nY1n9F4MwvlRN/zpPbzY+JY0obhv/dgNc81zVf6AKmXe6fTCw1K35fxdNbYOeVg/uRe8JOU479He0v3gPGSPBvntHSie9WfudUN0YQZ4R4puA7ufPIWmnYRvvU7ePaap3izwLpRjMq2bcb6YDGC7CYoAeK2anTyFAzY4D/mwTXeCh00RQAR4L7KHaJxBsdP75iH6CrUu7ssxTJfGHQKqtT6EYui230uDsw/Cjzgu3YtP+gHAfrWKWYH1lHzb4riMRfQwuNXNEUAEeONi95Cjo6NwlD1Ez+7Gui6/Yic3xWzqt9NR7IGnxRkfs3wTLkM7loPTpRwS4NmzMQMAh2b0+zYMK4b+YXvGQAQAAAAAKmmBCAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAK8GoNwfnQUjq9H5edXYHQdjrNjPposx9fhRY9+uv/z7OoBAACwDQHEV+sVBhLgzZMvV7Quwv14HMbZcndWfgcAAMCrIoAI7NTo+njW+ixfjoP4Gi9mGsC8CSflV29PdeC6yH9l68vl1qAryyRvFttbabW5tjXnKFwfZ9+da98JAABvmQAivEMLQYWdKYIQ7csQeg9FC7RiuQqPX1+4Gyuw0Bp0PH4IvU72XacXHqbf3YeLVrFqoR++zwUlR79vw7D8NwAA8L68rwBi3nqiaGGx2CpqXYuoslXFE+vk28jH85q01iiW9d0XF9eJy0qDjYTjm33fzap2IQwv23Prrdv38n7XnSvvyeA8BvWGodP7ttMWWYPzmEY7ofewHJQ4CTf3F2Hhq2fkt/UtnzZvLyW/Ta1puTW/2/r5MuH4csvrFdvf2obziGpd54TtJRmc579b3YJu+foklGvLx/bk2IsNl5PJ1+WQy+cs7/bOwvD2d3ndRuH3bQi9u172kz/h78EcJwAA8BLeYQvEYbhsH4X249VcK4zsu4UWUbFS1w6Xn+7KdcbhoRfy31upBA4vQzur1Ie7sgVHseLienllshv+9B7mttcJ/e66SuXTx9e6uC+/vwtxOLHO3Dbjcr8QqYnn0Q39s9l5aP31nhXBqG4/hLMsvS6mlYZlaf57jHSdXS0FD9d5Rn6Lg+r1u0vr1dheannwNYSf5baK7cX8Owv21M+XCceXlxvtcHs6v61i+9vZfB5Tqdc5dXubnHzLW8PNglUTg/DjcriUjuL1SyjXksdeTNxestTr0vR+GzT6G/7E/3/+Es6Gt+F3PKDR73AbTsPnHRYbAADA4XqfXZhjhe1m0vaqFT6fxprr47TSNrr+HvpZNf1uuk6si/7MK7j9XysRiDwYM9vcRbiKde3vk0rgKFx/vQzDbJ/zQYQYcMjr5NP15mw4vmRlJfDsy3w7s3Wtv3j7YrCiHS6HsUXgXHrdldFj3tVxMe2t96z8dvIlD6j9mWsOVXd7m/Pbap5pXVxlexiG2zyyUk/q8Q1+ZOVGpxd+NhborXcem65zs9dlct3LYNXE4Ffe4nIhHTVdrjVeTiZel1dRPp+Eb72QH3dMj5+uDunYAACAl7TbAGK/O9c1Ky7/hH8WPlctqettN4bbclCjaD00GWA/dtOKLV6+lJ8nWuHDp+x/f/4uBfLOwtLmQuvjXAAittrINtf5uFrtOvmSVcmXK8yZp4+vhtaHEA95fUtH3o2yBWw/H+tsuTvxboz+5u2X5sQA5rq8+/z8Fg0fJ2vV3V6D+S1J6vENwq9+Vm6cft5TwGbTdW7euiDbIL8IvfBt/liaLtf2VU7uYL/LXefjEocreI7W59MQbr+G7/35NDEMO0wKAADAAdptAHGha1Zc/g3/LnyuWlLX20UlfxQeY31rJfhZdP3cqiVg5tOHfYQBTsJNOVB+rKQW53FIY2zxEvKWbPEfCWk3jo+4nO7Xj+n5tFYeDZsX02KRbxe7kzad33aTfxfHSIxLN28ZV99uji9Vc+dRaHZ7J6H4m8qkG3NVELXpcq35cjLtujS/3+Wu83GJ3adry1sQfwr5Y6v1OZxmn4YrQW8AAOA9eZ9dmJ/UCrEB4Wrwc7JsDlqOYoSg83Gh0rvY9a+w2kprF1rh4n5y7HH8tGLMNy0S34+Tm8m974fuhgBFse7istVYia2PIWajdel+0fPz26Kmt1cEg2Irrtild7adbccibP74UjV7Hs1vL1polZ13Xz4LV2vTX9PlWnPbq3ddXkP5XB7jpD972XISAAB4XwQQV1R3dUwzCnlc8NOHbEuZ2HqjExsWrQkg5oHGbQelLwIR9boUxhYvRUV2c2CHt2XS2umFAhSTdH/5Y8MwA8/Nb8ua3l7Z5Xi5G22lTfky9fjWb6cYP3Ebdc9jk6a3V8onU8nS6I9B0X05qdVb0+Xac7b3nOvyuspnzxAAAHhfBBDXOPnWC504C+kWUZbBeZyoYn6ChFa4yGdV6S50BY2tVLr9Tuj93HZQ+jIQ0f9e3aJscL7S/bQIQHTCqak036GiJVExoe52XZPTZfv6meWjvNXj02OVPie/rdPs9sp8Nj9W6WRMyfLjos35Mu34yklF5raTt2y7/RTOsq/rq3semzS9vYnJeXfzLt3L41Pmmi7XGt1ejetywOXzy7SOBwAAXhMBxHVaF+E+tgRZM07ZatAlBkhmP+/+iRNVLHVDPLkJ47uzhQHu25efwt0zJ7Q4uZm1KJtsd+H4sv3+DF+nPyv2G0Lv4WUm0uAwxW7KcVy0za0DnynPRzGNLuWRGEmZb1lWK78laHh7K/msfRtOH4qx69bZmC8Tj6+YqX22nfbjVRjffwsfy5/XVfc8NkndXgx8Ts6vGOfxMrTLz+vuRzGZSrR+Ipdsx0nlWvJ+Gy4nk6/zWyuf16Tno503dQYAAF7K/8bj8X/lvxsVJ2PohrvZuElvUDHWVQwE7m7cMoD3Jc7Y3Q1/eg/bjb8JAABA47RABOBgFN14qyZPAQAAYB8EEAE4DIPzEGcw7vS+adUNAABwQAQQAdir6XiF3X44uxvrugwAAHBgjIEIAAAAAFTSAhEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkA8RKPrcHx0FI6OzsOg/ArYEfntBQ3CeX6tZ8v58kXfdD+mPy+X4+swKn/0chLO410rrs/x9cvfmRej3DgsDd2P0fWxe8rMm8rnnlsTjefzuu8lG9KVcgg4ZAKI7Nm+KppN7/cdVJh5A/aZTuO+u6F/dhfG4/F0uTkpf5yqdRHuy9+9Oyu/e1ENnQe8OYf+HPSc5r3y3Nqpvb+XALwcAcRDNH0Q3QTPdl6r4i+oc3+RzZfjcHB1N/ntZQx+hX7ohN63DVf50O9H6nnwtik3Dov7wS68lXTluXVYlFfAKyaACO/Qy3WPOAt3+UtSsdydDcNl+wCDiOzc6O+f7L+fwodW8fm1eivnAcD74LkFQFPeVwAxH3OiCF4sto5aF9AYhevjlJZTy+sVy3wXmfXBmqIrzdH8ACSpY2g0eR7TbZXHky3xkKbbXTmGzdcl/93892bbzJe5c50ddzf0s8/Dy/ZsvWxZ7WK0tK2157pZ+n4n+1vez+L39c9j/wbnR6F9OQyd3rcX/8vnybde6IRhuP29fF3kt3z9N5bfakm9H7WkpqumLV+/qrEei+PZnK5qGJzPbStbVnYcbT6+lHQ1s3ydi3S2YvkeZ8vazSVrJn9MjmtdeV3cm7ntpqbTWve3ofsx3edsnfjj6f5XjnVf5cvT4jOq8rqWxzHdZsL92O74Fq9N3XOYqLp+K9t7ofSScq5r91nr+DYo79m6Yym2vbzNpo9v+fpVbW9uncr0GG1/P9LK53XnsAubz6PedU6VcH+j5XuSLWsvX5T0HIyayefJaqWraOnarCas9PuRcv1Stlde2/T8C7w177AFYmwBdRTaj1dlq6iH0Otk332dL8TjQ7QdLj/Nxgp56IX89xYK27ygbYfb04fpeuPxXdh6+Itpk/aUMTQaPI/M7dfv4eND3EYI/W653WzlzvAy/Jium7697A09tLOX9XBXrDeOJ9TvTtdrXdyX2yiuV6c3fw3H4f5i/s+kcb/Zi//C2C1X4XHhXNOk7/ck3MTzj9d5bj+D81gB6YTew32Iq9Y7j30rXkS6WQ3qLLsvh3Ns8ttbzW/zL/ExaJ0dVOhOX0izZf7ludb9SFHj+m1S5zzydNoNf+au8UN2o+N9Xt1vSrpKl7+8Zxk85u/Jvsdffi3ut87xbUhXueR8md2PryH8nK4z2e+2lY3m8kdofQ6nWV4c3v5euu6j8Ps2u9+d0/B5kkWaLjeavh+Zwy5fNmt9jDfjMS0PJNyP+scX83c7PF6V62QbjkHHleuSavn6FRd6zfZ2nF7W7rdOOZlwfCnq5LfGjy9uL+H5lprPG8+/iceX4lU8t1Lvb7Ze4vMj6TmYazifp6j1/Fg6vuzCdLL0shq4S0z3yc/fDds7+ZZ9ztZayb+D8COms7OrvH4EvF3vswtzfDBPRw5uhc/5m8zsZXV0/T0rts/C3dzowq2Ln3mB2f81e7IMflyGYacXfu6rpGzoPOLDYvgpFvit8OFT/Lx+nJT07RXiw3u66smX/MX9z9+VJ9Vmo78hdr44+zJ/TCfh5v4iO+sdig/6+ISfVLqyl7EYfOv0fr7Ch2P5kjaMwc99DZw9CtdfszyTpYSruQsov73h/HZys/Cymm01qzvNXmDHO8zDda/fk5LPo0zjWVqZD0rEAEZeV/y+poK1IV0lyyqAX/OWxQ+L+Ts79tnn+se3KV2l58vVNNS6uMq2t65F8mbN5o9WuLjKvhnehoVDGf0OMZ5xdvWMdPrk/W3+fhx8+ZKglR/4nzDZfN4icVKrL8unTzvui9n0+S5sL3u3iMmtfnnwzPSyZr+1y8knjy9Ven5r/PgafZ/cQf7dy/N3f8+t9Pub+PxIeg7OvGS5to11+Xc1cJfZeD9qPn+f3N7k81L+zcfZzH51Ie0Cb9FuA4j97uwvXfnyT/hn4XPVkrredmO4LRdu8SE5G8i2/Avo2Zfy80T5Mv7nb1mADsKvGEw6/bxQIL+kZs6jsLCthb/+TtTbXrZiWPcMGT4urpWk9SHEXaz/S+iOZS8d+QtUtu/4F834UJ1/wXoVyr8s97NK/sO4aDn5cub/4j35K/P8oNHym/y2C3WvX0PKCnDn42oqPfmSv/kvvnBnnk5X6Ua/b7OqwGJwfkXt49uUrvaVL3eQP/LK42JlanJN1/1uqifvb+P3o3DQ5UuK1seQVU9LRRoL/V9z73udsOaSNajp813dXlUry12nl8X91i8nmyqv0vLbDo6vyefbLvLvPp6/e3tuNf+cTnoOTjWdz5vWULmxhU3bWxd8HOQvA71gnh54+3YbQIx/wZj/a9f43/DvwueqJXW97QvHaqPwmD3Psqf3XKCyWGL8aF3BfZiaPo99XpeTcJM3oS9eqor9vtwYGyc3k+5461tyHLq8hVD8R8I9ylt6LN3f540JM/uLdwzEhv73pfsmv633fvNbM/abrnbdOmqdUX7CafZxfNHimEpxiUNCbGMX9/ckFHXlSeuOqspt817+fhx4uduKQZRhyOvxsVXLWS8rj/ohb5A0esx+8r4ng2guvez3ObM5v+0mnzf9fGs2/+7v+Xvo5VDK86POc/C9ae75Gy3n3/3/kR94Oe+zC/OTWiH+gWc1+DlZdhG03IWmz2Pf16UVLu4n+4oBvWKMjpf4C20+7mHnLJxtPcbLfp3cTK5ZbA349Itose7i0lSLyyIQu3wN5bf13m9+a8Z+r9+6LlDFLJi7k7dMSLSP44uVlzgOV+ySNbsPkz/O1LWb+5tP8jRpbZO3ynmZPxq9/P049HK3OL54XeJ16Hz8nHeZi10a8+vS+Zit8brlgY4tz+M56WVxv/tNB5vz266Or9nnW/P5dz/P30Muh1KfH3Weg+9Js8/fwkLr1Lz7cmrLT+C1E0Bckdp0vnjwLTd1L8bzSFCOc7I723cBWK/p7U2sv45Pi3+hLR586154spsQipnGnurinrbf+NDt9rOX2p834eZnMQj917URuG3O4yVN/qq9z0BQ8RfLxe4w8tt6ryi/HaRdXb8NJpMDrLm+RcV9XXfSZhTjxpWttKo0fnyp+bJsXdRY96Yd3d/8+hTdsvKucDu8X7m9pZdDKl/WKY5v+Pg7SzchnGYXofX5NHQmx/vpQ7bGNg7lOT0KeVym7nk8O70s73dP5eTExvz2Esf3jOfbi+TfF3j+Hnw5lP78SHoOvlrbtopv+vlbyidTyeoUPwZF9+UXaK0PHAYBxDWKv4pehvaTUZZyENm5LpmTv/Asy1985x9oMcDVLruV7lDaeaRrenuF8gVipWvrnMH5SjfaooLaySsWy4oxUKKnXiIS9lsOxjydUax1EX72OhWztCVsb++Kv2pPxnR8Xtfk7eRpKP41fTYVqPxW4bXkt0O1m+u3SZbH8lkKFmdKnP0hYpuB+hOVMyOuzKqY3c/ZJWj6+Cry5e2ncJZ9PVOmu/k/HkzGZi0/1rWr/BHPZ3j7I/zIKly77461v/Syt/IlUd6SqH8ZLkMZvGh9CJ+y4+2uKfPTHcZzenAeJzRbnDgizfPSy7r97qecnNic3xo/vkafbzvIv3t5/h56OVTj+ZH0HHydRtdfi4kQa0cBm3/+FibP/26WTmL8UPgQ3gsBxHXi7LvxL35Zobg4XsRi0CUOKnt3VrToij9r356Gh/IvhQvKwNN0PJP2Y7gqxziZFx/Wk/0U439kD9Xy8/ILRZLE80jW9PZKJzezlnFrt3dyE36Grwv7a1+G0HtYPyFIEUCK1g+QPPHkfqdBp+XZ4eLAwfF5udqKb+N5HIjYTTnOyDe8/PFEC80dydJQfE9dqMDJb+u9kvzWlNT7kXzfdnT9NoozX96dZYfVnu6vffkp3O18AqOsEpj/gWDx3h79+rIyG2WTx7eSLx+vwvj+W/hY/nxiJd21b8Ppw2q+TLaj+5s/P4b90B+u747VeLmxr/Syr/KlrmlrubIFe2Z+soe692M/z+n5ycSy4/wTJzTbsvttrfSSsN99lZOlTfmt8eNLfL4lp6um82/i8TXuwMuh9OdH4nOwYU2/v0z+ED6ff4t33u3uR+PP39KkTrSpvgW8Lf8bj8f/lf9uVJyMoRvu5qaBfy8G4fyoG/70Hl7fbL3w6shvABymGDAoAjG7HU9w2b72C7wn3sHhPdICEQAAAEhSdK83eQq8NwKIAAAAwGaD8xDHIe/0vmnlDO+MACIAAABQaTqOY7cfzu7Gui7DO2QMRAAAAACgkhaIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIsCbMwjnR0fh+HpUft61Yn9Hc8v5oPwRTxtdh+P8mp1nV/FwjK6P6x3TgZ7HrhXXKTtvCR4AgDdOABGAZ4jBw27on92F8Xg8XW5Oyh9zgF46wEwz3DcAAPZHABHYnWmrpNmioc4bM/gV+qETet9EDLfSugj3edD1JrzqK/hWzqOm1sV9ETQXMQcA4I0TQIR3qHb3xG0MzsNR+zKE3sO0VVpcvvw6DhrQvB2jv3+y/34KH1rFZwAAAN6edxhAXB6ra00wI281VXw/Hd+oat0oBkrmt6mJFQdscH4U2pfD0Ol922FLoVG4/t4P2U7Cz4vFyNLJzX1Y+iqzeQy99UHP8vfmV95J/s3O53hunapt1ZG0383XpbB8fN2QXf01dnAeSRLOo+592yA5veSWj69qnw1ev+XWucfX2dZX5eeR/2zpGNcnhCfNrmuRPoaX7dn2smV919jFc15ZJ/E8CqnXebOq67L++FLT1fLxVZVDi+vEJeXaPXm+T5QH2903AABo1jsLIMbKwfJYXVfh8eu6Cs8wXLaPQvvxqlzvIfQ62XdL6+Yv9t1+OLubtbAaf/mlmyYHqKjMZsk1T6/3q1G85g0f1+StJXkFvxv+zLVUfOh1Qr/7nO7OTebfWG60w+WnWbnx0Av59rc9vqT9pl6XfL12uD2db+l5F87KH880eB5zwY4YjA6hH7pzAY2FIFKt+5t235qV+lxoOB1Mu/yOw93qzVo0vAzt7BjDJL3EX+h3a+932t22TB+dpdbBq2VCvK/t8Hg1228MXi3sN/k86jx/Ey1fl+KGbJeuEtPp7BpOlnV5LUpPL5vKg/r3DQAAmve+AoijvyF2tjv7Mt/u6iTc3F+Eta/fsaIzHdeoFT6fdrJ6yFxAJKtwfM1bcj0sThhwcmMCAQ5MWZkddkLv4SUmuGiFi6tY1S0DS6s1+tIoXH+9DMMsr81XgmOFOY+RfH9GcKGh/Du6/p6dxVm4m/uydfEz9LLN9X9VndcTkvabfl0GP7L11rT0XNboeWTHOglexCBLdrHD3VxAYzwtU7e4v5vuW9MSnwuNp4OaYnBpuuuTL3kg6c/fnV2Vqcb2W/f5m2jh+FoXIRY79dNV8+VQcnrxHgEAwCux2wBivztrkZIv/4R/Fj5XLanr1RzDrfUhfMr+l9qyabGiU1Qm5geIH/2+DcOsgnC1oeIOe1W2rOl3euFhvK778I7kQabY0if797QsWOrCN/odbochdD6uHtTJl6zmPrwNv7eIVUTN5N9R+B0P8OzL9PcKrfAhFiZ//tYPLKTsN/m6DMKv2FP89POGIEzz55Fki/u76b41Lum5sKfrN3UWli5Lbvi42702ut+az980q8fX+rg+4Pxkumq8HEpPL94jAAB4LXYbQIx/8Z9vlTL+N/y78LlqSV2vbqXyJNyUAY1YiVkb0Khh9JhVEODA5S3U4j8SWnHF8REXg/TPHV+rFS7uy/z60Aud7EhiF77lbX7awwwcafl3FPLVVv4YUnQFT7mmy+qUG81dl+bPo4593N90Kc+F/V6/t6HZ5+8u7CO/eY8AAOC1eIeTqMwFNPLxhIqAxjYtIvKWDnDgTm4maT12J366wl6su7g0Nr5WPlZaEUAY3v5eCLis6xJZzO67O2n5txXy1Vb+GDJZ6reMq1NuNHddmj+POvZxf+vZ9FzY7/V7O5p7/lbJA3Kdj9me6ttHfvMeAQDAa/EOA4jzYouIYlDybcZ0auV9kfrhBYa/gmeatP5pvsL+LK3PoRiKbE3FPQ8EnIbPT0UCynHVtpGWf5vvopq03+TrUgQqltcrxl+b91JdbZc89/42LSm9rHsu7On67cz6dPOynvf8XW8U8njfpw/ZGdbQeDpNTy/13iNq3Ld86IrY6rHmUC8AAFDhfQUQB+crXSeLinYnnG5Tiz35VnbHWmrVle3nYAI0MFW0/sknBeg+t2vyJoNwPj8Tb2l0/TXESXtnY/Zlx5TPetBdOJ44K2m33wm9n7MJFlqfT7OcOlfRjhXkdtk9exuJ+ffkWy904myvTWXqpP2mXpdyMoj+9+m24jrt20/hLPt6XuPnkST9/jYtOb0kPhf2c/12pQxwzaWbnWv6+bvG4DxOFLU4cUma5tNpcnqp9R6Rft+KsRWj1OAkAAA87X0FEE9uws/wdWE8oqw+GXoP204sMQnIFK26ptv99cXsiRys2E05zpw7vPyxw5YpJ+HmZwhf5/LaLL8tdYuOk63cnWXH055b71O4W57wpXURfmbHPR0/rf0Yrsou0dtJzL951+u7cLZmPLPlgEiaxP0mXpdiptjZttqPV2F8/y18LH8+1fh5JEq9v01LTS+pz4WGr18MTk1+vxgX7zK0n7G9uk5uZi2SX+Q8Gn/+RuUs7+XS/RMnitqyO3nT6TQ5vdR7j0i9b0UAPVo/EQ4AANT1v/F4/F/570bFyRi64S6MRdIAgAblLW3zAN+ex58sW7aG3kNz48UCAMABeudjIAIAbGn0mHcVPuyZxgEA4PkEEAEANhqF6/P5sV0H4Tz22+70wjedLQAAeOMEEAEANmqFi2/zY7t2Q//sLozvdzcZEAAAHApjIAIAAAAAlXYWQHwrBEIBAAAAeM90YQYAAAAAKgkgAgAAAACVdhZAjF1/j47nZysEAAAAAF4bLRABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEo7CyCe3IzD+P4itMrPAAAAAMDrowUiAAAAAFBJABEAAAAAqCSACAAAAABU2lkAcXB+FI6Or8Oo/AwAAAAAvD5aIL5lo+twfHQUjibLSwd0p/s/D4Pyq52qe74vfXzs0SCcZ/f6+HqLHPDe8tFaz7h+L2I/xxf/UHa41yRFcd2maTtbzpcT2ab0t8f8Mbo+fj9/qEy9znu6H/m92PuzNCE9H7qDKO85jPTM1g4iHx36exPAdgQQ37LWRbgfj8M4W+7Oyu/esvd2vrwM6Yo1YvCw2++E08+t8pvXJlZuuqF/dpen7clyc1L+ONUe80fr82noDC9D+9VFibaQep13cj9eQ0W4ofS8EwIJL8N1fp3cN4DXRACR3ZlWZG7CQbzDLzv043sXihfHlb8Sa4UxI50ensF56PZDOLu7DxevNn74K/RDJ/S+bUhVh5z+4rHFKFm/+/pamtGs1PR86JT38HzyEcDOCCAClV6uG08/fJ/76/Po920Ylv+GwzII50X08EBaN21n9PdP9t9P4cNrDYBOnNzkLe36XX9seM/eTHoGADhgAoivTd4y6zhcjyYtt4oxfopAT/Z5h+Mdzcabmu07LivdDhLHYEreXm4Uro/ntplfg/JHdT3z+I7WNnVp8PgOROyi2b4chk7v247/gtsJvd5ZGN7+Lu/DKPy+DaF318t+8if8XbiOS/ciW9a3PFq+H93QL38ysT44Wm5/2+ZMy2krW1Y2Nc3Dk2OYrLuUZuqk0/n1psu6NJiaTjdfv9oG53Pby5a117i5+1toPl+Orr9n+zoLd8+MHiaVf+U1W1cmFvf9+edTKTH91ZN6P5bTwdPnefItlhWLf4R4VVLKjT2YlS1F/hpetheOMeVZvfPneaLiXDaX91X5svKZ8ES5Vuv61cpvm8vJeudRL79Vmj7bZtuLu5teh7lzSr0fhaePr9Z1rm0pra69fs2l56r7tvYcUsqN6T0ptz1dd80xbtpejfub27C9Wvct9TyW91mRjxa3Mb+suS7J93d5var3EoDXTQDxlbr9+j18fHgIvU5seXEU2o9XYfyQVaKGl+HHuvebpsTxprKHYriLXQOyJdtnyB76Cy8Z064DCWMwpWwvf5Fqh8tPs7GNitXWvCyleM7xre0u1/Dx7V3xElQ0shqH+1320Rz9DbHdSPj8JZwNb8Pv+FI2+h1uw2lYGVoufzHshj+9h7nr3MnT/8J1ztdrh9vT2Xrj8V3YdKufL0sHX0P4Od3n5PjWvWwO8/SR59t83ZiXs+++zr3sJqbT1sX9dH+zbYXQ6f1c6l6bmE53cP3yl/UsQcX0NN3ml19r9tvk/d1FvozB7WGWMb40E1TfVP6dfMvv5Sy4PjEIPy7jcVzV60I9F+yIfxyILX+708pOtsxXtuqUk0lS70dcL6t4LYxldxUe5/PGsuxYr7JjXL1Or0GdcuNlzcqWIn915vJmXFafDTE9tcPjVblOlnBiUGD1/jaUL+uk5zqSnvuby7Va1y81v6WWk1Hy+0vN/LZBs++nm4+vfjpNtZSe4zlk128xmNdgep7Y9FzI1Sk3Et43amwv7f5u3l79+/a+3psADpUA4qs0DMNPseLYCh8+xc8vO+5PfFmeNr4pK27979u/bG7a3roWP62Ln8XLy69t39DSLRzfyZf8heDPXLO4fR9fs8qXpWGWph5ecgD6k/Ateym7/T0Kgx+X4dPVRVh8dRyF66+XYZhVIuZfKuNLYF4nmksv8feHnV74uXWlYVsn4eZ+8bhbF1dZyhjm57UiVohmCT98Ps0SzPBx63w0Mbr+Gi7D6vmnptPGr1/2Yv01b8n6sJieTm7mPjd/f3eSL2Nwe5iVuB8bujaZp8u/Sboog+sT+Xhv2e9+mb+gCbJrPqsEZduN12da2cmWpfTbpOT7Uf5RYfHcVvPWstbHNdfpVahZbhy4F31e7jA9bzqPtHKtaenl5MTm89guv1Vr+P208eOrZ135PP+Hil29/z39XIiaft9I3V7q/d1RufZe3psADpgAYop+d/YX7Xz5J/yz8LlqSV2v/thNCy9TnTWttXbmLCzXWYuK27YP8E3bq2rxU768/Pn77BeHp60eXzR8nOx138fXoLJlQz97CXoYv/zkEHFG1XD7NXzvz1/zYcgv9ROBm5Mv+Rt9GTgYhF/9mCU+L7y4HqLl4E/xF/FnDvhdVmrP1gRg09Jp89evGM/yLFw9laAav787ypejx+xcQvjU2EBrm8vTdZWuQX4ReuH1zBdR4360PoT41doWVU9o5Rsqywv25K08LzedRyyyEsq1piWXkxObz2Pb/PaURt9Pd3B86TaVz7tKz5ufC3U1+b6xv/pHs+cxdYDvTQCHTAAxRfyL1/xftcf/hn8XPlctqeuZJazaKDxmz+/VIG7RxfY5L1TNOPTjS5f/BTX+I+GY4/iIy+e71ThDeUCmHPi+9TmcZp+GKy9rM80FbnZjdVydlxwDZ9Y6Ze6P5aX9pdNRvuM0zd3fXZ5vJzTYADHBSShiA5NWL6+xslLnfpyEm7I7WQwaFOsldOdtfczuzPOklmuNlX+l/ZYbL+ntPC/rlGtNa/Y5uGV+ezGHfHz7Tc9NlxuHvr3dOMz3JoBDJoDIs+Uv0p2PjVVmF7fXCvEPr6tB3EMJvh768aU7uYnHG8dsieP+PP2CXqy7uGw/ztBEK1zcZ9uavMW1ipYH8xa6XpWK2Tf3L74sx3G4Ytej2XV5uTFw8i44w8WuNjP7S6d5y4lEzd3fXZ7vblu5rStPF1oX5d2XX7jl07PVvR9lWZD/LOahYuyrJ1sg5X+MeF5wN7Vca7L823e58bLezvOyTrnWtOafg1vktxd1qMf3cul5+bnQdLlx6NvblUN9bwI4ZAKIPNMo5O+tnz5MX2yeZ3l7h9a1admhH19dk7/2H04FIq8sxdaJ2YvcQterUvFiPelGU7zwLa9XjGOTYDKpS21lV5d9dSuddMG5q3qhTU2nz7x+axRdS/vhyeGgGr+/O8qXZSu3dRX4FfmQALG1Qp0hKirK03wylSxP/hgU3ZefaKV7mJ5zP2KZVFQ8n7ruh/KHhHr2XG4kW5/v6juw5+XW5X12Jinl2lRD1y+5nHyOtPy2E0n346njayqdVlnu0vpS6Xn5udB0ubHv7e36vlU44PcmgEMmgMizDM7jhBtVf72rb932Tr4Vs7u1D+fP4QsO/fjqK/7aX0zY+LyueZukV/qzY8pHEV+cATH+lbvb74Tez8nYNeWg2v3v0xaU+V/Cbz+Fs+zreXHMxc58BTAGfNplN+7ayhfN+TGo8gDSS3TZeaoLzkxaOk2/fsnKmYRXZnMcnM8FqJu/vzvJl09U4JcVY6RFqUGGp8rTyXl3865Ty+NAvQbJ9yNLF8tlTlER64TTJ6IjzQVQXtI+y406yuOcy3fb2tfzstnyPpNUrk00df1Sy8katsxvz5V8P2odX3PpdJ2itdripCEvkZ5XnwtNlxv73t5u79t6B/7eBHDABBDfsPgAWxyvI3sIlp+XX8jSxe6tc+OA/IkTbiz+9a7efjdvL7Quwn38i3P20jxZb7Isv0RPvn9qv41fl8Tje21iN704s+Xw8keNFlQ7FGfcvDvLjqc9vb7ty0/hbmnCl2JGyqIFZb7O41UY338LH8ufT2X37Wd2ftNxldqP4aoca2leano5uZm13Cy2dxtOH1a3lyo9PcdKTfaPTekvMZ0mX79kk4D03LWJy68viy/uO7i/zefL1YpClaKCHK2fyKCQUP6VislUoqe215zGy9PU+5Glg5/h68LP25ch9B6emNgpq5x+z/b9GgexTy03Gr8fNa0c57bb29fzMrG8T5dYrpU2Xb/k+5ZYTibbJr81IfV+1Dy+ptLpJMA5Xz63b09XJ5jbSXre/Fxo+n1j39tr6r6ll5OH/t4EcLj+Nx6P/yv/3ag4yHj+0Fuaxv+1yc8j3M3GZXvH4oO5eFGtau5fT9PbA9i9QTiPLSniuEjPeC7UL/+K/f7pPTQw3ujbkj+n+2eeJcCr5r0YgEOnBSIAJDsJ32Izin73RccILbrtvbbJU17A4DxvaVI9jhUAANAEAUQAqKHorhRjiE/PVt6YwXmIM1p2et8EyebFcbWK6OGT41gBAADPJ4D4rsUucYtjeqxdjq83zEAG8L5Mxmy6nY4S37zpeE7dfji7G+u6vCSfqOaZXckBAIA0OxsD8a0wBiIAAAAA75kAIiQQSAYAAADeK12YAQAAAIBKAogAAAAAQKWdBRBjl8+3MPnGWzkPAAAAANiGFogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQKX/jcfj/8p/AxUG50ehG+7C+Oak/AYAAADgfdACEQB4Nf7v//6v/BcAAPBSBBABAAAAgEoCiAAAAABApZ0FEOOYcUfH12FUfn6t3sp5AAAAAMA2tECEN2UQzo+OwvH1M0Leo+twnG3j6Og829omxf6O5pbzzb8EAAAAvCICiMCWYvCwG/pnd2E8Hk+XxYmqGwhoAgAAAHslgAgsal2E+zwYeBMWYoHLBr9CP3RC79uTawEAAACvnAAisJXR3z/Zfz+FD63iMwAAAPA2CSDCqzYK18fzYxB2Q7/8yVQ+puFxiL2IR9fHc+sW301Nxz4sl2dMHjTbT3E8w8v2bLvZsk2X5nyb+TEtjru4sK3BeeX2i2NaOmcAAABgIwFEeK3ygF873J4+zI1BeBfOyh8vGobL9lFoP16V6z2EXif77utckHDadXkc7tZvZBqgi0v7cph90Q/d8nO+lEHH1sX9wvF0evPHOA73F1s2WxxehvZRN4S7clsPvZCd2GzilpNv2Xllq93+Xgp+DsKPeLxnV2HbXQMAAMB7JYAIr9Tgx2UYdnrhZ2pELE52Mp3hpBU+n8ZI22O9VoYnN9Mg4EOM1IWzcFd+zpf7i2zLu3V2NzdRS+siXJ2F0P8+CYROzus2/J4/sXy8xux3vxivEQAAAOoSQIRU/e6spV2+/BP+WfhctaSudx4mDek2G4Rf/RA6p5+TA3bLwbOileCGiVIOzllYjgG2Pi4GQlsXV9law3A7F0Ec5BerF8z3AgAAAPUJIEKq2IJvvrXd+N/w78LnqiV1vdcWzDtUJ+HL2Xw35vrBVgAAAGBGABF41UaPwxA6HxeCgydFBLHoxpx3Xz4LVwY/BAAAgK0IIMKr1ApFz935gf7ivCrfV2dh3qv1x7nWdBboOl25R+Hvn+x/nz4sti7MJ1MZhssfg6L78tkXrTsBAABgSwKI8CqVk4X0v4frMjY3uj4O7dtP4Sz7+nC0wodP2f/mjrPK6PdtiPM6x5mdfyVGEAfn7XA5PAt301lVJibXpxu6efxQ+BAAAAC2JYAIr1ScBOXubBgu28UkLO3HqzC+/xY+lj+vKwYgJxO6xKBbGF6Gdvn5eFP07wknNw9Fa8DyOKu21/p8GorY5+pEKTP90C23kR/nn154qBg7sphMJXpqewAAAMAm/xuPx/+V/27U4Lys3N9fLHYtfGXeynnwPHk6CHdhvNLSjZeSt7C8/BTukiebGYTzo27403sI98Y/hDfj//7v/8L/+//+v+UnAADgJWiBCLxJxXiQJk8BAACA5xJABN6ewXloXw5Dp/fN5CkAAADwTAKIwJsxHcex2w9nd2NdlwEAAKABOxsDEd4SYyACHAZjIAIAwMvTAhEAeDUEDwEA4OUJIAIAAAAAlQQQAQAAAIBKOwsgxjHjjo6vw6j8/Fq9lfMAAAAAgG1ogQgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVPrfeDz+r/w3UGFwfhS64S6Mb07KbwAAAADeBy0QAQAAAIBKAogAAAAAQCUBRAAAAACg0s4CiHHMuKPj6zAqP79Wb+U8AAAAAGAbWiACb9PoOhwfHYWjo/MwKL8CAAAA6hNABAAAAAAqCSACb1PrItyPx2E8vgkn5VcAAABAfQKIAAAAAEAlAUR4pUbXx+UEP4Nwno/1VyzH10tT/uRjAR6H+HX+O9N1i+8WLW4rLudVAwgOzhfWO1q74ihcH8+ts3af0fJ+n7HedOzDcqmYBKnq+q09j+VtTpeq46yWdN/Ka7tyLzPFPay/XwAAANiWACK8ZsPL0D7qhnAXu+pmy0MvhMv2mqDfMPv6KLQfr4r1xg+h18m++zoXXMuDZN3wp/dQrjMOD71O6HdXg4h5EKvbD2eT/cbly6+l9WKArB0uP93NbS8e3vL24nrd0D+brTceX4XH+WPLJa437bo8Dndn5XdVlq9f/IV+d/H44nVpX4YwvS7x2sUfdLKv7sNFK1+rnk337eRbvo/h7e+Va/DjchjC2dV2+wUAAIAtCCDCKxeDeDeTQf5aF+EqxsC+LwffMjHwNlsxfD6NEarHcr1RuP56GYbZOvdzkanWxX0ehFvY3ug6fL0chk7vYbbf6ORm4fPo+nvoh7NwN/dl6+JnHhjr/5qL0I3+hj/Z/86+LGws3NxfZEc5J3W9mhau38mX7IhD+PN3dvVGv2/DMPv2anpdWuHiZy90sm8fVy5yuqfv2+T+3Ibf8/sY/Mqu6fI1AAAAgN0SQIRU/e60u2mx/BP+WfhctaSudx4WGuYlOQvLsaTWx/nA4Mxy0CkGB6cTjIx+h9thCJ2Pq6G4ky9nC4Gs1YDaOqPwO27w7MvSBCat8OFT9r8/f2fH1/oQ4lfrWjouSF2vltXrFw2fExlMsvm+tS6usrWG4XYugjj41c9uUi98W3PMAAAAsCsCiJBqoetsXP4N/y58rlpS19v/bMGfPmxuyzd6HJb/esoo5KutBF2PQjc2oVsIcJ6Em7JbcAwOFuutG+Mvdb1mtT6fhk7oh+/THZWtNSuCj805CUXsdtKNeRDy+OHp52e1uAQAAIC6BBDhjckDfJ2PWwWZ5rvuToz+xo7DM3lLuY1aIV9tJehaFSxthYv7yc/u8pZ3q2MlRqnrNWj0mO0lhOFluwxatsNl6IWHhgO+6+7bQuvPvPvyppafAAAA0DwBRHhTRiGP9336UC+A2PociiER1wQQ88DWafhcbrCV90Huh/lhDFet6aqcLLY0jMHB9QHNmdT1nmMUrr/3VwOhT427OJ2xuU6X9Ir7lk+mMgyXPwZF9+WVLuEAAACwewKI8IYMztvhcrg4cUmaVrjIZ/HohuO5PsFxtuVuvxN6P+cCZuUMwf3uUvfhwflCS8CTb73QibMNb2oemP3e/D6jYgKWTjidRC2j1PUaVQZC+7+Sg4HFGJHRpiDrTPV9KydTye5L7Ppt8hQAAAD2QQARXrV+6Oat3Yql++cZXWtPbsL47myuq+5RaF9+Cnfj+7DYa7boRnx3VnQfnqx79OvL4qzMrYtwH1sIrhkHcSEQmO33Z/i68PP2ZQi9h6X9Jq4Xg56TnxfjLV6Gdvl5OQCZ4uQmtnJcvM6TZV1stBgzMXpqjMT0+1ZMphLtesxFAAAAWO9/4/H4v/LfjRqcl5Xip7r6vQJv5Tx4njwdhLswrt2yb3dioKwI8O1/8pW3axDOj7qh31ktA/I0EVtnLgc6N6h/34pj+NN7CPfGPwQAAGAPtEAEqDL6G+LQhOtmPi4mkxmGNcNGNqropm3yFAAAAPZHABGgSutDiEMgDm9/L00GMwg/LuPkMr3wbZfNPwfnoZ3tp9P7ppUpAAAAeyOACFApzvT8EHphNo5isXRDP87MvKOhDabjOHb74exurOsyAAAAe7WzMRDhLTnEMRABAAAAXoIWiAAAAABAJQFEAAAAAKCSACIAAAAAUGlnAcQ4ZtzR8fXSzKWvz1s5DwAAAADYhhaIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAECl/43H4//KfwMVBudHoRvuwvjmpPwGAAAA4H3QAhEAAAAAqCSACAAAAABUEkAEAAAAACrtLIAYx4w7Or4Oo/Lza/VWzgMAAAAAtqEFIrCd0XU4PjoKR0fnYVB+BQAAALw9Aojwqg3C+dFROL7WRhYAAADYDQFEYDuti3A/Hofx+CaclF8BAAAAb48AIgAAAABQSQARXqHR9XE4yscf7IZ+9nl42S4/F8tCl+Z8rMLjEL+a/V5ciu8WTMc1nC3nywMcLq9TMclQvq/8Z0U36+n6KxtMU7W9hXMdnK9+VyrOfc05AwAAAE8SQIRXqHVxH8Z59+G7cJZ97vQeys/Fcn/RKlacGobL9lFoP16V6zyEXif77ut88G8Qzr+G8HNuOw+9Tuh3l4Ju067L43AXd/6U4WVoH3VDuCu3GX+h310NSqZa3t5DL2QnNtveybfsvLLVbn8vBTUH4cflMISzq7ByaQAAAIAnCSDCe3F2F8Y3k9EKW+HzaYy0Pc4F2k7Czf1F9pOZ1sVVOAvDcPt7+2Z7Z3fjMN3tyZc84Pnnb0Pba12EqxiT/D4JhE7O6zYsHPLgV95S8+yL0RoBAACgLgFESNXvTrvNFss/4Z+Fz1VL6nrnYduGeSmWg2dFK8ZdT4ByFtbF7IaP2wYQV7fX+rgYCF0X9Bz86sdmmuGb+CEAAADUJoAIqWILvrLrbrH8G/5d+Fy1pK63/9mMF8dIjEsxxuLrchK+nM13Yx6EPH54+nmhdSUAAACQRgARyMXgYftymHcRngU1izEWD9nocRhC5+NCcPCkiCAW3Zjz7stn4crghwAAALAVAUR41Vqh6MG7bZfgiVH4fRsDcXvs5jud3blOV+5R+Psn+9+nD4utC/PJVIbh8seg6L589mXvrTsBAADgtRJAhFetFT58yv7X/744U3Jt5XbmJx/JA3ov14V59Ps2DPN/9cOvxAji4LwdLodn4W46q8pEOZlKvxu6efxQ+BAAAAC2JYAIr9zJzUPR2q49G7vweIto4sp22rfh9CF+V65Qmh8nMQbnwvAytJ+x34nW59NQ7Gr9xCuFfuiW+8r3/6cXHirGjiwmU4me2h4AAACwyf/G4/F/5b8bNTgvK/f3F4tdC1+Zt3IePE+eDsJdGK+0dOOlFGM0fgp3yZPNDML5UTf86T2Ee+MfAgAAwNa0QATepNH1d5OnAAAAQAMEEIG3Z3Cezyjd6X0zeQoAAAA8kwAi8GZMx2fs9sPZ3VjXZQAAAGjAzsZAhLfEGIgAAADAe6UFIgAAAABQSQARAAAAAKgkgAgAAAAAVNpZADGOGXd0fB1G5efX6q2cBwAAAABsQwtEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKDS/8bj8X/lv4EKg/Oj0A13YXxzUn4DAAAA8D5ogQgAAAAAVBJABAAAAAAqCSACAAAAAJV2FkCMY8YdHV+HUfn5tXor5wEAAAAA29ACEahndB2Oj47C0WQRYAcAAIA3TQARqKd1Ee7H4zDOlruz8rsXMwjnR0fh+FrIEgAAAF6KACIAAAAAUEkAEQAAAACoJIAIr9ooXB/PjUd4dBwWeveW4xWu6/I7uj6uXH+2vaNwPih/VlOx/fOw+OtFF+Sj5Y1u2G+xrfh9N/Szz8PL9sK6q+e34brUkO87H+exPPZ1+xycr35XWnudAQAA4BURQIRXKwa02uHy010+HmFcHnohXLbngm+tz+G0E8Lw9vfSRCej8Pt2GELnNHxulV/F7X0N4We5rWJ7ndDv7jr4tXm/rYv78md3IQ672Ok9TNeNy/3F9CQyCdelruFlaB91Q7gr91lscLa9k28hO+Q113kQflxm1/nsKiwcIgAAALwiAojwSo2uv4d+OAt3NyflNzHQ9jMPZPV/TSOI4eLqLEa2wu/5yNbod4jxw7Ori2yNiZNwcz//OW7vKtvDMNwu/HLTmt1v2nWp7+xuHKabbF2EeFn73yczULfC5yJSu3idB7/yFpNnX2bHAgAAAK+NACKk6nen3VeL5Z/wz8LnqiV1veXuvk8pWxCefQmLoalW+PAp+9+fv7OWcCdfVoJxo9+32Tdn4e3FtWpcl1pWr1XrYwwYPk63ty7oOfjVj80lwzfxQwAAAF4xAURIdTbrElss/4Z/Fz5XLanr3SwFvZ4yCo/D7H8rQc2j0M0HCZwFtmILvy95I8RJ99qqIFv2k+lYg5OlGHNw15rbb53r0rTl6zwIefzw9PNC60oAAAB4bQQQ4VVqhdgAbjWouT4YefKtFzqT7rV59+VO6C01i4tBvPblMO+qO9tOMebgLjW733rX5TlGMVLZ+bgQHDwpIojFdc67L5+FK4MfAgAA8MoJIMKrVLNLbj6ZStG9Nu++vDB5SjSZVGXH3W1Hf8Of8p+FuvstAoTDx6qz3qKr8nQG6HpdyP/GE/n0YSGAWEymMgyXPwZF9+U1rTwBAADgtRFAhFeqaFV4GdpJUwsXk3wMb3+EH7fDNd1qy8Db/CQgeWBt+y7Mrc+noZP99nTekri99mWIPYxn6u63XL//vXJm6HrXJdtdPh5kNHesGwzO2+FyuDhRS6GcTKXfzbtMmzwFAACAt0AAEV6r1kW4j11914z3d7wmupYH9Ib90B+u71Z7cvNQtJ5rl9tp34bTh/hduUJpfrzCYlzBy9Bet9/s+H5mv9zvTrb3GK7Gq9tL3e/Eyvpr9lv7uuT/empSmX7ozm2n+6cXHiq6QxeTqURvcZIaAAAA3qP/jcfj/8p/N2pwXlay7y+WWjq9Lm/lPHiePB2EuzBeaXHGW1eM0fgp3CWPnzgI50fd8Kf3EO6NfwgAAMAboAUiQING19+DyVMAAAB4SwQQAZoyOM9nlO70vpk8BQAAgDdDABHgmabjQnb74exurOsyAAAAb8rOxkCEt8QYiAAAAMB7JYAIAAAAAFTShRkAAAAAqCSACAAAAABU2lkAMY4Zd3R8HUbl59fqrZwHzyMdAAAAAO+VFogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQCUBRAAAAACgkgAiAAAAAFBJABEAAAAAqCSACAAAAABUEkAEAAAAACoJIAIAAAAAlQQQAQAAAIBKAogAAAAAQKX/jcfj/8p/AwAAAAAs0AIREgzOj8LR+aD8BAAAAPB+CCACAAAAAJUEEAEAAACASjsLIOZdPo+vw6j8/Fq9lfMAAAAAgG1ogQgsGl2H46OjcHR0HtaO+jj9ebkIsAMAAMCbJoAI78IgnB8dhePrBkJ9rYtwPx6HcbbcnZXfAQAAAG+WACKwaBogvAkn5VcAAADA+yWACAAAAABUEkCEV2p0fbxmnMKiq/LRefFtsU4cq7Ab+tnn4WV7NnbhcpfmnYxtWB7PdDkO2/aizs8lP6bFbS6cw+B89btScS223z8AAAC8VwKI8Ia1Lu7zsQrH47sQhyvs9B7Kz8Vyf9EqVowaH9swBvq6oX92N7fPq/D49RmByeFlaGfbDHfl9h56IVy2QxkvDeHkW+h1stVufy/tYxB+XA5DOLsK86cMAAAAbCaACOzG6G/4k/3v7Mv8SIon4eb+Ijwnhnd2Nw43k022LsLVWQj975OgZCt8Ps0jiOH3fARx8Ctvgbl4LAAAAEAKAURI1e9Ou80Wyz/hn4XPVUvqesvdkV+51ofwKftfv3s0ayH4bGdhOQbY+hgDho/TFoeti6tsrWG4nYsgDn71Y/PL8E38EAAAAGoTQIQEJzeTLrjzy7/h35Xv1i2p6721WY9Pws34Ie9SHIOIRZD0JcYgPAlfzua7MQ9CHj88/fyslo8AAADwXgkgAjvUChf3kwBpHIdxGC7bTbZIDGH0OAyh83EhOHhSRBCLbsx59+WzcGXwQwAAANiKACK8JeW4g6taoejpu/Pmf0+ILRKLyVz+/F1zHNNZoOt05R6Fv/GEP31YbF2YT6YyDJc/BkX35bMvb6x1JwAAALwcAUR4pVqfT0Mn9MOvSbQtBuDal2FYflzUCh/yAQm/v0AX4tLgPBwv7Wx0/T074k44/bzaGnD0+7Y89rlz2mBw3g6Xw7NwN51VZaKcTKXfDd08fih8CAAAANsSQITXqnURfvY6s/EF24/hqhxzcJ2Tm/izogvxZOKW+QDf6Pp4+n0MuoXhZWg/Y71sh+Fn+DpdNy7tyxB6D/dhXW/iIiAarU6UMtMP3bntdf/0wkPF2JHFZCrRU9sDAAAANvnfeDz+r/x3owbnZeX+/mKxa+Er81bOA167GLhsX34Kd8mTzQzC+VE3/Ok9hHvjHwIAAMDWtEAE3qSiu7TJUwAAAOC5BBCBt2dwHtqXw9DpfTN5CgAAADyTACLwZkzHZ+z2w9ndWNdlAAAAaMDOxkAEAAAAAF4/LRABAAAAgEoCiAAAAABAJQFEAAAAAKDSzgKIg/OjcHR8HUbl59fqrZwHAAAAAGxDC0QAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoJIAIgAAAABQSQARAAAAAKgkgAgAAAAAVBJABAAAAAAqCSACAAAAAJUEEAEAAACASgKIAAAAAEAlAUQAAAAAoNL/xuPxf+W/AQAAAAAWaIG4weD8KBydD8pPAAAAAPC+CCACAAAAAJUEEAEAAACASjsLIOZdf4+vw6j8DAAAAAC8PlogUt/oOhwfHYWjo/NgdEgO1l7T6SCc5/ueLa9uKNVXns9H18ev9ti3Mr1f5VL1B7zU9Rp2GPcjIV9uSvd7un7beQPl0Cb7vh+b0kvT6p7vSx8f66XetxdJz0W5cHx9uCUXAIdLABGgUfHlvBv6Z3dhPB5Pl5uT8sd7peLwZrUuwn2Z1u7Oyu/WSV2vlteQrhrKlzu5frtwyOVQg17N/WjIezvft2Kv5TMANEcAkfqmLzg34a3VRXhpReBhpXVEE60m9pVOB79CP3RC79srzx3yOW9Jar58K+n+rZRDh+7Q04tyHABokAAiUOnluh32w/e51kuj37dhWP77tRn9/ZP991P40Co+A/v33vKlcggAgKYJIL5ao3B9PDdOytFxWO49lgd/8vFTJq28ymXdIEjL465Ml7ntJo7NUmu/CeeRqmq/C93qBuer35WKYNn2+39r4kRI7cth6PS+7bjlQif0emdhePu7TE+j8Ps2hN5dL/vJn/B3/n6sSafVY5iVSyPptDlFOlsOypbHMLfv2sdXpu116xX7jN93Qz/7PLxsL6y7kB8Sr19h6diyZfnwmrzOT0/OVWx/NW8vljHru9kmlEP5dSm+n13PinVTtldHSrrfg1rpaqqh+9G0Wuk+Vep5LOejlzrfYj+b0/OGfD7d1my9+PPpdutey6a3V0O+jzXl1Uo6TUwvydvLNZjun3l868vnPZRr07RQHut03XX73kG58cRzdSrlPHYi9XyX1yvKawDYhgDiqxRf9Nrh8tNsbKOHXgiX7TUvLcPL0M5eFsJdsd44DqrS7y6uF19+2pch9B7K7T2EXif+oJN9dR8uJi0Ypl1hEsZmSdlvnfNItbzfYoOz7Z18y89tFqyaGIQfl8MQzq5m5/tuFS+b3ewN8yy7jve7vCCjvyG2kwmfv4Sz4W34HW/K6He4Dafh88pus/TyNYSfZVop0ksnS1ZLL82Np9MEc5WMGHSNLSq705f1bKmovG2UeHx5xSq7YfF+Ta7N+Muv6Xqti/vy+7sQL0lnmteLZeEep16/vNLUDX/mtlXcjy3LoQStjzHzPta4lvE+tMPj1Wy/Mci1uN865dAw/779eFWuG8vK7Luv8/e36XItMd3vQa10lWv6fmxQJ1/WKTeSpJ5HXG95rMKr8LiQphLVLocS0nONfH779Xv4+FC8P8Sf59vNTrqT5f8fW6T9preXbLm8Km7c4vk+5zmzbntNlxsH/76WbS+5XNtHubv5uVqocx5NSjzfPP+2w+3pfNlclNcAsA0BxFdodP09qxachbu50dBbFz+Ll+xfq29K8eVnuurJl/zF4c9cs66iu+hZuJpFCsPFz9j6axgen/ECtHG/Nc8j1cJ+s5foq/gu/H3yotkKn09jEKIMVk3k40Vlv/tldizvU/lSOozB45cccP8kfMtefm+zmzL4cRk+XV1kd2rZSbi5X/y+dXGVpaBh/nvb2pROk5zczL3EZ+krpuvpy3q2LB13HRuPL6sgfM1bij4s3q/smHZ3/0bh+utlGJ7dLQSJYkApr3tO89tME9e59eFT9t9Zq9S8ReKktlQGoz8t9dlsvByKgZ7pupPyZBbUbL5c202635fG78dTdpgvN0k+jzLdLj57Vu95km3O98n0XCefD8PwU/wDXCvk2fTZ4y82vb16FtLpyntEfZu2t6v3oVQvmi9zNcu1ly53k5+r+ymfU883vk8NO73wc+WPOQCwHQHEFP3u7C/4+fJP+Gfhc9WSul6dMeZi987YUu5L9toyr3zJ/vN36QX3LKyLiQ2fExlMsmm/dc8j1ep+l1strXu5G/zqx+Yz4V2PN1+2NOln1+FhPNfy9IW0Pp/GJifhe3/+Hj4viL3ZvvJHqs3Ht/oHgBcQW4lm2bfzcXWfJ1+yqudygL6p69z6GLLcXBqEmG2z2tJc+dkJi4e0ab/1y6HlPzIUrfAmExTsqlx7K5q/H4epxnm0PoT41dqWuy/gyfRcM58vbKuzrhV5PU1vL91qOq3f+nnepu3tO90ffr586XJ3L8/VZKnnWzwjO6efs58AQDMEEDc4uZn76/10+Tf8u/LduiV1vTqz443CY/besBrULLqcbvOCGwM3nTA/iUXZ6qDipbIZzZ9HupNQ1H0m3Zi9ZEX5X6rjPxKufd7ya+m+rR/TaYPRY7bPcqD/1udwmn0arrwUzyyOgRQXY/mM8oy0H8ut/XYuD7aUQeXYavisF3qdfsgbXMynpWRNl0O7KdfeT7rf53OhSXXO4yTclMOGxCBisd7+u6fPe/F8/u4cero/9HKt+eOr81x9+fL5rZSTALxGAoivTivEP1zn3TmeHYws5RXv7J3jcjIAfjtchtgKbYttJdvBeVTIXwQ7HxeCgwutJ/Luy4f6l+aXUwTL49g4cdyspyuw6wLrzx8rsRUu7rNtTbrktIqWORPxJT2O6xW7Ws32ayyfvCXLnqzrglzM/rorRbkR9xv30/n4Oe/KFrts5ftdyuebNV0ONV+uva90/3LPhd2qex5l2Zf/LN7bYsy3fbRIXOfl8/nhWfce8RyL2zv0dH/o5Vrzx5f6XN1P+fxWykkAXiMBxFen6S4jo3D9vb/6IrLDsaEKW5xH3sU2BjjrdfnO6zmfPiyeTz6ZSlZJ+zEoui8/0ertfZm0hjmcCmxReS277LzVbuaTyWS2UIwLWLbC26ioeNTuOrwsthat2E5RMd5Vd8Oi3Bg+/s7SQwin2U7yFtSTcmQ5n2/UdHnafPn8OtJ9Q+mq8eu3L885j1gGFwGIdYG7F7W3fH5oKt4jtra8vUNP94derjV//dKeq/sqn1PPd325XIyfCADbEUB8hU6+FbMQthuJ7pQvIgvjiL2MuudRjEkTpQZLYlfbOCHI4kDThXIQ7n437/KxPL7O+1a0hikmYtyya3Ki9FYsZTqdH3NrMmZj+fG1mAwZME3D8TzaZffxbZQzi6/M+jg4XxMAnuT378/sIpmlkXwWgO5C+oitMbr9Tuj93N0fIPKWIf3LcDmZqTu2VM3Kke7ldlew2fJ0R+Xzwaf7ptJV8/djX5LPI8uny2VsUcHv5AHy/dpfPj8k1e8R21m3vUNP94derjV+/ZKeq/srn9POd/KeOyuX8xaTt5/CWfY1AGxDAPE1al2E+9hCIXupXx7/ZJtgz8lNbO0Qu60ubisu8+8m8cVj8n0xzkr28lJ+3irIVPM8isBL9NTYjIvn0f1T3RW7mEwl2uVYj69X7KYcZ/IcXv548eDyOic3s5aR+f1t34bTh2LssHmNp9OmZen+Z3bQ0/HO2o/hqhwDbTuTgO/ctYnLry9Ls0UWVq7j0nVJvn5xxte7s7mhD45C+/JTuHupCXimrXeKMU2jdZM9bNRwedp8+dxsut9V/tiUrpI1fT8SNX79Us8jy0c/w9eFn7cvQ+g9vPxEVmvtO59vsJv0vPk9ot5+E95LEtNL4+k01Z7KtWSNlxtpz9W9lc+J51vMmD47vvbjVRjffwsfy58DQF3/G4/H/5X/blScZCF/Sdp5V1ieZxDO419L48y7S/cqv4exlcGhVGQ2iC9eRcUmdfyX4tz/9B4aGL8PAHjN6r9HPK3p7QEA7JMWiO9dOfbauhmIi0GkyxlP36Cim5jJUwAAAACeIoD43rWKmW6Ht7+XBmMehB9xTLG3OmnF4DzEmfM6vW9aBQAAAAA8QQDx3Stn3Q2zcVaKpRv6cWbmN9YFfTq+TLcfzu7Gui4DwJsVhyqZf7epWI6vG5vBFwDgrdrZGIgAAAAAwOunBSIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKgkgAgAAAACVBBABAAAAgEoCiAAAAABAJQFEAAAAAKCSACIAAAAAUEkAEQAAAACoJIAIAAAAAFQSQAQAAAAAKoTw/wdUeRkKWYppLgAAAABJRU5ErkJggg==)
    

---


## **Data Reading from Different Sources**


### **Files**

In many cases, the data is stored in the local system. To read the data from the local system, specify the correct path and filename.

### **CSV Format**
Comma-separated values, also known as CSV, is a specific way to store data in a table structure format. The data used in this project is stored in a CSV file. Download the data for the project.


Use following code to read data from csv file using pandas. 
```
import pandas as pd
csv_file_path= "D:/ProjectPro/Telecom Machine Learning Project to Predict Customer Churn/data/Telecom_data.csv"
df = pd.read_csv(csv_file_path)
```
With appropriate csv_file_path, pd.read_csv() function will read the data and store it in df variable.
 
If you get *FileNotFoundError or No such file or directory*, try checking the path provided in the function. It's possible that python is not able to find the file or directory at a given location.


### **Colab - CSV Format**

```
# mount the google drive
from google.colab import drive
drive.mount('/content/drive')
csv_file_path= '/content/drive/MyDrive/project_pro/Telecom Machine Learning Project to Predict Customer Churn/Telecom_data.csv'
df = pd.read_csv(csv_file_path)
```

### **AWS S3 - CSV**
Use the S3 public link to read the CSV file directly into a pandas DataFrame
```
s3_link = 'https://s3.amazonaws.com/projex.dezyre.com/telecom-machine-learning-project-for-customer-churn/materials/Telecom_data.csv'
df = pd.read_csv(s3_link)
```

### **Database**
Most organizations store their data in databases such as MS SQL. Microsoft SQL Server (MS SQL) is a relational database management system developed by Microsoft. A BAK file in Microsoft SQL Server is a backup file that contains a copy of a SQL Server database at a specific point in time. It is essentially a binary representation of the database and includes all its data, tables, schema, indexes, stored procedures, and other objects.

#### **Installing MS SQL Management Studio**
To install Microsoft SQL Server Management Studio, you can follow these steps:

* Go to the Microsoft SQL Server Downloads page (https://www.microsoft.com/en-us/sql-server/sql-server-downloads) and click on the "Download now" button for the version of SQL Server Management Studio that you want to install.
* Follow the instructions on the screen to download the installation file to your computer.
* Once the download is complete, locate the installation file and double-click on it to start the installation process.


#### **Restore a BAK file in MS SQL**

* Open SQL Server Management Studio and connect to the SQL Server instance to which you want to upload the BAK file.
* Right-click on the Databases folder in the Object Explorer pane and select "Restore Database..." from the context menu.
* In the "Restore Database" dialog box, select the "Device" option under the "Source" section.
* Click on the "..." button to open the "Select backup devices" dialog box.
In the "Select backup devices" dialog box, click on the "Add" button to add the BAK file that you want to upload.
* In the "Locate Backup File" dialog box, browse to the location where the BAK file is stored in the Telecom Data Analysis Project to Improve Service Quality directory under the ‘data’ folder, select the file, and click on the "OK" button.
* Back in the "Select backup devices" dialog box, the BAK file you added should now be listed under "Backup media:".
* Click on the "OK" button to close the "Select backup devices" dialog box.
In the "Restore Database" dialog box, you should see the BAK file listed in the "Backup sets to restore" section.
* By default, the original database name and file locations from the BAK file will be used. If you want to restore the database with a different name or to a different location, you can modify the "To database" and "Restore as" options under the "General" section.
* Click the "Options" tab for additional restore options.
* If you want to overwrite an existing database with the same name, you can select the "Overwrite the existing database (WITH REPLACE)" option under the "Restore options" section.
* Click on the "OK" button to start the restore process.
* Once the restore process is complete, you should see a message indicating that the restore was successful.

#### **Read Data from DB to Python**
 The data can be accessed by secret credentials, which will be in the following format.
```
import pyodbc
import pandas as pd
connection = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};\
                       SERVER=server_name;\
                       PORT=1433;\
                       DATABASE=database_name;\
                       UID=admin;\
                       PWD=password')
```
#### **Steps to install ODBC driver**
* Go to the Microsoft Download Center page for the ODBC Driver 17 for SQL Server: https://www.microsoft.com/en-us/download/details.aspx?id=56567
* Select the download button that corresponds to the operating system you are using.
* Select the language you want to use for the installer, then click the download button.
* Once the download is complete, run the installer.
* Accept the license terms, then select the features you want to install.
Choose a location to install the driver, or use the default location.
* Complete the installation process by following the instructions provided by the installer.
* Once the installation is complete, you can use the ODBC Driver 17 for SQL Server to connect to SQL Server databases from applications that support ODBC connectivity.

#### **Query to read the data into Pandas**

```

query = '''select * from Processed_month_1_data
           UNION ALL
            select * from Processed_month_2_data
            UNION ALL
            select * from Processed_month_3_data
            UNION ALL
            select * from Processed_month_4_data
            UNION ALL
            select * from Processed_month_5_data
            UNION ALL
            select * from Processed_month_6_data
            UNION ALL
            select * from Processed_month_7_data
            UNION ALL
            select * from Processed_month_8_data
            UNION ALL
            select * from Processed_month_9_data
            UNION ALL
            select * from Processed_month_10_data
            UNION ALL
            select * from Processed_month_11_data
            UNION ALL
            select * from Processed_month_12_data
            UNION ALL
            select * from Processed_month_13_data
            UNION ALL
            select * from Processed_month_14_data'''
processed_data = pd.read_sql(query,connection)
processed_data.head()

```
