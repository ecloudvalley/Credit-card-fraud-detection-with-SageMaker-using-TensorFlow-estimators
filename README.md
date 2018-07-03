# Credit card fraud detection with SageMaker
![kaggle_data_pic.png](/images/kaggle_data_pic.png)
**This lab using credit card fraud detection dataset on Kaggle**

## Scenario

Various of companies want to make their data more valuable because data analysis is one way to make better quality of service to client. Valuable data also have a great insight to get more business opportunities. Thus, many startups or companies decide to build machine learning workflow but the process is so complicated that can easily cause several problems. One of main problems is due to lack of data scientist or data engineer team, so the training process seems too difficult to create and make sense. Another problem is deployment of machine learning backend server that contains lots of trouble to engineer which impossible to fulfill workflow in a short time.
 
AWS plays an important role of machine learning solution on cloud. We will focus on Amazon SageMaker to quickly create training job and deploy machine learning model. You can invoke Amazon SageMaker endpoint to do A/B test for production. The architecture also integrate with several services to invoke endpoint with serverless application which contains S3, Lambda, API Gateway.

## Use Case in this Lab 
* Dataset: Credit Card Fraud Detection <br>
https://www.kaggle.com/mlg-ulb/creditcardfraud



![preview_data1.png](/images/preview_data1.png)<br>
![preview_data2.png](/images/preview_data2.png)<br>

## Lab Architecture
![lab_architecture.png](/images/lab_architecture.png)

As illustrated in the preceding diagram, this is a big data processing in this model:<br><br>
* 1.&nbsp;&nbsp; 	Developer uploads dataset to S3 and then loads dataset to SageMaker<br><br>
* 2.&nbsp;&nbsp; 	Developer train machine learning job and deploy model with SageMaker<br><br>
* 3.&nbsp;&nbsp; 	Create endpoint on SageMaker that can invoked by Lambda<br><br>
* 4.&nbsp;&nbsp; 	Create API with API Gateway in order to send request between Application and API Gateway <br><br>
* 5.&nbsp;&nbsp; 	API Gateway send request to Lambda that invoke prediction job on SageMaker endpoint<br><br>
* 6.&nbsp;&nbsp; 	SageMaker response the result of prediction from API Gateway back to Application<br><br>

## AWS SageMaker introduction
Amazon SageMaker is a fully managed machine learning service. With Amazon SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers. It also provides common machine learning algorithms that are optimized to run efficiently against extremely large data in a distributed environment. 

## Deep Learning model in this Lab
TensorFlow DNN classifier using estimators<br><br>
https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator<br>
TensorFlow's high-level machine learning API (tf.estimator) makes it easy to configure, train, and evaluate a variety of machine learning models.

## Prerequisites
1.	Sign-in a AWS account, and make sure you have select **N.Virginia region**<br>
2.	Make sure your account have permission to create IAM role for following services: **S3**, **SageMaker**, **Lambda**, **API Gateway**<br>
3.	Download **this repository** and unzip, ensure that folder including two folders and some files:<br>
Folder: **Flask-app** (contains the code to build application of demo)<br>
Folder: **data** including **train_data.csv**, **test_data.csv** (training and testing data of machine learning job of SageMaker)



## Train and build model on SageMaker


#### Create following IAM roles

The role that allow Lambda trigger prediction job with Amazon SageMaker by client request from application<br><br>
The role that allow Amazon SageMaker to execute full job and get access to S3, CloudWatchLogs (create this role in SageMaker section)<br><br>


 
* 	On the **service** menu, click **IAM**.<br>
* 	In the navigation pane, choose **Roles**.<br>
* 	Click **Create role**.<br><br>
* 	For role type, choose **AWS Service**, find and choose **Lambda**, and choose **Next: Permissions**.<br>
* 	On the **Attach permissions policy** page, search and choose **AmazonSageMakerFullAccess**, and choose **Next: Review**.<br>
* 	On the **Review** page, enter the following detail: <br>
**Role name**: **invoke_sage_endpoint**<br>
* 	Click **Create role**.<br>
After above step you successfully create the role for Lambda trigger prediction job with Amazon SageMaker<br>
![iam_role1.png](/images/iam_role1.png) <br>
* 	On the service menu, click **Amazon SageMaker**<br>
* 	In the navigation pane, choose **Notebook**.<br>
* 	Click **Create notebook instance** and you can find following screen<br>
 
* 	In **IAM role** blank select **Create a new role**<br>
* 	For **S3 buckets you specify**, choose **Any S3 bucket** and click **Create role**<br>
![iam_role2.png](/images/iam_role2.png) <br>
* 	Back to **IAM** console, click **Roles**<br>
* 	Click the role **AmazonSageMaker-ExecutionRole-xxxxxxxxxxxxxxx** you just created by before step<br>
 
* 	In **Permissions** tab below click **Attach policy**<br>
* 	Search and choose policy name **CloudWatchLogsFullAccess** then click **Attach policy**<br>
* 	You will see below screen on console<br>
![iam_role3.png](/images/iam_role3.png)  <br>
You successfully create the role for Amazon SageMaker to execute full job and get access to S3, CloudWatchLogs<br>



#### Create S3 bucket to store data
Create a bucket to store train_data.csv, test_data.csv which also provide the bucket location for SageMaker jobs to store result<br>
* 	On the **service** menu, click **S3**.<br>
* 	Click **Create bucket**.<br>
* 	Enter the **Bucket name “yourbucketname-dataset” (e.g., tfdnn-dataset)** and ensure that the bucket name is unique so that you can create.<br>
* 	For Region choose **US East (N. Virginia)**.<br>
* 	Click **Create**.<br>
* 	Click **“yourbucketname-dataset”** bucket.<br>
* 	Click **Upload**.<br>
* 	Click **Add files**.<br>
* 	Select file **train_data.csv** and **test_data.csv** in **data** folder then click **Upload**.<br>
* 	Make sure that your S3 contain this bucket **“yourbucketname-dataset”** and the region is **US East (N. Virginia)**<br>

Congratulations! now you can start building notebook instance on **SageMaker**<br>

#### Create notebook instance on SageMaker
* 	On the **Services** menu, click **Amazon SageMaker**.<br>
* 	In the navigation pane, choose **Notebook**.<br>
* 	Click **Create notebook instance**<br>
* 	Enter **Notebook instance name** for **“ML-cluster”**<br>
* 	For **Notebook instance type**, select **“ml.t2.medium”**<br>
* 	For IAM role, choose **AmazonSageMaker-ExecutionRole-xxxxxxxxxxxxxxx** you just created by before step<br>
* 	Leave other blank by default and click **Create notebook instance**<br><br>
![create_notebook1.png](/images/create_notebook1.png) <br>
* 	Notebook instance will pending for a while until it in service<br><br>
![create_notebook2.png](/images/create_notebook2.png) <br>
* 	Until the status of **ML-cluster** transform into **InService** click **open**<br><br>
![create_notebook3.png](/images/create_notebook3.png) <br>
* 	The jupyter notebook screen show up like below<br><br>
![create_notebook4.png](/images/create_notebook4.png) <br>
* 	Click on **sample-notebooks** in file tab<br>
* 	You can find that different machine learning sample is out there<br><br>
![create_notebook5.png](/images/create_notebook5.png) <br>
In this workshop, we use **TensorFlow DNN Classifier** in **sagemaker-python-sdk** that is suitable for our data<br>
* 	Click **sagemaker-python-sdk**<br>
* 	Click **tensorflow_iris_dnn_classifier_using_estimators**<br><br>
![create_notebook6.png](/images/create_notebook6.png)<br>
* 	Click **tensorflow_iris_dnn_classifier_using_estimators.ipynb** to open kernel<br>

#### Setting training input and train model on SageMaker

* 	The kernel name **conda_tensorflow_p27** will show up on the upper right and a tutorial for this SDK in this page<br><br>
![model_setting1.png](/images/model_setting1.png) <br>
You can find that the tutorial example is for **Iris data set** to predict flower species based on sepal/petal geometry<br><br>
![model_setting2.png](/images/model_setting2.png) <br>
* 	In the cell (code blank) below **“Let us first initialize variables”** heading, change the bucket location with **“yourbucketname-dataset”**.  For example:<br>
custom_code_upload_location = 's3://tfdnn-dataset/customcode/'<br>
model_artifacts_location = 's3://tfdnn-dataset/artifacts'<br><br>
![model_setting3.png](/images/model_setting3.png) <br>
After changing bucket name<br><br>
![model_setting4.png](/images/model_setting4.png) <br>
**Remember to press shift + enter to run this cell**<br>
 
In next step we will create a deep neural network model with a source code:<br>
**The framework is TensorFlow estimator**<br>
**Neural network classification model with DNN Classifier** <br>
**TensorFlow estimator hint below:**<br>
https://www.tensorflow.org/get_started/premade_estimators<br>
* 	Back to **Home** of jupyter notebook and click **iris_dnn_classifier.py**<br><br>
![model_setting5.png](/images/model_setting5.png) <br>
You will see this python code<br><br>
![model_setting6.png](/images/model_setting6.png) <br>
First, we need to modify the parameter **shape** which represent the label or the column we want to train in our credit card fraud detection dataset<br>
 
In function **estimator_fn** modify shape from 4 to **29** that means we train **V1, V2, V3 … V28 and Amount label (total: 29)** and predict the Class<br><br>
In function serving_input_fn modify shape from 4 to **29** that means we train **V1, V2, V3 … V28 and Amount label (total: 29)** and predict the Class<br><br>
 

Second, we need to modify the predict label **n_classes** which means how many type or class for predict label<br>
 
In function **estimator_fn** modify **n_classes** from 3 to **2** that means our fraud detection class only contain two types : **0 (normal)** or **1 (Fraud)**<br>
 


Third, we modify the name of training data and test data that match to **train_data.csv** and **test_data.csv**<br>
 
In function **train_input_fn** modify **iris_training.csv** to **train_data.csv**<br><br>
In function **eval_input_fn** modify **iris_test.csv** to **test_data.csv**<br><br>
 
Last, let’s modify the function that load csv with header. Our dataset is **without header** after preprocessing<br>
 
In function **_generate_input_fn** modify<br>
**tf.contrib.learn.datasets.base.load_csv_with_header** to<br>
**tf.contrib.learn.datasets.base.load_csv_without_header**<br>
 
The origin source code below<br><br>
![model_setting7.png](/images/model_setting7.png) <br>
**The code we modify for our dataset below**<br><br>
![model_setting8.png](/images/model_setting8.png) <br>
When you finish the code click **File** and click **Save**<br>
* 	Back to **tensorflow_iris_dnn_classifier_using_estimators.ipynb** console<br>
Run the cell that contain **!cat "iris_dnn_classifier.py"** you will see **iris_dnn_classifier.py**<br><br>
![model_setting10.png](/images/model_setting10.png) <br>
In following cells, modify the parameter that we have done in **iris_dnn_classifier.py**<br>
**(shape=[29], n_classes=2, train_data.csv for training set ,load_csv_without_header)**<br>
 
Cell below **“Using a tf.estimator in SageMaker”** heading<br>
Origin code<br><br>
![model_setting11.png](/images/model_setting11.png) <br>
After modify<br><br>
![model_setting12.png](/images/model_setting12.png) <br>
 
Cell below **“Describe the training input pipeline”** heading<br>
Origin code<br><br>
![model_setting13.png](/images/model_setting13.png) <br>
After modify<br><br>
![model_setting14.png](/images/model_setting14.png) <br>
 
Cell below **“Describe the serving input pipeline”** heading<br>
Origin code<br><br>
![model_setting15.png](/images/model_setting15.png) <br>
After modify<br><br>
![model_setting16.png](/images/model_setting16.png) <br>
 
Remember to run those cells to ensure whether an error occur in model<br>
Next part we train our model and get the result of training<br>


#### train the model and get the result

* 	Move on to the cell below **“Train a Model on Amazon SageMaker using TensorFlow custom code”** heading<br>
* 	Ensure that **entry_point** parameter is **iris_dnn_classifier.py**<br>
* 	We choose **ml.c4.xlarge** for instance type as default<br><br>
![train_model1.png](/images/train_model1.png) <br>
Press shift + enter to run the cell<br><br>
![train_model2.png](/images/train_model2.png)<br> 
* 	Move on to the cell below that contain code including boto3 library<br><br>
![train_model3.png](/images/train_model3.png) <br>
* 	Modify the **train_data_location** to **'s3://yourbucketname-dataset'** as below<br><br>
![train_model4.png](/images/train_model4.png) <br>
**The most important, the code in this cell is start training job and running with instance.** Make sure not ignore any previous steps before run this cell.<br>
* 	**Run this cell and you need to wait for a while**<br><br>
![train_model5.png](/images/train_model5.png) <br>
**The console output some results when it is about to finish training**<br><br>
![train_model6.png](/images/train_model6.png) <br>
**TensorFlow evaluation step output**<br><br>
![train_model7.png](/images/train_model7.png) <br>

**Important Statistic value of training process**<br><br>
![train_model8.png](/images/train_model8.png) <br> 
**In this training process:**<br><br>
**Accuracy = 0.99953127**<br>
**AUC = 0.9594897**<br>
**Average loss = 0.0026860863**<br>
 
**Summary: Those value indicate that this machine learning model perform well on our dataset**<br>


#### Deploy the trained model and get the result

* 	Move on to the cell below **“Deploy the trained Model”** heading<br>
* 	When you run this cell represent that **creates an endpoint**** which serves prediction requests in real-time with an instance<br>
* 	Set **instance_type** as **'ml.m4.xlarge'** as default and run the cell<br><br>
![deploy_model1.png](/images/deploy_model1.png) <br> 
Wait for a while until it finish deployment<br><br>
![deploy_model2.png](/images/deploy_model2.png) <br>  


#### Invoke the endpoint

* 	The cell below **“Invoke the Endpoint to get inferences”** heading do prediction job and output the prediction result<br>
* 	Modify the data in **iris_predictor.predict()**<br>
Change to this data below:<br>
 
[-15.819178720771802,8.7759971528627,-22.8046864614815,11.864868080360699,-9.09236053189517,-2.38689320657655,-16.5603681078199,0.9483485947860579,-6.31065843275059,-13.0888909176936,9.81570317447819,-14.0560611837648,0.777191846436601,-13.7610179615936,-0.353635939812489,-7.9574472262599505,-11.9629542349435,-4.7805077876172,0.652498045264831,0.992278949261366,-2.35063374523783,1.03636187430048,1.13605073696052,-1.0434137405139001,-0.10892334328197999,0.657436778462222,2.1364244708551396,-1.41194537483904,-0.3492313067728856]<br>
 <br>
**This data is selected from real fraud data that can test accuracy of our model for prediction (output contain class 0 % and class 1 % )**<br><br>
![invoke_endpoint1.png](/images/invoke_endpoint1.png) <br>  
Then run the cell to get the output<br><br>
![invoke_endpoint2.png](/images/invoke_endpoint2.png) <br>   
**The result shows the probability of this credit card data is 0.9594** (for example of this model)<br>

* 	Now don’t run the cells below **“(Optional) Delete the Endpoint”** heading. That means to delete this endpoint. You can delete it after this workshop.<br>
* 	Back to **SageMaker** console, click **Endpoints** you will find the endpoint you just created and click **Models** you will find the model you deployed<br><br>
![invoke_endpoint3.png](/images/invoke_endpoint3.png) <br>   
![invoke_endpoint4.png](/images/invoke_endpoint4.png) <br>   
**Congratulations! You successfully deploy a model and create an endpoint on SageMaker.**<br>
Next step you will learn how to create a Lambda function to invoke that endpoint<br>


## Integrate with serverless application

#### Create a Lambda function to invoke SageMaker endpoint

* 	On the **Services** menu, click **Lambda**.<br>
* 	Click **Create function**.<br>
* 	Choose **Author from scratch**.<br>
* 	Enter function Name **endpoint_invoker**.<br>
* 	Select **python 3.6** in Runtime blank.<br>
* 	Select **Choose an existing role** in **Role** blank and choose **invoke_sage_endpoint** as **Existing role**.<br><br>
![lambda_function1.png](/images/lambda_function1.png) <br>   
* 	Click **Create function** and you will see below screen.<br><br>
![lambda_function2.png](/images/lambda_function2.png) <br>  
* 	Click **endpoint_invoker** blank in **Designer** and replace original code that existing in **Function code** editor with below code. Remember to modify **ENDPOINT_NAME** to **your SageMaker endpoint name**<br><br>
*
      import boto3
      import json
      client = boto3.client('runtime.sagemaker')
      ENDPOINT_NAME = 'sagemaker-tensorflow-xxxx-xx-xx-xx-xx-xx-xxx'

      def lambda_handler(event, context):
          # TODO implement
          print(event['body'])
          print(type(event['body']))
          # list(event['body'])
          target = json.loads(event['body'])
          result = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,Body=json.dumps(target))
          response = json.loads(result['Body'].read())

          print(response)

          http_response = {
              'statusCode': 200,
              'body': json.dumps(response),
              'headers':{
                  'Content-Type':'application/json',
                  'Access-Control-Allow-Origin':'*'
              }
          }
          return http_response


<br>![lambda_function3.png](/images/lambda_function3.png) <br> 
* 	Click **Save** to save the change of function.
 



#### Create an API with API Gateway

* 	On the **Service menu**, click **API Gateway**.<br>
* 	Click **Get Started** if you are first time to this console<br>
* 	Choose **new API** of **Create new API**<br>
* 	Enter API name **predict_request** and set **Endpoint Type** as **Reginal** then click **Create API**<br>
* 	Click **Actions** and select **Create Resource**<br>
* 	Enable **Configure as proxy resource** and enable **API Gateway CORS**<br><br>
![api_gateway1.png](/images/api_gateway1.png) <br> 
* 	Click **Create Resource**<br>
* 	In proxy setup, choose **Lambda Function Proxy** for **Integration type**, **Lambda Region** select **us-east-1**, select **“endpoint_invoker”** for **Lambda Function** then click **Save**.<br><br>
![api_gateway2.png](/images/api_gateway2.png) <br>  
Click **OK** in order to add permission to Lambda function<br><br>
![api_gateway3.png](/images/api_gateway3.png) <br>  
You will see this screen<br><br>
![api_gateway4.png](/images/api_gateway4.png) <br>  
* 	Click **Method Response**<br>
* 	Click **Add Response** and enter **200** and save<br>
* 	Spread **HTTP 200 status** blank<br><br>
![api_gateway5.png](/images/api_gateway5.png) <br>  
Click **Add Header** and enter **Access-Control-Allow-Origin** then save<br>
Click **Add Response Model** and enter **application/json** for **Content type**<br>
Select **Empty** in **Model** then save<br><br>
![api_gateway6.png](/images/api_gateway6.png) <br>  
* 	Click **Actions** then select **Deploy API**<br>
* 	Choose **New Stage** for **Deployment stage**<br>
* 	Enter **dev** for Stage name and click **Deploy**<br>
* 	You will see the **Invoke URL** for this API<br><br>
![api_gateway7.png](/images/api_gateway7.png) <br>  
Now you finish API deployment and you can try the demo on application<br>



#### Demo it

* 	For this workshop, we build a **Flask** web application to call API<br>
Detail: http://flask.pocoo.org/<br>
**Make sure that you have installed python2.7** in your system<br>
https://www.python.org/downloads/<br>
You also need to setup environment for **Flask**<br>
Run **pip install Flask** in terminal<br>
![demo1.png](/images/demo1.png) <br> 
* 	The source code of application is inside the **flask-app** folder<br>
* 	Modify the file **app.py** that inside **flask-app** with your editor<br>
![demo2.png](/images/demo2.png) <br> 
In function **predict_result()** find the post URL in about line 33<br>
Modify the URL to your **invoke URL{proxy+}** that have created on API Gateway<br>
![demo3.png](/images/demo3.png) <br> 
 
In terminal go to the same directory path of **flask-app** folder<br>
e.g., Mac example below<br><br>
![demo4.png](/images/demo4.png) <br> 
Run below two commands to run Flask web application<br>
**export FLASK_DEBUG=1**    <br>
**FLASK_APP=app.py flask run**
<br><br>
![demo5.png](/images/demo5.png) <br> 
Python will run it on **localhost:5000**<br>
We prepare 10 input data that is real fraud data for prediction at the screen below (in app.py)<br><br>
![demo6.png](/images/demo6.png) <br> 
![demo7.png](/images/demo7.png) <br> 
You can copy and paste each data in input area and click **predict**<br><br>
![demo8.png](/images/demo8.png) <br> 
Then you will get the prediction result on http://localhost:5000/result<br>
**Default input data is “input_data”**<br><br>
![demo9.png](/images/demo9.png) <br> 
Change input data to input_data5<br><br>
![demo10.png](/images/demo10.png) <br> 
You will get different result<br><br>
![demo11.png](/images/demo11.png) <br> 
For non-fraud transaction data in the real world, we also prepare 5 transactions to test<br>
You can see the results of prediction<br><br>
![demo12.png](/images/demo12.png) <br> <br>
![demo13.png](/images/demo13.png) <br> <br>
![demo14.png](/images/demo14.png) <br> <br>
* Let's try those different data to explore the fraud detection result<br><br>
![demo15.png](/images/demo15.png) <br> <br>
![demo16.png](/images/demo16.png) <br> <br>
![demo17.png](/images/demo17.png) <br> <br>

## Appendix

In the real-world example, when the system detects the fraud you may want to inform your client by sending the message through mobile phone or email, so actually, you can integrate with **SNS** service in this architecture

* First you need to create a topic and subscribe that topic in SNS dashboard<br>
![sns1.png](/images/sns1.png) <br> <br>
* There are various types of target to send the message<br><br>
![sns2.png](/images/sns2.png) <br> <br>
* Back to your Lambda function and add some code as below<br><br>
* 
      import boto3
      import json
      from time import gmtime, strftime
      import time
      import datetime

      client = boto3.client('runtime.sagemaker')
      sns = boto3.client('sns')

      ENDPOINT_NAME = 'YourSageMakerEndpointName'

      def lambda_handler(event, context):
          # TODO implement
          # print(event['body'])
          # print(type(event['body']))
          # list(event['body'])
          target = json.loads(event['body'])
          result = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,Body=json.dumps(target))
          response = json.loads(result['Body'].read())

          print(response)
          fraud_rate = response['result']['classifications'][0]['classes'][1]['score']
          fraud = float(fraud_rate)*100

          if fraud>=90:
              now = datetime.datetime.now()
              tdelta = datetime.timedelta(hours=8)
              mytime = now + tdelta

              mail_response = sns.publish(
              TopicArn='YourSNSTopicArn',
              Message='Do you remember this transaction?' + '\n' + mytime.strftime("%Y-%m-%d %H:%M:%S") + '\n Please check your credit card account \n it might be a fraud transaction',
              Subject='Transaction Alert')

          http_response = {
              'statusCode': 200,
              'body': json.dumps(response),
              'headers':{
                  'Content-Type':'application/json',
                  'Access-Control-Allow-Origin':'*'
              }
          }
          return http_response 
          
 * Remember to change **ENDPOINT_NAME** of SageMaker and **TopicArn** of SNS<br>
 * In this example, SNS will push a message to my Email if the fraud rate of prediction is over 90%<br><br>
 
  ![alert1.png](/images/alert1.png) <br> <br>
 
 * You will recieve an alert message from your target of SNS topic<br>
 * You can also check the time that transaction occurred &nbsp; (Email for this example)<br><br>
  ![alert2.png](/images/alert2.png) <br> <br>
