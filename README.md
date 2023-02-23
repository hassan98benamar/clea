# Predictive-Maintenance-CbV
This project contains predictive maintenance codes which predicts defects of various controls on hourly and daily basis.

Frequency of Defects
This predicts the frequency of defects that might occur calculated based on the workorder type of the data.

# Prerequisites :
Make Sure You have Python  installed in your System & Open CMD.  
Download "df_conformity.pkl" and "df_conformity_global.pkl" on this link then put in the folder data: 
https://capgemini.sharepoint.com/sites/PSAFactory/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=QpDkW8&OR=Teams%2DHL&CT=1658219785763&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIyNy8yMjA2MDYxNDgwNSIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D&cid=4ba51f37%2Dd813%2D4194%2D8661%2D01bcf9267ecc&FolderCTID=0x012000085597097FC3DA4B9E8B6EEDEC74D220&id=%2Fsites%2FPSAFactory%2FShared%20Documents%2FGeneral%2FCOMMUNICATION%2FOnboarding%2FAI%2DMLOPS&viewid=0e1165b4%2Ddc3d%2D4297%2D87a2%2D4deb0063c903
 
# Steps :

 1. First make sure to update pip:  
     $ sudo pip install --upgrade pip

 2. If you probably want to work in a virtualenv (optional):  
     $ sudo pip install --upgrade virtualenv  
 Or if you prefer you can install virtualenv using your favorite packaging system. E.g., in Ubuntu:  
     $ sudo apt-get update && sudo apt-get install virtualenv  
 Then:  
     $ cd $my_work_dir  
     $ virtualenv my_env  
     $ . my_env/bin/activate

 Make sure to install the dependencies first.  

 3. Install these requirements in PredMntec_CbV_AI:
     $ pip install --upgrade -r requirements.txt
	 
	Make sure to create/have a folder named saved_model in /data before next step  
	Make sure to run Train APIs first before predict

 4. Navigate to PredMntec_CbV_AI Folder :    
     $ cd $my_work_dir of PredMntec_CbV_AI  
     $ python setup.py install

 5. Return to the root :  
     $ python -m PredMntec_CbV_Restapp.launch

 6. You will get the URL & now navigate to URL to choose necessary action
 
## a. Predict

	1. Prediction By Control Name on Daily Basis :
		- Parameters : 
			- ctrl_name
			- period

	2. Prediction for All Controls on Hourly Basis :
		- Parameters : 
			- period		

	3. Prediction on Global Data Daily Basis :
		- Parameters : 
			- dataset_name
			- period
	
	4. Prediction on Global Data Hourly Basis :
		- Parameters : 
			- dataset_name
			- period

	5. Prediction By Control Name on Hourly Basis :
		- Parameters : 
			- ctrl_name
			- period

	6. Prediction for All Controls on Daily Basis :
		- Parameters : 
			- period
	7. Prediction Count of all results by Control Name
## b. Train
	Similar to predict. Call the train APIs.


## Cahier Expérimentation : 

Make sure to create/have a folder named saved_model in /data  
Make sure to run Train APIs first before predict

### APIs Predict with pkl files
All APIs Predict work  
- /predict/{ctrl_name}/{period} : predict the crtl name trained by specify the period  
- /predict_hourly/{ctrl_name}/{period} : predict the crtl name trained by specify the period in hourly  
But don't know the difference between :
	- /predict_global_daily/{usecase}/{period} and /predictall/{period}   
	- /predict_global_hourly/{usecase}/{period} and /predict_all_hourly/{period}

### APIs Train with pkl files
All APIs Train work  
- /train/ : train all ctrl name in data pkl files  
- /train/{ctrl_name} : train a specific ctrl name  
- /train_ctrl_hourly/{ctrl_name} : train a specific ctrl name hourly  
- /train_delete_model/{modelname} : delete a modelname already train  
- /train_get_all_model : get information about all ctrl name trained
- /train_get_specific_model/{modelname} : get information about a specific ctrl name trained  
But don't know the difference meaning between :
	- /train/ and /train_globally_daily/{usecase}
	- /train_ctrl_all_hourly and /train_globally_hourly/{usecase}

### API image
Didn't test this API

### API records
Didn't test this API

### Connexion with MongoDB
Download MongoDB and MongoDB Compass
Have to import the pkl file converted to csv file in MongoDB Compass
Don't know if the application is connected to MongoDB because no logs

For more information view the KT videos about Predictive-Maintenance-CbV : 
https://capgemini.sharepoint.com/sites/PSAFactory/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=QpDkW8&OR=Teams%2DHL&CT=1658219785763&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIyNy8yMjA2MDYxNDgwNSIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D&cid=4ba51f37%2Dd813%2D4194%2D8661%2D01bcf9267ecc&FolderCTID=0x012000085597097FC3DA4B9E8B6EEDEC74D220&id=%2Fsites%2FPSAFactory%2FShared%20Documents%2FGeneral%2FCOMMUNICATION%2FOnboarding%2FAI%2DMLOPS&viewid=0e1165b4%2Ddc3d%2D4297%2D87a2%2D4deb0063c903


	


