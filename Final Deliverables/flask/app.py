# Import Libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
from sklearn.preprocessing import LabelEncoder
import requests
## Integrate with IBM cloud API
# API_KEY = "8MQKoaCV7PXi-bUaQppxftdf1gASUWX9QOSHs-dBSWkJ"


def APIConnect(apikey):
	API_KEY = apikey
	token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
	mltoken = token_response.json()["access_token"]
	header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
	
	return API_KEY, token_response, mltoken, header

API_KEY, token_response, mltoken, header = APIConnect("8MQKoaCV7PXi-bUaQppxftdf1gASUWX9QOSHs-dBSWkJ")



app = Flask(__name__)#initiate flask app

## for locally saved model
def load_model(file='../Model Building/results/resale_model.sav'):#load the saved model
	return pickle.load(open(file, 'rb'))

@app.route('/')
def index():#main page
	return render_template('car.html')

@app.route('/predict_page', methods=['GET','POST'])
def predict_page():#predicting page
	return render_template('value.html')

def getInputs(request):
	reg_year = int(request.args.get('regyear'))
	powerps = float(request.args.get('powerps'))
	kms= float(request.args.get('kms'))
	reg_month = int(request.args.get('regmonth'))
	gearbox = request.args.get('geartype')
	damage = request.args.get('damage')
	model = request.args.get('model')
	brand = request.args.get('brand')
	fuel_type = request.args.get('fuelType')
	veh_type = request.args.get('vehicletype')

	return reg_year, reg_month, powerps, kms, gearbox, damage, model, brand, fuel_type, veh_type

@app.route('/predict', methods=['GET','POST'])
def predict():
	# get input data from form
	reg_year, reg_month, powerps, kms, \
		gearbox, damage, model, brand,\
		 fuel_type, veh_type = getInputs(request=request)

	#create a dictionary
	new_row = {'yearOfReg':reg_year, 'powerPS':powerps, 'kilometer':kms,
				'monthOfRegistration':reg_month, 'gearbox':gearbox,
				'notRepairedDamage':damage,
				'model':model, 'brand':brand, 'fuelType':fuel_type,
				'vehicletype':veh_type}

	print(new_row)


	# perform Label encoding on the input data
	# so we can give it to model
	new_df = pd.DataFrame(columns=['vehicletype','yearOfReg','gearbox',
		'powerPS','model','kilometer','monthOfRegistration','fuelType',
		'brand','notRepairedDamage'])
	new_df = new_df.append(new_row, ignore_index=True)
	labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicletype']
	mapper = {}
	for i in labels:
		mapper[i] = LabelEncoder()
		mapper[i].classes = np.load('../Model Building/results/'+str('classes'+i+'.npy'), allow_pickle=True)
		transform = mapper[i].fit_transform(new_df[i])
		new_df.loc[:,i+'_labels'] = pd.Series(transform, index=new_df.index)

	labeled = new_df[['yearOfReg','powerPS','kilometer','monthOfRegistration'] + [x+'_labels' for x in labels]]

	X = labeled.values.tolist()
	print('\n\n', X)

	#### locally saved model
	# predict = reg_model.predict(X)
	# predict[0]*=84.82
	# predict=round(float(predict[0]),2)


	month_dict = {	1:'Janauary',
		2:'February',
		3:'March',
		4:'April',
		5:'May',
		6:'June',
		7:'July',
		8:'August',
		9:'September',
		10:'October',
		11:'November',
		12:'December'		}
	reg_month=month_dict[reg_month]


	# manually define and pass the array(s) of values to be scored in the next line
	payload_scoring = {"input_data": [{"fields": [['yearOfReg', 'powerPS', 'kilometer', 'monthOfRegistration','gearbox_labels', 'notRepairedDamage_labels', 'model_labels','brand_labels', 'fuelType_labels', 'vehicletype_labels']], "values": X}]}

	# send inout data to model API end point
	response_scoring = requests.post('https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/660695dd-d0d8-419f-8850-76ccf4172103/predictions?version=2022-11-18', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
	#retrieve response
	predictions = response_scoring.json()
	print(response_scoring.json())
	predict = predictions['predictions'][0]['values'][0][0]
	#convert to INR
	predict*=84.82
	predict=round(float(predict),2)

	# send relevant variables to predit.html
	return render_template('predict.html',**locals())

if __name__=='__main__':
	#reg_model = load_model()#load the saved model
	app.run(host='localhost', debug=True, threaded=False)
