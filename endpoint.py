import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://e732de18-6017-48a8-805c-d4da4a27c398.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'kKi8HnSbksKHhMzKMRMKJa365q5WGU5U'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
     		"age": 17,
      		"job": "blue_collar",
      		"marital": "married",
		"education": "university.degree",
		"default": "no",
 	        "housing": "yes",
	        "loan": "yes",
	        "contact": "cellular",
	        "month": "may",
	        "day_of_week": "mon",
	        "duration": 971,
  	        "campaign": 1,
	        "pdays": 999,
	        "previous": 1,
	        "poutcome": "failure",
	        "emp.var.rate": -1.8,
	        "cons.price.idx": 92.893,
	        "cons.conf.idx": -46.2,
	        "euribor3m": 1.299,
	        "nr.employed": 5099.1
    },
          {
      		"age": 87,
	        "job": "blue_collar",
	        "marital": "married",
	        "education": "university.degree",
	        "default": "no",
	        "housing": "yes",
	        "loan": "yes",
	        "contact": "cellular",
	        "month": "may",
	        "day_of_week": "mon",
	        "duration": 471,
	        "campaign": 1,
	        "pdays": 999,
	        "previous": 1,
	        "poutcome": "failure",
	        "emp.var.rate": -1.8,
	        "cons.price.idx": 92.893,
	        "cons.conf.idx": -46.2,
	        "euribor3m": 1.299,
	        "nr.employed": 5099.1
          },
      ],
	"method": "predict"
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, str.encode(input_data), headers=headers)
print(resp.json())


