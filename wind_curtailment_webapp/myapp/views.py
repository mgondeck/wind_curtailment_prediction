from django.shortcuts import render
import requests

def index(request):

    base_url = "https://redispatch-run.azurewebsites.net/api/operations/get"
    params = {
        "networkoperator": "shn",          # change as needed (ava (Avacon), bag (Bayernwerk), edi (Edis))
        "type": "finished",
        "orderDirection": "desc",
        "orderBy": "start",
        "chunkNr": 1
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    context = {'api_data': data}
    return render(request, 'myapp/index.html', context)

