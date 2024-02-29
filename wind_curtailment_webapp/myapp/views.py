from django.shortcuts import render
import requests
import pandas as pd

def index(request):

    ### GET THE DATA VIA API ####
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

    ### CLEAN DATA
    df = pd.DataFrame(data)
    df_operations = pd.DataFrame(df['operations'].tolist())
    columns_to_keep = ['start', 'end', 'duration', 'controlStage', 'location', 'reason', 'assetKey']
    column_mapping = {
        'start': 'Start', 
        'end': 'End', 
        'duration': 'Duration', 
        'controlStage': 'Control Stage', 
        'location': 'Location', 
        'reason': 'Reason', 
        'assetKey': 'Asset Key'
    }
    df_operations = df_operations[columns_to_keep].rename(columns=column_mapping)

    translation_dict = {
    'Netzengpass': 'Grid Congestion',
    'Funktionsnachweis': 'Functional Proof',
    'Kundenfunktionstest': 'Customer Function Test',
    'Test': 'Test',
    'Direktvermarkter': 'Direct Marketer',
    'Sonstige': 'Other',
    'Vorgelagerter Netzbetreiber': 'Upstream Grid Operator'
    }

    df_operations['Reason'] = df_operations['Reason'].map(translation_dict).fillna(df_operations['Reason'])

    ### CREATE HTML TABLE AND RENDER 
    html_table = df_operations.to_html(classes='table table-striped', index=False)
    context = {'html_table': html_table}
    return render(request, 'index.html', context)

