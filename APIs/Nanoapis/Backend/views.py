from django.shortcuts import render
from rest_framework.views import APIView
from django.http import JsonResponse
from .ai import AutoencoderConfig
import json

# Create your views here.


class pred_score(APIView):
    
    def post(self,request):
        
        if request.method == 'POST':
            #print('>>>>>>>>>>>>>>>>>>>>> : ',request)
            data = json.load(request)
            #print('Data : ',data)
            try:
                label = data['label']
            except:
                label = 'text'
            text = [str(data[label])]
            #print('Text is : ',text)
            obj = AutoencoderConfig()
            #print('Upp')
            resp = obj.predict(text)
            #print('Final Ans : ',resp)
            p = {'score':str(resp)}
            
        return JsonResponse(p)
            
            