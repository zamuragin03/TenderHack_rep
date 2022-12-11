import shutil
from fastapi import FastAPI, UploadFile, File
from schemas import Info, Result
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import Tender_model 
from pathlib import Path

model =Tender_model.Model()
way= Path(__file__).parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.post('/calculate', response_model=Result)
async def calculate(item:Info):
    
    try:
        res =model.predict_object({"id":[item.id],
                  "Наименование КС": [f'{item.session_name}'],
                  "ОКПД 2": [f'{item.OKPD}'],
                  "Регион": [f'{item.Region}'],
                  "КПГЗ": [f'{item.KPGZ}'],
                  "НМЦК": [f'{item.start_price}'],
                  "Дата": [f'{item.date}'],
                  "ИНН":[f"{item.INN}"]
                  }) 
        temp_res=Result()
        temp_res.percent=res[0]
        temp_res.participants=res[1]
        return temp_res
    except:
        return {"error":"unknown format"}
    
    

@app.post('/calculate_csv')
async def calculate_csv(file:UploadFile=File(...)):
    with open(file.filename,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)    
    try:
        model.predict_file(Path(way,file.filename))
    except:
        return {"error":"unknown format"}
    return FileResponse(Path(way,'files','Mister MISISter_2191574_TenderHack_Moscow.csv'), media_type="text/csv", filename='Mister MISISter_2191574')
    
    
if __name__== "__main__":
    # uvicorn.run("main:app",host='0.0.0.0', port=8000, reload=True)
    uvicorn.run("main:app", reload=True)
    


