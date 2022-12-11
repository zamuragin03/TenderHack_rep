from pydantic import BaseModel

class Info(BaseModel):
    id:int=None
    session_name:str =None
    OKPD:str =None
    KPGZ:str=None
    Region:str=None
    start_price :float=None
    date: str=None
    INN: str=None
    
class Result(BaseModel):
    percent:float=None
    participants:int=None
    
