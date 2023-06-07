from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
import whisper
import time
import shutil
from fastapi.responses import JSONResponse
app = FastAPI()


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(audio_file: UploadFile = File(...)):
    file = '/opt/code/tt/tmp/'+str(time.time())
    with open(file, 'wb+') as file_object:
        shutil.copyfileobj(audio_file.file, file_object)
    model = whisper.load_model("base")
    options_dict = {"language": f'zh'}
    result = model.transcribe(file, **options_dict)
    return JSONResponse(result)
