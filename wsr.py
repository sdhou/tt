import whisper

model = whisper.load_model("base")
options_dict = {"language": f'zh'}
result = model.transcribe("/Users/sdhou/Downloads/1.m4a", **options_dict)
print(result)
