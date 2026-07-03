## to install the dependencies :
```
pip install uv
uv pip install requirements.txt
```


## to run this file :
```
cd /WAYCON
python -m forensics.person_creation.run \
  --name mohamed \
  --videos "/forensics/person_creation/videos/mohamed.mp4" \
  --output "/forensics/person_creation/person_db" \
  --every 5
```