## to install the dependencies :
```
pip install uv
uv pip install -r requirements.txt
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

## to open the web interface
```
cd /frontend
npm install
npm run dev
```
```
cd /WAYCON
python -m forensics.person_creation.service
```




