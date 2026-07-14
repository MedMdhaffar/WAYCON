```
cd WAYCON/WAYCON
export PERSON_CREATION_PROFILE=1
export PERSON_CREATION_PROFILE_CUDA_SYNC=1
export PERSON_CREATION_PROFILE_SQL=1
python -m forensics.person_creation.service
```

```
forensics/person_creation/videos/4.mp4
```


```
cd WAYCON/WAYCON
python -m forensics.face_engine.app
```

```
cd ~/WAYCON/WAYCON/forensics/person_creation/frontend
chmod +x node_modules/.bin/vite

npm install
npm run dev
```

```
export PERSON_CREATION_PROFILE=1
export PERSON_CREATION_PROFILE_CUDA_SYNC=1
export PERSON_CREATION_PROFILE_SQL=1
PERSON_CREATION_GPU_PIPELINE=1 python -m forensics.person_creation.service
```