# butterknife

Identify suicidal or not using stream of text.

## Features

- [x] Basic identification
  - [x] Emotion Layer
  - [x] Context Layer
- [ ] Improvement Points
  - [x] Add test dataset
  - [x] Epoch support
  - [x] Automatic Loss Stop (save best model)
  - [x] Optimizer
  - [ ] make mild suicidal to 50% suicidal
  - [ ] seperate message of user that changed (non-suicidal <-> suicidal)
  - [ ] seperate multi-modal version
  - [ ] More datasets
    - [x] X
    - [ ] Bluesky
    - [ ] Mastodon
    - [ ] Instagram
- [ ] Multi Lang support
  - [ ] English
  - [ ] Japanese
  - [ ] Chinese

## Models

- [v0 (Embedding-Based)](https://minio.misile.xyz/noa/models/cabinet_v0.pth.zst)
  - [v0.1 (28% error)](https://minio.misile.xyz/noa/models/cabinet_v0.1.pth.zst)
- v1 (Electra-Based)
  - [v1 (26% error)](https://minio.misile.xyz/noa/models/cabinet_v1.pth.zst)
