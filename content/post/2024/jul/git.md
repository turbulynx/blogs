---
title: Maintaining multiple Github accounts on your local.
date: 2024-07-15
tags: ["git"]
image : "/img/posts/github-multiple-faces.jpg"
Description  : "Multiple git credentials for different accounts"
---
# Seperate folder structure
```  
  Code
  ├── eigenai
  │   ├── ...
  │   └── .gitconfig-eigenai
  ├── moversly
  │   ├── ...
  │   └── .gitconfig-moversly
  └── personal
      ├── ...
      └── .gitconfig-personal
```

# edit your - .gitconfig-personal
```
[credential]
  helper = store
[user]
  name = turbulynx
  email = turbulynx@gmail.com
[credential "https://github.com"]
  username = turbulynx
```
# git config
```bash
git config --global --edit
cat ~/.gitconfig
```
```
[includeIf "gitdir:~/Code/work/"]
    path = ~/Code/work/.gitconfig-work
[includeIf "gitdir:~/Code/personal/"]
    path = ~/Code/personal/.gitconfig-personal
```
# token
```bash 
cat ~/.git-credentials 
```
```
https://turbulynx:ghp_abcdefghijklmnopqrstuvwxyz0123456789@github.com
https://achuth:ghp_abcdefghijklmnopqrstuvwxyz0123456789@github.com
```


