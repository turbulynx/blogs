## First time
### Install hugo
```bash
sudo apt install hugo
```

### setup theme
```bash
ls -la themes/lightbi-hugo
git submodule deinit themes/lightbi-hugo
git rm themes/lightbi-hugo
rm -rf .git/modules/themes/lightbi-hugo
git submodule add https://github.com/binokochumolvarghese/lightbi-hugo themes/lightbi-hugo
```

### run
```
hugo server
```