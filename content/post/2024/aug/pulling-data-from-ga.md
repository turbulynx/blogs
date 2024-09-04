---
title: Google Ads Integration with AppFlow
date: 2024-07-15
tags: ["app-flow","google ads"]
image : "/img/posts/apache-superset.jpg"
Description  : "Google Ads pulling Data into S3."
---
# Installation
## Accept invitation
![](/blogs/img/posts/accept-invitation-google-ads.png)

## Enable Google ads API
![](/blogs/img/posts/enable-google-ads-api.png)
https://console.cloud.google.com/apis/api/googleads.googleapis.com/metrics?project=eigenai

# Google Ads Developer token
* [Sign into manager account and create API Center Manager Access](https://ads.google.com/home/tools/manager-accounts/)
![](/blogs/img/posts/api-center-manager-access.png)
* Create an API token
![](/blogs/img/posts/api_token.png)

* Apply for Basic Token [read](https://developers.google.com/google-ads/api/docs/access-levels#applying_for_basic_access)
* Fill the [form here](https://support.google.com/adspolicy/contact/new_token_application)

# API & Services
* (https://console.cloud.google.com/welcome?pli=1&project=yummy-kqaa)
![](/blogs/img/posts/api-and-services.png)

# Create Credentials
* [Setting up Oauth2.0](https://support.google.com/cloud/answer/6158849?hl=en#zippy=)
* create Credentials
![](/blogs/img/posts/settingup-oath2.png)
![](/blogs/img/posts/create-credentials.png)
![](/blogs/img/posts/credentials-created.png)

# Setting up OAuth 2.0
![](/blogs/img/posts/oauth-consent-screen.png)
![](/blogs/img/posts/google-ads-api-scope.png)
# APP Flow
* Connect to Google Ads
![](/blogs/img/posts/connect-to-google-ads.png)
* You might get an error
![](/blogs/img/posts/redirect-url-google-error-upon-login.png)
https://ap-southeast-2.console.aws.amazon.com/appflow/oauthflowName=GeneralOAuthFlow

if so then update the credentials to accept the above redirect_uri
* Create a Flow 
![](/blogs/img/posts/app-flow-connected.png)

* You might get the error when you try to select an API Object
![](/blogs/img/posts/error_without_developer_token_basic_access.png)


# References
https://docs.aws.amazon.com/appflow/latest/userguide/connectors-google-ads.html