---
title: Deployment Infrastructure.
date: 2024-07-15
tags: ["hugo","Netlify"]
image : "/img/posts/deployment_infrastructure_swirling_cloud.jpg"
Description  : "Deployment Infrastructure"
---
## Objective
As a startup, optimizing operational expenditure is crucial to prioritize funds for innovation, ensuring our customers achieve their objectives. We must maintain scalability while upholding our standards such as security, reliability, availability and Performance SLAs. Simultaneously, empowering developers to focus on customer use-cases instead of infrastructure management is paramount. Therefore, herewe are currently assessing and comparing various deployment infrastructure options to make informed decisions for both current and future needs.

## Yardsticks to compare the various options
1. Deployment simplicity
2. Scalability potential
3. Cost effectiveness
4. Cloud neutrality
5. Others such as Performance metrics, Security feature & Compliance.
6. Flexibility & customization options.

## As such there are 3 options
1. Using ECS
    - EC2 
    - Fargate
2. using Kubernetes 
    - EC2 
    - EKS
3. GCP

## Using ECS- EC2 
Given that we have some credits in GPC this is a great option.

## Using ECS- Fargate

## Using K8s with EC2

## Using K8s with EKS


## Reference
### Setting up a K8s cluster on EC2 instance.
https://phoenixnap.com/kb/install-kubernetes-on-ubuntu
Note:
following the tutorials will give an error - **E: The repository ‘https://apt.kubernetes.io kubernetes-xenial Release’ does not have a Release file**

```shell
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt update
```
hence use the following instead:
```shell
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

sudo apt update
```
see: [discuss.kubernetes.io](https://discuss.kubernetes.io/t/e-the-repository-https-apt-kubernetes-io-kubernetes-xenial-release-does-not-have-a-release-file/28121
)
