---
title: Deployment Infrastructure.
date: 2024-07-15
tags: ["hugo","Netlify"]
image : "/img/posts/cloud-swirl.jpg"
Description  : "Assessing Deployment Infrastructure Options to Optimize Expenditure and Empower Innovation in a Startup Environment."
---
# Objective
One of the principal ideas of our startup is optimizing operational expenditure through frugality, enabling us to prioritize funds for innovation and ensuring our customers achieve their objectives. We must maintain scalability while upholding standards such as security, reliability, availability, and Performance SLAs. Simultaneously, empowering developers to focus on customer use-cases instead of infrastructure management is paramount. Therefore, we are currently assessing and comparing various deployment infrastructure options to make informed decisions for imminent and future needs.

# Yardsticks to compare the various options
1. Deployment simplicity
2. Scalability potential
3. Cost effectiveness
4. Cloud neutrality
5. Others such as Performance metrics, Security feature & Compliance.
6. Flexibility & customization options.

# As such we have 3 options:
---
## Using AWS ECS
### EC2 
### Fargate
---
## using Kubernetes 
### EC2 
![Architecture with EC2 K8s](/blogs/img/posts/k8s-ec2.jpg)
### EKS
Master node is managed by AWS
Pros:
- Security and best practices taken care by AWS.
- highly available master node. atleast:
    - 2 API Servers & 3 etcd nodes across 3 AZ within region.
    - auto detect unhealthy control plane and restarts them

Cons:
- $75 just for keeping Master node running.

There are 3 approaches to EKS
1. Self Managed Nodes
    Cons:
    - team is responsible for provisioning and maintaining the instances, registering with control plane.
    - have to install (but can be automated):
        - Kubelet
        - Kubeproxy
        - Container Runtime.
    - Security and Updates is our responsibility.
2. Managed Node Groups:
    - EKS optimised images
    - provisioing, lifecycle mgmt and registration is automated with minimal interaction.
    - backed by autoscaling group.
3. Fargate
    - serverless.
    - scale out on its own based on container requirements.
    - pay per use.

### installing 
https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html


https://eksctl.io/installation/


---
## using Kubernetes on GCP
Given that we have some credits in GPC this is a great option.

## Using K8s with EC2


### Using K8s with EKS

# Reference
## Teams K8s Learning path
* [K8s Tutorials for Beginners](https://www.youtube.com/watch?v=X48VuDVv0do)
* [Label & Selectors](https://www.youtube.com/watch?v=y_vy9NVeCzo)
* [The Kubernetes Book](https://www.amazon.com.au/Kubernetes-Book-Nigel-Poulton/dp/1916585000/ref=asc_df_1916585000/?tag=googleshopdsk-22&linkCode=df0&hvadid=650005042738&hvpos=&hvnetw=g&hvrand=2574643024603298371&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9071729&hvtargid=pla-2186628604682&psc=1&mcid=75741d3b554b30568cddcd46d7440922)
* [Setting of Ingress to route web traffic](https://www.youtube.com/watch?v=H9RCxniXT_k)
## Setting up a K8s cluster on EC2 instance.
follow: https://mrmaheshrajput.medium.com/deploy-kubernetes-cluster-on-aws-ec2-instances-f3eeca9e95f1

## FAQ
1. **E: The repository ‘https://apt.kubernetes.io kubernetes-xenial Release’ does not have a Release file**

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

2. Accessing kubectl from local.

