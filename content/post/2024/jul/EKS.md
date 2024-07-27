---
title: Deployment Infrastructure.
date: 2024-07-15
tags: ["hugo","Netlify"]
image : "/img/posts/k8s-eks.jpeg"
Description  : "EKS"
---
# Objective

# Installation
## CLI
### awscli
### kubectl
https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html
### eksctl
https://eksctl.io/installation/
### EKS
Master node is managed by AWS


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
## EKS Core objects
- EKS Cluster parts:
    - EKS Control plan 
        - etcd, kubeapiserver, kube-controller (aws managed)
        - Security and best practices taken care by AWS.
        - highly available master node. atleast:
        - 2 API Servers & 3 etcd nodes across 3 AZ within region.
        - auto detect unhealthy control plane and restarts them
    - Worker Nodes & Node Groups (group of EC2 instances for workloads)
        - connect to contorl plane via cluster API server endpoint 
        - node group = 1 or more EC2 instances in a EC2 Autoscaling group
        - all instance must be same instance type.
        - same API
    - Fargate Profiles (workloads on serverless fargate)
        - on demand, right sized compute capacity.
        - TODO: see other adv. of fargate..
    - VPC (for secure n/w standards)
        - restrict traffic between control plane to within a single cluster.

## Getting started.
Create the Cluster
```shell 
eksctl create cluster --name=eigenaik8scluster \
                      --region=ap-southeast-2 \
                      --zones=ap-southeast-2a,ap-southeast-2b \
                      --version="1.29" \
                      --without-nodegroup 
```
Create the OIDC IAM policies and associate it with the cluster
```shell
eksctl utils associate-iam-oidc-provider \
    --region ap-southeast-2 \
    --cluster eigenaik8scluster \
    --approve
```
Create the node Group.
```shell
eksctl create nodegroup --cluster=eigenaik8scluster \
                        --region=ap-southeast-2 \
                        --name=eigenaik8scluster-ng-private1 \
                        --node-type=t3.medium \
                        --nodes-min=2 \
                        --nodes-max=4 \
                        --node-volume-size=20 \
                        --ssh-access \
                        --ssh-public-key=k8s-key \
                        --managed \
                        --asg-access \
                        --external-dns-access \
                        --full-ecr-access \
                        --appmesh-access \
                        --alb-ingress-access \
                        --node-private-networking 
```
Verify the resources
```shell
eksctl get cluster
eksctl get nodegroup --cluster=eigenaik8scluster
```

Ensure no service account with cluster
```shell
eksctl get iamserviceaccount --cluster=eigenaik8scluster
```

Configure kubeconfig for kubectl
```shell
aws eks --region ap-southeast-2 update-kubeconfig --name eigenaik8scluster
kubectl get nodes
```
Configure IAM Policy
```shell
curl -o iam_policy_latest.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json
```
Create Policy
```shell
aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy \
    --policy-document file://iam_policy_latest.json
```
Take a note of the Policy ARN or go to aws management console and grab the policy ARN.
Verify no Service Account Role Exists.
```shell
kubectl get sa -n kube-system
kubectl get sa aws-load-balancer-controller -n kube-system
```
Create a Service Account. AWS will create an STS token with AWS via the service Account.
```shell
eksctl create iamserviceaccount \
  --cluster=eigenaik8scluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn=<Replace with IAM Role> \
  --override-existing-serviceaccounts \
  --approve
```
Verify Service Account
```shell
eksctl  get iamserviceaccount --cluster eigenaik8scluster
kubectl get sa -n kube-system
kubectl get sa aws-load-balancer-controller -n kube-system
kubectl describe sa aws-load-balancer-controller -n kube-system
```
Install AWS Load Balancer Controller using [HelmV3](https://helm.sh/docs/intro/install/)
```shell
helm version
helm repo add eks https://aws.github.io/eks-charts
helm repo update
```
get image repository latest from [here](https://docs.aws.amazon.com/eks/latest/userguide/add-ons-images.html)
```shell
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=eigenaik8scluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller \
  --set region=ap-southeast-2 \
  --set vpcId=<replace this with your cluster associated vpc id> \
  --set image.repository=<choose your repo based on region from https://docs.aws.amazon.com/eks/latest/userguide/add-ons-images.html>
```
Full Verification
```shell
kubectl -n kube-system get deployment 
kubectl -n kube-system get deployment aws-load-balancer-controller
kubectl -n kube-system describe deployment aws-load-balancer-controller
kubectl get deployment -n kube-system aws-load-balancer-controller
kubectl -n kube-system get svc aws-load-balancer-webhook-service
kubectl -n kube-system get svc aws-load-balancer-webhook-service -o yaml
kubectl -n kube-system get deployment aws-load-balancer-controller -o yaml
kubectl get pods -n kube-system
kubectl -n kube-system logs -f  <aws load balancer controller pod>
```
[ingress-class.yaml](https://github.com/turbulynx/eks-deployment-poc/blob/main/ingress-class/ingress-class.yaml)
[ingress.yaml](https://github.com/turbulynx/eks-deployment-poc/blob/main/resources/ingress.yaml)
[nginx.yaml](https://github.com/turbulynx/eks-deployment-poc/blob/main/resources/nginx-service.yaml)
[moversly-service.yaml](https://github.com/turbulynx/eks-deployment-poc/blob/main/resources/moversly-service.yaml)

Create Ingress Class
```shell
kubectl apply -f ingress-class
```
Create the rest of the resources
```shell
kubectl apply -f resources/
```
Verify if the Application Load balancer was created. Associate the Route 53 subdomain with Load Balancer. in our case dev.eigenai.co
![Route53](/blogsimg/post/route-53-dev.png)