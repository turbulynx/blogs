---
title: Deployment Infrastructure.
date: 2024-07-15
tags: ["hugo","Netlify"]
image : "/img/posts/k8s-eks.jpeg"
Description  : "Setting up an EKS Cluster and CI/CD"
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
                      --region=<<put your region here>> \
                      --zones=<<put your region here>>a,<<put your region here>>b \
                      --version="1.29" \
                      --without-nodegroup 
```
Create the OIDC IAM policies and associate it with the cluster
```shell
eksctl utils associate-iam-oidc-provider \
    --region <<put your region here>> \
    --cluster eigenaik8scluster \
    --approve
```
Create the node Group.
```shell
eksctl create nodegroup --cluster=eigenaik8scluster \
                        --region=<<put your region here>> \
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
aws eks --region <<put your region here>> update-kubeconfig --name eigenaik8scluster
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
  --set region=<<put your region here>> \
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

## Writing your deployment descriptors:
# TODO Needs addtn. working on.
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
![Route53](/blogs/img/post/route-53-dev.png)


---

# Export your Account ID
export ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

# Set Trust Policy
TRUST="{ \"Version\": \"2012-10-17\", \"Statement\": [ { \"Effect\": \"Allow\", \"Principal\": { \"AWS\": \"arn:aws:iam::${ACCOUNT_ID}:root\" }, \"Action\": \"sts:AssumeRole\" } ] }"

# Verify inside Trust policy, your account id got replacd
echo $TRUST

# Create IAM Role for CodeBuild to Interact with EKS
aws iam create-role --role-name EksCodeBuildKubectlRole --assume-role-policy-document "$TRUST" --output text --query 'Role.Arn'

# Define Inline Policy with eks Describe permission in a file iam-eks-describe-policy
echo '{ "Version": "2012-10-17", "Statement": [ { "Effect": "Allow", "Action": "eks:Describe*", "Resource": "*" } ] }' > /tmp/iam-eks-describe-policy

# Associate Inline Policy to our newly created IAM Role
aws iam put-role-policy --role-name EksCodeBuildKubectlRole --policy-name eks-describe --policy-document file:///tmp/iam-eks-describe-policy

# Verify what is present in aws-auth configmap before change
kubectl get configmap aws-auth -o yaml -n kube-system

# Set ROLE value
ROLE="    - rolearn: arn:aws:iam::<<put your account-id here>>:role/EksCodeBuildKubectlRole\n      username: build\n      groups:\n        - system:masters"

# Get current aws-auth configMap data and attach new role info to it
kubectl get -n kube-system configmap/aws-auth -o yaml | awk "/mapRoles: \|/{print;print \"$ROLE\";next}1" > /tmp/aws-auth-patch.yml

# Patch the aws-auth configmap with new role
kubectl patch configmap/aws-auth -n kube-system --patch "$(cat /tmp/aws-auth-patch.yml)"

# Verify what is updated in aws-auth configmap after change
kubectl get configmap aws-auth -o yaml -n kube-system
----
1. Create your code in github.com
2. 

## have the following folder structure
```
.
├── app.py
├── buildspec.yml
├── deployment-manifest
│   ├── charts
│   ├── Chart.yaml
│   ├── templates
│   │   └── service.yaml
│   └── values.yaml
├── Dockerfile
└── requirements.txt
```

buildspec.yml
```yml
version: 0.2
phases:
  install:
      commands:
        - echo "Install Helm"
        - curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
        - helm repo add stable https://charts.helm.sh/stable
        - helm repo update
        - helm version
        - echo "Verify AWS CLI Version..."
        - aws --version
  pre_build:
      commands:
        - NAME=<<put your project name here>>
        - ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
        - REGION=<<put your region here>>
        - REPOSITORY_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$NAME"
        - TAG="$(date +%Y-%m-%d.%H.%M.%S).$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | head -c 8)"
        - echo "Login in to Amazon ECR..."
        - aws ecr get-login-password | docker login --username AWS --password-stdin $REPOSITORY_URI
        - export KUBECONFIG=$HOME/.kube/config
  build:
    commands:
      - echo "Build started on `date`"
      - echo "Building the Docker image..."
      - docker build --tag $REPOSITORY_URI:$TAG .
      - echo "Pushing the Docker image to ECR Repository"
      - docker push $REPOSITORY_URI:$TAG
      - echo "Docker Image Push to ECR Completed -  $REPOSITORY_URI:$TAG"  
      - echo "Build completed on `date`"
  post_build:
    commands:
      - echo "Setting Environment Variables related to AWS CLI for Kube Config Setup"    
      - CREDENTIALS=$(aws sts assume-role --role-arn $EKS_KUBECTL_ROLE_ARN --role-session-name codebuild-kubectl --duration-seconds 900)
      - export AWS_ACCESS_KEY_ID="$(echo ${CREDENTIALS} | jq -r '.Credentials.AccessKeyId')"
      - export AWS_SECRET_ACCESS_KEY="$(echo ${CREDENTIALS} | jq -r '.Credentials.SecretAccessKey')"
      - export AWS_SESSION_TOKEN="$(echo ${CREDENTIALS} | jq -r '.Credentials.SessionToken')"
      - export AWS_EXPIRATION=$(echo ${CREDENTIALS} | jq -r '.Credentials.Expiration')
      - echo "Update Kube Config"      
      - aws eks update-kubeconfig --name $EKS_CLUSTER_NAME
      - echo "Apply changes to deployment-manifest using Helm"            
      - helm upgrade --install $NAME ./deployment-manifest --set image=$REPOSITORY_URI --set tag=$TAG
      - echo "Helm deployment completed"
artifacts:
  files: 
    - deployment-manifest/*
```

templates/service.yaml
```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name:  {{ .Release.Name }}
  labels:
    app: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: {{ .Release.Name }}
          image: {{ .Values.image }}:{{ default "latest" .Values.tag }}
          ports:
            - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-service
  labels:
    app: {{ .Release.Name }}
  annotations:
    alb.ingress.kubernetes.io/healthcheck-path: /health
spec:
  type: NodePort
  selector:
    app: {{ .Release.Name }}
  ports:
    - port: 80
      targetPort: 5000
```

values.yaml
```yaml
replicaCount: 1
```

## Code Pipeline
Create a Code Pipline with new service role.
Go to Code pipeline and in the process create a new project and create a code-build.
![](/blogs/img/post/code-pipeline.png)
These have to be set in the env variables of code-pipeline and code-build both.
```
REPOSITORY_URI=<<put your account-id here>>.dkr.ecr.<<put your region here>>.amazonaws.com/<<put your project name here>>
EKS_KUBECTL_ROLE_ARN=arn:aws:iam::<<put your account-id here>>:role/EksCodeBuildKubectlRole
EKS_CLUSTER_NAME=eigenaik8scluster
```
