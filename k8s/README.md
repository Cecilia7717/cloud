Three files in this directory:

1) interactive.yaml 

You can launch a pod in interactive mode. It also supports JupyterNotebook if you prefer to use it

Notes: 
1-1) if you want to create multiple pods like this in one namespace, make sure they have unique names. 
1-2) current yaml file has settings for "volumeMounts" if you do not have the external storage setup remove that part or you have to create the storage before launching the pod.


2) storage.yaml

This file creates a storage that can be mounted on your pods. It will not be deleted if a pod gets deleted. You can use it to keep your code and results. You need to run it only once, i.e., `kubectl create -f storage.yaml`


3) job.yaml

This file is for running a job in a non-interactive mode.

If your job pulls from a private Git repo, you should create a Gitlab Personal Access Token (instructions here: https://ucsd-prp.gitlab.io/userdocs/running/jobs/) of type `read_repository` and put it in your namespace secret:

```
kubectl create secret generic gitlab-secret --from-literal=user=USERNAME --from-literal=password=TOKEN
```

Notes: 
3-1) if you increase the resources, it may take a while for the server to allocate that resource. So adjust them wisely...
3-2) current file is setup to mount external storage
3-3) in "args" you can place commands that you want to run. cloning a repo, installing a package, and running the code. 

UCSD quick start: https://ucsd-prp.gitlab.io/userdocs/start/quickstart/

In the following examples, I have setup my default name space. You can pass the namespace name with this flag: --namespace=<insert-namespace-name-here>

************************************
example use for interactive.yaml:

Create a Pod: kubectl create -f interactive.yaml
Checking pod status: kubectl get pods
Connecting to the pod: kubectl exec -it interactive bash
Run Jupyter: jupyter notebook --ip='0.0.0.0'
Port forwarding (in a new terminal): kubectl port-forward interactive 8888:8888
Delete the Pod: kubectl delete pod interactive

**********************************
example use for job.yaml:

Get jobs: kubectl get jobs
Create a job: kubectl apply -f job.yaml
Status of the job: kubectl describe jobs/brev
Associated Pod: kubectl get pods
Get output: kubectl -f logs pod-name
Delete the job: kubectl delete jobs/brev
Delete the pod: kubectl delete pod pod-name

******************************************
how to copy from storage to local disk:

create a pod with mounted storage and then: `kubectl cp <name space>/<pod name>:/<persistent volume name>/<dir> <name of file>`
