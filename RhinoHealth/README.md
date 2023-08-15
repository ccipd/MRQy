# Running MRQy via the Rhino Health Federated Computing Platform (FCP)
<br/>

### **Description**

Rhino Health's Federated Computing Platform can be used to run MRQy on data without copying it to any other location (whether it's your data or data from a collaborating site). This folder contains the assets needed in order to build the MRQy code to be used with FCP.

Additional resources and examples can be found at: https://github.com/RhinoHealth/user-resources

<br/><br/>

### **Resources**
- `Dockerfile` - This is the Dockerfile to be used for building the container image.
- `runprep.sh` - The entrypoint shell script for the docker container, which runs mrqy/QC.py
<br><br>

### **Build Instructions**
From the root of this repository, run the following command to build the container image:

```<path_to_fcp_user-resources>/rhino-utils/docker-push.sh -f RhinoHealth/Dockerfile rhino-gc-workgroup-<your-workgroup> <unique-image-tag>```
<br><br>


# Getting Help
For additional support, please reach out to [support@rhinohealth.com](mailto:support@rhinohealth.com).
