import yaml
# from numpy.f2py.crackfortran import endifs
import os
import docker
from prompt_toolkit.utils import to_str
currentNumberScript = 0

registrationList = ["fs3d32","fs3d64","fs3d128","fs3d32ICP","fs3d64ICP","fs3d128ICP","ICP"]
numberOfSkips = [1,2,5,10,15,20,30]
robot = ["Alpha","Bob","Carol"]
scanRadiusMax = [15,25,35]

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

yaml.add_representer(str, quoted_presenter)



client = docker.from_env()



for numberOfSkips_ in numberOfSkips:
    for robot_ in robot:
        for scanRadiusMax_ in scanRadiusMax:
            for registration in registrationList:


                config = {
                    "/odometrypublisher": {
                        "ros__parameters": {
                            "number_of_skips": numberOfSkips_,
                            "pcl_topic_name": '/'+str(robot_)+'/velodyne_points',
                            "pose_topic_name": '/'+str(robot_)+'/poseArray',
                            "gt_topic_name": '/'+str(robot_)+'/gt_xyz',
                            "time_until_save": 10,  # after 5 minutes
                            "which_registration": str(robot_),
                            "scan_radius_max": scanRadiusMax_
                        }
                    }
                }

                configFileNameHost = '/Users/timhansen/Documents/ros_ws/src/UnderwaterSlam/params/pythonGeneratedParams/config'+str(currentNumberScript)+'.yaml'
                configFileNameDocker = '/home/tim-external/ros_ws/src/UnderwaterSlam/params/pythonGeneratedParams/config'+str(currentNumberScript)+'.yaml'
                bashFileNameHost = '/Users/timhansen/Documents/ros_ws/src/UnderwaterSlam/bashScript/bashScriptsPythonGenerated/myscript'+str(currentNumberScript)+'.sh'
                bashFileNameDocker = '/home/tim-external/ros_ws/src/UnderwaterSlam/bashScript/bashScriptsPythonGenerated/myscript'+str(currentNumberScript)+'.sh'
                with open(configFileNameHost, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)

                with open(bashFileNameHost, "w") as file:
                    file.write("#!/bin/bash\n")
                    file.write("ROS_DOMAIN_ID="+str(currentNumberScript)+"\n")
                    file.write("source /opt/ros/humble/setup.bash\n")
                    file.write("source /home/tim-external/ros_ws/install/setup.bash\n")
                    file.write("ros2 run fsregistration ros2ServiceRegistrationFS3D & >/dev/null 2>&1\n")
                    file.write("ros2 run underwaterslam conversionGPStoXYZ.py & >/dev/null 2>&1\n")
                    file.write("ros2 run underwaterslam odometryTest --ros-args --params-file "+configFileNameDocker+" & >/dev/null 2>&1\n")
                    file.write("pid1=$!\n")
                    file.write("\nsleep 60\n")
                    file.write("ros2 bag play /home/tim-external/dataFolder/S3E/S3Ev1/S3E_Campus_Road_1/ -r 0.2\n")
                    file.write("wait $pid1\n")

                # Make the script executable


                os.chmod(bashFileNameHost, 0o755)

                #run bash script in docker
                while 1:
                    containers = client.containers.list()
                    total_memory_usage = sum(
                        stats['memory_stats']['usage'] / (1024 ** 3) for c in containers if
                        (stats := c.stats(stream=False)))
                    print("Memory usage is: ", total_memory_usage)

                    if (total_memory_usage < 70):
                        print("running container number: ", currentNumberScript)
                        container = client.containers.run(
                            image='computationimageodometry',
                            command=bashFileNameDocker,
                            volumes={
                                '/Users/timhansen/Documents/ros_ws/cache/humble/build': {
                                    'bind': '/home/tim-external/ros_ws/build', 'mode': 'cached'},
                                '/Users/timhansen/Documents/ros_ws/cache/humble/install': {
                                    'bind': '/home/tim-external/ros_ws/install', 'mode': 'cached'},
                                '/Users/timhansen/Documents/ros_ws/cache/humble/log': {
                                    'bind': '/home/tim-external/ros_ws/log', 'mode': 'cached'},
                                '/Users/timhansen/Documents/ros_ws/configFiles': {
                                    'bind': '/home/tim-external/ros_ws/configFiles', 'mode': 'cached'},
                                '/Users/timhansen/Documents/ros_ws/src': {'bind': '/home/tim-external/ros_ws/src',
                                                                          'mode': 'cached'},
                                '/Users/timhansen/Documents/dataFolder': {'bind': '/home/tim-external/dataFolder',
                                                                          'mode': 'cached'}
                            },
                            # network='devcontainer'+str(i)+'_net',
                            detach=True,
                            remove=True
                        )
                        break











                currentNumberScript=currentNumberScript+1













