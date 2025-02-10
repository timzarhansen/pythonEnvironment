from time import sleep

import docker
from prompt_toolkit.utils import to_str

client = docker.from_env()

# print(containers)

# total_memory_usage = sum(stats['memory_stats']['usage'] / (1024**3) for c in containers if (stats := c.stats(stream=False)))
# print(total_memory_usage)
i=1
while i<=30:
    containers = client.containers.list()
    total_memory_usage = sum(
        stats['memory_stats']['usage'] / (1024 ** 3) for c in containers if (stats := c.stats(stream=False)))
    print("Memory usage is: ", total_memory_usage)
    if(total_memory_usage<70):

        print("running container number: ",i)
        container = client.containers.run(
            image='computationimageodometry',
            command='/home/tim-external/ros_ws/src/UnderwaterSlam/bashScript/runningDockerBenchmarks/input'+str(i)+'.sh',
            volumes={
                '/Users/timhansen/Documents/ros_ws/cache/humble/build': {'bind': '/home/tim-external/ros_ws/build', 'mode': 'cached'},
                '/Users/timhansen/Documents/ros_ws/cache/humble/install': {'bind': '/home/tim-external/ros_ws/install', 'mode': 'cached'},
                '/Users/timhansen/Documents/ros_ws/cache/humble/log': {'bind': '/home/tim-external/ros_ws/log', 'mode': 'cached'},
                '/Users/timhansen/Documents/ros_ws/configFiles': {'bind': '/home/tim-external/ros_ws/configFiles', 'mode': 'cached'},
                '/Users/timhansen/Documents/ros_ws/src': {'bind': '/home/tim-external/ros_ws/src', 'mode': 'cached'},
                '/Users/timhansen/Documents/dataFolder': {'bind': '/home/tim-external/dataFolder', 'mode': 'cached'}
            },
            # network='devcontainer'+str(i)+'_net',
            detach=True
        )
        i=i+1
    sleep(100)

















