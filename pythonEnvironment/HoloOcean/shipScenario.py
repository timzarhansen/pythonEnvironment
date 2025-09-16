import holoocean
import matplotlib.pyplot as plt
import numpy as np
import json
from pynput import keyboard

# "location": [35.0, -40.5, -282.5], shipwreck
# "location": [350, 900, -282.5], plane
# "location": [350, 820, -282.5], shipwreck2


with open('scenarios/OpenWaterScenario.json') as f:
    scenario = json.load(f)


    config = scenario['agents'][0]['sensors'][-1]["configuration"]
    azi = config['Azimuth']
    minR = config['RangeMin']
    maxR = config['RangeMax']
    binsR = config['RangeBins']
    binsA = config['AzimuthBins']

    #### GET PLOT READY
    plt.ion()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
    ax.set_theta_zero_location("N")
    ax.set_thetamin(-azi/2)
    ax.set_thetamax(azi/2)

    theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
    r = np.linspace(minR, maxR, binsR)
    T, R = np.meshgrid(theta, r)
    z = np.zeros_like(T)

    plt.grid(False)
    plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    pressed_keys = list()
    force = 25


    def on_press(key):
        global pressed_keys
        if hasattr(key, 'char'):
            pressed_keys.append(key.char)
            pressed_keys = list(set(pressed_keys))


    def on_release(key):
        global pressed_keys
        if hasattr(key, 'char'):
            pressed_keys.remove(key.char)


    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    #### RUN SIMULATION
    def parse_keys(keys, val):
        command = np.zeros(8)
        if 'i' in keys:
            command[0:4] += val
        if 'k' in keys:
            command[0:4] -= val
        if 'j' in keys:
            command[[4, 7]] += val/5
            command[[5, 6]] -= val/5
        if 'l' in keys:
            command[[4, 7]] -= val/5
            command[[5, 6]] += val/5

        if 'w' in keys:
            command[4:8] += val
        if 's' in keys:
            command[4:8] -= val
        if 'a' in keys:
            command[[4, 6]] += val
            command[[5, 7]] -= val
        if 'd' in keys:
            command[[4, 6]] -= val
            command[[5, 7]] += val

        return command


    np.set_printoptions(suppress=True)
    #### RUN SIMULATION
    # command = np.array([0,0,0,0,20,20,20,20])
    with holoocean.make(scenario_cfg=scenario) as env:
        while True:
            if 'q' in pressed_keys:
                break
            command = parse_keys(pressed_keys, force)

            # send to holoocean
            env.act("auv0", command)
            state = env.tick()

            if 'ImagingSonar' in state:
                s = state['ImagingSonar']
                plot.set_array(s.ravel())

                fig.canvas.draw()
                fig.canvas.flush_events()
                if 'PoseSensor' in state:
                    print(np.asarray(state['PoseSensor']))

    print("Finished Simulation!")
    plt.ioff()
    plt.show()