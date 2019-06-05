import numpy as np
import math

# HANDLE THE GENERATION OF VEHICLES IN ONE EPISODE
class TrafficGenerator:
    def __init__(self, max_steps):
        self._n_cars_generated = 1000  # how many cars per episode
        self._max_steps = max_steps

    # generation of routes of cars
    def generate_routefile(self, seed):
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/custom.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_E" edges="w_t t_e"/>
            <route id="N_S" edges="n_t t_s"/>
            <route id="E_W" edges="e_t t_w"/>
            <route id="S_N" edges="s_t t_n"/>
            """, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                route_straight = np.random.randint(1, 5)  # choose a random source & destination
                if route_straight == 1:
                    print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_straight == 2:
                    print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_straight == 3:
                    print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:
                    print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
