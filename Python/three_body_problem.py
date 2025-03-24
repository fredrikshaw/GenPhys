import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

class Body:
    def __init__(self,  ID=None, mass=0, radius=0, x_pos=0, y_pos=0, x_vel=0, y_vel=0, T=0):
        self.m = mass
        self.r = radius
        self.x = x_pos
        self.y = y_pos
        self.v_x = x_vel
        self.v_y = y_vel
        self.T = T
        if ID == None:
            self.ID = random.randint(0, 1e9)
        else:
            self.ID = ID
        self.F_x = 0
        self.F_y = 0

    def law_of_gravity(self, body):
        G = 6.6743e-11
        dx = body.x - self.x
        dy = body.y - self.y
        r = np.sqrt(dx ** 2 + dy ** 2)

        if r == 0:
            return {'F_x': 0, 'F_y': 0}

        tot_F = G * self.m * body.m / (r ** 2)
        F_x = tot_F * (dx / r)
        F_y = tot_F * (dy / r)
        return {'F_x': F_x, 'F_y': F_y}

    def calc_forces(self, list_of_bodies):
        F_x = 0
        F_y = 0
        for body in list_of_bodies:
            if body.ID != self.ID:
                temp_F = self.law_of_gravity(body)
                F_x += temp_F['F_x']
                F_y += temp_F['F_y']
        self.F_x = F_x
        self.F_y = F_y

    def verlet_time_step(self, t):
        # Use Verelt algorithm to find next point
        a_x = self.F_x/self.m
        a_y = self.F_y/self.m
        self.x = self.x + self.v_x*t + 0.5*a_x*t**2
        self.y = self.y + self.v_y*t + 0.5*a_y*t**2
        self.v_x = self.v_x + a_x*t
        self.v_y = self.v_y + a_y*t

    def update_temperature_equilibrium(self, list_of_bodies):
        # This calculation is done using the formula T=(0.25*SUM(R_sun^2*T_sun^2/dist^2))^0.25
        # This formula derived from stefan boltzman law and the inverse square law
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²K⁴)
        star_threshold_max = 1e28  # ~50x Jupiter's mass
        albedo = 0.3
        cum_absorbed_power = 0

        if self.m >= star_threshold_max:
            return

        for sun in list_of_bodies:
            if sun.ID != self.ID and sun.m >= star_threshold_max:
                d = np.sqrt((self.x - sun.x) ** 2 + (self.y - sun.y) ** 2)
                solar_constant = sun.r**2 * sigma * sun.T**4 / (d**2)
                cum_absorbed_power += (1-albedo) * solar_constant / (4*sigma)
        self.T = cum_absorbed_power**0.25
        return

class BodySystem:
    def __init__(self, list_of_bodies):
        self.bodies = list_of_bodies
        self.coords = {}
        self.temps = {}
        self.times = np.array([0])
        for body in list_of_bodies:
            self.coords[body.ID] = np.array([[body.x], [body.y]])
            self.temps[body.ID] = body.T

    def time_step(self, dt):
        self.times = np.append(self.times, self.times[-1] + dt)
        for body in self.bodies:  # Calculate all forces with current position
            body.calc_forces(self.bodies)
        for body in self.bodies:
            body.update_temperature_equilibrium(self.bodies)
            body.verlet_time_step(dt)
            self.coords[body.ID] = np.append(self.coords[body.ID], [[body.x], [body.y]], axis=1)
            self.temps[body.ID] = np.append(self.temps[body.ID], body.T)

if __name__ == "__main__":
    body_list = [
        Body("Earth", 5.972e24,6.371e6, 1.496e11, 0, 0, 29780, 288),
        Body("Sun", 1.989e30,6.96e8, 0, 0, 0, 0, 5778),
        Body("Jupiter with mass of sun", 1.989e30,6.96e8, 7.78e11, 0, 0, 1305.74, 5778),
    ]

    solar_system = BodySystem(body_list)
    for i in range(1000):
        solar_system.time_step(3600*24)

    au_in_m = 1.496e11
    for body in body_list:
        temp_coords = solar_system.coords[body.ID]
        plt.plot(temp_coords[0]/au_in_m, temp_coords[1]/au_in_m, label=body.ID)
    plt.legend()
    plt.xlabel("X-axis [AU]")
    plt.ylabel("Y-axis [AU]")
    plt.title('Map of motion of planets')
    plt.axis("equal")
    plt.show()

    plt.plot(solar_system.times/(3600*24), solar_system.temps["Earth"])
    plt.ylabel('Temperature of earth')
    plt.xlabel("Time [days]")
    plt.show()
