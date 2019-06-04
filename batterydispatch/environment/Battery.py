class Battery:
    def __init__(self, capacity=1000, power=100, name='battery'):
        self.capacity = capacity  # maximum capacity of the battery in kWh
        self.power = power
        self.leakage = 0  # not implemented - how much of the stored energy is lost each time step?
        self.name = name
        self.charge = self.capacity

    def deploy(self, rate, time_period=1, affect_state=True):
        '''Deploy the battery by charging (negative value) or discharging (positive value)
        time_period is the length of time to discharge in hours where a default is 1 hour.
        If affect_state = False, the actual charge of the battery (internal state of Battery class) is not changed.
        Returns: kW deployed (negative means drawing from the bus, and positive means outputing to the bus)
        '''
        if self.charge - rate * time_period < 0:
            # Instruction would deplete the battery below 0.
            rate = self.charge / time_period
        elif self.charge - rate * time_period > self.capacity:
            # Instruction would over-charge battery.
            rate = -1 * (self.capacity - self.charge) / time_period

        if affect_state:
            self.charge -= rate * time_period

        if abs(rate) > self.power:
            raise ValueError("charge/discharge of {} exceeds power!".format(rate))
        return rate

    def soc(self):
        return self.charge / self.capacity