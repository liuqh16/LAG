import numpy as np


class BaseAgent:
    def __init__(self, break_value: float, delta_t: float):
        self.x = None
        self.y = None
        self.phi = None  # angle of the velocity vector
        self.theta = None  # direction of the agent
        self.speed = None
        self.delta_t = delta_t
        self.break_value = break_value

    def accelerate(self, value: float) -> None:
        raise NotImplementedError

    def break_(self) -> None:
        raise NotImplementedError

    def turn(self, value: float) -> None:
        raise NotImplementedError

    def reset(self, x: float, y: float, direction: float) -> None:
        self.x = x
        self.y = y
        self.speed = 0
        self.theta = direction

    def _step(self) -> None:
        angle = self.theta if self.phi is None else self.phi
        self.x += self.delta_t * self.speed * np.cos(angle)
        self.y += self.delta_t * self.speed * np.sin(angle)


class MovingAgent(BaseAgent):
    def __init__(self, break_value: float, delta_t: float):
        super(MovingAgent, self).__init__(break_value, delta_t)

    def accelerate(self, value: float) -> None:
        self.speed += value
        self._step()

    def break_(self) -> None:
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        self._step()

    def turn(self, value: float) -> None:
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()


class SlidingAgent(BaseAgent):
    def __init__(self, break_value: float, delta_t: float):
        super(SlidingAgent, self).__init__(break_value, delta_t)
        self.phi = 0

    def accelerate(self, value: float) -> None:
        # Adding two polar vectors: https://math.stackexchange.com/a/1365938/849658
        speed = np.sqrt(value**2 + self.speed**2 + 2*value*self.speed*np.cos(value - self.speed))
        angle = self.theta + np.arctan2(value*np.sin(self.phi-self.theta),
                                        self.theta + self.phi*np.cos(self.phi - self.theta))
        self.speed = speed
        self.phi = angle
        self._step()

    def break_(self) -> None:
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        self.phi = self.theta if self.speed == 0 else self.phi  # not sure it is needed
        self._step()

    def turn(self, value: float) -> None:
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()