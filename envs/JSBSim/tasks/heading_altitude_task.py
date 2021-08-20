from .heading_task import HeadingTask
from ..termination_conditions import ExtremeState, LowAltitude, Overload, \
                                      Timeout, UnreachHeading, UnreachHeadingAndAltitude
class HeadingAndAltitudeTask(HeadingTask):
    def __init__(self, config):
        super().__init__(config)
        self.termination_conditions = [
            UnreachHeadingAndAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]