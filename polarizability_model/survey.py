import numpy as np

from discretize.utils import mkvc
from geoana.em.static import LineCurrentFreeSpace
from SimPEG.survey import BaseSrc, BaseSurvey, BaseRx

class MagneticControlledSource(BaseSrc):

    def __init__(self, receiver_list=None, location=None, current=1):
        super().__init__()
        self._receiver_list = receiver_list
        self._current = current
        if location.shape[1] != 3:
            raise ValueError(
                f"The location must be (npoints, 3), but the input shape is {location.shape}"
            )
            if not np.allclose(location[0, :], location[-1, :]):
                warnings.warn("This is not a closed loop")

        self._location=location

    @property
    def current(self):
        return self._current

    @property
    def location(self):
        return self._location

    @property
    def receiver_list(self):
        return self._receiver_list

    def eval(self, locations):
        if getattr(self, "_line_current_source", None) is None:
            self._line_current_source = LineCurrentFreeSpace(nodes=self.location, current=self.current)

        # if same as last time, don't re-evaluate
        if getattr(self, "_eval_locations", None) is not None:
            if np.allclose(self._eval_locations, locations):
                return self._magnetic_flux_density

        self._eval_locations = locations
        self._magnetic_flux_density = mkvc(self._line_current_source.magnetic_flux_density(locations))

        return self._magnetic_flux_density

class MagneticUniformSource(BaseSrc):
    def __init__(self, receiver_list=None, orientation="z", amplitude=1):
        super().__init__()
        self._receiver_list = receiver_list
        self._amplitude = amplitude
        self._orientation = orientation.lower()

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def orientation(self):
        return self._orientation

    @property
    def receiver_list(self):
        return self._receiver_list

    def eval(self, locations):
        src = np.zeros_like(locations)
        if self.orientation == "x":
            src[:, 0] = self.amplitude
        elif self.orientation == "y":
            src[:, 1] = self.amplitude
        elif self.orientation == "z":
            src[:, 2] = self.amplitude
        return src.flatten()

class MagneticFluxDensityReceiver:
    def __init__(self, locations, components=None):
        if locations.shape[1] != 3:
            raise ValueError(
                f"The location must be (npoints, 3), but the input shape is {locations.shape}"
            )
        self._locations = locations

        if components is None:
            components = ["x", "y", "z"]
        else:
            for component in components:
                if component.lower() not in ["x", "y", "z"]:
                    raise ValueError(
                        f"Components must be 'x', 'y' or 'z', not {component}"
                    )
        self._components = components

    @property
    def locations(self):
        return self._locations

    @property
    def components(self):
        return self._components

    @property
    def nD(self):
        if getattr(self, "_nD", None) is None:
            self._nD = self.locations.shape[0] * len(self.components)
        return self._nD


class Survey(BaseSurvey):

    @property
    def nD(self):
        if getattr(self, "_nD", None) is None:
            nD = 0
            for src in self.source_list:
                for rx in src.receiver_list:
                    nD += rx.nD
            self._nD = nD
        return self._nD

