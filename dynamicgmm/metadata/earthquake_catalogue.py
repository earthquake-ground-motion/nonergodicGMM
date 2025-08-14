"""
Earthquake Catalogue Related Classes and Methods
"""
import json
import logging
from typing import Dict, Optional, List, Union, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
import obspy


logging.basicConfig(level=logging.INFO)


def _author_from_creation_info(
    creation_info: obspy.core.event.base.CreationInfo
) -> Union[str, None]:
    """
    """
    if not creation_info:
        return None
    if creation_info["author"]:
        return creation_info["author"]
    elif creation_info["author_uri"]:
        return creation_info["author_uri"].id.split("/")[-1]
    elif creation_info["agency_id"]:
        return creation_info["agency_id"]
    elif creation_info["agency_uri"]:
        return creation_info["agency_uri"].id.split("/")[-1]
    else:
        return None


"""
Elemental earthquake catalogue objects
"""
class Magnitude():
    """Class to store and I/O earthquake magnitude information

    Args:
        id: Unique identifier
        value: Magnitude value
        mag_type: Magnitude unit
        author: Author reporting the magnitue
        uncertainty: Uncertainty on the magnitude
        origin_id: Origin to which the magnitude is associated
        metadata: Dictionary of additional metadata (optional)
    """
    def __init__(
        self,
        id: str,
        event_id: str,
        value: float,
        mag_type: str,
        author: str,
        uncertainty: Optional[obspy.core.event.QuantityError] = None,
        origin_id: Optional[str] = "",
        metadata: Optional[Dict] = None
    ):
        self.id = id
        self.event_id = event_id
        self.origin_id = origin_id
        self.value = value
        self.mag_type = mag_type
        self.author = author
        self.metadata = metadata
        if isinstance(uncertainty, obspy.core.event.QuantityError):
            self.uncertainty = uncertainty["uncertainty"]
        else:
            self.uncertainty = uncertainty

    def __repr__(self):
        return "{:s}: {:.2f} {:s} (+/- {:.3f}) {:s}".format(
            self.id,
            self.value,
            self.mag_type,
            0.0 if self.uncertainty is None else self.uncertainty,
            self.author
        )

    @classmethod
    def from_quakeml(
        cls,
        magnitude: obspy.core.event.Magnitude,
        event_id: str,            
        id_splitter: Dict
    ):
        """Instantiate the object from an obspy.core.event.Magnitude object
        """
        mag_splitter = id_splitter["magnitude"]
        i_d = magnitude.resource_id.id.split(mag_splitter)[-1]
        mag = magnitude.mag
        author = _author_from_creation_info(magnitude.creation_info)
        mag_type = magnitude.magnitude_type
        if magnitude.origin_id:
            origin_splitter = id_splitter["origin"]
            origin_id = magnitude.origin_id.id.split(origin_splitter)[-1]
        else:
            origin_id = ""
        metadata = {
            "creation_time": str(magnitude.creation_info["creation_time"]) \
                if magnitude.creation_info else None
        }
        if magnitude.method_id:
            method_splitter = id_splitter.get("method", "method_id=")
            metadata["method_id"] = magnitude.method_id.id.split(method_splitter)[-1]
        if len(magnitude.comments):
            metadata["comments"] = "|".join(magnitude.comments)

        return cls(i_d, event_id, mag, mag_type, author,
                   magnitude.mag_errors, origin_id, metadata)

    def to_dict(self) -> Dict:
        """Parse to a dictionary (json compatible). In this case the private
        dictionary method is json compatible
        """
        return self.__dict__
        


class Origin():
    """Class to store and I/O earthquake origin information
    """
    def __init__(
        self,
        id: str,
        event_id: str,
        time: str,
        longitude: float,
        latitude: float,
        depth: float,
        author: str,
        longitude_error: float = None,
        latitude_error: float = None,
        depth_errors: float = None,
        depth_type: Optional[str] = None
    ):
        """
        """
        self.id = id
        self.event_id = event_id
        self.time = time
        self.longitude = longitude
        self.latitude = latitude
        self.depth = depth
        self.author = author
        self.longitude_error = longitude_error
        self.latitude_error = latitude_error
        self.depth_error = depth_errors
        self.depth_type = depth_type

    def __repr__(self):
        return "{:s}: {:s} ({:.4f} E, {:.4f} N, {:.2f} Z) {:s}".format(
            self.id,
            str(self.time),
            self.longitude,
            self.latitude,
            self.depth,
            self.author
        )

    @classmethod
    def from_quakeml(
        cls,
        origin: obspy.core.event.Origin,
        event_id: str, 
        id_splitter: Dict,
        depth_scaling_factor: float = 1.0,
        ):
        """Parse the origin information from an obspy.core.event.Origin object
        """
        i_d = origin.resource_id.id.split(id_splitter["origin"])[-1]
        time = str(origin.time)
        author = _author_from_creation_info(origin.creation_info)
        return cls(
            i_d,
            event_id, 
            time,
            origin.longitude,
            origin.latitude,
            (origin.depth * depth_scaling_factor) if origin.depth else None,
            author,
            origin.longitude_errors["uncertainty"] \
                if origin.longitude_errors["uncertainty"] else None,
            origin.latitude_errors["uncertainty"] \
                if origin.latitude_errors["uncertainty"] else None,
            (origin.depth_errors["uncertainty"] * depth_scaling_factor) \
                if origin.depth_errors["uncertainty"] else None,
            origin.depth_type
        )

    def to_dict(self) -> Dict:
        """Parse the origin information to a json compatible dictionary. In this case the
        private __dict__ is sufficient
        """
        return self.__dict__

def _parse_nodal_planes(nodal_planes: obspy.core.event.source.NodalPlanes) -> Dict:
    """Parses the obspy.core.event.source.NodalPlanes class to a simple dictionary
    """
    output = { 
        "nodal_plane_1": nodal_planes.get("nodal_plane_1", {}),
        "nodal_plane_2": nodal_planes.get("nodal_plane_2", {}),
    }
    for npl in output:
        if output[npl]:
            output[npl] = {
                "strike": output[npl].get("strike", None),
                "dip": output[npl].get("dip", None),
                "rake": output[npl].get("rake", None)
            }
    return output
        

def _parse_principal_axes(principal_axes: obspy.core.event.PrincipalAxes) -> Dict:
    """Parses the obspy.core.event.source.PrincipalAxes class to a simple dictionary
    """
    return {
        "T": {
            "azimuth": principal_axes["t_axis"]["azimuth"],
            "plunge": principal_axes["t_axis"]["plunge"],
            "length": principal_axes["t_axis"]["length"],
        },
        "P": {
            "azimuth": principal_axes["p_axis"]["azimuth"],
            "plunge": principal_axes["p_axis"]["plunge"],
            "length": principal_axes["p_axis"]["length"],
        },
        "N": {
            "azimuth": principal_axes["n_axis"]["azimuth"],
            "plunge": principal_axes["n_axis"]["plunge"],
            "length": principal_axes["n_axis"]["length"],
        }
    }

def _parse_moment_tensor(moment_tensor: obspy.core.event.source.MomentTensor) ->\
        Tuple[Dict, Dict]:
    """Parses the obspy.core.event.source.MomentTensor object to a simple dictionary
    """
    tensor = {
        "m_rr": moment_tensor.tensor["m_rr"],
        "m_tt": moment_tensor.tensor["m_tt"],
        "m_pp": moment_tensor.tensor["m_pp"],
        "m_rt": moment_tensor.tensor["m_rt"],
        "m_rp": moment_tensor.tensor["m_rp"],
        "m_tp": moment_tensor.tensor["m_tp"],
    }
    tensor_error = {
        "m_rr": moment_tensor.tensor["m_rr_errors"]["uncertainty"],
        "m_tt": moment_tensor.tensor["m_tt_errors"]["uncertainty"],
        "m_pp": moment_tensor.tensor["m_pp_errors"]["uncertainty"],
        "m_rt": moment_tensor.tensor["m_rt_errors"]["uncertainty"],
        "m_rp": moment_tensor.tensor["m_rp_errors"]["uncertainty"],
        "m_tp": moment_tensor.tensor["m_tp_errors"]["uncertainty"],
    }
    return tensor, tensor_error


class FocalMechanism():
    def __init__(
        self,
        i_d: str,
        event_id: str,
        scalar_moment: float,
        author: str,
        nodal_planes: Optional[Dict] = None,
        principal_axes: Optional[Dict] = None,
        tensor: Optional[Dict] = None, 
        tensor_error: Optional[Dict] = None,
        tensor_id: Optional[str] = None
    ):
        self.id = i_d
        self.event_id = event_id
        self.moment = scalar_moment
        self.author = author
        self.nodal_planes = nodal_planes
        self.principal_axes = principal_axes
        self.tensor = tensor
        self.tensor_error = tensor_error
        self.tensor_id = tensor_id

    def __repr__(self):
        has_nodal_planes = "Y" if self.nodal_planes else "N"
        has_moment_tensor = "Y" if self.tensor else "N"
        has_principal_axes = "Y" if self.principal_axes else "N"
        return "{:s}: Mo {:.0E} Nm {:s} (Nodal Planes: {:s}, Moment Tensor: {:s}, Principal Axes: {:s})".format(
            self.id,
            self.moment if self.moment else np.nan,
            self.author,
            has_nodal_planes,
            has_moment_tensor,
            has_principal_axes
        )

    @classmethod
    def from_quakeml(cls, focal_mechanism, event_id: str, id_splitter: Dict):
        """
        """
        i_d = focal_mechanism.resource_id.id.split(id_splitter["focal_mechanism"])[-1]
        author = _author_from_creation_info(focal_mechanism.creation_info)
        nodal_planes = None
        if focal_mechanism.nodal_planes:
            nodal_planes = _parse_nodal_planes(focal_mechanism.nodal_planes)
        principal_axes = None
        if focal_mechanism.principal_axes:
            principal_axes = _parse_principal_axes(focal_mechanism.principal_axes)
        tensor = None
        tensor_error = None
        tensor_id = None
        if focal_mechanism.moment_tensor:
            moment = focal_mechanism.moment_tensor["scalar_moment"]
            if focal_mechanism.moment_tensor.tensor:
                tensor_id = focal_mechanism.moment_tensor.resource_id.id.split(
                    id_splitter["moment_tensor"]
                )[-1]
                tensor, tensor_error = _parse_moment_tensor(focal_mechanism.moment_tensor)
        else:
            moment = None
        return cls(i_d, event_id, moment, author, nodal_planes, principal_axes,
                   tensor, tensor_error, tensor_id)

    def to_dict(self) -> Dict:
        """Exports the information to a json-compatible dictionary. In this case the
        private __dict__ is sufficient
        """
        return self.__dict__


class Event():
    """
    """
    def __init__(self, i_d: str, author: str, origins: List, magnitudes: List,
                 focal_mechanisms: Optional[List] = None,
                 preferred_origin_id: Optional[str] = None,
                 preferred_magnitude_id: Optional[str] = None,
                 preferred_focal_mechanism_id: Optional[str] = None,
                 event_descriptions: Optional[Dict] = None):
        """
        """
        self.id = i_d
        self.author = author
        self.origins = origins
        self.origin_ids = [orig.id for orig in self.origins]
        self.magnitudes = magnitudes
        self.magnitude_ids = [mag.id for mag in self.magnitudes]
        self.focal_mechanisms = focal_mechanisms if focal_mechanisms is not None else []
        self.focal_mechanism_ids = [foc_mech.id for foc_mech in self.focal_mechanisms]
        # print(self.focal_mechanism_ids, self.focal_mechanisms)
        if preferred_origin_id:
            self.pref_orig_loc = self.origin_ids.index(preferred_origin_id)
        else:
            self.pref_orig_loc = None
        if preferred_magnitude_id:
            self.pref_mag_loc = self.magnitude_ids.index(preferred_magnitude_id)
        else:
            self.pref_mag_loc = None
        if preferred_focal_mechanism_id:
            self.pref_foc_mech_loc = self.focal_mechanism_ids.index(preferred_focal_mechanism_id)
        else:
            self.pref_foc_mech_loc = None
        self.event_descriptions = event_descriptions

    def __repr__(self):
        return f"{self.id} - {self.author}"

    @property
    def preferred_origin(self):
        """
        """
        if self.pref_orig_loc is None:
            raise ValueError("No preferred origin defined for {:s}".format(str(self)))
        return self.origins[self.pref_orig_loc]

    @property
    def preferred_magnitude(self):
        """
        """
        if self.pref_mag_loc is None:
            raise ValueError("No preferred magnitude defined for {:s}".format(str(self)))
        return self.magnitudes[self.pref_mag_loc]

    @property
    def preferred_focal_mechanism(self):
        if not len(self.focal_mechanisms):
            return None
        if self.pref_foc_mech_loc is None:
            raise ValueError("No preferred focal_mechanism defined for {:s}".format(str(self)))
        return self.focal_mechanisms[self.pref_foc_mech_loc]

    @property
    def preferred_solution(self):
        """Returns the preferred origin, magnitude and (if 
        """
        try:
            pref_foc_mech = self.preferred_focal_mechanism
        except ValueError:
            # No preferred focal mechanism
            pref_foc_mech = None
        return self.preferred_origin, self.preferred_magnitude, pref_foc_mech

    @classmethod
    def from_quakeml(cls, event, id_splitter: Dict, depth_scaling_factor: float = 1.0):
        """
        """
        i_d = event.resource_id.id.split(id_splitter["event"])[-1]
        author = _author_from_creation_info(event.creation_info)
        origins = [Origin.from_quakeml(orig, i_d, id_splitter, depth_scaling_factor)
                   for orig in event.origins]
        preferred_origin_id = event.preferred_origin_id.id.split(id_splitter["origin"])[-1]
        magnitudes = [Magnitude.from_quakeml(mag, i_d, id_splitter)
                      for mag in event.magnitudes]
        preferred_magnitude_id = event.preferred_magnitude_id.id.split(
            id_splitter["magnitude"]
        )[-1]
        if len(event.focal_mechanisms):
            focal_mechanisms = [FocalMechanism.from_quakeml(foc_mech, i_d, id_splitter)
                                for foc_mech in event.focal_mechanisms]
            preferred_focal_mechanism_id = \
                event.preferred_focal_mechanism_id.id.split(id_splitter["focal_mechanism"])[-1]
        else:
            focal_mechanisms = None
            preferred_focal_mechanism_id = None
        event_descriptions = {"creation_agency": event.creation_info["agency_id"]}
        return cls(
            i_d,
            author,
            origins,
            magnitudes,
            focal_mechanisms,
            preferred_origin_id, preferred_magnitude_id, preferred_focal_mechanism_id,
            event_descriptions
        )

    def to_dict(self) -> Dict:
        """
        """
        output = {
            "id": self.id,
            "author": self.author,
            "event_descriptions": self.event_descriptions,
            "origins": [],
            "magnitudes": [],
            "focal_mechanisms": []
        }
        preferred_origin, preferred_mag, preferred_foc_mech = self.preferred_solution
        # Get the origins and label the preferred one as such
        for orig in self.origins:
            orig_dict = orig.to_dict()
            orig_dict["preferred"] = orig.id == preferred_origin.id
            output["origins"].append(orig_dict)
        # Likewise for the magnitudes
        for mag in self.magnitudes:
            mag_dict = mag.to_dict()
            mag_dict["preferred"] = mag.id == preferred_mag.id
            output["magnitudes"].append(mag_dict)
        for foc_mech in self.focal_mechanisms:
            foc_mech_dict = foc_mech.to_dict()
            if preferred_foc_mech:
                foc_mech_dict["preferred"] = foc_mech.id == preferred_foc_mech.id
            output["focal_mechanisms"].append(foc_mech_dict)
        return output


class Catalogue():
    """
    """
    def __init__(self, name: str, events: List, data_source: Optional[str] = None):
        """
        """
        self.name = name
        self.events = events
        self.source = data_source
        self.event_ids = [ev.id for ev in self.events]

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        return "Catalogue {:s}: {:g} Events".format(self.name, len(self))

    def __iter__(self):
        for ev in self.events:
            yield ev

    def __getitem__(self, key: Union[int, str]) -> Event:
        if isinstance(key, int):
            return self.events[key]
        elif isinstance(key, str):
            loc = self.event_ids.index(key)
            if loc:
                return self.events[loc]
            else:
                raise ValueError("Cannot retrieve event %s (not found)" % key)
        else:
            raise ValueError("Key type not recognised")
        return

    @classmethod
    def from_quakeml(
        cls, 
        catalogue: obspy.core.event.catalog.Catalog,
        name: str,
        id_splitter: Dict,
        data_source: Optional[str] = None,
        depth_scaling_factor: float = 1.0
    ):
        """
        """
        events = []
        for event in catalogue:
            events.append(Event.from_quakeml(event, id_splitter, depth_scaling_factor))
        return cls(name, events, data_source)

    @classmethod
    def from_multiple_quakemls(
        cls,
        catalogues: List,
        name: str,
        id_splitter: Dict,
        data_source: Optional[str] = None,
        depth_scaling_factor: float = 1.0
    ):
        """
        """
        all_events = []
        for catalogue in catalogues:
            try:
                sub_catalogue = Catalogue.from_quakeml(
                    catalogue,
                    "",
                    id_splitter,
                    data_source,
                    depth_scaling_factor
                )
            except:
                for i, event in enumerate(catalogue):
                    print(event.resource_id, event.preferred_origin().time)
                    _ = Event.from_quakeml(event, id_splitter, depth_scaling_factor)
            all_events.extend(sub_catalogue.events)
        return cls(name, all_events, data_source)

    def count_magnitude_types_agencies(self, print_output=False) -> Dict:
        """Returns the different magnitude types and agencies and the number
        of times they appear.

        Args:
            print_output: If True then it prints a formatted output to the log
        """
        summary = {}
        for event in self:
            for mag in event.magnitudes:
                mag_info = (mag.mag_type, mag.author)
                if mag_info in summary:
                    summary[mag_info] += 1
                else:    
                    summary[mag_info] = 1
        if print_output:
            for (mag_type, author), count in summary.items():
                logging.info("{:s} - {:s}: {:g}".format(mag_type, author, count))
        return summary

    def to_dict(self) -> Dict:
        """Exports the Catalogue to a dictionary
        """
        output = {"name": self.name,
                  "source": self.source,
                  "events": {}}
        for event in self.events:
            output["events"][event.id] = event.to_dict()
        return output

    def to_json(self, filename: str):
        """Exports the catalogue to a json file
        """
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)
        logging.info("Exported to file: %s" % filename) 
        return
