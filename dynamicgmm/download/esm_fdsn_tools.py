"""
Tools to access and download ESM data
"""
import io
# import os
# import json
import logging
import requests
import datetime
from requests.exceptions import HTTPError
from copy import deepcopy
from typing import Dict, Union, List, Optional
import numpy as np
import pandas as pd
from openquake.hazardlib.geo import Point, PlanarSurface, MultiSurface
from openquake.hazardlib.geo.surface.base import BaseSurface
import obspy

logging.basicConfig(level=logging.INFO)

VALID_FORMATS = ["xml", "text", "shapefile"]
VALID_ORDERBY = ["time", "time-asc", "magnitude", "magnitude-asc"]
VALID_MAGTYPE = ["any", "mw", "ml", "ms", "md", "mb"]
VALID_CATALOG = ["ESM", "EMSC", "USGS", "ISC", "IGV"]
VALID_LEVELS = ["network", "station", "channel"]
VALID_DATA_FORMATS = ["hdf5", "mseed", "sac", "ascii"]
VALID_DATA_TYPES = ["ACC", "VEL", "DIS", "SA", "SD"]
VALID_PROCESSING_TYPES = ["CV", "MP", "AP"]


FDSN_SPECS = {
    "starttime": (str, ),
    "endtime": (str, ),
    "minlatitude": (float, -90.0, 90.0),
    "maxlatitude": (float, -90.0, 90.0),
    "minlongitude": (float, -180.0, 180.0),
    "maxlongitude": (float, -180.0, 180.0),
    "latitude": (float, -90.0, 90.0),
    "longitude": (float, -180.0, 180.0),
    "minradius": (float, 0.0, np.inf),
    "maxradiue": (float, 0.0, np.inf),
    "mindepth": (float, 0.0, np.inf),
    "maxdepth": (float, 0.0, np.inf),
    "minmagnitude": (float, -np.inf, np.inf),
    "maxmagnitude": (float, -np.inf, np.inf),
    "format": (str, VALID_FORMATS + VALID_DATA_FORMATS),
    "orderby": (str, VALID_ORDERBY),
    "magnitudetype": (str, VALID_MAGTYPE),
    "includeallmagnitudes": (bool, ),
    "includeallorigins": (bool, ),
    "includeallfocalmechanisms": (bool, ),  # GEOFON only
    "includefocalmechanism": (bool, ),  # GEOFON only
    "limit": (int, 0, np.inf),
    "eventid": (str, ),
    "catalog": (str, VALID_CATALOG),
    "level": (str, VALID_LEVELS),
    "network": (str,),
    "station": (str,),
    "location": (str,),
    "channel": (str,),
    "processing-type": (str, VALID_PROCESSING_TYPES),
    "data-type": (str, VALID_DATA_TYPES),
    "add-xml": (bool,),
    "add-auxiliary-data": (bool,),

}


# BASE URLS FOR CATALOGUE SERVICES

# Base URL for the Event Query
ESM_EVENT_QUERY_BASE_URL = "https://esm-db.eu/fdsnws/event/1/query?"

# Base URL for the station query
ESM_STATION_QUERY_BASE_URL = "https://esm-db.eu/fdsnws/station/1/query?"

# Base URL for waveform data query
ESM_DATA_QUERY_BASE_URL = "https://esm-db.eu/esmws/eventdata/1/query?"

# Base URL for Seismic Portal Query
SEISMIC_PORTAL_QUERY_BASE_URL = "https://www.seismicportal.eu/fdsnws/event/1/query?"

# Base URL for ISC Bulletin Query
ISC_BULLETIN_QUERY_BASE_URL = "https://www.isc.ac.uk/fdsnws/event/1/query?"

# Base URL for GEOFON event query
GEOFON_EVENT_QUERY_BASE_URL = "https://geofon.gfz.de/fdsnws/event/1/query?"


def construct_query_url(query_type: str, config: Dict, base_url: str) -> str:
    """Constructs the url for any ESM FDSN query

    Args:
        query_type: Is either an "event", "station" or "waveform" query
        config: Arguments for the query

    Returns:
        FDSN query URL
    """
    query = []
    for fdsnkey, arg in config.items():
        if fdsnkey not in FDSN_SPECS:
            raise ValueError(f"{fdsnkey} not a valid FDSN Option")
        fdsn_spec = FDSN_SPECS[fdsnkey]
        if isinstance(arg, str) and ("," in arg):
            vals = arg.split(",")
        else:
            vals = [arg]
        for val in vals:
            assert isinstance(val, fdsn_spec[0]), \
                "FDSN option %s has invalid type" % fdsnkey
            if isinstance(val, str) and len(fdsn_spec) > 1:
                assert val in fdsn_spec[1], \
                    "Categorical value {:s} for option {:s} not in valid list: {:s}".format(
                    val, fdsnkey, str(fdsn_spec[1])
                )
            if (isinstance(val, float) or isinstance(val, int)) and (len(fdsn_spec) == 3):
                # Verify within upper limits
                assert (val >= fdsn_spec[1]) & (val <= fdsn_spec[2])
        query.append("{:s}={:s}".format(fdsnkey, str(arg)))
    return base_url + "&".join(query)


class ESMEventWebService():
    """
    """
    BASE_URL = ESM_EVENT_QUERY_BASE_URL
    SERVICE = "ESM Webservice Event Catalogue"
    EVENT_ID_STRIP = "smi:esm-db.eu/fdsnws/event/1/query?event_id="

    def __init__(self, config):
        """
        """

        self.config = deepcopy(config)
        self.config["format"] = "xml"
        self.url = construct_query_url("event", config, self.BASE_URL)
        self.catalogue = []
        self.event_ids = []

    def __len__(self):
        return len(self.catalogue)

    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, str):
            return self.catalogue[self.event_ids.index(key)]
        else:
            return self.catalogue[key]

    def __iter__(self):
        for ev_id, event in zip(self.event_ids, self.catalogue.events):
            yield ev_id, event
        return

    def __repr__(self):
        return "%s (%g Events)" % (self.SERVICE, len(self))

    def get_events(self):
        """
        """
        logging.info(f"Query URL: {self.url}")
        # Query the event webservice
        try:
            catalogue = obspy.read_events(self.url, format="QUAKEML")
        except HTTPError as he:
            # If a 204 HTTP Error is returned then there is no content
            if str(he).startswith("204 HTTP Error: No Content for url"):
                logging.info("Catalogue contains no events")
                return
            else:
                # Something else is wrong, so raise the error!
                raise
        if not len(catalogue):
            logging.info("Catalogue contains no events")
            return
        ev_times = []
        for ev in catalogue:
            ev_times.append(np.datetime64(ev.preferred_origin().time))
        idx = np.argsort(np.array(ev_times))
        events = [catalogue[i] for i in idx]
        self.catalogue = obspy.Catalog(
            events,
            resource_id=catalogue.resource_id,
            description=catalogue.description,
            comments=catalogue.comments,
            creation_info=catalogue.creation_info
            )
        logging.info("Retreived catalogue contains {:g} events".format(len(self.catalogue)))
        for ev in self.catalogue:
            self.event_ids.append(ev.resource_id.id.replace(self.EVENT_ID_STRIP, ""))
        return


class ESMStationWebservice():
    """
    """
    BASE_URL = ESM_STATION_QUERY_BASE_URL
    SERVICE = "ESM Webservice Station Inventory"

    def __init__(self, config):
        """
        """
        self.config = deepcopy(config)
        self.config["format"] = "xml"
        self.url = construct_query_url("station", config, self.BASE_URL)
        self.stations = {}
        self.station_ids = []
        logging.info("Query URL: %s" % self.url)
        self._get_stations()

    def _get_stations(self):
        """
        """
        raw_stations = obspy.read_inventory(self.url, format="STATIONXML")
        # Obspy inventory object
        self.station_ids = []

        for ntw in raw_stations.networks:
            self.stations[ntw.code] = {}
            for chn in ntw.get_contents()["channels"]:
                chn_data = chn.split(".")
                assert chn_data[0] == ntw.code
                stn_id = chn_data[1]
                chan_id = chn_data[3]
                if stn_id not in self.stations[ntw.code]:
                    self.stations[ntw.code][stn_id] = raw_stations.get_coordinates(chn)
                    self.stations[ntw.code][stn_id]["channels"] = [chan_id,]
                    self.station_ids.append(f"{ntw}-{stn_id}")
                else:
                    self.stations[ntw.code][stn_id]["channels"].append(chan_id)
        return

    def __repr__(self):
        return "%s (%g stations from %g networks)" %\
            (self.SERVICE, len(self.station_ids), len(self.stations))

    def __len__(self):
        return len(self.station_ids)

    def __iter__(self):
        for ntw in self.stations:
            for stn in self.stations[ntw]:
                yield ntw, stn


class ESMWaveformWebService():
    """
    """
    BASE_URL = ESM_DATA_QUERY_BASE_URL
    SERVICE = "ESM Webservice Waveforms"

    def __init__(self, config, filename):
        """
        """
        self.config = deepcopy(config)
        self.config["format"] = "hdf5"
        self.filename = filename if filename.endswith(".hdf5") else (filename + ".hdf5")
        self.url = construct_query_url("waveform", config, self.BASE_URL)

    def download_waveforms(self):
        """
        """
        logging.info("Query URL: %s" % self.url)
        response = requests.get(self.url)
        if response.status_code == 200:
            # Successful download
            with open(self.filename, "wb") as file:
                file.write(response.content)
            logging.info("---- Successfully downloaded to %s" % self.filename)
        else:
            logging.info("---- Failed download")
            logging.info("---- %s" % response.reason)
        return response.status_code, response.reason


class SeismicPortalWebService(ESMEventWebService):
    """
    """
    BASE_URL = SEISMIC_PORTAL_QUERY_BASE_URL
    SERVICE = "Seismic Portal Event Catalogue Webservice"
    EVENT_ID_STRIP = "quakeml:eu.emsc/event/"


class ISCBulletinWebService(ESMEventWebService):
    """
    """
    BASE_URL = ISC_BULLETIN_QUERY_BASE_URL
    SERVICE = "ISC Bulletin Webservice"
    EVENT_ID_STRIP = "smi:ISC/evid="


class GEOFONEventWebService(ESMEventWebService):
    """
    """
    BASE_URL = GEOFON_EVENT_QUERY_BASE_URL
    SERVICE = "GEOFON Event Webservice"
    EVENT_ID_STRIP = "smi:org.gfz-potsdam.de/geofon/"


ESM_FAULT_WS_SPECS = {
    "eventid": (str, ),
    "output-format": (str, {"json", "text", }),
    "empirical-fault": (bool, ),
    "indent": (bool, ),
    "includeallfaults": (bool, ),
}


class ESMFaultWebservice():
    """Class to manage download and processing of fault data from the ESM Fault Service
    https://esm-db.eu/esmws/fault/1/
    """
    BASE_URL = "https://esm-db.eu/esmws/fault/1/query?"

    def __init__(self, config):
        self.config = config
        self.config["output-format"] = config.get("output-format", "json")
        assert self.config["output-format"] in ["json", "text", "shapefile"], \
            "Fault file format must be one of 'json', 'text' or 'shapefile'"
        self.config["indent"] = config.get("indent", True)
        self.config["includeallfaults"] = config.get("includeallfaults", True)
        self.faults = {}

    def _build_url(self, eventid: Optional[str] = None):
        """Constructs the URL from the query options
        """
        if eventid:
            query_opts = [f"eventid={eventid}",]
        else:
            query_opts = []
        for key, val in self.config.items():
            if key == "eventid" and eventid is not None:
                continue
            query_opts.append(f"{key}={str(val)}")
        return self.BASE_URL + "&".join(query_opts)

    def get_fault_data(self, event_ids: List):
        """Downloads the fault data for a set of events and parses the faults into
        and OpenQuake Planar or MultiSurface object where available.

        Args:
            event_ids: List of event IDs
        """
        raw_fault_data = self._get_raw_fault_data(event_ids)
        if not raw_fault_data:
            logging.info("No raw fault data obtained")
            return
        for ev_id, raw_fault in raw_fault_data.items():
            if not raw_fault:
                continue
            if self.config["output-format"] == "json":
                self.faults[ev_id] = self.parse_json_fault_data_to_OQ_surface(raw_fault)
            else:
                raise NotImplementedError("Parser for fault data type not supported")
        return

    def _get_raw_fault_data(self, event_ids: List) -> Dict:
        """Download the raw fault data from the webservice
        """
        raw_fault_info = {}
        for event_id in event_ids:
            event_url = self._build_url(event_id)
            logging.info("Querying event ID %s" % event_id)
            logging.info(event_url)
            raw_data = requests.get(event_url)
            if raw_data.status_code != 200:
                logging.info("Unsuccessful - status code %s (%s)"
                             % (raw_data.status_code, raw_data.reason))
                raw_fault_info[event_id] = None
            else:
                if self.config["output-format"] == "json":
                    raw_fault_info[event_id] = raw_data.json()
                elif self.config["output-format"] == "text":
                    raw_fault_info[event_id] = pd.read_csv(
                        io.StringIO(raw_data.content.decode("utf-8")),
                        sep=";"
                        )
                else:
                    raise NotImplementedError
                logging.info("Successful!")
        return raw_fault_info

    @staticmethod
    def parse_json_fault_data_to_OQ_surface(fault_data: Dict) -> BaseSurface:
        """Converts the JSON fault data from the ESM Fault Webservice into an OpenQuake
        planar fault surface

        Args:
            fault_data: Fault data for an individual rupture in the json dict format
        Returns:
            OpenQuake Planar or MultiSurface
        """
        pref_id = fault_data["preferred_source_id"]
        fault_surfaces = {}
        for source in fault_data["sources"]:
            source_id = source["source_id"]
            fault_surfaces[source_id] = {
                "ID": source_id,
                "Name": source["source_name"],
                "preferred": source_id == pref_id,
                "surface": None,
                "rake": None,
            }
            segments = []
            for seg_id, segment in source["segment_id"].items():
                if segment["Z_top"] is None or segment["Z_bottom"] is None:
                    # Fault surface cannot be constructed
                    logging.info(f"---- Source {source_id} segment {seg_id} missing a depth")
                    break
                top_left = Point(segment["UL_lon"], segment["UL_lat"], segment["Z_top"])
                top_right = Point(segment["UR_lon"], segment["UR_lat"], segment["Z_top"])
                bottom_right = Point(segment["LR_lon"], segment["LR_lat"], segment["Z_bottom"])
                bottom_left = Point(segment["LL_lon"], segment["LL_lat"], segment["Z_bottom"])
                try:
                    surface = PlanarSurface.from_corner_points(
                        top_left, top_right, bottom_right, bottom_left
                    )
                except ValueError:
                    logging.info("Surface construction failed for Event-Source-Segment "
                                 f"{fault_data['Event_id']}-{source['source_id']}-{seg_id}")
                    # Failed rupture - so reset the number of segments to none
                    segments = []
                    break
                orig_strike = segment["strike"]
                orig_dip = segment["dip"]
                orig_length = segment["Length"]
                orig_width = segment["Width"]
                sfc_strike = surface.get_strike()
                sfc_dip = surface.get_dip()
                sfc_width = surface.get_width()
                sfc_length = surface.get_area() / sfc_width
                logging.info(
                    "Comparison:\n Strike {:.2f}|{:.2f}, Dip {:.2f}|{:.2f} "
                    "Length {:.2f}|{:.2f}, Width {:.2f}|{:.2f}".format(
                        orig_strike, sfc_strike,
                        orig_dip, sfc_dip,
                        orig_length, sfc_length,
                        orig_width, sfc_width
                    )
                )
                segments.append((surface, segment["rake"]))
            if not len(segments):
                continue
            if len(segments) > 1:
                surfaces = []
                areas = []
                rakes = []
                for sfc, rake in segments:
                    surfaces.append(sfc)
                    areas.append(sfc.get_area())
                    if rake is None:
                        rakes.append(np.nan)
                    else:
                        rakes.append(rake)
                areas = np.array(areas)
                weights = areas / np.cumsum(areas)
                rakes = np.array(rakes)
                rakes[np.isnan(rakes)] = np.nanmean(rakes)
                fault_surfaces[source_id]["surface"] = MultiSurface(surfaces)
                fault_surfaces[source_id]["rake"] = float(np.sum(weights * rakes))
            else:
                fault_surfaces[source_id]["surface"] = segments[0][0]
                fault_surfaces[source_id]["rake"] = segments[0][1]
        return fault_surfaces


class ESMEventProcessingUpdateWebservice():
    """Retreives a dictionary containing the list of events whose data and/or metadata
    have been updated after a specific date
    """
    BASE_URL = "https://esm-db.eu/esmws/event-processing-update/1/query?"

    def __init__(self, config):
        self.config = config
        self.config["ptype"] = config.get("ptype", "all")
        self.info = None

    def _build_query_url(self):
        """Constructs the query url
        """
        query_args = []
        for key, val in self.config.items():
            query_args.append(f"{key}={str(val)}")
        return self.BASE_URL + "&".join(query_args)

    def get_update_information(self, updated_after: Union[str, datetime.date]):
        """Get the list of events updated after a specific date
        """
        self.config["updatedafter"] = str(updated_after)
        query_url = self._build_query_url()
        logging.info("Querying Event-Processing Update Webservice: \n%s" % query_url)
        raw_data = requests.get(query_url)
        if raw_data.status_code == 200:
            # Successful query
            self.info = raw_data.json()
            return
        # Error handling
        if raw_data.status_code == 204:
            # No content error
            logging.info("Successful query - No content")
        else:
            # Other error
            logging.info("Unsuccessful - status code %s (%s)"
                         % (raw_data.status_code, raw_data.reason))
        return
