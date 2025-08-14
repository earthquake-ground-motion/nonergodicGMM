"""
Tools to access and download ESM data
"""

import os
import logging
import requests
from requests.exceptions import HTTPError
from copy import deepcopy
from typing import Dict, Union
import numpy as np
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
#    if query_type == "event":
#        base_url = EVENT_QUERY_BASE_URL
#    elif query_type == "station":
#        base_url = STATION_QUERY_BASE_URL
#    elif query_type == "waveform":
#        base_url = DATA_QUERY_BASE_URL
#    else:
#        raise ValueError(f"Query type {query_type} not recognised -"
#                         " must be one of 'event', 'station' or 'waveform'")
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
                    "Categorical value {:s} for FDSN option {:s} not in valid list: {:s}".format(
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

        # self._get_events()

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
        return


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

#    def download_waveforms(self):
#        """
#        """
#        logging.info("Query URL: %s" % self.url)
#        download_command = ["curl", "-X", "POST", self.url, "-o", self.filename]
#        logging.info("Running command %s" % " ".join(download_command))
#        completed_process = subprocess.run(download_command, check=True)
#        logging.info("Run. Return Code %g" % completed_process.returncode)
#        assert os.path.exists(self.filename)
#        return
