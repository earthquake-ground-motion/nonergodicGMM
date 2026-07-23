"""
Main code to control the download and storage of ground motion data from the ORFEUS services
ESM and RRSM
"""
import os
import logging
import datetime
import shutil
import pathlib
import requests
from copy import deepcopy
from typing import Dict, List, Tuple
import toml
import h5py
import numpy as np
import pandas as pd
from obspy.core.event import Event
from dynamicgmm.download.download_utils import get_start_end_times
from dynamicgmm.download.esm_fdsn_tools import (
    ESMEventWebService, ESMStationWebservice, ESMWaveformWebService,
    ESM_STATION_QUERY_BASE_URL, ESM_DATA_QUERY_BASE_URL
)
from dynamicgmm.download.rrsm_fdsn_tools import (
    RRSMEventStationWebService,
    RRSMWaveformWebService
)


logging.basicConfig(level=logging.INFO)


def _datetime_to_isoformat(config: Dict) -> Dict:
    """Convert all datetime objects to strings in ISO format
    """
    for key in config:
        if isinstance(config[key], datetime.datetime):
            config[key] = config[key].isoformat()
    return config


def spawn_configs(
    template_file: pathlib.Path,
    time_limits: List,
    config_directory: pathlib.Path,
    output_directory: pathlib.Path
):
    """Spawn the set of configuration files and sub-directories to download by month

    Args:
        template_file: General template TOML config file (dates and folders will change)
        time_limits: List of tuples of (starttime, endtime, target_folder) for query results
        config_directory: Directory to store the config files
        output_directory: Directory to store the data
    """
    with open(str(template_file), "r") as f:
        template_config = toml.load(f)
    # Remove existing output directory and make a new one
    if output_directory.exists():
        shutil.rmtree(str(output_directory))
    output_directory.mkdir()
    # Remove existing config directory and make a new one
    if config_directory.exists():
        shutil.rmtree(str(config_directory))
    config_directory.mkdir()
    for start_time, end_time, target_dir in time_limits:
        config = deepcopy(template_config)
        s_t = datetime.datetime.fromisoformat(start_time)
        e_t = datetime.datetime.fromisoformat(end_time)
        if "ESM" in list(config):
            config["ESM"]["event"]["starttime"] = s_t
            config["ESM"]["event"]["endtime"] = e_t
        if "RRSM" in list(config):
            if s_t.year < 2005:
                del config["RRSM"]
            else:
                config["RRSM"]["event"]["starttime"] = s_t
                config["RRSM"]["event"]["endtime"] = e_t
        if s_t.year < 1990:
            # Handle pre-1990 case (no subdirectory per month)
            year_directory = output_directory / "pre_1990"
            if not year_directory.exists():
                year_directory.mkdir()
            config["output-folder"] = str(year_directory)
            fname = str(config_directory / "config_pre_1990.toml")
        else:
            year_directory = output_directory / f"{s_t.year:g}"
            if not year_directory.exists():
                year_directory.mkdir()
            month_string = str(s_t.month).zfill(2)
            year_month_directory = year_directory / month_string
            if not year_month_directory.exists():
                year_month_directory.mkdir()
            config["output-folder"] = str(year_month_directory)
            fname = str(config_directory / f"config_{s_t.year:g}_{month_string}.toml")
        with open(fname, "w") as f:
            toml.dump(config, f)
    return


class ORFEUSStrongMotionDownloader():
    """Main class to run a download operation from a ORFEUS strong motion services

    Attributes:
        config: Config file as import from toml format
        output_directory: Output directory to store the waveforms
    """
    def __init__(
            self,
            config: Dict,
            run_type: str = "wet",
            split_dists: Tuple = (0, 500, 100),
    ):
        """
        """
        self.config = deepcopy(config)
        self.output_directory = pathlib.Path(config["output-folder"])
        
        logging.info("---- Exports to directory %s" % str(self.output_directory))
        if not self.output_directory.exists():
            self.output_directory.mkdir()
        self.esm_catalogue = None
        self.rrsm_catalogue = None
        assert run_type in ["wet", "damp", "dry"], \
            "Run type must be one of 'wet', 'damp' or 'dry' (%s given)" % run_type
        self.run_type = run_type
        self.distances = split_dists

    def run(self):
        """
        """
        if "ESM" in self.config:
            # Run the ESM downloader
            logging.info("Running the ESM download process")
            target_dir = self.output_directory / self.config["ESM"]["esm-output"]
            target_dir.mkdir()
            if self.run_type != "dry":
                self.run_esm_download(target_dir)
        if "RRSM" in self.config:
            logging.info("Running the RRSM download process")
            target_dir = self.output_directory / self.config["RRSM"]["rrsm-output"]
            target_dir.mkdir()
            if self.run_type != "dry":
                self.run_rrsm_download(target_dir)
        return

    def run_esm_download(self, target_dir: pathlib.Path):
        """Run the download process for ESM data
        """
        # Get the events from the webservice
        self.config["ESM"]["event"] = _datetime_to_isoformat(
            self.config["ESM"]["event"]
        )
        event_ws = ESMEventWebService(self.config["ESM"]["event"])
        event_ws.get_events()
        self.esm_catalogue = event_ws.catalogue
        if not len(event_ws.event_ids):
            logging.info("No events found in ESM service")
            return
        if self.run_type == "damp":
            # Running just the event query only, not the waveform download
            return
        for ev_id in event_ws.event_ids:
            # Download the waveforms for a specific event
            wf_config = deepcopy(self.config["ESM"]["waveform"])
            wf_config["eventid"] = ev_id
            logging.info("---- Downloading for event %s" % ev_id)
            event_target_file = target_dir / f"{ev_id}.{wf_config['format']}"
            wf_downloader = ESMWaveformWebService(wf_config, str(event_target_file))
            if self.run_type == "wet":
                resp_code, resp_reason = wf_downloader.download_waveforms()
                if (resp_code == 413) and (resp_reason == "Request Entity Too Large"):
                    # Too many records for this event - split by distance ring!
                    logging.info("---- Request Entity Too Large for %s" % ev_id)
                    # Gets the minimum and maximum distances for each sub-query, plus the url
                    network_station_split = self._split_esm_download_by_distance(
                        ev_id, event_ws[ev_id], wf_config
                    )
                    for (min_r, max_r), split_data in network_station_split.items():
                        logging.info(f"---- Querying distance band {min_r} km to {max_r} km")
                        logging.info(split_data["url"])
                        output_filename = target_dir / f"{ev_id}_DIST_{min_r}_{max_r}.hdf5"
                        response_split = requests.get(split_data["url"])
                        if response_split.status_code == 200:
                            # Successful download
                            with open(str(output_filename), "wb") as file:
                                file.write(response_split.content)
                            logging.info(
                                f"---- ---- Successfully downloaded to {str(output_filename)}"
                            )
                        elif response_split.status_code == 204:
                            # Successful download but no data
                            logging.info("---- ---- No data")
                        else:
                            # Failed download
                            # Show the url of the split
                            logging.info(split_data["url"])
                            logging.info(f"---- ---- Failed download: {response_split.reason}")
                else:
                    pass
        return

    def _split_esm_download_by_distance(
            self,
            ev_id: str,
            event: Event,
            wf_config: Dict
    ) -> Dict:
        """When a request returns a 413 "Request entity too large" error we can try to
        split the request into a smaller number entities based on the number of stations
        within concentric rings of epicentral distance. We use the FDSN station service
        to list the stations within successively increasing epicentral distance ranges
        and return the url with the list of stations and networks for each distance range

        Args:
            ev_id: Event ID
            event_params: Event parameters as a row (pd.Series) from the source table
            config: Event data service configuration options
            distances: Distance (km) range in terms of
                       (min_distance, max_distance, distance_increment)

        Returns:
            Dictionary containing the list of networks, stations and query url for each
            distance band, e.g. {
                  (0, 100): {"networks": [ ... ], "stations": [...], "url": ...},
                  (100, 200): {"networks": [ ... ], "stations": [...], "url": ...},
                  ...
                  }

        """
        network_station_split = {}
        min_dist, max_dist, inc_dist = self.distances
        distance_bins = [(low, low + inc_dist) for low in range(min_dist, max_dist, inc_dist)]
        for (min_r, max_r) in distance_bins:
            # Query FDSN station service
            pref_orig = event.preferred_origin()
            station_query_url = ESM_STATION_QUERY_BASE_URL + "&".join([
                "level=station",
                f"longitude={pref_orig.longitude}",
                f"latitude={pref_orig.latitude}",
                f"minradius={min_r / 111.0}",
                f"maxradius={max_r / 111.0}",
                "format=text",
            ])
            logging.info(station_query_url)
            station_list = pd.read_csv(station_query_url, sep="|")
            if station_list.shape[0] > 0:
                network_station_split[(min_r, max_r)] = {
                    "networks": pd.unique(station_list["network_code"]).tolist(),
                    "stations": pd.unique(station_list["station_code"]).tolist()}
                query_string = []
                for key, val in wf_config.items():
                    if val:
                        query_string.append("{:s}={:s}".format(key, str(val)))
                query_string.append("network={:s}".format(
                    ",".join(network_station_split[(min_r, max_r)]["networks"])
                    ))
                query_string.append("station={:s}".format(
                    ",".join(network_station_split[(min_r, max_r)]["stations"])
                ))
                network_station_split[(min_r, max_r)]["url"] = \
                    ESM_DATA_QUERY_BASE_URL + f"eventid={ev_id}&" + "&".join(query_string)
            else:
                network_station_split[(min_r, max_r)] = None
                continue
        return network_station_split

    def run_rrsm_download(self, target_dir: str):
        """Run the download process for RRSM data
        """
        self.config["RRSM"]["event"] = _datetime_to_isoformat(
            self.config["RRSM"]["event"]
        )
        # Get the events and stations from the webservice
        rrsm_downloader = RRSMEventStationWebService(self.config["RRSM"]["event"])
        rrsm_downloader.get_events_stations()
        if rrsm_downloader.events_stations is None:
            # Download failed - no content
            return
        self.rrsm_catalogue = rrsm_downloader.events_stations
        # Get target directory
        by_station = self.config["RRSM"]["waveform"].get("by-station", False)
        wf_downloader = RRSMWaveformWebService(rrsm_downloader.events_stations,
                                               target_dir,
                                               by_station=by_station)
        rrsm_downloader.to_json(os.path.join(target_dir, "events_stations.json"))
        if self.run_type == "wet":
            wf_downloader.download_waveforms()
        return


def build_orfeus_strong_motion_full_download(
    config_directory: str,
    output_directory: str,
    template_config: str,
    run_type: str = "wet"
):
    """Runs a full clean download of data from ESM and/or RRSM

    Args:
        config_directory: Path to folder to place config files
        output_directory: Path to folder to place the downloaded data
        template_config: The master config file (dates/times within the file will change per
                         download
        run_type: "wet" is a full run executing the download, "dry" creates the files but
                  doesn't execute the api download download
    """

    # Get the start and end times
    start_end_times = get_start_end_times()
    config_directory = pathlib.Path(config_directory)
    output_directory = pathlib.Path(output_directory)
    template_config = pathlib.Path(template_config)

    # Build the configurations and output directories
    logging.info("Setting up configs and output directories")
    spawn_configs(template_config, start_end_times, config_directory, output_directory)

    # Start to run the download jobs
    config_files = list(config_directory.iterdir())
    config_files.sort()
    for config_file in config_files:
        job_start = datetime.datetime.now()
        logging.info("Running download job for config file %s (Job began at %s)"
                     % (str(config_file), str(job_start)))
        with open(str(config_file), "r") as f:
            config = toml.load(f)
        downloader = ORFEUSStrongMotionDownloader(config, run_type=run_type)
        downloader.run()
        job_end = datetime.datetime.now()
        d_t = job_end - job_start
        logging.info("--- Finished at %s (DT = %s hh:mm:ss.sss)"
                     % (str(job_end), str(d_t).zfill(15)[:11]))
    logging.info("Download complete")
    return


if __name__ == "__main__":
    # CONFIG DIRECTORY
    CONFIG_DIRECTORY = "./europe_download_configs"
    # Output
    OUTPUT_DIRECTORY = "./europe_orfeus_data_122025"
    # Template file
    TEMPLATE_CONFIG = "./example_download_config.toml"
    build_orfeus_strong_motion_full_download(
        CONFIG_DIRECTORY,
        OUTPUT_DIRECTORY,
        TEMPLATE_CONFIG,
        run_type="dry"
    )
