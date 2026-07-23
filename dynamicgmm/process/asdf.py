"""
Class to handle data in ASDF and retreive information to put into a flatfile
format
"""
import os
import io
import h5py
import json
import numpy as np
import logging
from typing import Union, List, Dict, Optional, Tuple
from scipy.integrate import cumtrapz
import pandas as pd
from scipy.constants import g
from obspy import read_events, read_inventory
from dynamicgmm.process.base import (
    ResponseSpectrum, FourierSpectrum, Waveform, Record, # get_horizontal_spectrum,
    DEFAULT_PERIODS,
    )
import dynamicgmm.process.intensity_measures as ims

logging.basicConfig(level=logging.INFO)


class ASDFEventHandler():
    """Class to manage access and processing of ground motion data stored in
    ASDF format

    Attributes:
        fname: Path to the asdf file
        events: Dictionary of events information (as :class:obspy.?)
        stations: Dictionary of station metadata extracted from the station XMLs
        events_stations: Dictionary mapping the available record IDs to the corresponding
                         networks and events
    """

    FLATFILE_MAPPING = {
        "event_id": "event_id",
        "epicentral_distance_km": "repi",
        "hypocentral_distance_km": "rhypo",
        "fault_distance_km": "rjb",
        "rupture_distance_km": "rrup",
        "hw_xx_distance_km": "rx",
        "hw_yy_distance_km": "ry0",
        "focal_mechanism": "focal_mechanism",
        "network": "network",
        "vs30_m_s": "vs30",
        "vs30_m_s_from_topography": "vs30_topography",
        "vs30_m_s_from_vs_profile": "vs30_vs_profile",
        "station_code": "station_code",
        "station_name": "station_name",
        "low_cut_frequency_hz": "low_cut_freq",
        "high_cut_frequency_hz": "high_cut_freq",
        "filter_type": "filter_type",
        "filter_order": "filter_order",
        "class": "class",
        }

    def __init__(self, fname: str, calculate_response_spectrum: bool = True,
                 periods: Optional[np.ndarray] = None,
                 pseudo: bool = True,
                 verbose: bool = True):
        """
        """
        self.fname = fname
        self.verbose = verbose
        fle = h5py.File(self.fname, "r")
        self.events = {}
        self.parse_event_data(fle)
        self.stations = {}
        self.parse_station_metadata(fle)
        fle.close()
        self.events_stations = self.get_events_stations()
        self.data = None
        self.calculate_response_spectrum = calculate_response_spectrum
        self.periods = periods if periods is not None else DEFAULT_PERIODS
        self.pseudo = pseudo

    def parse_event_data(self, dstore):
        """Sorts the events into a dictionary
        """
        with io.BytesIO(dstore["QuakeML"][:].tobytes().strip()) as buf:
            catalog = read_events(buf, format="quakeml")
        if self.verbose:
            logging.info("%g events in file" % len(catalog))
        for eq in catalog:
            event_id = eq.resource_id.id.split("event_id=")[1]
            if event_id not in self.events:
                self.events[event_id] = eq
        return

    def get_events_stations(self):
        """Creates a mapping between each event and the available records per network
        """
        events_stations = {}
        fle = h5py.File(self.fname, "r")
        for event in list(self.events):
            events_stations[event] = {}
        for ntw_stn in list(self.stations):
            ntw, stn = ntw_stn.split(".")
            if ntw not in list(events_stations[event]):
                events_stations[event][ntw] = []

        for ntw_stn in list(self.stations):
            recs = list(fle["Waveforms/{:s}".format(ntw_stn)])
            ntw, stn = ntw_stn.split(".")
            for rec in recs:
                channel_code = rec.split("__")[0]
                if rec == "StationXML":
                    continue
                else:
                    evid = fle[
                        "Waveforms/{:s}/{:s}".format(ntw_stn, rec)
                        ].attrs["event_id"].decode().split("event_id=")[-1]
                    if (evid in list(events_stations)) and\
                            (ntw_stn not in events_stations[event]):
                        if channel_code[:-1] not in list(events_stations[event][ntw]):
                            events_stations[event][ntw].append(channel_code[:-1])
        fle.close()
        return events_stations

    def parse_station_metadata(self, dstore):
        """Sorts the station metadata from the XMLs into a dictionary
        """
        self.stations = {}
        self.st_lon = []
        self.st_lat = []
        self.st_elevation = []
        for stn in list(dstore["Waveforms"]):
            stn_data = dstore["Waveforms/{:s}/StationXML".format(stn)][:].tobytes().strip()
            with io.BytesIO(stn_data) as buf:
                station_data = read_inventory(buf, format="stationxml")
                channel = station_data.get_contents()["channels"][0]
                coords = station_data.get_coordinates(channel)

                self.stations[stn] = {
                    "lon": coords["longitude"],
                    "lat": coords["latitude"],
                    "elevation": coords["elevation"],
                    "local_depth": coords["local_depth"],
                    "inventory": station_data
                }
        return

    def get_record(self, event_id: str, network: str, station_code: str) -> Dict:
        """Parses the three waveform components for a given event, network and station returns
        them as a dictionary of Record objects nested by location and channel

        Args:
            event_id: The ID of the event
            network: Selected network of the record
            station_code: The three- or four-letter code of the station

        Returns:
            records: Dictionary of Record objects according to location and channel

        """
        if event_id not in self.events_stations:
            raise ValueError("No event with ID %s" % event_id)
        if network not in self.events_stations[event_id]:
            raise ValueError("No network %s in records for %s" % (network, event_id))
        ntw_stn = ".".join([network, station_code])
        record_codes = []
        for rec in self.events_stations[event_id][network]:
            if rec.startswith(ntw_stn):
                record_codes.append(rec)
        if not len(record_codes):
            raise ValueError("No record %s for event %s" % (ntw_stn, event_id))
        event = self.events[event_id]
        fle = h5py.File(self.fname)
        stn = fle["Waveforms/{:s}".format(ntw_stn)]
        waveforms = {}
        for rec in list(stn):
            if rec == "StationXML":
                # Ignore the station xml
                continue
            if stn[rec].attrs["event_id"].decode() == event.resource_id.id:
                # Is a ground motion file corresponding to this event
                stn_full, start_time, end_time, rec_id = tuple(rec.split("__"))
                rate = stn[rec].attrs["sampling_rate"]
                start_time_int = stn[rec].attrs["starttime"]
                nanosec_diff = int(start_time_int % 1E9)
                start_time = np.datetime64(int((start_time_int - nanosec_diff) / 1E9), "s") +\
                    np.timedelta64(nanosec_diff, "ns")
                timeseries = stn[rec][:]
                # Now get auxilliary information
                auxiliary_key = "AuxiliaryData/Headers/{:s}_{:s}/{:s}".format(network,
                                                                              station_code,
                                                                              rec_id)
                units = fle[auxiliary_key].attrs["units"]
                # Add the metadata
                metadata = dict([(key, val) for key, val in fle[auxiliary_key].attrs.items()])

                if (not self.calculate_response_spectrum) and \
                        "Spectra" in list(fle["AuxiliaryData"]):
                    # If the response spectrum is present then add this too
                    spectra_key = "AuxiliaryData/Spectra/{:s}_{:s}/{:s}".format(network,
                                                                                station_code,
                                                                                rec_id)

                    spectrum = ResponseSpectrum(spectrum=fle[spectra_key][1, :],
                                                periods=fle[spectra_key][0, :],
                                                units=units,
                                                damping=fle[spectra_key].attrs["damping"])
                else:
                    # Calculate the response spectrum directly from the time-series
                    periods = self.periods if self.periods is not None else DEFAULT_PERIODS
                    spectrum = ResponseSpectrum.from_timeseries(
                        acceleration=timeseries,
                        time_step=1.0 / rate,
                        units=units,
                        periods=periods,
                        pseudo=self.pseudo)

                _, _, locn, comp = tuple(stn_full.split("."))
                component = comp[-1]
                channel = comp[-3:-1]
                if locn not in waveforms:
                    waveforms[locn] = {channel: {}}
                elif channel not in waveforms[locn]:
                    waveforms[locn][channel] = {}
                else:
                    pass
                waveforms[locn][channel][component] = Waveform(event_id=event_id,
                                                               station=stn_full,
                                                               timeseries=timeseries,
                                                               sampling_rate=rate,
                                                               start_time=start_time,
                                                               units=units,
                                                               metadata=metadata,
                                                               response_spectrum=spectrum)
        record = {}
        for locn, channels in waveforms.items():
            for channel, wfs in channels.items():
                key = ".".join([network, station_code, locn, channel])
                record[key] = Record(event_id,
                                     network,
                                     station_code,
                                     locn,
                                     channel,
                                     waveforms=[w_f for comp, w_f in wfs.items()],
                                     units=units)
        return record

    def get_records(self) -> Dict:
        """Returns all the records in the file ordered by event ID and station code
        """
        records = {}
        for ev_id in self.events_stations:
            for network in self.events_stations[ev_id]:
                for station in self.events_stations[ev_id][network]:
                    ntw, station_code, locn, channel = tuple(station.split("."))
                    records["{:s}|{:s}".format(ev_id, station)] = self.get_record(
                        ev_id,
                        network,
                        station_code
                        )
        return records

    def __iter__(self):
        """Iterator alternative to get_records, return the event_id, the station code
        and the corresponding record
        """
        for ev_id in self.events_stations:
            for network in self.events_stations[ev_id]:
                for station in self.events_stations[ev_id][network]:
                    ntw, station_code, locn, channel = tuple(station.split("."))
                    yield ev_id, station, self.get_record(ev_id, network, station_code)
        return

    @staticmethod
    def get_preferred_magnitude(event):
        """
        """
        magnitudes = {}
        for mag in event.magnitudes:
            type_author = "{:s}|{:s}".format(mag.magnitude_type.upper(),
                                             mag.creation_info["author"])
            magnitudes[mag.magnitude_type.upper()] = (mag.mag, type_author)

        if "MW" in list(magnitudes):
            mag = magnitudes["MW"][0]
            mtype, mauthor = magnitudes["MW"][1].split("|")
        elif "MS" in list(magnitudes):
            mag = magnitudes["MS"][0]
            mtype, mauthor = magnitudes["MS"][1].split("|")
        else:
            key = list(magnitudes)[0]
            mag = magnitudes[key][0]
            mtype, mauthor = magnitudes[key][1].split("|")
        return mag, mtype, mauthor

#    def build_egsim_parametric_table(
#            self,
#            horizontal_component: str,
#            periods: Optional[np.ndarray] = None,
#            accel_units: str = "g"
#        ) -> pd.DataFrame:
#        """
#        """
#        table_entries = []
#        #for event_id in list(self.events):
#        #    for stn in list(self.stations):
#        #        ntw, stn_code = stn.split(".")
#        for event_id in self.events_stations:
#            for ntw in self.events_stations[event_id]:
#                for stn_code in self.events_stations[event_id][ntw]:
#                    try:
#                        waveforms, event, station = self.get_record(event_id, ntw, stn_code)
#                    except ValueError as ve:
#                        logging.info("Event %s  Station: %s - No value (%s)"
#                                     % (event_id, stn, str(ve)))
#                        continue
#                    components = list(waveforms)
#                    if not len(components):
#                        # No waveform for this station and event
#                        print("No components for event %s station %s!"
#                              % (event_id, stn))
#                        #raise ValueError("No components for event %s station %s!"
#                        #                 % (event_id, stn))
#                        continue
#                    record_id = "|".join([event_id,
#                                          "{:s}-{:s}".format(ntw, stn_code),
#                                          components[0][:2]])
#                    logging.info("Processing Record %s" % record_id)
#                    origin = event.origins[0]
#                    mag, mtype, mauthor = self.get_preferred_magnitude(event)
#                    metadata = waveforms[components[0]].metadata
#                    units = metadata["units"]
#                    table_entry = {
#                        "gmid": record_id,
#                        "event_id": event_id,
#                        "event_time": str(origin.time),
#                        "event_longitude": origin.longitude,
#                        "event_latitude": origin.latitude,
#                        "event_depth": origin.depth,
#                        "mag": mag,
#                        "mag_type": mtype,
#                        "mag_source": mauthor,
#                        "SoF": metadata["focal_mechanism"],
#                        "repi": metadata["epicentral_distance_km"],
#                        "rhypo": metadata["hypocentral_distance_km"],
#                        "rjb": np.nan,
#                        "rrup": metadata["fault_distance_km"],
#                        "rx": metadata["hw_xx_distance_km"],
#                        "ry0": metadata["hw_yy_distance_km"],
#                        "network": metadata["network"],
#                        "station_code": metadata["station_code"],
#                        "station_longitude": station["lon"],
#                        "station_latitude": station["lat"],
#                        "vs30_m_s": metadata["vs30_m_s"],
#                        "vs30_geology": metadata["vs30_m_s_from_geology"],
#                        "vs30_topo": metadata["vs30_m_s_from_topography"],
#                        "vs30_profile": metadata["vs30_m_s_from_vs_profile"],
#                        }
#
#                    # Add on the spectra (geometric mean)
#                    spectra, pga, pgv, pgd, out_periods = self.get_horizontal_intensity_measures(
#                        waveforms, horizontal_component, accel_units, periods)
#                    table_entry["PGA"] = pga
#                    table_entry["PGV"] = pgv
#                    table_entry["PGD"] = pgd
#                    for period, sa in zip(out_periods, spectra):
#                        table_entry["SA({:.3f})".format(period)] = sa
#                    table_entries.append(table_entry)
#
#        return pd.DataFrame(table_entries)
#
#    @staticmethod
#    def get_horizontal_intensity_measures(
#            waveforms: Dict,
#            horizontal_component: str,
#            accel_units: str,
#            periods: Optional[np.ndarray] = None
#        ):
#        """
#        """
#        if accel_units == "g":
#            conv = 1.0 / (100.0 * g)
#        elif accel_units in ("m/s/s", "m/s^2"):
#            conv = 1.0 / g
#        else:
#            conv = 1.0
#        spectra = []
#        pga = []
#        pgv = []
#        pgd = []
#        horizontal_waveforms = []
#        for key in waveforms:
#            if not key.endswith("Z"):
#                horizontal_waveforms.append(waveforms[key])
#        if horizontal_waveforms[0].response_spectrum is not None:
#            periods_1 = horizontal_waveforms[0].response_spectrum.periods
#        else:
#            periods_1 = []
#        if horizontal_waveforms[1].response_spectrum is not None:
#            periods_2 = horizontal_waveforms[1].response_spectrum.periods
#        else:
#            periods_2 = []
#        if periods is None:
#            if len(periods_1) and len(periods_2) and (len(periods_1) == len(periods_2)):
#                periods = np.array(periods_1)
#            else:
#                raise ValueError("No periods specfies and the pre-calculated periods of "
#                                 "the two horizontal spectra are either not specified or not"
#                                 " the same!")
#        sah, pga, pgv, pgd, periods = get_horizontal_spectrum(
#            horizontal_waveforms[0],
#            horizontal_waveforms[1],
#            periods,
#            horizontal_component)
#        pga *= conv
#        sah *= conv
#        return sah, pga, pgv, pgd, periods
