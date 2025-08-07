import os
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Union
import logging
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from dynamicgmm.process.base import (
    DEFAULT_FREQUENCIES,
    DEFAULT_PERIODS,
    get_im_set_from_record
)
from dynamicgmm.process import asdf


logging.basicConfig(level=logging.INFO)


RESP_IMS = ("geometric", "envelope", "larger pga", "random")


def ims_to_array_set(
        im_record_set: Dict,
        im_config: Dict,
        periods: np.ndarray,
        frequencies: np.ndarray,
        grp: Optional[h5py.Group] = None
):
    """Converts a set of intensity measures from mutiple records into single arrays and
    (optionally) stores these to hdf5

    Args:
        im_record_set: Dictionary containing the intensity measures for each record
        im_config: Dictionary containing the configuration options for the intensity measures
        periods: Periods (s) used for the response spectra
        frequencies: Frequencies used for the Fourier spectra
        grp: h5py.Group object to store the arrays

    Returns:
        None (if storing to file) or a dictionary containing the full arrays per intensity
        measure
    """
    record_ids = np.array(list(im_record_set), dtype=np.str_)
    nrec = len(im_record_set)
    nper = len(periods) + 2
    nfreq = len(frequencies)
    response_spectra_ims = []
    fas_ims = []
    scalar_ims = []
    output = {
        "record_id": record_ids.astype(bytes),
        "periods": periods,
        "frequencies": frequencies
    }
    for i_m in im_config:
        if i_m in RESP_IMS or i_m.lower().startswith("rotd"):
            output[i_m] = np.zeros([nrec, nper])
            response_spectra_ims.append(i_m)
        elif i_m.upper().startswith("EAS"):
            output[i_m] = np.zeros([nrec, nfreq])
            fas_ims.append(i_m)
        else:
            scalar_ims.append(i_m)
    scalar_dtypes = np.dtype([(key, float) for key in scalar_ims])
    output["scalar_ims"] = np.zeros(nrec, scalar_dtypes)
    for i, (record_id, im_set) in enumerate(im_record_set.items()):
        for i_m, values in im_set.items():
            if i_m in response_spectra_ims:
                output[i_m][i, 0] = values["PGV"]
                output[i_m][i, 1] = values["PGA"]
                output[i_m][i, 2:] = values["SA"]
            elif i_m in fas_ims:
                output[i_m][i, :] = values["EAS"]
            elif i_m in scalar_ims:
                output["scalar_ims"][i_m][i] = values
            else:
                pass
    if grp:
        # Add to a group object
        for key, data in output.items():
            if key == "record_id":
                rid_dset = grp.create_dataset(
                    key, data.shape,
                    dtype=h5py.string_dtype()
                )
                rid_dset[:] = data
            else:
                dset = grp.create_dataset(key, data.shape, dtype=data.dtype)
                dset[:] = data
                if key == "EAS_smoothed":
                    # For the smoothed EAS then store the config parameters controlling
                    # the smoothing
                    dset.attrs["bandwidth"] = \
                        im_config[key]["konno_ohmachi_kwargs"].get("bandwidth", 40)
                    dset.attrs["normalize"] = \
                        im_config[key]["konno_ohmachi_kwargs"].get("normalize", False)
                elif (key in im_config) and im_config[key]:
                    # If there are further configuration parameters for the data then store
                    # these
                    for subkey, val in im_config[key].items():
                        if subkey == "limits":
                            # Is a tuple of (lowe, upper), so store separately
                            dset.attrs["lower_limit"] = val[0]
                            dset.attrs["upper_limit"] = val[1]
                        else:
                            dset.attrs[subkey] = val
                else:
                    pass

        return
    else:
        return output


class DatastoreByEvent():
    """Build the datastore with records grouped by event

    Attributes:
        dbname: Name of the file for the datastore
        data_provider: Name of the data provider
        verbose: If True then report details of the data processing steps
    """
    def __init__(
        self,
        dbname: str,
        data_provider: str,
        verbose: bool = True,
    ):
        """
        """
        self.dbname = dbname
        self.db = None
        self.data_provider = data_provider
        self.verbose = verbose

    def add_events(
        self,
        fnames: List,
        intensity_measure_config: Dict,
        periods: Optional[np.ndarray] = DEFAULT_PERIODS,
        frequencies: Optional[np.ndarray] = DEFAULT_FREQUENCIES,
        response_spectrum_units: Optional[str] = "cm/s/s",
        fas_units: Optional[str] = "cms/s/s",
        significant_duration_definition: Optional[Tuple] = (0.05, 0.95),
        cav_threshold: Optional[float] = 0.0,
        damping: Optional[float] = 0.05,
        num_proc: Optional[int] = None,
        verbose: bool = True
    ):
        """Adds data from a set of events, each event a single ASDF file
        containing the records from multiple stations

        Args:
            fnames: List of paths to the event files
            intensity_measure_config: Dictionary containing the required intensity measures as
                                      keys and, where necessary, additional configuration
                                      parameters for the items, e.g.
                {"geometric: {},
                    "RotD50": {},
                 "EAS_smoothed": {"konno_ohmachi_kwargs": {"bandwidth": 40,
                                                           "normalize": True}},
                 "
                 }
            periods: Numpy array of periods for response spectra (takes ESM defaults if not
                     provided)
            frequencies: Numpy array of frequencies
            response_spectrum_units: Units of acceleration for the response spectra
            fas_units: Units of acceleration for the Fourier spectra IMs
            significant_duration_definition: Tuple containing fractions of total Arias
                                             Intensity used to define the significant duration
            cav_threshold: Threshold acceleration (g) to calculate CAV
            damping: Fractional damping for response spectra [0, 1]
            num_proc: Number of processors to use for response spectra calculations
        """
        for fname in fnames:
            if not os.path.exists(fname):
                logging.info("File %s not found - skipping!" % fname)
                continue

            if fname.endswith("hdf5") or fname.endswith("asdf"):
                # Use the ASDF parser
                # Get the event metadata
                if self.verbose:
                    logging.info("Processing records from file %s" % fname)
                handler = asdf.ASDFEventHandler(fname)
                # Extracting the intensity measures
                self.intensity_measures_from_asdf(
                    handler,
                    intensity_measure_config,
                    periods,
                    frequencies,
                    response_spectrum_units=response_spectrum_units,
                    fas_units=fas_units,
                    significant_duration_definition=significant_duration_definition,
                    cav_threshold=cav_threshold,
                    damping=damping,
                    num_proc=num_proc
                )
                event_metadata = self.event_metadata_from_asdf(handler, verbose=self.verbose)
                event_id = list(handler.events)[0]
                with pd.HDFStore(self.dbname, "a") as store:
                    store.put("events/{:s}/{:s}/metadata".format(event_id, self.data_provider),
                              event_metadata)
                # Get the waveform data
            elif fname.endswith("mseed"):
                # miniseed parser
                raise NotImplementedError("mseed not yet supported")
            else:
                raise ValueError("File type %s not supported (%s)"
                                 % (os.path.splitext(fname)[-1], fname))
        return

    @staticmethod
    def event_metadata_from_asdf(
        handler: asdf.ASDFEventHandler,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Extract the event metadata from ASDF (usually Obspy objects) and return as
        a pandas Dataframe
        """
        # handler = asdf.ASDFEventHandler(fname, verbose=verbose)
        metadata = []
        for event_id, station, records in handler:
            # pref_origin_id = handler.events[event_id].preferred_origin_id
            # pref_mag_id = handler.events[event_id].preferred_magnitude_id
            pref_origin = handler.events[event_id].preferred_origin()
            pref_mag = handler.events[event_id].preferred_magnitude()
            event_metadata = {
                "event_id": event_id,
                "event_time": str(pref_origin.time),
                "event_longitude": pref_origin.longitude,
                "event_latitude":  pref_origin.latitude,
                "event_hypo_depth": pref_origin.depth,
                "event_origin_author": pref_origin.creation_info.author,
                "event_preferred_mag": pref_mag.mag,
                "event_preferred_mag_type": pref_mag.magnitude_type,
                "event_preferred_mag_author": pref_mag.creation_info.author
            }
            for rec_id, record in records.items():
                record_metadata = deepcopy(event_metadata)
                record_metadata["network"] = record.network
                record_metadata["station"] = record.station
                record_metadata["location"] = record.location
                record_metadata["channel"] = record.channel
                ntw_stn = ".".join([record.network, record.station])
                record_metadata["station_id"] = ".".join(
                    [record.network, record.station, record.location, record.channel]
                )
                record_metadata["station_longitude"] = handler.stations[ntw_stn]["lon"]
                record_metadata["station_latitude"] = handler.stations[ntw_stn]["lat"]
                record_metadata["station_elevation"] = handler.stations[ntw_stn]["elevation"]
                record_metadata["station_depth"] = handler.stations[ntw_stn]["local_depth"]
                record_metadata["station_id"] = ".".join(
                    [record.network, record.station, record.location, record.channel]
                )
                if record.h1.metadata is not None:
                    for key_in, key_out in handler.FLATFILE_MAPPING.items():
                        record_metadata[key_out] = record.h1.metadata[key_in]
                record_metadata["record_id"] = "|".join([event_id,
                                                         record_metadata["station_id"]])
                metadata.append(record_metadata)
        metadata = pd.DataFrame(metadata)
        if metadata.duplicated("record_id").any():
            metadata.drop_duplicates("record_id", inplace=True, ignore_index=True)
        return metadata

    def intensity_measures_from_asdf(
            self,
            handler: asdf.ASDFEventHandler,
            intensity_measure_config: Dict,
            periods: Optional[np.ndarray] = DEFAULT_PERIODS,
            frequencies: Optional[np.ndarray] = DEFAULT_FREQUENCIES,
            response_spectrum_units: Optional[str] = "cm/s/s",
            fas_units: Optional[str] = "cms/s/s",
            significant_duration_definition: Optional[Tuple] = (0.05, 0.95),
            cav_threshold: Optional[float] = 0.0,
            damping: Optional[float] = 0.05,
            num_proc: Optional[int] = None,
    ):
        """Calculates the intensity measures and metadata and then stores these to an hdf5 file
        """
        db = h5py.File(self.dbname, "a")
        # Data stored in the group events
        if "events" not in list(db):
            events_group = db.create_group("events")
        else:
            events_group = db["events"]
        intensity_measures = {}
        for ev_id, station, record in handler:
            for rec_id, rec in record.items():
                if rec_id in intensity_measures:
                    continue
                if self.verbose:
                    logging.info(".... Processing record: {:s}|{:s}".format(ev_id, rec_id))
                intensity_measures[rec_id] = get_im_set_from_record(
                    rec,
                    intensity_measure_config,
                    periods,
                    frequencies,
                    response_spectrum_units=response_spectrum_units,
                    fas_units=fas_units,
                    significant_duration_definition=significant_duration_definition,
                    cav_threshold=cav_threshold,
                    damping=damping,
                    num_proc=num_proc
                )
        # Join together the intensity measure set and store to hdf5
        if ev_id not in list(events_group):
            event_group = events_group.create_group(ev_id)
        else:
            event_group = events_group[ev_id]

        if self.data_provider not in list(event_group):
            provider_group = event_group.create_group(self.data_provider)
        else:
            provider_group = event_group[self.data_provider]
        ims_to_array_set(intensity_measures,
                         intensity_measure_config,
                         periods, frequencies,
                         provider_group)
        if self.verbose:
            logging.info(
                ".... Stored to database entry: /events/{:s}/{:s}".format(
                    ev_id, self.data_provider
                )
            )
        db.close()
        return

    def build_flatfile(
            self,
            spectra_types: List,
            data_sources: List,
            output_dir: Optional[str] = None,
            periods: Union[List, np.ndarray] = DEFAULT_PERIODS,
            frequencies: Union[List, np.ndarray] = DEFAULT_FREQUENCIES
    ):
        """
        From the metadata and ground motion values in the database this function
        combines them into a single flatfile for each intensity measure type. If an output
        directory is specified then these are exported to the directory

        Args:
            spectra_types: List of intensity measure spectral measures (e.g. geometric,
                           RotD50 etc.)
            data_sources: List of data sources to choose from (e.g. ESM, RRSM)
            output_dir: Path to output directory to export the flatfiles
            periods: List of periods (s) for the response spectra metrics (will interpolate to
                     target periods if different from the stored periods in the database)
            frequencies: List of frequencies (Hz) for the Fourier spectra metrics (will
                         interpolate to target periods if different from the stored periods in
                         the database)
        """
        fle = h5py.File(self.dbname, "r")
        assert "events" in list(fle), "No events in database file %s" % self.dbname
        event_ids = list(fle["events"])
        all_metadata = []
        all_record_ids = []
        ims = dict([(key, []) for key in spectra_types])
        ims["scalar"] = []
        sa_headers = ["PGV", "PGA"] + ["{:.5f}".format(per) for per in periods]
        fas_headers = ["{:.5f}".format(freq) for freq in frequencies]
        for ev_id in event_ids:
            for dsrc in data_sources:
                if dsrc in list(fle["events/{:s}".format(ev_id)]):
                    path_key = "events/{:s}/{:s}/".format(ev_id, dsrc)
                    station_id = fle[f"{path_key}/record_id"][:].astype(str)
                    record_id = pd.Series(["{:s}|{:s}".format(ev_id, sid)
                                           for sid in station_id])
                    # Read the metadata into a dataframe
                    metadata = pd.read_hdf(self.dbname, key=f"{path_key}/metadata")
                    metadata["data_source"] = [dsrc] * metadata.shape[0]
                    all_metadata.append(metadata)
                    # Read the record IDs
                    all_record_ids.append(record_id)
                    for im_type in spectra_types:
                        if im_type not in list(fle[path_key]):
                            # Spectra type not found - skipping
                            continue
                        gmvs = fle[f"{path_key}/{im_type}"][:]
                        if im_type.startswith("EAS") or im_type.startswith("FAS"):
                            input_xvals = fle[f"{path_key}/frequencies"][:]
                            if not np.allclose(input_xvals, frequencies):
                                # Interpolate to target values
                                spl = make_interp_spline(np.log10(input_xvals),
                                                         np.log10(gmvs),
                                                         k=1, axis=1)
                                gmvs = 10.0 ** (spl(np.log10(frequencies)))
                            df = pd.DataFrame(gmvs, columns=fas_headers,
                                              index=record_id)
                            df["sid"] = station_id
                        else:
                            input_xvals = fle[f"{path_key}/periods"][:]
                            if not np.allclose(input_xvals, periods):
                                # Interpolate to target values
                                spl = make_interp_spline(np.log10(input_xvals),
                                                         np.log10(gmvs[:, 2:]),
                                                         k=1, axis=1)
                                gmvs[:, 2:] = 10.0 ** (spl(np.log10(periods)))
                            df = pd.DataFrame(gmvs, columns=sa_headers,
                                              index=record_id)
                            df["sid"] = station_id
                        ims[im_type].append(df)
                    if "scalar_ims" in list(fle[path_key]):
                        scalar_ims = fle[f"{path_key}/scalar_ims"][:]
                        scalar_headers = scalar_ims.dtype.names
                        ims["scalar"].append(pd.DataFrame(fle[f"{path_key}/scalar_ims"][:],
                                                          columns=scalar_headers,
                                                          index=record_id))
        fle.close()
        # Concatenate into single arrays
        metadata = pd.concat(all_metadata, axis=0, ignore_index=True)
        record_ids = pd.concat(all_record_ids, axis=0, ignore_index=True)
        metadata.set_index(record_ids, inplace=True, drop=True)
        for key in ims:
            ims[key] = pd.concat(ims[key], axis=0)
            ims[key] = pd.concat([metadata, ims[key]], axis=1)
        if output_dir:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for key in ims:
                fname = os.path.join(output_dir, f"flatfile_{key}.csv")
                ims[key].to_csv(fname, sep=",", index_label="wfid")
                logging.info("Flatfile for IMs %s written to %s" % (key, fname))
            return
        return metadata, ims
