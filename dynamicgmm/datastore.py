import os
import io
from copy import deepcopy
import datetime
import tomllib
import pathlib
from typing import List, Dict, Tuple, Optional, Union
import logging
from multiprocessing import cpu_count
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from obspy import read_events
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
        grp: Optional[h5py.Group] = None,
        append: bool = False
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
        if "record_id" in list(grp):
            # Append to an existing data set
            for key, data in output.items():
                if key in ["frequencies", "periods",]:
                    # Static: don't touch these here
                    continue
                else:
                    # Resize the dataset with the new data
                    orig_size = grp[key].shape[0]
                    grp[key].resize(orig_size + data.shape[0], axis=0)
                    grp[key][orig_size:] = data
        else:
            # Add to a clean, empty group object
            for key, data in output.items():
                if key in ["frequencies", "periods"]:
                    # Not resizable
                    dset = grp.create_dataset(key, data.shape, dtype=data.dtype)
                    dset[:] = data
                elif key == "record_id":
                    # Can be re-sizable but takes string datatype
                    rid_dset = grp.create_dataset(
                        key, data.shape,
                        maxshape=(None,),
                        chunks=True,
                        dtype=h5py.string_dtype()
                    )
                    rid_dset[:] = data
                elif key == "scalar_ims":
                    # Single dimension but structured array, resizable
                    dset = grp.create_dataset(key, data.shape,
                                              maxshape=(None,),
                                              chunks=True,
                                              dtype=data.dtype)
                    dset[:] = data
                else:
                    # Spectra (2D array), resizable along 0 axis
                    dset = grp.create_dataset(key, data.shape,
                                              maxshape=(None, data.shape[1]),
                                              chunks=True,
                                              dtype=data.dtype)
                    dset[:] = data
                    # Add on attributes
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


# For storing strings in metadata this determines a fixed itemsize
# This avoids pandas.HDFStore raising an error if appending a new metadata
# table to an existing one but one of the text columns in the new table has
# an entry longer than the maximum entry of the original table
METADATA_MIN_ITEMSIZE = {
    "event_id": 60,
    "event_time": 30,
    "event_origin_author": 80,
    "event_preferred_mag_type": 12,
    "event_preferred_mag_author": 80,
    "network": 6,
    "station": 8,
    "channel": 3,
    "station_id": 16,
    "station_code": 10,
    "station_name": 120,
    "filter_type": 20,
    "class": 10,
    "record_id": 76
}


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
        fas_units: Optional[str] = "cm/s/s",
        significant_duration_definition: Optional[Tuple] = (0.05, 0.95),
        cav_threshold: Optional[float] = 0.0,
        damping: Optional[float] = 0.05,
        num_proc: Optional[int] = None,
        verbose: bool = True,
        skip_existing: bool = False
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
                if skip_existing:
                    event_id = asdf.extract_event_ids(fname)[0]
                    fle = h5py.File(fname, "r")
                    if ("events" in list(fle)) and (event_id in list(fle["events"])):
                        # Event is already in the file
                        logging.info(f"Event {event_id} already in datastore - skipping")
                        fle.close()
                        continue

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
                with pd.HDFStore(self.dbname, mode="a") as store:
                    key = f"events/{event_id}/{self.data_provider}/metadata"
                    if key in store:
                        # Existing metadata for this event, so concatenate the old and new
                        # dataframes
                        store.append(key, event_metadata, format="table", index=False)
                    else:
                        store.append(key, event_metadata, format="table", index=False,
                                     min_itemsize=METADATA_MIN_ITEMSIZE)
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
        # Data stored in the group 'events'
        if "events" not in list(db):
            # Clean file, no events groupd created
            events_group = db.create_group("events")
        else:
            events_group = db["events"]
        intensity_measures = {}
        for ev_id, station, record in handler:
            # Join together the intensity measure set and store to hdf5
            if ev_id not in list(events_group):
                # Event not in file yet, create the groups for the event and the data provider
                event_group = events_group.create_group(ev_id)
                provider_group = event_group.create_group(self.data_provider)
                # No record IDS yet
                rec_ids = []
            else:
                # Event is already in the group, add on info
                event_group = events_group[ev_id]
                provider_group = events_group[f"{ev_id}/{self.data_provider}"]
                if "record_id" in list(provider_group):
                    rec_ids = provider_group["record_id"][:].tolist()
                else:
                    rec_ids = []

            for rec_id, rec in record.items():
                if rec_id in rec_ids:
                    # Record is already in the data file, skip
                    continue
                if rec_id in intensity_measures:
                    # Record is alreadty in the IMS set
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


def parse_ims_config_from_toml(
    ims_dict: Dict
) -> Tuple[Dict, str, str, np.ndarray, np.ndarray, float]:
    """Parse the intensity measures from the TOML representation
    to the IM config required for the processing, filling in
    defaults where necessary

    Args:
        ims_dict: Dictionary of the intensity measure config (from the TOML)

    Returns:
        ims_config: Updated configuration parameters
        sa_units: Units for the spectral acceleration
        fas_units: Units for the Fourier Amplitude Spectrum
        periods: Numpy array of target periods
        frequencies: Numpy array of target frequencies
        damping: Damping (%) for the response spectrum acceleration
    """
    ims_config = {}
    sa_dict = ims_dict.pop("SA", {})
    if sa_dict:
        sa_units = sa_dict.pop("units", "cm/s/s")
        periods = sa_dict.pop("periods", DEFAULT_PERIODS)
        damping = sa_dict.pop("damping", 0.05)
        if not len(periods):
            # If empty list defined then use defaults
            periods = DEFAULT_PERIODS
        for key, val in sa_dict.items():
            ims_config[key] = val if val else {}
    else:
        sa_units = "cm/s/s"
        periods = DEFAULT_PERIODS
        damping = 0.05
    fas_dict = ims_dict.pop("FAS", {})
    if fas_dict:
        fas_units = fas_dict.pop("units", "cm/s/s")
        frequencies = fas_dict.pop("frequencies", DEFAULT_FREQUENCIES)
        if not len(frequencies):
            frequencies = DEFAULT_FREQUENCIES
        for key, val in fas_dict.items():
            ims_config[key] = val if val else {}
    else:
        fas_units = "cm/s/s"
        frequencies = DEFAULT_FREQUENCIES
    # Any other IMS
    for key, val in ims_dict.items():
        ims_config[key] = val if val else {}
    return ims_config, sa_units, fas_units, periods, frequencies, damping


def get_file_list(data_folder: pathlib.Path, sort: bool = True) -> List:
    """Get the list of files within all subdirectories of the target data folder

    Args:
        data_folder: Path to find the ground motion data
        sort: Sort the files into ascending order
    Returns:
        The list of files for analysis
    """
    assert data_folder.exists(), f"{str(data_folder)} not found"
    file_list = []
    for root, dirs, files in data_folder.walk():
        for fname in files:
            if fname.endswith(".hdf5") or fname.endswith(".asdf"):
                file_list.append(str(root / fname))
    if sort:
        file_list.sort()
    return file_list


def event_within_times_asdf(
        fname: str,
        starttime: Optional[datetime] = datetime.datetime.min,
        endtime: Optional[datetime] = datetime.datetime.max
) -> bool:
    """
    For each event file check if the event falls within the selected start and
    end time if defined

    Args:
        fname: Path to the event record ASDF
        starttime: Earliest time for selection window
        endtime: Latest time for selection window

    Returns:
        Event is within the time window (True) or not (False)
    """
    with h5py.File(fname, "r") as dstore:
        with io.BytesIO(dstore["QuakeML"][:].tobytes().strip()) as buf:
            catalog = read_events(buf, format="quakeml")
    event_in_times = []
    for ev in catalog:
        event_time = ev.preferred_origin().time.datetime
        event_in_times.append((event_time >= starttime) & (event_time <= endtime))
    return any(event_in_times)


class GMDataProcessor:
    """Class to setup and execute ground motion data processing from a
    user-specfied configuration
    """
    def __init__(
        self,
        data_folder: str,
        data_store: str,
        data_sources: List,
        ims_config: Dict,
        flatfile_directory: Optional[str] = None,
        flatfile_config: Optional[Dict] = None,
        clean: bool = True,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        skip_existing: bool = True,
        periods: Optional[Union[List, np.ndarray]] = DEFAULT_PERIODS,
        frequencies: Optional[Union[List, np.ndarray]] = DEFAULT_FREQUENCIES,
        response_spectrum_units: str = "cm/s/s",
        fas_units: str = "cm/s/s",
        sa_damping: float = 0.05,
        number_processes: Optional[int] = None
    ):
        """
        """
        self.data_folder = pathlib.Path(data_folder)
        if not self.data_folder.exists():
            raise OSError(f"Data folder {data_folder} not found!")
        self.data_store = pathlib.Path(data_store)
        if clean and self.data_store.exists():
            logging.info(f"Removing existing data store {str(self.data_store)}")
            self.data_store.unlink()
        self.data_sources = data_sources
        self.ims_config = ims_config
        self.flatfile_directory = pathlib.Path(flatfile_directory)
        self.flatfile_config = flatfile_config
        self.starttime = start
        self.endtime = end
        self.skip_existing = skip_existing
        self.periods = periods
        self.frequencies = frequencies
        self.response_spectrum_units = response_spectrum_units
        self.fas_units = fas_units
        self.sa_damping = sa_damping
        self.number_processes = number_processes if number_processes else cpu_count()
        logging.info(str(self))

    def __repr__(self):
        dsource_string = ", ".join(self.data_sources)
        return f"Event-by-Event Data processor for {str(self.data_folder)}" +\
            f"\n---- Data sources: {dsource_string}"

    @classmethod
    def from_toml_config(cls, config_file: str, clean: bool = True):
        """Instantiate the class from a TOML config file

        Args:
            config_file: Path to the TOML config
            clean: Removes an existing datastore
        """
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
        ims_config, sa_units, fas_units, periods, frequencies, sa_damping = \
            parse_ims_config_from_toml(config["process"]["ims"])
        # Flatfile config
        flatfile_config = config["process"].get("flatfile", {})
        if flatfile_config:
            flatfile_directory = flatfile_config.pop("flatfile_directory", None)
            flatfile_config["periods"] = flatfile_config.get("periods", periods)
            flatfile_config["frequencies"] = flatfile_config.get("frequencies", frequencies)
        else:
            flatfile_directory = None
            flatfile_config = None
        return cls(
            data_folder=config["process"]["data_folder"],
            data_store=config["process"]["data_store"],
            data_sources=config["process"]["data_sources"],
            ims_config=ims_config,
            flatfile_directory=flatfile_directory,
            flatfile_config=flatfile_config,
            clean=clean,
            start=config["process"].get("start", None),
            end=config["process"].get("end", None),
            skip_existing=config["process"].get("skip_existing", True),
            periods=periods,
            frequencies=frequencies,
            response_spectrum_units=sa_units,
            fas_units=fas_units,
            sa_damping=sa_damping,
            number_processes=config["process"].get("number_processes", None)
        )

    def run(self, verbose: bool = True):
        """Run the data processing

        Args:
            verbose: Provide event-by-event logging
        """

        initial_file_list = get_file_list(self.data_folder, sort=True)
        logging.info(f"Found {len(initial_file_list)} files in {str(self.data_folder)}")
        if self.starttime or self.endtime:
            # Can filter to just the subset of files within the time limit
            file_list = []
            for fname in initial_file_list:
                if event_within_times_asdf(fname, self.starttime, self.endtime):
                    file_list.append(fname)
        else:
            file_list = initial_file_list
        for provider in self.data_sources:
            logging.info(f"Adding events from data source: {provider}")
            dstore = DatastoreByEvent(str(self.data_store), provider, verbose)
            dstore.add_events(file_list,
                              self.ims_config,
                              self.periods,
                              self.frequencies,
                              response_spectrum_units=self.response_spectrum_units,
                              fas_units=self.fas_units,
                              damping=self.sa_damping,
                              num_proc=self.number_processes,
                              skip_existing=self.skip_existing,
                              )
        logging.info(f"All events added to {str(self.data_store)}")
        if self.flatfile_directory:
            logging.info("Building flatfiles")
            # Build the flatfile
            dstore.build_flatfile(
                self.flatfile_config["horizontal_definitions"],
                data_sources=self.data_sources,
                output_dir=str(self.flatfile_directory),
                periods=self.flatfile_config["periods"],
                frequencies=self.flatfile_config["frequencies"]
            )
        logging.info(f"Saved flatfiles in {str(self.flatfile_directory)}")
        return
