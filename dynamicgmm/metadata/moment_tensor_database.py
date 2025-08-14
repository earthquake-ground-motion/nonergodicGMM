import os
import json
import urllib
import logging
from copy import deepcopy
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import obspy
from obspy.core.event import (
    Catalog, Event, Origin, Magnitude, ResourceIdentifier, CreationInfo,
    NodalPlane, NodalPlanes, FocalMechanism, MomentTensor, Tensor, PrincipalAxes, Axis
)
from openquake.hazardlib.geo import geodetic


logging.basicConfig(level=logging.INFO)


M0_TO_MW = lambda m0: (2.0 / 3.0) * (np.log10(m0) - 9.05)
MW_TO_M0 = lambda mw: 10.0 ** ((1.5 * mw) + 9.05)


ISC_WEBSERVICES_BASE_URL = "http://www.isc.ac.uk/fdsnws/event/1/query?"


def build_emsc_mtwebservice_url(config: Dict) -> str:
    """
    Builds an EMSC (SeismicPortal) moment tensor webservice FDSN query url from a config
    """
    url_base = "https://seismicportal.eu/mtws/api/search?"
    url_options = []
    for key, val in config.items():
        if key in ("minmag", "maxmag", "minlon", "maxlon", "minlat", "maxlat"):
            url_options.append("{:s}={:.3f}".format(key, val))
        elif key in ("limit",):
            url_options.append("{:s}={:g}".format(key, val))
        else:
            url_options.append("{:s}={:s}".format(key, val))
        query_url = url_base + "&".join(url_options)
    return query_url


def get_emsc_moment_tensors_by_year(start_year: int, end_year: int, config: Dict) -> Dict:
    """
    Creates a dictionary of moment tensor queries on a year-by-year bases
    """
    catalogues = {}
    years = np.arange(start_year, end_year, 1)
    for year in years:
        year_config = deepcopy(config)
        year_config["starttime"] = "{:g}-01-01T00:00:000".format(year)
        year_config["endtime"] = "{:g}-01-01T00:00:00".format(year + 1)
        query_url = build_emsc_mtwebservice_url(year_config)
        logging.info(query_url)
        if config["format"] == "quakeml":
            catalogues[year] = obspy.read_events(query_url, format="quakeml")
        elif config["format"] == "json":
            req = urllib.request.Request(query_url)
            raw_data = urllib.request.urlopen(req)
            catalogues[year] = json.load(raw_data)
        else:
            raise ValueError("File format %s not supported" % config["format"])
        logging.info("Found %g events" % len(catalogues[year]))
    return catalogues


class MomentTensorDatabase():
    """

    """
    def __init__(self, identifier: str, events: Optional[List] = None,
                 data_source: Optional[str] = None, comments: Optional[str] = None):
        """
        """
        self.id = identifier
        self.events = events if events is not None else []
        self.data_source = data_source
        self.comments = comments
        self.event_ids = [event.resource_id.id for event in self.events]
        self._dataframe = None

    def __repr__(self):
        return "Moment Tensor Database %s (%g events)" % (self.id, len(self))

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        for ev in self.events:
            yield ev
        return

    def __getitem__(self, key: Union[int, str]):
        """Returns the event information accoding to the event IDs or the position
        """
        if isinstance(key, str):
            # Retreive according to the key
            assert key in self.event_ids, "No event with ID %s found" % key
            i = self.event_ids.index(key)
            return self.events[i]
        else:
            # Retreive by position
            return self.events[key]

    @property
    def dataframe(self):
        """
        """
        if isinstance(self._dataframe, pd.DataFrame) and\
                (self._dataframe.shape[0] == len(self)):
            return self._dataframe
        DF_HEADERS = [
            "evt_id", "evt_time", "evt_longitude", "evt_latitude", "evt_depth", "evt_region",
            "evt_origin_author", "evt_moment", "evt_moment_magnitude", "evt_magnitude_author",
            "strike_1", "dip_1", "rake_1", "strike_2", "dip_2", "rake_2",
            "eigen_t", "plunge_t", "strike_t", "eigen_n", "plunge_n", "strike_n",
            "eigen_p", "plunge_p", "strike_p", "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp",
            "focal_mechanism_author"]
        self._dataframe = dict([(hdr, []) for hdr in DF_HEADERS])
        for evnt in self:
            self._dataframe["evt_id"].append(evnt.resource_id)
            for origin in evnt.origins:
                if origin.resource_id == evnt.preferred_origin_id:
                    # Add origin attributes
                    self._dataframe["evt_time"].append(str(origin.time))
                    self._dataframe["evt_longitude"].append(origin.longitude)
                    self._dataframe["evt_latitude"].append(origin.latitude)
                    self._dataframe["evt_depth"].append(origin.depth / 1000.0)
                    self._dataframe["evt_region"].append(origin.region)
                    self._dataframe["evt_origin_author"].append(origin.creation_info.author)
                    break
            for mag in evnt.magnitudes:
                if mag.resource_id == evnt.preferred_magnitude_id:
                    assert mag.magnitude_type.upper() == "MW", \
                        "Preferred magnitude for %s is not MW" % evnt.resource_id
                    moment = 10.0 ** ((1.5 * mag.mag) + 9.05)
                    self._dataframe["evt_moment"].append(moment)
                    self._dataframe["evt_moment_magnitude"].append(mag.mag)
                    self._dataframe["evt_magnitude_author"].append(
                        mag.creation_info.author)
                    break
            for foc_mec in evnt.focal_mechanisms:
                if foc_mec.resource_id == evnt.preferred_focal_mechanism_id:
                    self._dataframe["focal_mechanism_author"].append(
                        foc_mec.resource_id.resource_id.split("/")[-1]
                        )
                    # Add focal mechanism info
                    # Nodal planes
                    self._dataframe["strike_1"].append(
                        foc_mec.nodal_planes.nodal_plane_1.strike)
                    self._dataframe["dip_1"].append(
                        foc_mec.nodal_planes.nodal_plane_1.dip)
                    self._dataframe["rake_1"].append(
                        foc_mec.nodal_planes.nodal_plane_1.rake)
                    self._dataframe["strike_2"].append(
                        foc_mec.nodal_planes.nodal_plane_2.strike)
                    self._dataframe["dip_2"].append(
                        foc_mec.nodal_planes.nodal_plane_2.dip)
                    self._dataframe["rake_2"].append(
                        foc_mec.nodal_planes.nodal_plane_2.rake)
                    # Principal axes
                    self._dataframe["eigen_t"].append(foc_mec.principal_axes.t_axis.length)
                    self._dataframe["plunge_t"].append(foc_mec.principal_axes.t_axis.plunge)
                    self._dataframe["strike_t"].append(foc_mec.principal_axes.t_axis.azimuth)
                    self._dataframe["eigen_p"].append(foc_mec.principal_axes.p_axis.length)
                    self._dataframe["plunge_p"].append(foc_mec.principal_axes.p_axis.plunge)
                    self._dataframe["strike_p"].append(foc_mec.principal_axes.p_axis.azimuth)
                    self._dataframe["eigen_n"].append(foc_mec.principal_axes.n_axis.length)
                    self._dataframe["plunge_n"].append(foc_mec.principal_axes.n_axis.plunge)
                    self._dataframe["strike_n"].append(foc_mec.principal_axes.n_axis.azimuth)
                    if foc_mec.moment_tensor:
                        # If the moment tensor is given
                        for mt_key in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
                            self._dataframe[mt_key].append(
                                foc_mec.moment_tensor.tensor[mt_key]
                            )
                    else:
                        # otherwise np.nan
                        for mt_key in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
                            self._dataframe[mt_key].append(np.nan)
                    break
        #self._dataframe = pd.DataFrame(self._dataframe)
        #self._dataframe["evt_time"] = pd.to_datetime(self._dataframe["evt_time"], utc=True)
        return self._dataframe

    @classmethod
    def from_quakeml(cls, identifier: str, path_or_url: str,
                     data_source: Optional[str] = None, comments: Optional[str] = None):
        """
        Builds a moment tensor database from QuakeML (either a url for a FDSN webservice or a
        file)

        Args:
            path_or_url: URL for a FDSN webservice request or path to QuakeML xml file
        """
        mt_catalogue = obspy.read_events(path_or_url, format="quakeml")
        events = []
        for i, event in enumerate(mt_catalogue):
            if hasattr(event, "focal_mechanisms") and len(event.focal_mechanisms):
                for foc_mech in event.focal_mechanisms:
                    assert isinstance(foc_mech, FocalMechanism), \
                        "Focal mechanism not instance of class obspy.core.source.FocalMechanism"
                events.append(event)
            else:
                logging.info("Event %g: %s has no focal mechanism" % (i, event.resource_id))

        return cls(identifier=identifier, events=events, data_source=data_source,
                   comments=comments)

    @classmethod
    def from_isc_fmquakeml(cls, identifier: str, path_or_url: str,
                           data_source: Optional[str] = None, comments: Optional[str] = None,
                           force=False):
        """Builds a moment tensor database form ISC's FMQuakeML format. Effectively this is
        the same as the QuakeML format but in the ISC Moment Tensor database the magnitudes
        are missing, so this assigns the corresponding moment magnitude from the scalar
        moment information
        """
        mt_catalogue = cls.from_quakeml(identifier, path_or_url, data_source, comments)
        for event in mt_catalogue:
            event.magnitudes = []
            for f_m in event.focal_mechanisms:
                if not f_m.moment_tensor:
                    # Wont be able to get an Mw value for this tensor
                    continue
                f_m_id = f_m.resource_id.id
                f_m_author = f_m.creation_info.get("author", "ISC")
                mag_id = f_m_id.replace("smi:ISC/fault_planes;fmid=", "/magid=") +\
                    f"_Mw_{f_m_author}"
                mag_id = ResourceIdentifier(mag_id, prefix="smi:ISC")
                m_w = Magnitude(
                    resource_id=mag_id,
                    mag=M0_TO_MW(f_m.moment_tensor.scalar_moment),
                    magnitude_type="Mw",
                    origin_id=f_m.moment_tensor.derived_origin_id,
                    creation_info=f_m.creation_info,
                )
                if event.preferred_focal_mechanism_id.id and \
                        (mag_id.id == event.preferred_focal_mechanism_id.id):
                    event.preferred_magnitude_id = mag_id
                event.magnitudes.append(m_w)
            if force and not len(event.magnitudes):
                cls.query_isc_bulletin_for_single_event(event)
        return mt_catalogue

    @staticmethod
    def query_isc_bulletin_for_single_event(event):
        """
        """
        logging.info("Downloading magnitude and origin information for Event: %s"
                     % event.resource_id.id)
        # Get all the magnitudes for this event
        evid = event.resource_id.id.split("evid=")[-1]
        event_url = ISC_WEBSERVICES_BASE_URL + "&".join([
            f"eventid={evid}",
            "includeallmagnitudes=true",
            "includeallorigins=true",
            "format=xml"])
        logging.info("URL: %s" % event_url)
        isc_event = []
        try:
            isc_event = obspy.read_events(event_url, format="QuakeML")
        except:
            pass
        if len(isc_event):
            for f_m in event.focal_mechanisms:
                for mag in isc_event[0].magnitudes:
                    if (f_m.creation_info["author"] == mag.creation_info["author"]) and \
                            (mag.magnitude_type.upper() == "MW"):
                        event.magnitudes.append(mag)
                        break
        else:
            logging.info("Query failed")
        if not len(event.magnitudes):
            logging.info("No viable Mw values found for event %s" % evid)
        return

    @classmethod
    def from_seismicportal_json(
            cls,
            identifier: str,
            path_or_url: str,
            data_source: Optional[str] = None,
            comments: Optional[str] = None
            ):
        """
        Builds a moment tensor database from QuakeML (either a url for a FDSN webservice or a
        file)

        Args:
            path_or_url: URL for a FDSN webservice request or path to QuakeML xml file

        """
        if path_or_url.endswith("json"):
            # Is a file
            with open(path_or_url, "r") as f:
                catalogue = json.load(f)
        else:
            # Is a url
            req = urllib.request.Request(path_or_url)
            raw_data = urllib.request.urlopen(req)
            catalogue = json.load(raw_data)
        # Need to group dictionaries by event it
        event_groups = {}
        for mt in catalogue:
            if mt["ev_unid"] not in event_groups:
                event_groups[mt["ev_unid"]] = [mt]
            else:
                event_groups[mt["ev_unid"]].append(mt)

        events = []
        # Loop through the event groups
        for i, (event_id, evnts) in enumerate(event_groups.items()):
            event_id = ResourceIdentifier(id=event_id, prefix="")
            origins = []
            magnitudes = []
            focal_mechanisms = []
            preferred_origin_id = None
            preferred_magnitude_id = None
            preferred_focal_mechanism_id = None
            for evnt in evnts:
                agency = evnt["mt_source_catalog"]
                is_preferred = evnt["mt_preferred"]
                if evnt["ev_event_time"].endswith("UTC"):
                    # Non ISO compliant
                    orig_time = obspy.core.UTCDateTime(evnt["ev_event_time"].rstrip("UTC"))
                    cent_time = obspy.core.UTCDateTime(evnt["mt_centroid_time"].rstrip("UTC"))
                else:
                    # ISO compliant
                    orig_time = obspy.core.UTCDateTime(evnt["ev_event_time"])
                    cent_time = obspy.core.UTCDateTime(evnt["mt_centroid _time"])
                # Origins
                origins.extend([
                    # Hypocenter
                    Origin(
                        resource_id=ResourceIdentifier(
                            event_id.resource_id + f"/origin/hypocenter/{agency}"
                            ),
                        force_resource_id=False,
                        time=orig_time,
                        longitude=evnt["ev_longitude"],
                        latitude=evnt["ev_latitude"],
                        depth=evnt["ev_depth"] * 1000.0,  # km to m
                        region=evnt["ev_region"],
                        creation_info=CreationInfo(author=agency)
                    ),
                    # Centroid
                    Origin(
                        resource_id=ResourceIdentifier(
                            event_id.resource_id + f"/origin/centroid/{agency}"
                            ),
                        force_resource_id=False,
                        time=cent_time,
                        longitude=evnt["mt_longitude"],
                        latitude=evnt["mt_latitude"],
                        depth=evnt["mt_depth"] * 1000.0,  # km to m
                        region=evnt["mt_region"],
                        creation_info=CreationInfo(author=agency)
                    )
                ])
                # Add the moment magnitude for the tensor
                m_w = Magnitude(
                    resource_id=ResourceIdentifier(
                        event_id.resource_id + f"/magnitude/Mw/{agency}"
                        ),
                    mag=evnt["mt_mw"], magnitude_type="Mw",
                    creation_info=CreationInfo(author=agency)
                    )
                magnitudes.append(m_w)

                if evnt["ev_mag_type"].lower() != "mw":
                    # if event magnitude is not a moment magnitude then add this too
                    magnitudes.append(Magnitude(
                        resource_id=ResourceIdentifier(
                            event_id.resource_id +
                            "/magnitude/{:s}/{:s}".format(evnt["ev_mag_type"], agency)
                            ),
                        mag=evnt["ev_mag_value"],
                        magnitude_type=evnt["ev_mag_type"]
                        )
                    )
                mo_exp = 10.0 ** evnt["mt_m0_exp"]
                axe_exp = 10.0 ** evnt["mt_axe_exp"]
                tensor_exp = 10.0 ** evnt["mt_tensor_exp"]
                # Nodal planes
                nodal_planes = NodalPlanes(
                    nodal_plane_1=NodalPlane(strike=evnt["mt_strike_1"],
                                             dip=evnt["mt_dip_1"],
                                             rake=evnt["mt_rake_1"]),
                    nodal_plane_2=NodalPlane(strike=evnt["mt_strike_2"],
                                             dip=evnt["mt_dip_2"],
                                             rake=evnt["mt_rake_2"]),
                )
                # Principal axes
                principal_axes = PrincipalAxes(
                    t_axis=Axis(
                        length=evnt["mt_tval"] * axe_exp,
                        azimuth=evnt["mt_taz"],
                        plunge=evnt["mt_tplung"]
                    ),
                    p_axis=Axis(
                        length=evnt["mt_pval"] * axe_exp,
                        azimuth=evnt["mt_paz"],
                        plunge=evnt["mt_pplung"]
                    ),
                    n_axis=Axis(
                        length=evnt["mt_nval"] * axe_exp,
                        azimuth=evnt["mt_naz"],
                        plunge=evnt["mt_nplung"]
                    )
                )
                # Tensor
                if ("mt_mrr" in evnt) and ("mt_mtt" in evnt) and ("mt_mpp" in evnt):
                    # Has a tensor
                    tensor = Tensor(
                        m_rr=evnt["mt_mrr"] * tensor_exp,
                        m_tt=evnt["mt_mtt"] * tensor_exp,
                        m_pp=evnt["mt_mpp"] * tensor_exp,
                        m_rt=evnt["mt_mrt"] * tensor_exp,
                        m_rp=evnt["mt_mrp"] * tensor_exp,
                        m_tp=evnt["mt_mtp"] * tensor_exp,
                    )
                    mom_ten = MomentTensor(
                        resource_id=ResourceIdentifier(
                            id=event_id.resource_id + f"/moment_tensor/{agency}"
                            ),
                        force_resource_id=False,
                        derived_origin_id=event_id.resource_id +
                        f"/origin/hypocenter/{agency}",
                        scalar_moment=evnt["mt_m0"] * mo_exp,
                        tensor=tensor,
                    )
                else:
                    # No tensor
                    mom_ten = None

                focal_mechanisms.append(FocalMechanism(
                    resource_id=ResourceIdentifier(
                        id=event_id.resource_id + f"/focal_mechanism/{agency}"
                    ),
                    force_resource_id=False,
                    nodal_planes=nodal_planes,
                    principal_axes=principal_axes,
                    moment_tensor=mom_ten)
                )
                if is_preferred:
                    preferred_origin_id = event_id.resource_id + f"/origin/hypocenter/{agency}"
                    preferred_magnitude_id = event_id.resource_id + f"/magnitude/Mw/{agency}"
                    preferred_focal_mechanism_id = event_id.resource_id +\
                        f"/focal_mechanism/{agency}"

            events.append(
                Event(
                    resource_id=event_id,
                    force_resource_id=False,
                    origins=origins,
                    magnitudes=magnitudes,
                    focal_mechanisms=focal_mechanisms,
                    preferred_origin_id=preferred_origin_id,
                    preferred_magnitude_id=preferred_magnitude_id,
                    preferred_focal_mechanism_id=preferred_focal_mechanism_id
                )
            )
        return cls(identifier, events, data_source=data_source, comments=comments)

    @classmethod
    def from_rcmt_csv(cls, identifier: str, filename: str,
                      data_source: Optional[str] = None, comments: Optional[str] = None):
        """Creates a moment tensor database from the RCMT csv format. The RCMT catalogue
        is available from here: http://rcmt2.bo.ingv.it/
        """
        events = []
        catalogue = pd.read_csv(filename, sep=",")
        for i, row in catalogue.iterrows():
            event_id = ResourceIdentifier(id=row.ev_id, prefix="")
            yr, mo, dy = row.date.split("-")
            hh, mm, ss = row.time_orig.split(":")
            orig_time = obspy.core.UTCDateTime(int(yr), int(mo), int(dy),
                                               int(hh), int(mm), int(ss))
            # Parse origin
            origin_hypo = Origin(
                resource_id=ResourceIdentifier(row.ev_id + "/origin/hypocenter"),
                force_resource_id=False,
                time=orig_time,
                longitude=row.long_orig,
                latitude=row.lat_orig,
                depth=row.depth * 1000.0,  # Convert km to m
                region=row.region,
                creation_info=CreationInfo(author="RCMT|{:s}".format(row.source_type))
            )
            origin_centroid = Origin(
                resource_id=ResourceIdentifier(row.ev_id + "/origin/centroid"),
                force_resource_id=False,
                time=orig_time + row.centroid_time,
                longitude=row.centroid_long,
                latitude=row.centroid_lat,
                depth=row.centroid_depth * 1000,  # Convert km to m
                region=row.region,
                creation_info=CreationInfo(author="RCMT|{:s}".format(row.source_type))
            )
            # Parse magnitudes
            magnitudes = [
                # Mw
                Magnitude(
                    resource_id=ResourceIdentifier(row.ev_id + "/magnitude/Mw"),
                    mag=row.Mw,
                    magnitude_type="Mw",
                    creation_info=CreationInfo(author="RCMT")
                ),
            ]
            if row.Mb > 0.0:
                magnitudes.append(
                    Magnitude(
                        resource_id=ResourceIdentifier(row.ev_id + "/magnitude/mb"),
                        mag=row.Mb,
                        magnitude_type="mb"
                    )
                )
            if row.Ms > 0.0:
                magnitudes.append(
                    Magnitude(
                        resource_id=ResourceIdentifier(row.ev_id + "/magnitude/Ms"),
                        mag=row.Ms,
                        magnitude_type="Ms"
                    )
                )
            exp_fact = 10.0 ** (row.exp - 7)  # Converting from dyne-cm to Nm
            # Nodal Planes
            nodal_planes = NodalPlanes(
                nodal_plane_1=NodalPlane(strike=row.strike1, dip=row.dip1, rake=row.rake1),
                nodal_plane_2=NodalPlane(strike=row.strike2, dip=row.dip2, rake=row.rake2)
            )
            # Principal Axes
            principal_axes = PrincipalAxes(
                t_axis=Axis(
                    length=row.eigen_t * exp_fact,
                    azimuth=row.strike_t,
                    plunge=row.plunge_t
                ),
                p_axis=Axis(
                    length=row.eigen_p * exp_fact,
                    azimuth=row.strike_p,
                    plunge=row.plunge_p
                ),
                n_axis=Axis(
                    length=row.eigen_n * exp_fact,
                    azimuth=row.strike_n,
                    plunge=row.plunge_n
                )
            )
            # Moment tensor
            tensor = Tensor(
                m_rr=row.mrr * exp_fact,
                m_tt=row.mss * exp_fact,
                m_pp=row.mee * exp_fact,
                m_rt=row.mrs * exp_fact,
                m_rp=row.mre * exp_fact,
                m_tp=row.mse * exp_fact,
            )
            mom_ten = MomentTensor(
                resource_id=ResourceIdentifier(row.ev_id + "/moment_tensor/RCMT"),
                force_resource_id=False,
                derived_origin_id=origin_hypo.resource_id,
                scalar_moment=row.scalar_moment * exp_fact,
                tensor=tensor,)
            foc_mech = FocalMechanism(
                resource_id=ResourceIdentifier(row.ev_id + "/focal_mechanism/RCMT"),
                force_resource_id=False,
                nodal_planes=nodal_planes,
                principal_axes=principal_axes,
                moment_tensor=mom_ten
            )
            event = Event(
                resource_id=event_id,
                force_resource_id=False,
                origins=[origin_hypo, origin_centroid],
                magnitudes=magnitudes,
                focal_mechanisms=[foc_mech],
                preferred_origin_id=origin_hypo.resource_id,
                preferred_magnitude_id=magnitudes[0].resource_id,
                preferred_focal_mechanism_id=foc_mech.resource_id
            )
            events.append(event)
        return cls(identifier, events, data_source=data_source, comments=comments)

    def to_quakeml(self, output_file):
        """Exports the catalogue to QuakeML by creating an instance of an
        obspy.core.event.Catalog class and using the xml writer
        """
        resource_id = ResourceIdentifier(self.id)
        creation_info = CreationInfo(author=self.data_source) if self.data_source else None
        description = self.comments if self.comments else None
        catalogue = Catalog(events=self.events,
                            resource_id=resource_id,
                            description=description,
                            creation_info=creation_info)
        catalogue.write(output_file, "QUAKEML")
        logging.info("Written to file %s" % output_file)
        return

    def get_all_agency_magnitude_pairs(self):
        """Retrieves a dictionary containing the dataset of magnitudes associated with
        each commonly reported agencies for each agency and magnitude pair found in the
        database
        """
        agency_magnitude_pairs = {}
        for event in self:
            nmag = len(event.magnitudes)
            if nmag < 2:
                # Only one value, skip
                continue
            for i in range(nmag - 1):
                mag_i = event.magnitudes[i]
                if mag_i.magnitude_type.upper() != "MW":
                    continue
                val_i = mag_i.mag
                author_i = mag_i.creation_info["author"]
                for j in range(i + 1, nmag):
                    mag_j = event.magnitudes[j]
                    if mag_j.magnitude_type.upper() != "MW":
                        continue
                    val_j = mag_j.mag
                    author_j = mag_j.creation_info["author"]
                    #print(author_i, val_i, author_j, val_j)
                    if ((author_i, author_j) in agency_magnitude_pairs):    
                        agency_magnitude_pairs[(author_i, author_j)].append([val_i, val_j])
                    elif (author_j, author_i) in agency_magnitude_pairs:
                        agency_magnitude_pairs[(author_j, author_i)].append([val_i, val_j])
                    else:
                         agency_magnitude_pairs[(author_i, author_j)] = [[val_i, val_j]]
        output = {}
        for (agcy1, agcy2) in agency_magnitude_pairs:
            if not len(agency_magnitude_pairs[(agcy1, agcy2)]):
                continue
            dset1 = np.array(agency_magnitude_pairs[(agcy1, agcy2)])
            if (agcy2, agcy1) in agency_magnitude_pairs:
                # Turn to numpy array and swap column order
                dset2 = np.array(agency_magnitude_pairs[(agcy2, agcy1)])[:, ::-1]
                output[(agcy1, agcy2)] = np.column_stack([dset1, dset2])
                agency_magnitude_pairs[(agcy2, agcy1)] = []
            else:
                output[(agcy1, agcy2)] = dset1
        return output

    def find_matching_focal_mechanisms(
            self,
            time_window_s: int,
            distance_window_km: float,
            identifiers: Union[List, np.ndarray, pd.Series],
            times: Union[List, np.ndarray, pd.Series],
            lons:  Union[List, np.ndarray, pd.Series],
            lats:  Union[List, np.ndarray, pd.Series],
        ):
        """
        """
        d_t = np.timedelta64(time_window_s, "s")
        # Build catalogue dataframe of early and latest time windows per event
        early_times = []
        late_times = []
        for evt in self.events:
            low_vals = []
            high_vals = []
            for orig in evt.origins:
                orig_dt = np.datetime64(str(orig.time))
                low_vals.append(orig_dt - d_t)
                high_vals.append(orig_dt + d_t)
            early_times.append(np.array(low_vals, dtype="datetime64").min())
            late_times.append(np.array(high_vals, dtype="datetime64").max())
        event_times = pd.DataFrame({"early": early_times, "late": late_times},
                                   index=pd.Series(self.event_ids))
        matched_events = {}
        for eq_id, eq_time, lon, lat in zip(identifiers, times, lons, lats):
            if isinstance(eq_time, str):
                eq_time = np.datetime64(eq_time)
            elif isinstance(eq_time, pd.Timestamp):
                eq_time = eq_time.to_numpy()
            else:
                pass
            idx = (eq_time >= event_times["early"]) & (eq_time <= event_times["late"])
            if not np.any(idx):
                # No moment tensor found within time window
                logging.info("No moment tensor within time window of event %s" % eq_id)
                continue
            matched_events[eq_id] = []
            candidate_events = event_times.index[idx].to_list()
            has_candidate = False
            for candidate in candidate_events:
                orig_lons = []
                orig_lats = []
                #orig_depths = []
                candidate_eq = self[candidate]
                for orig in candidate_eq.origins:
                    orig_lons.append(orig.longitude)
                    orig_lats.append(orig.latitude)
                #    orig_depths.append(orig.depth / 1000.0) # Convert to km)
                dists = geodetic.distance(lon, lat, 0.0,
                                          np.array(orig_lons), np.array(orig_lats), 0.0)
                didx = dists <= distance_window_km
                if np.any(didx):
                    matched_events[eq_id].append(candidate_eq)
                    has_candidate = True
            if has_candidate:
                if len(matched_events[eq_id]) > 1:
                    logging.info("Event %s matches with %g candidate moment tensors"
                                 % (eq_id, len(matched_events[eq_id])))
                else:
                    logging.info("Event %s matches with moment tensor %s"
                                 % (eq_id, matched_events[eq_id][0].resource_id.id))
            else:
                logging.info("No moment tensor within distance window of event %s" % eq_id)
        return matched_events


#    def find_matching_focal_mechanisms(
#            self,
#            catalogue: pd.DataFrame,
#            time_delta: float = 30.0,
#            distance_delta: float = 50.0
#            ) -> pd.DataFrame:
#        """Given an input earthquake catalogue, creates a dataframe with focal mechanism
#        information and assigns the appropriate focal mechanism parameters to those earthquakes
#        with corresponding focal mechanims in the database
#
#        Args:
#            catalogue: Target catalogue for assigning focal mechanisms
#            time_delta: Maximum time difference in seconds for matching events
#            distance_delta: Maximum separation distance (in km) for matching events
#        """
#        n_out = catalogue.shape[0]
#        # Setup initial dataframe
#        output_headers = ["strike_1", "dip_1", "rake_1", "strike_2", "dip_2", "rake_2",
#                          "eigen_t", "plunge_t", "strike_t", "eigen_n", "plunge_n", "strike_n",
#                          "eigen_p", "plunge_p", "strike_p", "m_rr", "m_tt", "m_pp", "m_rt",
#                          "m_rp", "m_tp", "focal_mechanism_author", "focal_mechanism_event_id"]
#        output_dframe = {}
#        for hdr in output_headers:
#            if hdr in ("focal_mechanism_author", "focal_mechanism_event_id"):
#                output_dframe[hdr] = [None] * n_out
#            else:
#                output_dframe[hdr] = np.zeros(n_out)
#        output_dframe = pd.DataFrame(output_dframe)
#        #print(output_dframe)
#        catalogue_groups = catalogue.groupby("event_id")
#        for evid, evidx in catalogue_groups.groups.items():
#            n = len(evidx)
#            grp = catalogue_groups.get_group(evid)
#            dt = (grp["event_time"].iloc[0] - self.dataframe["evt_time"]).dt.total_seconds()
#            tidx = np.fabs(dt) <= time_delta
#            if not np.any(tidx):
#                # No event within time window
#                continue
#
#            candidate = self.dataframe[tidx]
#            distance = geodetic.distance(
#                grp["event_longitude"].iloc[0], grp["event_latitude"].iloc[0], 0.0,
#                candidate["evt_longitude"], candidate["evt_latitude"], 0.0)
#            ridx = distance <= distance_delta
#            if np.any(ridx):
#                # A match is found!
#                candidate = candidate[ridx]
#                if candidate.shape[0] > 1:
#                    # Find the nearest event in time
#                    dt_sel = dt[tidx][ridx]
#                    min_loc = np.argmin(dt_sel)
#                    candidate = candidate.iloc[min_loc]
#                if verbose:
#                    logging.info("{:s}: {:s} {:s}".format(evid, evidx, str(candidate)))
#                for key in output_headers:
#                    if key == "focal_mechanism_author":
#                        output_dframe[key][evidx] = pd.Series([candidate[key].iloc[0]] * n,
#                                                              index=evidx)
#                    elif key == "focal_mechanism_event_id":
#                        output_dframe[key][evidx] = pd.Series([candidate["evt_id"].iloc[0]] * n,
#                                                              index=evidx)
#                    else:
#                        output_dframe[key][evidx] += candidate[key].iloc[0]
#        return output_dframe
