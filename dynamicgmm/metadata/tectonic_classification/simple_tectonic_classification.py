"""
Simple Tectonic Classifier - deterministic
"""
import pathlib
import tomllib
import logging
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import LinearNDInterpolator
from openquake.hazardlib.geo import (
    ComplexFaultSurface, PlanarSurface, SimpleFaultSurface, Line, Point, geodetic, Mesh
)

logging.basicConfig(level=logging.INFO)

DATA_DIR = pathlib.Path(__file__).parent / "tectonic_classification_data"


def vrancea_classification(
    events: pd.DataFrame,
    vrancea_upper_depth: float = 50.0,
    vrancea_lower_depth: float = 300.0
) -> pd.DataFrame:
    """Determines whether events are within the Vrancea deep seismic zone according
    to the ESHM20 definition. Adds the column "is_Vrancea" on to the events dataframe
    indicating whether the event is within the Vrancea region or not

    Args:
        events: Dataframe for the events
        vrancea_upper_depth: Upper depth of the Vrancea zone (km)
        vrancea_lower_depth: Lower depth of the Vranea zone
    """
    vrancea_sources = gpd.GeoDataFrame.from_file(
        str(DATA_DIR / "eshm20_asz_deep_winGT_fs017_dm02_results.shp"),
        crs='EPSG:4326'
    )
    vrancea_sources = vrancea_sources[vrancea_sources["TECTO_ID"] == "TSZ002"]
    ev_lon_lat = gpd.points_from_xy(
        events["event_longitude"],
        events["event_latitude"],
        crs="EPSG:4326"
    )
    vrancea_cart = vrancea_sources.to_crs("EPSG:3035")
    event_df = gpd.GeoDataFrame(events[["event_id", "event_longitude", "event_latitude",
                                        "event_hypo_depth", "event_preferred_mag"]],
                                geometry=ev_lon_lat,
                                crs="EPSG:4326")
    event_df_cart = event_df.to_crs("EPSG:3035")
    assignment = gpd.sjoin(event_df_cart, vrancea_cart)
    logging.info(f"Identified {assignment.shape[0]} Vrancea events in the database")
    events["is_Vrancea"] = (
        events["event_id"].isin(assignment["event_id"]) &
        (events["event_hypo_depth"] > vrancea_upper_depth) &
        (events["event_hypo_depth"] <= vrancea_lower_depth)
    )
    return events


def volcanic_classification(
    events: pd.DataFrame,
    max_volc_dist: float = 20.0,
    max_volc_depth: float = 10.0,
    exception_list: Optional[List] = None
):
    """Classifiea events as volcanic according to their location with respect to the
    distance to the nearest volcano according to the Smithsonsian Institution Volcano database
    for Europe. Adds the columns: "is_Volcanic" indicating whether the event is volcanic or
    not. "volcano" the name of the volcano to which it is associated. "dist_to_volcano" the
    distance to the volcano location (km).

    Args:
        events: Dataframe for the events
        max_volc_dist: Maximum distance from the volcano location for an event to be classed
                       as volcanic
        max_volc_depth: Maximum depth of events within that region associated to volcanic
                        seismicity
        exception_list: List of names of volcanos to be excluded for consideration

    """
    if exception_list is None:
        exception_list = []
    volcano_database = pd.read_csv(
        str(DATA_DIR / "europe_volcano_smithsonian_db.csv"),
        sep=","
    )
    volcano_database.set_index("Volcano Name", drop=True, inplace=True)
    # Drop any excluded volcanoes
    for exc in exception_list:
        volcano_database.drop(exc, inplace=True)
    neq = events.shape[0]
    volcanic_assignment = pd.DataFrame(
        {
            "is_Volcanic": np.zeros(neq, dtype=bool),
            "volcano": [None] * neq,
            "dist_to_volcano": np.inf * np.ones(neq, dtype=float)
        },
        index=events.index
    )
    for volc_name, volc_data in volcano_database.iterrows():
        dist_to_volc = geodetic.distance(
            volc_data["Longitude"], volc_data["Latitude"], 0.0,
            events["event_longitude"].to_numpy(),
            events["event_latitude"].to_numpy(),
            np.zeros(neq)
        )
        idx = np.logical_and(dist_to_volc <= max_volc_dist,
                             events["event_hypo_depth"] <= max_volc_depth)
        if np.any(idx):
            logging.info(f"Volcano {volc_name}: {np.sum(idx)} events")
            # print(volc_name, np.sum(idx))
            volcanic_assignment["is_Volcanic"][idx] = True
            nearer = np.logical_and(
                idx,
                dist_to_volc <= volcanic_assignment["dist_to_volcano"].to_numpy()
            )
            volcanic_assignment["volcano"][nearer] = volc_name
            volcanic_assignment["dist_to_volcano"][nearer] = dist_to_volc[nearer]
        else:
            continue

    volcanic_assignment["dist_to_volcano"][
        np.isinf(volcanic_assignment["dist_to_volcano"])
    ] = np.nan
    for key in volcanic_assignment.columns:
        events[key] = volcanic_assignment[key].copy()
    return events


SUB_AZIMUTHS = {"GRSS001": 280.0, "CYSS001": 290.0, "ITSS001": 220.0, "MASS001": 190.0}


def _setup_subduction_data(sub_zone_ids: List, mesh_spacing: float = 2.0):
    """Sets up the subduction zone data loading in the lattice and the interface
    points according to the European Seismogenic Fault Source Model. Builds the complex
    fault sources for the interfaces and returns the lattice points for the slab.
    """
    subduction_lattice = gpd.GeoDataFrame.from_file(
        str(DATA_DIR / "efsm20_is_lattice.geojson"),
    )
    interface_mesh_points = gpd.GeoDataFrame.from_file(
        str(DATA_DIR / "EFSM20_Meshes_efsm20_si_points.geojson"),
    )
    sz_depth_groups = interface_mesh_points.groupby(["idfs", "pntdep"])
    subduction_interfaces = dict([(sub_id, []) for sub_id in sub_zone_ids])
    for (sub_id, sub_dep) in list(sz_depth_groups.groups):
        if sub_id not in sub_zone_ids:
            continue
        sub_az = SUB_AZIMUTHS[sub_id]
        pnts = sz_depth_groups.get_group((sub_id, sub_dep))
        edge = Line([
            Point(lon, lat, -depth / 1000.0)
            for (lon, lat, depth) in zip(pnts["lon"], pnts["lat"], pnts["pntdep"])
            ])
        if np.fabs(edge.azimuth - sub_az) > 45.0:
            # Needs reversal
            edge = [
                Point(lon, lat, -depth / 1000.0)
                for (lon, lat, depth) in zip(pnts["lon"], pnts["lat"], pnts["pntdep"])
            ]
            edge = Line(edge[::-1])
        subduction_interfaces[sub_id].append(edge)
    for key in subduction_interfaces.keys():
        # Reverse order to have shallowest first
        subduction_interfaces[key] = subduction_interfaces[key][::-1]
        logging.info(f"Building interface fault surface for {key}")
        try:
            subduction_interfaces[key] = ComplexFaultSurface.from_fault_data(
                subduction_interfaces[key], mesh_spacing
            )
        except ValueError:
            logging.info(f"Building complex fault surface failed for {key}")
            del subduction_interfaces[key]
            continue

    return subduction_interfaces, subduction_lattice


class SimpleTectonicClassifier:
    """Simple deterministic classifier for European regions
    Default to Shallow Crust
    Steps:
    1) Vrancea: Within Vrancea polygons and depth > 50.0
    2) Volcanic: Within 20 km of a volcano location from the Smithsonian Database
    3) In-slab: Within EFSM subduction lattice and depth > 50 km
    4) Interface: Within EFSM subduciton lattice and RRup for the surface < 10 km
    5) Outer rise: Within the subduction lattice but the interface Rx < 0.0
    """

    EVENT_HEADERS = [
        'event_id',
        'event_time',
        'event_longitude',
        'event_latitude',
        'event_hypo_depth',
        'event_origin_author',
        'event_preferred_mag',
        'event_preferred_mag_type',
        'event_preferred_mag_author',
        'focal_mechanism',
    ]

    def __init__(self, flatfile: pd.DataFrame, config: Dict):
        """
        Args:
            flatfile: The ground motion flatfile
            config: The configuration dictionary

        """
        self.config = config
        self.crustal_depth = self.config.get("Crustal Depth", 50.0)
        self.flatfile = flatfile.copy()
        self.events = flatfile[self.EVENT_HEADERS].drop_duplicates("event_id", inplace=False)
        self.nobs = self.flatfile.shape[0]
        self.neq = self.events.shape[0]

        if "Vrancea" in self.config:
            logging.info("Searching for Vrancea events")
            self.events = vrancea_classification(
                self.events,
                config["Vrancea"].get("upper_depth", 50.0),
                config["Vrancea"].get("lower_depth", 300.0)
            )
        if "Volcanic" in self.config:
            logging.info("Searching for Volcanic events")
            self.events = volcanic_classification(
                self.events,
                config["Volcanic"].get("maximum_volcanic_distance", 20.0),
                config["Volcanic"].get("maximum_volcanic_depth", 10.0),
                config["Volcanic"].get("exclude", [])
            )
        if "Subduction" in self.config:
            logging.info("Loading subduction data and preparing faults")
            self.subduction_interfaces, self.subduction_lattice = _setup_subduction_data(
                    self.config["Subduction"].get("zones", ["GRSS001", "CYSS001", "ITSS001"]),
                    self.config["Subduction"].get("fault_mesh_spacing", 2.0)
                )
            self.subduction_zones = list(self.subduction_interfaces)
        else:
            self.subduction_interfaces = None
            self.subduction_lattice = None
            self.interface_interpolators = {}
            self.subduction_zones = []
        self.events.set_index("event_id", inplace=True, drop=True)
        event_counts = self.flatfile.value_counts("event_id")
        self.events["Nrec"] = pd.Series(np.zeros(self.neq, dtype=int),
                                        index=self.events.index)
        for ev_id in self.events.index:
            self.events["Nrec"].loc[ev_id] += event_counts.loc[ev_id]

    def subduction_zone_info(self):
        """Returns the distances of the events to each interface and lattice, alongside
        the scipy.interpolate.LinearNDIntoerpolator class for each inferface
        """
        ev_mesh = Mesh(self.events["event_longitude"].to_numpy(),
                       self.events["event_latitude"].to_numpy(),
                       self.events["event_hypo_depth"].to_numpy())
        ev_sfc_mesh = Mesh(self.events["event_longitude"].to_numpy(),
                           self.events["event_latitude"].to_numpy())

        dists_to_interfaces = {}
        interpolators = {}
        for key, sub_if in self.subduction_interfaces.items():
            dists_to_interfaces[f"{key}-rrup"] = sub_if.get_min_distance(ev_mesh)
            dists_to_interfaces[f"{key}-rjb"] = sub_if.get_joyner_boore_distance(ev_mesh)
            dists_to_interfaces[f"{key}-rx"] = sub_if.get_rx_distance(ev_mesh)
            interpolators[key] = LinearNDInterpolator(
                np.column_stack((sub_if.mesh.lons.ravel(), sub_if.mesh.lats.ravel())),
                sub_if.mesh.depths.ravel(),
            )
        dists_to_lattices = {}
        for key in self.subduction_zones:
            lattice = self.subduction_lattice[self.subduction_lattice["idfs"] == key]

            lattice_mesh = Mesh(
                lattice["lon"].to_numpy(),
                lattice["lat"].to_numpy(),
                np.zeros(lattice.shape[0])
            )
            dists_to_lattices[f"{key}"] = \
                lattice_mesh.get_min_distance(ev_sfc_mesh)

        return dists_to_interfaces, dists_to_lattices, interpolators

    def classify_events(self, verbose=True) -> pd.DataFrame:
        """Applies the classification scheme returning a dataframe indicating the
        tectonic region class and the weight associated with the classification (which in
        the present case is always 1)
        """
        classification = pd.DataFrame({
            "Tectonic Region": ["Shallow Crust"] * self.neq,
            "Classification Weight": np.ones(self.neq, dtype=float)
        }, index=self.events.index.copy())
        if "Subduction" in self.config:
            # classify_subduction = True
            dists_to_if, dists_to_lattice, interpolators = self.subduction_zone_info()
            max_lattice_dist = self.config["Subduction"].get("lattice_distance", 10.0)
            max_interface_dist = self.config["Subduction"].get("max_interface_distance", 10.0)
            mesh_spacing = self.config["Subduction"].get("fault_mesh_spacing", 2.0)
            outer_rise_rx = self.config["Subduction"].get("outer_rise_rx", -25.0)
        logging.info("Running classifier")
        for i, (ev_id, event) in enumerate(self.events.iterrows()):
            if "is_Vrancea" in event and event["is_Vrancea"]:
                classification["Tectonic Region"].loc[ev_id] = "Vrancea"
                continue
            if ("is_Volcanic" in event) and event["is_Volcanic"]:
                classification["Tectonic Region"].loc[ev_id] = "Volcanic"
                continue

            classification["Tectonic Region"].loc[ev_id] = "Shallow Crust" \
                if event["event_hypo_depth"] <= self.crustal_depth else "Non-Subduction Deep"
            # Apply subduction classifications
            if "Subduction" in self.config:
                for key in dists_to_lattice:
                    r_lattice = dists_to_lattice[key][i]
                    if r_lattice > max_lattice_dist:
                        # Unconnected to the zone
                        continue
                    if event["event_hypo_depth"] > self.crustal_depth:
                        # Is within the lattice _AND_ deeper than 50 km -> Inslab
                        classification["Tectonic Region"].loc[ev_id] = "Subduction Inslab"
                        break
                    else:
                        # Need to assess where we are with respect to the interface
                        rrup_if = dists_to_if[f"{key}-rrup"][i]
                        if rrup_if <= max_interface_dist:
                            # Interface
                            classification["Tectonic Region"].loc[ev_id] = \
                                "Subduction Interface"
                            break
                        rx_if = dists_to_if[f"{key}-rx"][i]
                        rjb_if = dists_to_if[f"{key}-rjb"][i]

                        if rx_if < outer_rise_rx:
                            # Shallow on the outer rise
                            classification["Tectonic Region"].loc[ev_id] = "Outer Rise"
                            break

                        if rjb_if <= mesh_spacing:
                            # In the surface projection but not interface.
                            # If above the interface then this is crustal, otherwise in-slab
                            if_depth_at_epicentre = interpolators[key](
                                np.array([[event.event_longitude, event.event_latitude]])
                                )[0]
                            if np.isnan(if_depth_at_epicentre):
                                classification["Tectonic Region"].loc[ev_id] = \
                                    "Subduction Inslab" if rx_if < 0.0 else "Shallow Crust"
                            if event.event_hypo_depth <= if_depth_at_epicentre:
                                classification["Tectonic Region"].loc[ev_id] = "Shallow Crust"
                            else:
                                classification["Tectonic Region"].loc[ev_id] = \
                                    "Subduction Inslab"
                        else:
                            classification["Tectonic Region"].loc[ev_id] = "Subduction Inslab"
            if verbose:
                logging.info(
                    f"{i}: Event {ev_id} - {classification["Tectonic Region"].loc[ev_id]}"
                )
        return classification

    def add_classifications_to_flatfile(self, classification):
        """Given an event classification dataframe, this adds the classification to
        each corresponding event in the flatfile
        """
        logging.info("Adding classification to flatfile")
        record_classification = pd.DataFrame({
            "Tectonic Region": [None] * self.nobs,
            "Classification Weight": np.zeros(self.nobs)
            },
            index=self.flatfile.index.copy()
        )
        for ev_id, tect_class in classification.iterrows():
            idx = self.flatfile["event_id"] == ev_id
            record_classification["Tectonic Region"][idx] = tect_class["Tectonic Region"]
            record_classification["Classification Weight"][idx] = \
                tect_class["Classification Weight"]
        for key in record_classification.columns:
            self.flatfile[key] = record_classification[key].copy()
        return
