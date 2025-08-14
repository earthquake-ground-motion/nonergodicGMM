import os
import logging
from typing import Tuple, List, Dict, Optional
import h5py
import rtree
import rasterio
import numpy as np
import pandas as pd
from shapely import geometry
from pyproj import Transformer


logging.basicConfig(level=logging.INFO)


CELL_INFO_HEADERS = "cellid,cellname,q1X,q1Y,q1Z,q2X,q2Y,q2Z,q3X,q3Y,q3Z,q4X,q4Y,q4Z,q5X,q5Y,"\
    "q5Z,q6X,q6Y,q6Z,q7X,q7Y,q7Z,q8X,q8Y,q8Z,mptX,mptY,mptZ,UTMzone,q1Lat,q1Lon,q2Lat,q2Lon,"\
    "q3Lat,q3Lon,q4Lat,q4Lon,q5Lat,q5Lon,q6Lat,q6Lon,q7Lat,q7Lon,q8Lat,q8Lon,mptLat,mptLon"


class Grid3D(object):
    """Core class representing a spatial grid with standard gridding operations.
    This is mostly to be extended by subclasses to handle source model activity
    rates and catalogues.

    Attributes:

        llon: Western limit of the grid
        llat: Southern Limit of the grid
        ulon: Eastern limit of the grid
        ulat: Northern limit of the grid
        spcx: Grid spacing in the longitudinal axis (in decimal degrees)
        spcy: Grid spacing in the latitudinal axis (in decimal degrees)
        lons: 1D numpy array of grid longitude values
        lats: 1D numpy array of grid latitude values
        glons: 2D grid of longitude values (from meshgrid)
        glats: 2D grid of latitude values (from meshgrid)
        geo_to_cart: PyProj Transformer from geodetic coordinates to cartesian
        cart_to_geo: PyProj Transformer from cartesian coordinates to geodetic
        xylons: Cartesian coordinates of the grid longitude values
        xylats: Cartesian coordinates of the grid latitude values
        ny: Number of latitude values
        nx: Number of longitude values
        grid_tree: RTree spatial index of grid
        polygons: Cells of grid as a set of polygons in cartesian space

    """
    def __init__(
            self,
            bbox: Tuple[float, float, float, float],
            spcx: float,
            spcy: float,
            zbox: Optional[Tuple[float, float]] = None,
            spcz: float = 50.0,
            geodetic_crs: str = "EPSG:4326",
            cartesian_crs: str = "EPSG:3035",
            build_tree_polygons: bool = True
            ):
        """
        Instantiate grid with bounding box and spacings

        Args:
            bbox: Bounding box of region as tuple of (west, south, east, north) in decimal
                  degrees
            Bounding box as (llon, llat, ulon, ulat)
            spcx: Grid spacing in the longitudinal axis (in decimal degrees)
            spcy: Grid spacing in the latitudinal axis (in decimal degrees)
            geodetic_crs: Coordinate reference system to use for geodetic coordinates
                          (long., lat.) given in terms of a full EPSG code, e.g. WGS84
                          (default) is "EPSG:4326"
            cartesian_crs: Coordinate reference system to use for cartesian coordinates
                           (easting, northing) given in terms of a full EPSG code,
                           e.g. European Equal Area Projection (default) is "EPSG:3035"
            build_tree_polygons: If True then this will build the RTree for spatial indexing
                                 and the polygons, which are needed for building or adapti
                                 a grid from a source model. If False then this step is
                                 skipped, which may be more efficient if retreiving a grid
                                 from file.
        """
        # Setup basic grid
        self.llon, self.llat, self.ulon, self.ulat = bbox
        if zbox is not None:
            self.uz, self.lz = zbox
        else:
            self.uz, self.lz = (0.0, spcz)
        self.spcx = spcx
        self.spcy = spcy
        self.spcz = spcz
        self.lons = np.arange(self.llon, self.ulon + spcx, spcx)
        self.lats = np.arange(self.llat, self.ulat + spcy, spcy)
        self.glons, self.glats = np.meshgrid(self.lons, self.lats)
        self.depths = np.arange(self.uz, self.lz + self.spcz, self.spcz)
        self.cartesian_crs = cartesian_crs
        self.geo_to_cart = Transformer.from_crs(geodetic_crs, cartesian_crs, always_xy=True)
        self.cart_to_geo = Transformer.from_crs(cartesian_crs, geodetic_crs, always_xy=True)
        # Transform grid to a cartesian reference frame
        self.xylons, self.xylats = self.transform(self.glons, self.glats)
        self.ny, self.nx = self.glons.shape
        self.ny -= 1
        self.nx -= 1
        self.nz = len(self.depths) - 1
        # Build rtree
        if build_tree_polygons:
            self.grid_tree = rtree.index.Index()
            cntr = 0
            logging.info("Building rtree ...")
            for i in range(self.ny):
                for j in range(self.nx):
                    bbox = (self.xylons[i, j], self.xylats[i, j],
                            self.xylons[i, j + 1], self.xylats[i + 1, j])
                    self.grid_tree.insert(cntr, bbox)
                    cntr += 1
            logging.info("done")
            # Building polygon set
            self.polygons = self.to_polygon_set(as_xy=True)
        else:
            self.grid_tree = []
            self.polygons = []
        self._areas = None

    def transform(
            self,
            x: np.ndarray,
            y: np.ndarray,
            xy2geo: bool = False
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms a set of coordinates between geodetic and cartesian

        Args:
            x: X-values of grid (either longitudes or cartesian X)
            y: Y-values of grid (either latitudes or cartesian Y)
            xy2geo: Convert cartesian to geodetic (True) or geodetic to cartesian (False)

        Returns:
            Tuple of [X, Y] or [Longitude, Latitude]
        """
        if xy2geo:
            return self.cart_to_geo.transform(x, y)
        else:
            return self.geo_to_cart.transform(x, y)

    def __repr__(self):
        # Basic string to represent grid attributes
        return "GRID:{:.3f}/{:.3f}/{:.3f}/{:.3f} ({:.3f}, {:.3f})".format(
            self.llon, self.llat, self.ulat, self.ulon, self.spcx, self.spcy)

    def iter_polygons(self):
        """Iterate over the polygons, returning a unique polygon ID and the
        polygon

        """
        if not len(self.polygons):
            raise ValueError("No polygons for iteration")
        cntr = 0
        for i in range(self.ny):
            for j in range(self.nx):
                cntr += 1
                yield self.polygons[cntr - 1]


    def to_polygon_set(self, as_xy: bool = False) -> List:
        """Transforms the grid into a set of shapely polygons

        Args:
            as_xy: Create the polygon out of the Cartesian coordinates (True) or geodetic
                   coordinates (False)

        Returns:
            polygons: Grid cells as a list of Shapely polygons
        """
        polygons = []
        for i in range(self.ny):
            for j in range(self.nx):
                if as_xy:
                    polygon = [
                        (self.xylons[i, j], self.xylats[i, j]),
                        (self.xylons[i + 1, j], self.xylats[i + 1, j]),
                        (self.xylons[i + 1, j + 1], self.xylats[i + 1, j + 1]),
                        (self.xylons[i, j + 1], self.xylats[i, j + 1]),
                        (self.xylons[i, j], self.xylats[i, j])
                    ]
                else:
                    polygon = [
                        (self.glons[i, j], self.glats[i, j]),
                        (self.glons[i + 1, j], self.glats[i + 1, j]),
                        (self.glons[i + 1, j + 1], self.glats[i + 1, j + 1]),
                        (self.glons[i, j + 1], self.glats[i, j + 1]),
                        (self.glons[i, j], self.glats[i, j])
                    ]
                polygons.append(geometry.Polygon(polygon))
        return polygons

    @property
    def areas(self):
        """Returns the areas of the grid cells (in km ^ 2)
        """
        if self._areas is not None:
            return self._areas
        dlons = self.xylons[:, 1:] - self.xylons[:, :-1]
        dlats = self.xylats[1:, :] - self.xylats[:-1, :]
        self._areas = 1.0E-6 * (dlons[:-1, :] * dlats[:, :-1])
        return self._areas

    @property
    def area(self):
        """Returns the total area of the grid
        """
        return np.sum(self.areas)

    def to_cell_info(self, utmzone: str) -> pd.DataFrame:
        """Creates a cell info flatfile dataframe
        """
        dframe = dict(
            [(header, []) for header in CELL_INFO_HEADERS.split(",")]
            )
        dframe["UTMzone"] = [utmzone] * (self.nx * self.ny * self.nz)
        cntr = 0
        for k in range(self.nz):
            # Depth limits
            uz = self.depths[k]
            lz = self.depths[k + 1]
            mpt_z = (lz + uz) / 2.0
            for i, poly in enumerate(self.iter_polygons()):
                dframe["cellname"].append(f"c.{cntr}")
                dframe["cellid"].append(cntr)

                poly_x, poly_y = poly.exterior.xy

                vect = 0.5 * np.array([poly_x[2] - poly_x[0], poly_y[2] - poly_y[0]])
                mpt_x = poly_x[0] + vect[0]
                mpt_y = poly_y[0] + vect[1]
            
                poly_lon, poly_lat = self.cart_to_geo.transform(poly_x, poly_y)
                mpt_lon, mpt_lat = self.cart_to_geo.transform(mpt_x, mpt_y)

                # Polygons must be in order LL, LR, UL, UR but they were
                # originally input from clockwise LL LR UR UL
                for j, loc in enumerate([0, 3, 1, 2]):
                    x = poly_x[loc]
                    y = poly_y[loc]
                    lon = poly_lon[loc]
                    lat = poly_lat[loc]
                    # Lower point
                    dframe[f"q{j + 1}X"].append(x)
                    dframe[f"q{j + 1}Y"].append(y)
                    dframe[f"q{j + 1}Z"].append(-lz)
                    dframe[f"q{j + 1}Lon"].append(lon)
                    dframe[f"q{j + 1}Lat"].append(lat)
                    # Upper point
                    dframe[f"q{j + 5}X"].append(x)
                    dframe[f"q{j + 5}Y"].append(y)
                    dframe[f"q{j + 5}Z"].append(-uz)
                    dframe[f"q{j + 5}Lon"].append(lon)
                    dframe[f"q{j + 5}Lat"].append(lat)
                # Add the midpoint info
                dframe["mptX"].append(mpt_x)
                dframe["mptY"].append(mpt_y)
                dframe["mptZ"].append(-mpt_z)
                dframe["mptLon"].append(mpt_lon)
                dframe["mptLat"].append(mpt_lat)
                cntr += 1
        return pd.DataFrame(dframe)



class Flatfile():
    """
    """
    SOURCE_HEADERS = ["event_id", "event_time", "event_longitude", "event_latitude",
                      "event_hypo_depth", "event_origin_author", "event_preferred_mag",
                      "event_preferred_mag_type", "event_preferred_mag_author",]
    DISTANCE_HEADERS = ["repi", "rhypo", "rjb", "rrup", "rx", "ry0",]
    SITE_HEADERS = ["station_id", "network", "station", "location", "channel",
                    "station_longitude", "station_latitude", "station_elevation",
                    "station_depth", "vs30", "vs30_geology", "vs30_vs_profile",]

    def __init__(self,
        flatfile: pd.DataFrame,
        cartesian_crs: str = "EPSG:3035",
        geodetic_crs: str = "EPSG:4326",
        earthquake_source_point: str = "hypocenter",
        utmzone: Optional[str] = None,
        default_event_depth: float = 10.0
    ):
        """
        """
        self.data = flatfile.copy()
        self._get_event_station_record_identifiers(flatfile)
        self.cartesian_crs = cartesian_crs
        self.geodetic_crs = geodetic_crs
        self.geo_to_cart = Transformer.from_crs(
            self.geodetic_crs,
            self.cartesian_crs,
            always_xy=True
            )

        self.cart_to_geo = Transformer.from_crs(
            self.cartesian_crs,
            self.geodetic_crs,
            always_xy=True
            )
        self.default_depth = default_event_depth

        self.utmzone = utmzone if utmzone else cartesian_crs
        if earthquake_source_point == "hypocenter":
            # Source point for path definition is taken from the hypocenter
            self.eqx, self.eqy, self.eqz = self._earthquake_xyz_hypocenter()
        else:
            raise ValueError("Earthquake source point %s not supported"
                             % earthquake_source_point)

        self.ssx, self.ssy = self.geo_to_cart.transform(
            flatfile["station_longitude"].to_numpy(),
            flatfile["station_latitude"].to_numpy()
            )

        # Get the paths as a list of shapely linestrings
        self.xy_paths = []
        self.geo_paths = []
        for ex, ey, sx, sy, elon, elat, slon, slat in zip(
                self.eqx, self.eqy, self.ssx, self.ssy,
                flatfile["event_longitude"].to_numpy(),
                flatfile["event_latitude"].to_numpy(),
                flatfile["station_longitude"].to_numpy(),
                flatfile["station_latitude"].to_numpy()):
            self.xy_paths.append(geometry.LineString([(ex, ey), (sx, sy)]))
            self.geo_paths.append(geometry.LineString([(elon, elat), (slon, slat)]))


    def _get_event_station_record_identifiers(self, flatfile):
        """From the input flatfile retrieve the unique identifiers mapped to integers
        required by the flatfile format
        """
        logging.info("Building identifier data")
        self.eqid_mapping = dict([
            (eqid, i) for i, eqid in enumerate(pd.unique(flatfile["event_id"]).tolist())
            ])
        self.ssn_mapping = dict([
            (ssn, i) for i, ssn in enumerate(pd.unique(flatfile["station_id"]).tolist())
            ])
        self.rsn_mapping = dict([
            (rsn, i) for i, rsn in enumerate(flatfile["record_id"].to_list())
            ])
        self.eqids = list(self.eqid_mapping) # [val for key, val in self.eqid_mapping.items()]
        self.ssns = list(self.ssn_mapping) # [val for key, val in self.ssn_mapping.items()]
        self.rsns = list(self.rsn_mapping) #flatfile["gmid"].to_list()
        self.neq = len(self.eqids)
        self.nsta = len(self.ssns)
        self.nrec = len(self.rsns)
        self.identifiers = {
            "rsn": np.zeros(self.nrec, dtype=int),
            "eqid": np.zeros(self.nrec, dtype=int),
            "ssn": np.zeros(self.nrec, dtype=int)
            }
        for i, row in flatfile[["event_id", "station_id", "record_id"]].iterrows():
            self.identifiers["rsn"][i] = self.rsn_mapping[row.record_id]
            self.identifiers["eqid"][i] = self.eqid_mapping[row.event_id]
            self.identifiers["ssn"][i] = self.ssn_mapping[row.station_id]
        self.identifiers = pd.DataFrame(self.identifiers)
        return

    def _earthquake_xyz_hypocenter(self):
        """Defines the point representing the earthquake position in the
        path grid - assuming a hypocenter
        """
        eqz = self.data["event_hypo_depth"].to_numpy()
        # Zero depths lead to errors in the distance matrix, so these
        # should be rendered to a default reference depth
        eqz[np.isclose(eqz, 0.0)] = self.default_depth
        eqx, eqy = self.geo_to_cart.transform(
            self.data["event_longitude"].to_numpy(),
            self.data["event_latitude"].to_numpy()
            )
        return eqx, eqy, eqz

    def export_to_stan_flatfile(self, fname: Optional[str] = None,
            total_residual: Optional[np.ndarray] = None,
            x_2: Optional[np.ndarray] = None,
            x_3: Optional[np.ndarray] = None,
            m_to_km: bool = True) -> pd.DataFrame:
        """
        """

        fct = 1.0 / 1000.0 if m_to_km else 1.0
        # Get the midpoints (XY and Long, Lat)
        mptxy = np.column_stack([self.eqx, self.eqy, 1000.0 * self.eqz])
        vector = np.column_stack([self.ssx, self.ssy, np.zeros(self.nrec)]) - mptxy
        mptxy += (0.5 * vector)
        mptlon, mptlat = self.cart_to_geo.transform(mptxy[:, 0], mptxy[:, 1])
        # Build the output flatfile
        output = self.identifiers.copy()
        output["eqX"] = self.eqx * fct
        output["eqY"] = self.eqy * fct
        output["eqZ"] = self.eqz
        output["staX"] = self.ssx * fct
        output["staY"] = self.ssy * fct
        output["rsn_orig"] = self.data["record_id"].copy()
        output["eqid_orig"] = self.data["event_id"].copy()
        output["ssn_orig"] = self.data["station_id"].copy()
        output["event_time"] = self.data["event_time"].copy()
        output["mag"] = self.data["magnitude"].copy()
        output["Rrup"] = self.data["rrup"].copy()
        output["RJB"] = self.data["rjb"].copy()
        output["Repi"] = self.data["repi"].copy()
        output["Rhypo"] = self.data["rhypo"].copy()
        output["Rx"] = self.data["rx"].copy()
        output["Ry0"] = self.data["ry0"].copy()
        output["Vs30"] = self.data["vs30"].copy()
        output["eqLon"] = self.data["event_longitude"].copy()
        output["eqLat"] = self.data["event_latitude"].copy()
        output["staLon"] = self.data["station_longitude"].copy()
        output["staLat"] = self.data["station_latitude"].copy()
        output["mptLon"] = mptlon
        output["mptLat"] = mptlat
        output["mptX"] = mptxy[:, 0] * fct
        output["mptY"] = mptxy[:, 1] * fct
        output["UTMzone"] = [self.utmzone] * self.nrec
        if total_residual is not None:
            output["tot"] = total_residual
        if x_2 is not None:
            output["x_2"] = x_2
        if x_3 is not None:
            output["x_3"] = x_3
        if fname:
            output.to_csv(fname, sep=",", index=False)
        return output

    def event_catalogue(self):
        """Returns just the event catalogue as a dataframe
        """
        event_cat = self.data[self.SOURCE_HEADERS].copy()
        event_cat.drop_duplicates("event_id", inplace=True, ignore_index=True)
        event_cat.set_index("event_id", inplace=True)
        return event_cat

    def get_distance_matrix(self, grid: Grid3D, fname: Optional[str] = None):
        """
        """
        assert self.cartesian_crs == grid.cartesian_crs, "Cartesian CRS do not agree between"\
            f" flatfile ({self.cartesian_crs}) and grid ({grid.cartesian_crs})"

        distances = np.zeros([self.nrec, grid.nx * grid.ny, grid.nz], dtype="f")

        logging.info("Building cell distance matrix")
        markerpoint = self.nrec // 10

        for j, path in enumerate(self.xy_paths):
            if not j or not (j % markerpoint):
                logging.info("Running %g of %g" % (j, self.nrec))
            distances[j, :, :] = self._get_path_distances_through_cells(path, grid,
                                                                        self.eqz[j],
                                                                        j)[0]

        # Now have the entire distance grid, so convert this to the dataframe formatted
        # according to the distance matrix requirements
        logging.info("Building dataframe")
        distance_matrix = self.identifiers.copy() #[["rsn", "eqid", "ssn"]]
        cntr = 0
        for i in range(grid.nz):
            for j in range(grid.nx * grid.ny):
                distance_matrix[f"c.{cntr}"] = distances[:, j, i]
                cntr += 1
        if fname:
            logging.info(f"Distance matrix built, exporting to {fname}")
            distance_matrix.to_csv(fname, sep=",", index=False)
        return distance_matrix

    @staticmethod
    def export_distance_matrix_hdf5(distance_matrix: pd.DataFrame, filename: str):
        """Exports the distance matrix to an hdf5
        """
        nrow, ncol = distance_matrix.shape
        fle = h5py.File(filename, "a")
        for key in ["rsn", "eqid", "ssn"]:
            dset = fle.create_dataset(key, (nrow,), dtype="i")
            dset[:] = distance_matrix[key].to_numpy()
        dist_dset = fle.create_dataset("distance", (nrow, ncol - 3), dtype="f")
        dist_dset[:] = distance_matrix[:, 3:].to_numpy()
        fle.close()
        logging.info("Distance Matrix exported to %s" % filename)
        return

    @staticmethod
    def _get_path_distances_through_cells(path, grid, eqz, pos):
        """
        """
        locs = []
        paths_in_cell = []
        for loc in grid.grid_tree.intersection(path.bounds):
            # Get the path inside each specific polygon (if it intersects) 
            path_in_cell = grid.polygons[loc].intersection(path)
            if path_in_cell and isinstance(path_in_cell, geometry.LineString):
                locs.append(loc)
                paths_in_cell.append(path_in_cell)
        distances = np.zeros([grid.nx * grid.ny, grid.nz])
        if not len(locs):
            # Path doesn't intersect any cell - skip
            return distances, []
        #print(locs)
        if len(locs) == 1:
            #assert grid.polygons[locs[0]].contains(path), f"Pos: {pos}, Loc: {locs[0]}"
            # Both epicentre and site are in the same cell so divide the total
            # length in proportion to the depth change in the cell
            # Get the total length (approx. equal to rhypo)
            surface_length = path.length / 1000.0
            total_length = np.sqrt((path.length / 1000.0) ** 2. + eqz ** 2.0)
            #print(surface_length, total_length)
            for k in range(grid.nz):
                if eqz >= grid.depths[k + 1]:
                    # Earthquake is deeper than the lower depth of the cell
                    # so take the fraction of the path within the cell
                    dz = grid.depths[k + 1] - grid.depths[k]
                    distances[locs[0], k] = (dz / eqz) * total_length
                elif grid.depths[k] > eqz:
                    # Cell below earthquake - no contritbution
                    continue
                else:
                    # Earthquake is within the cell, so take the fraction
                    # distance between the earthquake and the upper depth
                    dz = eqz - grid.depths[k]
                    distances[locs[0], k] = (dz / eqz) * total_length
            return distances, locs
        # The path crosses multiple cells
        # Get the 3D vector for the path
        eqz_m = eqz * 1000.0
        eqxyz = np.array([path.xy[0][0], path.xy[1][0], eqz_m])  # EQ position (m)
        sitexyz = np.array([path.xy[0][1], path.xy[1][1], 0.0])  # Site position (m)
        vector_3d = sitexyz - eqxyz  # Vector (site - EQ) in m
        for loc, path_in_cell in zip(locs, paths_in_cell):
            surface_length = path_in_cell.length / 1000.0  # Surface length in km
            dz = (path_in_cell.length / path.length) * eqz  # Proportion of depth range in path
            # Total length of the vector in the column (in km)
            total_length = np.sqrt(surface_length ** 2. + dz ** 2.)
            # Get 3D vector from EQ point to intersection points
            try:
               p0 = np.array([path_in_cell.xy[0][0], path_in_cell.xy[1][0]]) - eqxyz[:2]
               p1 = np.array([path_in_cell.xy[0][1], path_in_cell.xy[1][1]]) - eqxyz[:2]
            except IndexError:
               print(path_in_cell, eqxyz, sitexyz, surface_length, total_length)
               raise
            # Get depths of the intersection points
            z0 = (eqxyz[2] - ((p0 / vector_3d[:2])[0] * eqxyz[2]))
            z1 = (eqxyz[2] - ((p1 / vector_3d[:2])[0] * eqxyz[2]))
            min_z = min(z0, z1) / 1000.0
            max_z = max(z0, z1) / 1000.0
            #print(loc, surface_length, dz, total_length)
            for k in range(grid.nz):
                #print("    ", grid.depths[k], grid.depths[k + 1], min_z, max_z)
                # Several cases here
                if (min_z >= grid.depths[k]) and (min_z < grid.depths[k + 1]) and\
                    (max_z >= grid.depths[k]) and (max_z < grid.depths[k + 1]):
                    # Both line end points are within the cell, which means that
                    # the total length requires only the surface length and the
                    # depth difference
                    dz_k = (max_z - min_z) / dz
                elif (min_z >= grid.depths[k]) and (min_z < grid.depths[k + 1]) and\
                    (max_z >= grid.depths[k + 1]):
                    # The shallowest depth is within the depth range of the cell
                    # but the largest depth is in a deeper cell
                    dz_k = (grid.depths[k + 1] - min_z) / dz
                elif (min_z < grid.depths[k]) and (max_z >= grid.depths[k]) and\
                    (max_z < grid.depths[k + 1]):
                    # The deeper point is within the depth range of the cell
                    # but the shallower point is within a shallower cell
                    dz_k = (max_z - grid.depths[k]) / dz
                else:
                    # No intersection between path and cell
                    continue
                distances[loc, k] = dz_k * total_length
        return distances, locs

## Rubbish below 
#    @classmethod
#    def from_egsim_flatfile(cls,
#        flatfile: pd.DataFrame,
#        cartesian_crs: str = "EPSG:3035",
#        geodetic_crs: str = "EPSG:4326"
#        ):
#        """
#        """
#        flatfile_new = flatfile.copy()
#        flatfile_new["eqid"] = flatfile["event_id"].copy()
#        flatfile_new["ssn"] = flatfile["station_id"].copy()
#        flatfile_new["rsn"] = flatfile["gmid"].copy()
#        flatfile_new["ztor"] = flatfile["depth_top_of_rupture"].copy()
#        idx = pd.isna(flatfile["depth_top_of_rupture"])
#        flatfile_new["ztor"][idx] = flatfile["event_depth"][idx]
#        return cls(flatfile_new, cartesian_crs)

#    @staticmethod
#    def _get_path_distances_through_cells(path, grid, eqz):
#        """
#        """
#        locs = list(grid.grid_tree.intersection(path.bounds))
#        distances = np.zeros([grid.nx * grid.ny, grid.nz])
#        if not len(locs):
#            # Path doesn't intersect bounding box at all - skip
#            return distances
#        print(locs)
#        if len(locs) == 1:
#            assert grid.polygons[locs[0]].contains(path)
#            # Both epicentre and site are in the same cell so divide the total
#            # length in proportion to the depth change in the cell
#            # Get the total length (approx. equal to rhypo)
#            total_length = np.sqrt((path.length / 1000.0) ** 2. + eqz ** 2.0)
#            print(total_length)
#            for k in range(grid.nz):
#                if (eqz > grid.depths[k]) and (eqz <= grid.depths[k + 1]):
#                    distances[locs[0], k] = ((eqz - grid.depths[k]) / eqz) * total_length
#            return distances
#
#        # Get the 3D vector for the path
#        eqxyz = np.array([path.xy[0][0], path.xy[1][0], eqz * 1000.])
#        sitexyz = np.array([path.xy[0][1], path.xy[1][1], 0.0])
#        vector_3d = sitexyz - eqxyz
#        print(eqxyz, sitexyz, vector_3d)
#        for loc in locs:
#            path_in_cell = grid.polygons[loc].intersection(path)
#            #print(loc, path_in_cell)
#            if path_in_cell:
#                # Intersection
#                surface_length = path_in_cell.length / 1000.0 # Length in cell in km
#                dz = (path_in_cell.length / path.length) * eqz 
#                total_length = np.sqrt(surface_length ** 2. + dz ** 2.)
#                print(loc, surface_length, total_length, dz)
#                # Get 3D vector from EQ point to intersection points
#                p0 = np.array([path_in_cell.xy[0][0], path_in_cell.xy[1][0]]) - eqxyz[:2]
#                p1 = np.array([path_in_cell.xy[0][1], path_in_cell.xy[1][1]]) - eqxyz[:2]
#                z0 = (eqxyz[2] - ((p0 / vector_3d[:2])[0] * eqxyz[2]))
#                z1 = (eqxyz[2] - ((p1 / vector_3d[:2])[0] * eqxyz[2]))
#                min_z = min(z0, z1) / 1000.0
#                max_z = max(z0, z1) / 1000.0
#                #print(p0, p1)
#                #print(z0, z1)
#                #print(min_z, max_z)
#                #continue
#                #print(p0, p1, np.linalg.norm(p0), np.linalg.norm(p1))
#                #if np.allclose(p0, 0.0):
#                #    # This is the first segment from the earthquake
#                #    dz = (eqz * 1000.0) - ((p1 / vector_3d[:2])[0] * (eqz * 1000.0))
#                #    max_z = eqz
#                #    min_z = eqz - dz
#                #else:
#                #    #print(p0, p1, p0 / vector_3d[:2], p1 / vector_3d[:2])
#                #    z0 = max(0, (eqz * 1000.0) - ((p0 / vector_3d[:2])[0] * (eqz * 1000.0)))
#                #    z1 = max(0, (eqz * 1000.0) - ((p1 / vector_3d[:2])[0] * (eqz * 1000.0)))
#                #    print(z0, z1)
#                #    min_z = min(z1, z0) / 1000.0
#                #    max_z = max(z1, z0) / 1000.0
#                #    dz = np.fabs(z1 - z0) / 1000.0
#
#                # Now have the total path length through the vertical
#                # column but need to know if is crosses multiple depth zones
#
#                for k in range(grid.nz):
#                    if eqz < grid.depths[k]:
#                        # Path should not intersect volume
#                        continue
#                    print(k, min_z, max_z, grid.depths[k], grid.depths[k + 1])
#                    if (min_z >= grid.depths[k]) and (min_z < grid.depths[k + 1]) and\
#                        (max_z >= grid.depths[k]) and (max_z < grid.depths[k + 1]):
#                        # If the depths of both of the intersecting points are
#                        # within the depth limits of the cell then the line is
#                        # entirely in the cell
#                        print(k, total_length)
#                        distances[loc, k] = total_length
#                    elif (min_z < grid.depths[k + 1]) and (max_z >= grid.depths[k + 1]):
#                        # Count the part of the path in the bin depths[k] <= depths[k + 1]
#                        distances[loc, k] = ((grid.depths[k + 1] - min_z) / (max_z - min_z)) *\
#                            total_length
#
#                    
#                       
#                        # Line is split across two or more depth layers
##                        if (min_z >= grid.depths[k]) and (min_z < grid.depths[k + 1]):
##                            # The line segment is between the minimum depth and the
##                            # deeper layer boundary
##                            dist = ((grid.depths[k + 1] - min_z) / dz) * total_length
##                        elif (max_z >= grid.depths[k]) and (max_z < grid.depths[k + 1]):
##                            # The line segment is between the maximum depth and
##                            # the shallower layer boundary
##                            dist = ((max_z - grid.depths[k]) / dz) * total_length
##                        else:
##                            # The entire path is in the cell
##                            dist = (grid.spcz / dz) * total_length
#                        distances[loc, k] = dist
#        return distances
