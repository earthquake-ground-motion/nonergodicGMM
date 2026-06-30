"""
Execute a non-ergodic GMM fitting workflow
"""
import os
import subprocess
from typing import Union, List, Tuple, Optional
from multiprocessing import cpu_count
import logging
import toml
import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from openquake.hazardlib import valid
from openquake.hazardlib.imt import from_string # PGA, PGV, SA
from dynamicgmm.nonergodic.grid_tools import Grid3D, Flatfile
from dynamicgmm.nonergodic import ergodic_gmms as gmms


AVAILABLE_GMMS = {
    "AbrahamsonEtAl2014": gmms.ASK2014Ergodic,
    "BooreEtAl2014": gmms.BSSA2014Ergodic,
    "CampbellBozorgnia2014": gmms.CB2014Ergodic,
    "ChiouYoungs2014": gmms.CY2014Ergodic,
    "KothaEtAl2020ESHM20": gmms.KothaEtAl2020ESHM20Ergodic,
    "BindiEtAl2014Rjb": gmms.BindiEtAl2014Ergodic
}


class Complete():
    returncode = 0


def setup_flatfile_grid(
        flatfile: pd.DataFrame,
        bbox: Union[List, Tuple],
        spcx: float, spcy: float,
        output_folder: str,
        query_string: str = "",
        **grid_kwargs
) -> Tuple[Flatfile, Grid3D]:
    """Sets up the flatfile and the grid needed for fitting the non-ergodic models,
    including the distance of each record path through the grid cells

    Args:
        flatfile: Ground motion data flatfile
        bbox: Bounding box of the grid as list/tuple of [west, south, east, north]
        spcx: Longitude spacing of the grid (decimal degrees)
        spcy: Latitude spacing of the grid (decimal degrees)
        output_folder: Folder for collecting results (must not exist)
        query_string: String to execute a query on the flatfile

    Returns:
        working_flatfile: Flatfile as instance of Flatfile class
        grid: Target grid as a Grid3D object
    """
    if os.path.exists(output_folder):
        raise OSError("Folder for outputs %s already exists" % output_folder)
    os.mkdir(output_folder)
    if query_string:
        # Query the flatfile if a query string is passed
        data = flatfile.query(query_string, inplace=False).copy()
        data.reset_index(inplace=True, drop=True)
    else:
        data = flatfile.copy()
    # bbox = [w, s, e, n] - ignore dateline issues for now
    assert bbox[2] > bbox[0], "East limit of bounding box should exceed west limit"
    assert bbox[3] > bbox[1], "North limit of bounding box should exceed south limit"
    zbox = grid_kwargs.get("zbox", None)
    spcz = grid_kwargs.get("spcz", 50.0)
    geodetic_crs = grid_kwargs.get("geodetic_crs", "EPSG:4326")
    cartesian_crs = grid_kwargs.get("cartesian_crs", "EPSG:3035")
    logging.info("Building grid")
    grid = Grid3D(bbox, spcx, spcy, zbox, spcz, geodetic_crs, cartesian_crs)
    logging.info("Grid built:")
    logging.info(str(grid))
    working_flatfile = Flatfile(data, utmzone="32N")
    # Get the cell crossing distances in the file
    logging.info("Building distance matrix")
    distance_matrix = working_flatfile.get_distance_matrix(grid)
    cell_dataframe = grid.to_cell_info("32N")
    # Export the files to the output folder
    cell_dataframe.to_csv(os.path.join(output_folder, "cellinfo.csv"))
    logging.info("Exporting distance matrix")
    working_flatfile.export_distance_matrix_hdf5(
        distance_matrix,
        os.path.join(output_folder, "distance_matrix.hdf5")
    )
    return working_flatfile, grid


def fit_nonergodic_inla_workflow(config_file: str, num_cores: Optional[int] = None):
    """Fits a non-ergodic ground motion model to a data set
    """
    config = toml.load(config_file)
    if not num_cores:
        num_cores = cpu_count()
    # setup the flatfile and grid
    logging.info("Setting up flatfile and grid")
    input_flatfile = pd.read_csv(config["flatfile"], sep=",")
    flatfile, grid = setup_flatfile_grid(
        input_flatfile,
        query_string=config["query"],
        output_folder=config["output_folder"],
        **config["grid"]
    )
    assert config["flatfile_acceleration_units"] in \
        ("g", "m/s", "m s^-1", "cm/s/s", "cm s^-2", "gal"), \
        "Flatfile acceleration units %s not recognised" % config["flatfile_acceleration_units"]
    for model in config["model"]:
        imts = [from_string(imt) for imt in model["imts"]]
        gmm = AVAILABLE_GMMS[model["ergodic_gmm"]]()
        logging.info("Running with ergodic model %s and non-ergodic type %g"
                     % (model["ergodic_gmm"], model["type"]))
        total_residuals = gmm.get_total_residual(
            flatfile=flatfile.data, imts=imts,
            function_type=model["type"],
            sa_units=config["flatfile_acceleration_units"]
            )
        for imt in imts:
            logging.info("--- IMT: %s" % str(imt))
            subfolder_name = os.path.join(
                config["output_folder"],
                f"{model['identifier']}_{str(imt)}_Type{model['type']}"
            )
            os.mkdir(subfolder_name)
            data_file = os.path.join(subfolder_name, "ground_motion_data.csv")
            flatfile.export_to_stan_flatfile(
                data_file,
                total_residual=total_residuals[str(imt)].to_numpy(),
                m_to_km=True
            )
            config_filename = os.path.join(subfolder_name, "config.toml")
            model["files"]["input"] = data_file
            model["files"]["coefficients"] = os.path.join(subfolder_name, "coefficients.csv")
            model["files"]["hyperparameters"] = os.path.join(
                subfolder_name, "hyperparameters.csv"
            )
            model["files"]["hyperposteriors"] = os.path.join(
                subfolder_name, "hyperposteriors.csv"
            )
            model["files"]["residuals"] = os.path.join(subfolder_name, "residuals.csv")
            if model["type"] > 1:
                # Also the attenuation files
                model["files"]["cell_mat"] = os.path.join(config["output_folder"],
                                                          "distance_matrix.hdf5")
                model["files"]["cell_info"] = os.path.join(config["output_folder"],
                                                           "cellinfo.csv")
                model["files"]["attenuation"] = os.path.join(subfolder_name, "attenuation.csv")

            with open(config_filename, "w") as f:
                # Save the configuration toml specific to this IMT and model
                toml.dump(model, f)
            # Run the INLA script
            logging.info("---- ---- Running INLA Regression with R")
            command_list = [
                "Rscript",
                os.path.join(os.path.dirname(__file__), "nonergodic_gmm_fit_INLA.R"),
                "--config", config_filename,
                "--num_cores", str(num_cores),
                ]
            logging.info("cmd >> %s" % subprocess.list2cmdline(command_list))
            complete = subprocess.run(command_list)
            # Just create the output files for now
#            open(model["files"]["coefficients"], "a").close()
#            open(model["files"]["hyperparameters"], "a").close()
#            open(model["files"]["hyperposteriors"], "a").close()
#            open(model["files"]["residuals"], "a").close()
#            if model["type"] > 1:
#                open(model["files"]["attenuation"], "a").close()
#            complete = Complete()

            if complete.returncode == 0:
                logging.info("---- ---- Run complete")
            else:
                logging.info("---- ---- Run failed!")
    return
