# sim/voltage_fits.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import constants
from trapping_variables import DriveKey, drive_colname


class VoltageFitsMixin:
    """
    Methods that *fit* local polynomial models to per-drive TotalV columns.
    Expects `self` to have:
      - self.total_voltage_df : pandas.DataFrame with x,y,z and per-drive TotalV columns
      - self.trapVariables    : Trapping_Vars
      - self.center_fits      : dict to store results per DriveKey
    """

    def update_center_polys(self, polyfit_deg=4):
        drives = self.trapVariables.get_drives()
        for drive in drives:
            self.center_fits[drive] = self.get_voltage_poly_for_drive_at_region(
                drive,
                region_x_low=-constants.center_region_x_um,
                region_x_high=constants.center_region_x_um,
                region_y_low=-constants.center_region_y_um,
                region_y_high=constants.center_region_y_um,
                region_z_low=-constants.center_region_z_um,
                region_z_high=constants.center_region_z_um,
                polyfit=polyfit_deg,
            )
        return

    def get_voltage_poly_for_drive_at_region(
        self,
        drive: DriveKey,
        region_x_low=-100,
        region_x_high=100,
        region_y_low=-10,
        region_y_high=10,
        region_z_low=-10,
        region_z_high=10,
        max_pnts=1e6,
        polyfit=4,
    ):

        # region cutout (bounds in µm; DF stores meters)
        df = self.total_voltage_df
        cutout = df[
            df["x"].between(region_x_low * 1e-6, region_x_high * 1e-6)
            & df["y"].between(region_y_low * 1e-6, region_y_high * 1e-6)
            & df["z"].between(region_z_low * 1e-6, region_z_high * 1e-6)
        ].copy()

        if drive.f_uHz == 0:
            column_name = "Static_TotalV"
        else:
            column_name = drive_colname(drive)

        if column_name not in cutout.columns:
            raise KeyError(
                f"Voltage column '{column_name}' not found. Available: {list(cutout.columns)}"
            )

        # drop rows with NaNs in inputs/target
        cutout.dropna(subset=["x", "y", "z", column_name], inplace=True)

        # (optional) cap point count
        max_pnts_int = int(max_pnts)
        if len(cutout) > max_pnts_int:
            cutout = cutout.sample(n=max_pnts_int, random_state=1)
            print("len(cutout) was cut to:", len(cutout))
        # print("len(cutout):", len(cutout))
        if len(cutout) == 0:
            raise ValueError("No points in region after filtering/sampling.")

        # features/target
        xyz_vals = cutout[["x", "y", "z"]].values
        voltage_vals = cutout[column_name].values

        poly = PolynomialFeatures(degree=polyfit, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals)

        model = LinearRegression(fit_intercept=False)  # avoid double bias term
        model.fit(X_poly, voltage_vals)

        r2 = model.score(X_poly, voltage_vals)
        if r2 < 0.999:
            print(
                "WARNING: Low R-squared for polynomial fit in get_voltage_poly_for_drive_at_region"
            )
            print("R-squared:", r2)

        return model, poly, r2



if __name__ == "__main__":
    print("hi")
