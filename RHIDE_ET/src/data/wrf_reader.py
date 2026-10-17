# wrf_reader.py
from __future__ import annotations
from typing import Optional
import numpy as np
try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None

class WRFReader:
    def __init__(self, nc_path: str, var_map: dict):
        if Dataset is None:
            raise ImportError("netCDF4 is required to read WRF files.")

        self.nc_path = nc_path
        self.var_map = var_map or {}

        # Open once to validate names that must exist
        ds = Dataset(self.nc_path, "r")

        # Required met fields (we keep these checks as before)
        must_have = ["XLAT", "XLONG"]
        missing = [k for k in must_have if self.var_map.get(k, k) not in ds.variables]
        if missing:
            ds.close()
            raise KeyError(f"Variables missing in WRF file: {missing}")

        # ---- Robust time handling ----
        # Many WRF files have a char variable 'Times' and a dimension 'Time'.
        # Older readers sometimes expect a variable 'Time' (numeric).
        var_names = set(ds.variables.keys())
        dim_names = set(ds.dimensions.keys())

        self._time_dim_name: Optional[str] = "Time" if "Time" in dim_names else None
        # Prefer 'Times' if present; else 'Time' variable if present; else None.
        if "Times" in var_names:
            self._time_var_name = "Times"
        elif "Time" in var_names:
            self._time_var_name = "Time"
        else:
            self._time_var_name = None  # fine; we will use dimension size

        if self._time_dim_name is None and self._time_var_name is None:
            # Extremely unusual WRF—still proceed, but T will become 1
            self._time_dim_name = None

        ds.close()

    def _read_times(self, ds):
        """Return (T, times_array_or_None). Works with char 'Times', numeric 'Time', or neither."""
        # Determine T from dimension if available
        if self._time_dim_name and self._time_dim_name in ds.dimensions:
            T = len(ds.dimensions[self._time_dim_name])
        else:
            # Fallback: try to infer from 'Times' variable first
            if self._time_var_name == "Times" and "Times" in ds.variables:
                T = ds.variables["Times"].shape[0]
            elif self._time_var_name == "Time" and "Time" in ds.variables:
                T = ds.variables["Time"].shape[0]
            else:
                T = 1  # last resort

        times = None
        if self._time_var_name == "Times" and "Times" in ds.variables:
            # Decode char array Times[Time, DateStrLen] -> array of strings
            carr = ds.variables["Times"][:]  # shape (T, DateStrLen)
            try:
                times = np.array(["".join(row.astype(str)).strip() for row in carr])
            except Exception:
                # NetCDF4 char arrays may come as bytes
                times = np.array(["".join([c.decode() if hasattr(c, "decode") else str(c) for c in row]).strip()
                                  for row in carr])
        elif self._time_var_name == "Time" and "Time" in ds.variables:
            # Numeric time; just return the raw values
            times = np.array(ds.variables["Time"][:])

        return int(T), times

    def read_all(self):
        ds = Dataset(self.nc_path, "r")

        # Shapes
        T, times = self._read_times(ds)
        ny = len(ds.dimensions[self.var_map.get("south_north", "south_north")]) \
             if self.var_map.get("south_north", "south_north") in ds.dimensions else \
             ds.variables[self.var_map.get("XLAT", "XLAT")].shape[-2]
        nx = len(ds.dimensions[self.var_map.get("west_east", "west_east")]) \
             if self.var_map.get("west_east", "west_east") in ds.dimensions else \
             ds.variables[self.var_map.get("XLAT", "XLAT")].shape[-1]

        # Read 2D features (examples)
        def v(name):  # helper: map logical->actual var name
            return self.var_map.get(name, name)

        out = {
            "meta": {
                "T": T,
                "ny": ny,
                "nx": nx,
                "times": times,  # can be None
            },
            # pull only what your pipeline needs; example:
            "XLAT": np.array(ds.variables[v("XLAT")][:]),
            "XLONG": np.array(ds.variables[v("XLONG")][:]),
        }

        # Add selected feature variables if present
        for key in ("RAINC", "RAINNC", "OLR", "SNOW", "SNOWH", "PSFC", "P", "U", "V", "T", "PB", "QVAPOR"):
            varname = v(key)
            if varname in ds.variables:
                out[key] = np.array(ds.variables[varname][:])

        ds.close()
        return out

