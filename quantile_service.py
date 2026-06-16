import numpy as np
import warnings
from typing import List, Union, Optional, Dict


class QuantileService:
    def __init__(self):
        self._valid_methods = {"linear", "midpoint", "nearest"}

    def _validate_inputs(
        self,
        data: Union[List[float], np.ndarray],
        quantiles: Union[float, List[float], np.ndarray],
        method: str,
    ) -> None:
        if method not in self._valid_methods:
            raise ValueError(
                f"Invalid interpolation method '{method}'. "
                f"Must be one of: {', '.join(sorted(self._valid_methods))}"
            )

        if len(data) == 0:
            raise ValueError("Data sequence cannot be empty")

        if isinstance(quantiles, (list, np.ndarray)):
            quantiles_arr = np.asarray(quantiles, dtype=float)
            if quantiles_arr.size == 0:
                raise ValueError("Quantiles list cannot be empty")
            if np.any((quantiles_arr < 0) | (quantiles_arr > 1)):
                raise ValueError("All quantiles must be in the range [0, 1]")
        else:
            q = float(quantiles)
            if q < 0 or q > 1:
                raise ValueError("Quantile must be in the range [0, 1]")

    def compute(
        self,
        data: Union[List[float], np.ndarray],
        quantiles: Union[float, List[float], np.ndarray],
        method: str = "linear",
    ) -> Union[float, np.ndarray]:
        self._validate_inputs(data, quantiles, method)

        data_arr = np.asarray(data, dtype=float)
        sorted_data = np.sort(data_arr)
        n = len(sorted_data)

        if n < 2:
            warnings.warn(
                f"Sample size is {n} (< 2). Interpolation is not meaningful; "
                f"returning the original value directly.",
                UserWarning,
                stacklevel=2,
            )
            single_val = float(sorted_data[0])
            if isinstance(quantiles, (list, np.ndarray)):
                q_arr = np.atleast_1d(np.asarray(quantiles, dtype=float))
                return np.full(q_arr.shape, single_val)
            return single_val

        if method == "linear":
            return self._linear_interpolation(sorted_data, quantiles)
        elif method == "midpoint":
            return self._midpoint_interpolation(sorted_data, quantiles)
        elif method == "nearest":
            return self._nearest_interpolation(sorted_data, quantiles)

    def _get_positions(self, sorted_data: np.ndarray, quantiles) -> np.ndarray:
        n = len(sorted_data)
        quantiles_arr = np.atleast_1d(np.asarray(quantiles, dtype=float))
        positions = quantiles_arr * (n - 1)
        return positions

    def _linear_interpolation(self, sorted_data: np.ndarray, quantiles) -> np.ndarray:
        positions = self._get_positions(sorted_data, quantiles)
        lower_idx = np.floor(positions).astype(int)
        upper_idx = np.ceil(positions).astype(int)
        frac = positions - lower_idx

        lower_vals = sorted_data[lower_idx]
        upper_vals = sorted_data[upper_idx]

        result = lower_vals + frac * (upper_vals - lower_vals)

        if result.size == 1 and not isinstance(quantiles, (list, np.ndarray)):
            return float(result[0])
        return result

    def _midpoint_interpolation(self, sorted_data: np.ndarray, quantiles) -> np.ndarray:
        positions = self._get_positions(sorted_data, quantiles)
        lower_idx = np.floor(positions).astype(int)
        upper_idx = np.ceil(positions).astype(int)

        lower_vals = sorted_data[lower_idx]
        upper_vals = sorted_data[upper_idx]

        result = (lower_vals + upper_vals) / 2.0

        if result.size == 1 and not isinstance(quantiles, (list, np.ndarray)):
            return float(result[0])
        return result

    def _nearest_interpolation(self, sorted_data: np.ndarray, quantiles) -> np.ndarray:
        positions = self._get_positions(sorted_data, quantiles)
        nearest_idx = np.round(positions).astype(int)
        result = sorted_data[nearest_idx]

        if result.size == 1 and not isinstance(quantiles, (list, np.ndarray)):
            return float(result[0])
        return result

    def compute_batch(
        self,
        data: Union[List[float], np.ndarray],
        quantiles: Union[List[float], np.ndarray],
        method: str = "linear",
    ) -> Dict[float, float]:
        if not isinstance(quantiles, (list, np.ndarray)):
            raise TypeError(
                "quantiles must be a list or ndarray for batch computation. "
                "Use compute() for a single quantile."
            )
        quantiles_arr = np.asarray(quantiles, dtype=float)
        values = self.compute(data, quantiles_arr, method=method)
        return {float(q): float(v) for q, v in zip(quantiles_arr, values)}

    def compute_multi_method(
        self,
        data: Union[List[float], np.ndarray],
        quantiles: Union[float, List[float], np.ndarray],
        methods: Optional[Union[List[str], np.ndarray]] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        if methods is None:
            methods_list = list(self._valid_methods)
        else:
            methods_list = list(methods)
            for m in methods_list:
                if m not in self._valid_methods:
                    raise ValueError(
                        f"Invalid interpolation method '{m}'. "
                        f"Must be one of: {', '.join(sorted(self._valid_methods))}"
                    )
        return {m: self.compute(data, quantiles, method=m) for m in methods_list}

    def compute_five_number_summary(
        self,
        data: Union[List[float], np.ndarray],
        method: str = "linear",
    ) -> Dict[str, float]:
        summary_qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        labels = ["min", "q1", "median", "q3", "max"]
        result = self.compute_batch(data, summary_qs, method=method)
        return {label: result[q] for label, q in zip(labels, summary_qs)}


def compute_quantile(
    data: Union[List[float], np.ndarray],
    quantiles: Union[float, List[float], np.ndarray],
    method: str = "linear",
) -> Union[float, np.ndarray]:
    service = QuantileService()
    return service.compute(data, quantiles, method)


def compute_quantile_batch(
    data: Union[List[float], np.ndarray],
    quantiles: Union[List[float], np.ndarray],
    method: str = "linear",
) -> Dict[float, float]:
    service = QuantileService()
    return service.compute_batch(data, quantiles, method)


def compute_quantile_multi_method(
    data: Union[List[float], np.ndarray],
    quantiles: Union[float, List[float], np.ndarray],
    methods: Optional[Union[List[str], np.ndarray]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    service = QuantileService()
    return service.compute_multi_method(data, quantiles, methods)


def compute_five_number_summary(
    data: Union[List[float], np.ndarray],
    method: str = "linear",
) -> Dict[str, float]:
    service = QuantileService()
    return service.compute_five_number_summary(data, method)
