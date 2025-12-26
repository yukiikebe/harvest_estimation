from pathlib import Path
import yaml


class Config:
    """
    Configuration holder for the harvest pipeline.
    """

    def __init__(
        self,
        cdl_yaml_path,
        *,
        allowed_crops=None,
        min_points=11,
        savgol_window=11,
        savgol_polyorder=3,
        output_root="./outputs",
        gt_windows=None,
    ):
        self.cdl_yaml_path = Path(cdl_yaml_path)
        self.output_root = Path(output_root)
        self.gt_windows = gt_windows or {}

        self._load_cdl_yaml()

        self.allowed_crops = (
            set(allowed_crops)
            if allowed_crops is not None
            else set(self.crop_dict.values())
        )

        self.min_points = min_points
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

        self._validate()

    def _load_cdl_yaml(self):
        if not self.cdl_yaml_path.exists():
            raise FileNotFoundError(
                f"CDL config not found: {self.cdl_yaml_path}"
            )

        with open(self.cdl_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # numeric label -> crop name
        self.crop_dict = {
            int(k): v for k, v in data.get("num2class", {}).items()
        }

        # numeric labels that exist in CDL
        self.valid_crop_labels = set(
            int(k) for k in data.get("crop_type", {}).keys()
        )

    def _validate(self):
        unknown = self.allowed_crops - set(self.crop_dict.values())
        if unknown:
            raise ValueError(
                f"Allowed crops not in CDL config: {unknown}"
            )

        if self.savgol_window % 2 == 0:
            raise ValueError("savgol_window must be odd")

        if self.savgol_window <= self.savgol_polyorder:
            raise ValueError(
                "savgol_window must be larger than savgol_polyorder"
            )
        
        for crop, (start, end) in self.gt_windows.items():
            if crop not in self.crop_dict.values():
                raise ValueError(f"GT window crop not in CDL: {crop}")

    def is_valid_crop_label(self, label):
        return label in self.valid_crop_labels

    def crop_name(self, label):
        return self.crop_dict.get(label, f"Crop_{label}")
