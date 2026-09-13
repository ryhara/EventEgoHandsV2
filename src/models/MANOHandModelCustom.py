import os
import torch
import smplx

from hot3d.hot3d.data_loaders.mano_layer import MANOHandModel, mano_joint_mapping

class MANOHandModelCustom(MANOHandModel):
    """
    Extended MANOHandModel with PCA conversion functionality.
    Converts hand_pose from PCA coefficients (15D) to full axis-angle representation (45D).

    Args:
        mano_model_files_dir: Path to directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl
        joint_mapper: Optional joint mapping list. If None, uses default mano_joint_mapping
    """

    def __init__(
        self,
        mano_model_files_dir: str,
        joint_mapper=mano_joint_mapping,
        num_pose_coeffs: int = 15,
        use_pose_pca: bool = True,
        num_shape_params: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            mano_model_files_dir,
            joint_mapper=joint_mapper,
            num_pose_coeffs=num_pose_coeffs,
        )

        self.use_pose_pca = use_pose_pca
        self.num_pose_coeffs = num_pose_coeffs
        self.num_shape_params = num_shape_params
        self.device = device
        self.dtype = dtype

        mano_left_filename = os.path.join(mano_model_files_dir, "MANO_LEFT.pkl")
        mano_right_filename = os.path.join(mano_model_files_dir, "MANO_RIGHT.pkl")

        self.mano_layer_left = smplx.create(
            mano_left_filename,
            "mano",
            use_pca=self.use_pose_pca,
            is_rhand=False,
            num_pca_comps=self.num_pose_coeffs,
        )
        self.mano_layer_left.to(self.device)

        self.mano_layer_right = smplx.create(
            mano_right_filename,
            "mano",
            use_pca=self.use_pose_pca,
            is_rhand=True,
            num_pca_comps=self.num_pose_coeffs,
        )
        self.mano_layer_right.to(self.device)

        # Fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if (
            torch.sum(
                torch.abs(
                    self.mano_layer_left.shapedirs[:, 0, :]
                    - self.mano_layer_right.shapedirs[:, 0, :]
                )
            )
            < 1
        ):
            self.mano_layer_left.shapedirs[:, 0, :] *= -1

    def to(self, device):
        """Move model to specified device."""
        self.device = device
        self.mano_layer_left = self.mano_layer_left.to(device)
        self.mano_layer_right = self.mano_layer_right.to(device)
        # Also update parent class device attribute if it exists
        if hasattr(super(), "device"):
            super().device = device
        return self

    def convert_hand_pose_pca_to_full(
        self,
        hand_pose_pca: torch.Tensor,
        global_orient: torch.Tensor = None,
        is_right_hand: bool = True,
    ) -> torch.Tensor:
        """
        Convert hand_pose from PCA coefficients (15D) to full axis-angle representation (45D).

        Args:
            hand_pose_pca: PCA coefficients, shape (15,) or (batch_size, 15)
            global_orient: Global orientation (not used in current implementation)
            is_right_hand: Whether this is for right hand (True) or left hand (False)

        Returns:
            hand_pose_full: Full axis-angle representation, shape (45,) or (batch_size, 45)
        """
        if is_right_hand:
            components = self.mano_layer_right.hand_components
        else:
            components = self.mano_layer_left.hand_components

        is_batched = len(hand_pose_pca.shape) == 2
        if not is_batched:
            hand_pose_pca = hand_pose_pca.unsqueeze(0)

        hand_pose_pca = hand_pose_pca.to(components.device)
        hand_pose_full = torch.einsum("bi,ij->bj", [hand_pose_pca, components])

        if not is_batched:
            hand_pose_full = hand_pose_full.squeeze(0)

        return hand_pose_full
