"""
Modality config for Unitree G1 + Dex3 hands (arm+hand only, 28 DOF).

State/Action layout (must match meta/modality.json):
  [0:7]   left_arm   - shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
  [7:14]  right_arm  - same joints, right side
  [14:21] left_hand  - thumb0/1/2, middle0/1, index0/1
  [21:28] right_hand - same fingers, right side

Register this config by passing it to finetune.sh:
  --modality-config-path examples/G1_Dex3/g1_dex3_config.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

g1_dex3_config = {
    # Video: current frame only (delta_indices=[0])
    # Keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    # State: current proprioceptive reading
    # Keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    # Action: predict 16 future steps (action horizon)
    # One ActionConfig per modality_key, in the same order
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[
            # Arms: RELATIVE = predict delta from current joint position
            # Better generalization across starting poses
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Hands: ABSOLUTE = predict target joint position directly
            # G1 Dex3 finger joints are controlled like binary/target signals
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction text
    # "annotation.human.task_description" maps to task_index via modality.json
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
