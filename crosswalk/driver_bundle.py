"""Select a hash-pinned set of trained city-driver components."""
import hashlib
import io
import json
from pathlib import Path

from .city_maneuvers import FEATURE_NAMES, VERSION, ManeuverModel
from .maneuver_choice import ACTIONS, FEATURES, ManeuverChoice
from .stationary_intent import StationaryIntent


FORMAT = "ado-city-driver-bundle-v1"


def _choice_from_bytes(payload):
    # The same checks as ManeuverChoice(path), applied to the exact hashed bytes.
    data = json.loads(payload)
    if data["format"] != "ado-route-maneuvers-1" or data["features"] != list(FEATURES) or data["actions"] != list(ACTIONS):
        raise ValueError("Incompatible route maneuver checkpoint")
    return ManeuverChoice(tree=data["tree"])


def load_bundle(path):
    path = Path(path).resolve(strict=True)
    bundle_bytes = path.read_bytes()
    definition = json.loads(bundle_bytes)
    if definition.get("format") != FORMAT or definition.get("schema") != VERSION:
        raise ValueError(f"Unsupported city driver bundle: {path.name}")

    def component(name):
        entry = definition.get(name, {})
        relative = Path(entry.get("path", ""))
        expected = entry.get("sha256", "")
        if relative.is_absolute() or relative == Path(".") or len(expected) != 64:
            raise ValueError(f"Invalid {name} entry in {path.name}")
        return (path.parent / relative).resolve(strict=True), expected.lower()

    def snapshot(name):
        selected, expected = component(name)
        payload = selected.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise ValueError(f"Selected {name} checksum differs from {path.name}")
        return selected, payload, observed

    pace_path, pace_bytes, pace_hash = snapshot("pace")
    choice_path, choice_bytes, choice_hash = snapshot("maneuver_choice")
    intent_path, intent_hash = component("stationary_intent")
    intent = StationaryIntent(intent_path)
    if intent.sha256 != intent_hash:
        raise ValueError(f"Selected stationary_intent checksum differs from {path.name}")

    driver = ManeuverModel.load(io.BytesIO(pace_bytes))
    driver.driver_bundle_checkpoint = path
    driver.driver_bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    driver.driver_bundle_bytes = bundle_bytes
    driver.pace_checkpoint, driver.pace_sha256 = pace_path, pace_hash
    driver.pace_checkpoint_bytes = pace_bytes
    driver.maneuver_choice_checkpoint, driver.maneuver_choice_sha256 = choice_path, choice_hash
    driver.maneuver_choice_bytes = choice_bytes
    driver.maneuver_choice_model = _choice_from_bytes(choice_bytes)
    driver.stationary_intent_model = intent
    driver.stationary_intent_checkpoint, driver.stationary_intent_sha256 = intent.checkpoint_path, intent.sha256
    return driver, FEATURE_NAMES, VERSION
