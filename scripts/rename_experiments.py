#!/usr/bin/env python3
"""Rename old experiments to deterministic naming convention.

Computes the mapping and performs the rename operations.
"""

import json
import shutil
from pathlib import Path

from phd.training.experiments import generate_experiment_name


def main():
    registry_path = Path("experiments/experiments_registry.json")
    experiments_dir = Path("experiments/train_nn1_cropped")

    with open(registry_path) as f:
        registry = json.load(f)

    # Identify old experiments (completed ones with random IDs)
    # These are experiments that need renaming
    old_experiments = {}
    for name, data in registry.items():
        status = data.get("status", "")
        # Old completed experiments have status like EARLY_STOPPING, NAN_VALUE_DETECTED, SKIPPED
        if status in ("EARLY_STOPPING", "NAN_VALUE_DETECTED", "SKIPPED"):
            config = data["config"]
            new_name = generate_experiment_name(config)
            # Only rename if the name doesn't already match
            if name != new_name:
                old_experiments[name] = {
                    "new_name": new_name,
                    "status": status,
                    "has_directory": (experiments_dir / name).exists(),
                }

    # Identify NOT_STARTED duplicates to delete
    # Build set of deterministic names for completed experiments
    deterministic_names = set(e["new_name"] for e in old_experiments.values())

    # Also include experiments that already have deterministic names
    for name, data in registry.items():
        status = data.get("status", "")
        if status in ("EARLY_STOPPING", "NAN_VALUE_DETECTED", "SKIPPED"):
            if name not in old_experiments:
                # Already has deterministic name
                deterministic_names.add(name)

    duplicates_to_delete = []
    for name, data in registry.items():
        if data.get("status") == "NOT_STARTED":
            config = data["config"]
            new_name = generate_experiment_name(config)
            # If this NOT_STARTED experiment has same deterministic name as a completed one
            if new_name in deterministic_names:
                duplicates_to_delete.append(name)

    # Print mapping
    print("=" * 60)
    print("RENAME MAPPING (completed experiments)")
    print("=" * 60)
    if not old_experiments:
        print("  (none)")
    for old_name, info in sorted(old_experiments.items()):
        print(f"  {old_name}")
        print(f"    -> {info['new_name']}")
        print(f"    status: {info['status']}, has_dir: {info['has_directory']}")
        print()

    print("=" * 60)
    print("DUPLICATES TO DELETE (NOT_STARTED)")
    print("=" * 60)
    if not duplicates_to_delete:
        print("  (none)")
    for name in sorted(duplicates_to_delete):
        print(f"  - {name}")

    # Ask for confirmation
    print()
    response = input("Proceed with rename? [y/N]: ")
    if response.lower() != "y":
        print("Aborted.")
        return

    # Perform renames
    new_registry = {}

    for name, data in registry.items():
        # Skip duplicates
        if name in duplicates_to_delete:
            print(f"Deleting duplicate: {name}")
            continue

        # Rename completed experiments
        if name in old_experiments:
            new_name = old_experiments[name]["new_name"]
            print(f"Renaming: {name} -> {new_name}")

            # Rename directory
            old_dir = experiments_dir / name
            new_dir = experiments_dir / new_name
            if old_dir.exists():
                if new_dir.exists():
                    print(f"  WARNING: Target directory exists, removing: {new_dir}")
                    shutil.rmtree(new_dir)
                old_dir.rename(new_dir)
                print("  Renamed directory")

            # Update config.json in the directory
            config_path = new_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                config["exp_name"] = new_name
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print("  Updated config.json")

            # Update registry entry
            data["config"]["exp_name"] = new_name
            new_registry[new_name] = data
        else:
            # Keep as-is
            new_registry[name] = data

    # Save updated registry
    with open(registry_path, "w") as f:
        json.dump(new_registry, f, indent=2)
    print(f"\nSaved updated registry with {len(new_registry)} experiments")

    # Print summary
    print("\nSummary:")
    print(f"  Renamed: {len(old_experiments)}")
    print(f"  Deleted: {len(duplicates_to_delete)}")
    print(f"  Total remaining: {len(new_registry)}")


if __name__ == "__main__":
    main()
