#!/usr/bin/env python3
"""
Dynamic Optuna storage URL generation.

Automatically detects the current PostgreSQL host and builds the connection string.
This allows the database to move between compute nodes without updating .env files.

Usage:
    from optuna_storage import get_optuna_storage_url

    storage = get_optuna_storage_url()
    study = optuna.create_study(storage=storage, ...)
"""
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load .env configuration
load_dotenv()

# Status file where PostgreSQL job writes its hostname
STATUS_FILE = Path.home() / ".optuna_db_host"


def get_db_host():
    """
    Auto-detect PostgreSQL database host.

    Tries in order:
    1. Status file written by PostgreSQL job
    2. SLURM queue query
    3. GNN_OPTUNA_STORAGE from .env
    4. Default fallback

    Returns:
        str: Database hostname (e.g., 'your-compute-node')
    """
    # Try status file first (fastest, but verify PostgreSQL is actually running)
    if STATUS_FILE.exists():
        try:
            host = STATUS_FILE.read_text().strip()
            if host:
                # Strip domain suffix if present (your-compute-node -> your-compute-node)
                host = host.split('.')[0]

                # Verify PostgreSQL is actually running (check SLURM queue)
                try:
                    result = subprocess.run(
                        ["squeue", "-u", os.getenv("USER"), "-o", "%j %N", "-h"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        # Check if postgres job is actually running on this node
                        for line in result.stdout.strip().split("\n"):
                            if not line:
                                continue
                            if "postgres" in line.lower() or "optuna" in line.lower() or "database" in line.lower():
                                # PostgreSQL is running, cache is valid
                                return host
                        # PostgreSQL not running - delete stale cache and continue to other methods
                        STATUS_FILE.unlink()
                except:
                    # If SLURM check fails, trust the cache (might be in different cluster)
                    return host
        except:
            pass

    # Try SLURM queue
    slurm_available = False
    postgres_found = False

    try:
        result = subprocess.run(
            ["squeue", "-u", os.getenv("USER"), "-o", "%j %N", "-h"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            slurm_available = True
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    job_name, node = parts
                    if any(keyword in job_name.lower() for keyword in ["postgres", "optuna", "database"]):
                        postgres_found = True
                        host = node.strip()
                        # Strip domain suffix if present (your-compute-node -> your-compute-node)
                        host = host.split('.')[0]
                        # Cache it
                        try:
                            STATUS_FILE.write_text(host + "\n")
                        except:
                            pass
                        return host
    except FileNotFoundError:
        # SLURM not available (different cluster or not installed)
        slurm_available = False
    except:
        pass

    # If SLURM is available but PostgreSQL is NOT running, fail loudly
    if slurm_available and not postgres_found:
        import sys
        print("ERROR: PostgreSQL Optuna database is not running!", file=sys.stderr)
        print("", file=sys.stderr)
        print("Start PostgreSQL with:", file=sys.stderr)
        print("  python3 pipeline.py postgres", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or check if it's running:", file=sys.stderr)
        print("  squeue -u $USER | grep postgres", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

    # Try .env setting (only if SLURM not available, e.g., different cluster)
    storage_url = os.getenv("GNN_OPTUNA_STORAGE", "")
    if "@" in storage_url and ":" in storage_url:
        try:
            after_at = storage_url.split("@")[1]
            host = after_at.split(":")[0]
            if host and host != "localhost":
                return host
        except:
            pass

    # Default fallback
    return "your-compute-node"


def get_optuna_storage_url():
    """
    Get complete Optuna storage URL with auto-detected host.

    Returns:
        str: PostgreSQL connection string
             Format: postgresql://user:password@host:port/database
    """
    # Get credentials from .env
    storage_template = os.getenv("GNN_OPTUNA_STORAGE", "")

    # If it's already a localhost tunnel, use as-is
    if "localhost" in storage_template:
        return storage_template

    # Otherwise, build URL with auto-detected host
    host = get_db_host()

    # Extract credentials from .env template
    if "@" in storage_template:
        # postgresql://user:pass@oldhost:port/db -> extract user:pass
        credentials = storage_template.split("@")[0].replace("postgresql://", "")
        port = "5432"
        database = "optuna"

        # Try to extract port and database from template
        try:
            after_at = storage_template.split("@")[1]
            if ":" in after_at:
                port = after_at.split(":")[1].split("/")[0]
            if "/" in after_at:
                database = after_at.split("/")[1]
        except:
            pass

        return f"postgresql://{credentials}@{host}:{port}/{database}"

    # Fallback: use .env as-is
    return storage_template


def get_optuna_storage_host_for_slurm():
    """
    Get the database host in a format suitable for SLURM job scripts.

    For Bio cluster: Returns actual compute node (e.g., your-compute-node)
    For CS cluster: Returns 'localhost' (assumes SSH tunnel)

    Returns:
        str: Database host
    """
    cluster = os.getenv("GNN_CLUSTER", "bio")

    if cluster == "cs":
        # CS cluster should use tunnel
        return "localhost"
    else:
        # Bio cluster: auto-detect
        return get_db_host()


def update_status_file(hostname):
    """
    Update the status file with current database hostname.

    Call this from your PostgreSQL startup script.

    Args:
        hostname: Current compute node hostname
    """
    try:
        STATUS_FILE.write_text(hostname + "\n")
        print(f"✓ Updated Optuna DB status file: {hostname}")
    except Exception as e:
        print(f"Warning: Could not update status file: {e}")


if __name__ == "__main__":
    # When run as script, print the storage URL
    print(get_optuna_storage_url())
