#!/usr/bin/env python3
"""
Auto-detect the PostgreSQL Optuna database host from running SLURM jobs.

This script checks:
1. A status file written by the PostgreSQL job
2. SLURM queue for running PostgreSQL job
3. Falls back to .env setting

Usage:
    python3 scripts_and_jobs/scripts/get_optuna_db_host.py
    # Outputs: your-compute-node
"""
import os
import subprocess
import sys
from pathlib import Path

# Status file where PostgreSQL job writes its hostname
STATUS_FILE = Path.home() / ".optuna_db_host"

def get_db_host_from_status_file():
    """Read database host from status file written by PostgreSQL job."""
    if STATUS_FILE.exists():
        try:
            host = STATUS_FILE.read_text().strip()
            if host:
                return host
        except Exception as e:
            print(f"Warning: Could not read status file: {e}", file=sys.stderr)
    return None

def get_db_host_from_slurm():
    """Query SLURM for running PostgreSQL job and get its node."""
    try:
        # Look for PostgreSQL job in queue
        result = subprocess.run(
            ["squeue", "-u", os.getenv("USER"), "-o", "%j %N", "-h"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                job_name, node = line.split(None, 1)
                # Look for postgres/optuna database job
                if "postgres" in job_name.lower() or "optuna" in job_name.lower() or "database" in job_name.lower():
                    return node.strip()
    except Exception as e:
        print(f"Warning: Could not query SLURM: {e}", file=sys.stderr)

    return None

def get_db_host_from_env():
    """Extract database host from .env file as fallback."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        storage_url = os.getenv("GNN_OPTUNA_STORAGE", "")
        if "@" in storage_url and ":" in storage_url:
            # Extract host from: postgresql://user:pass@host:port/db
            after_at = storage_url.split("@")[1]
            host = after_at.split(":")[0]
            return host
    except Exception as e:
        print(f"Warning: Could not parse .env: {e}", file=sys.stderr)

    return None

def main():
    """Get database host using multiple detection methods."""
    # Try status file first (fastest, most reliable)
    host = get_db_host_from_status_file()
    if host:
        print(host)
        return

    # Try SLURM queue (works if job is running)
    host = get_db_host_from_slurm()
    if host:
        print(host)
        # Cache for future use
        try:
            STATUS_FILE.write_text(host + "\n")
        except:
            pass
        return

    # Fall back to .env setting
    host = get_db_host_from_env()
    if host:
        print(host)
        return

    # Default fallback
    print("your-compute-node", file=sys.stderr)
    print("your-compute-node")

if __name__ == "__main__":
    main()
