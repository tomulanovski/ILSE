#!/usr/bin/env python3
"""
Cluster configuration helper for multi-cluster support.

This module provides a unified interface for generating SLURM job parameters
that work across different compute clusters with different configurations.

Supported clusters:
    - bio: Bio cluster with lab-specific/general-pool partitions
    - cs: CS cluster with lab-specific/killable partitions

Usage:
    from cluster_config import get_cluster_config

    config = get_cluster_config(partition_type='priority')
    slurm_params = config.to_slurm_dict()
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ClusterSlurmConfig:
    """SLURM configuration for a specific cluster partition."""
    partition: str
    account: Optional[str] = None
    qos: Optional[str] = None
    gpu_spec: str = "gpu:1"  # Default generic GPU spec

    def to_slurm_dict(self) -> Dict[str, str]:
        """Convert to dictionary for SLURM script generation."""
        params = {
            'partition': self.partition,
            'gres': self.gpu_spec,
        }

        if self.account:
            params['account'] = self.account

        if self.qos:
            params['qos'] = self.qos

        return params

    def to_sbatch_lines(self) -> str:
        """Generate #SBATCH lines for job script."""
        lines = [f"#SBATCH --partition={self.partition}"]

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")

        lines.append(f"#SBATCH --gres={self.gpu_spec}")

        return "\n".join(lines)


def get_cluster_type() -> Literal['bio', 'cs']:
    """Get the configured cluster type from environment."""
    cluster = os.getenv('GNN_CLUSTER', 'bio').lower()

    if cluster not in ['bio', 'cs']:
        raise ValueError(
            f"Invalid GNN_CLUSTER value: '{cluster}'. Must be 'bio' or 'cs'. "
            f"Update your .env file."
        )

    return cluster


def get_cluster_config(
    partition_type: Literal['priority', 'general'] = 'priority',
    gpu_type: Optional[str] = None
) -> ClusterSlurmConfig:
    """
    Get SLURM configuration for the current cluster and partition type.

    Args:
        partition_type: 'priority' for lab-specific partition, 'general' for shared
        gpu_type: Optional specific GPU type (e.g., 'A6000', 'A100').
                  Only used on Bio cluster. Ignored on CS cluster.

    Returns:
        ClusterSlurmConfig with appropriate SLURM parameters

    Examples:
        # Priority partition (default to lab partition)
        >>> config = get_cluster_config('priority')

        # General partition with specific GPU on Bio
        >>> config = get_cluster_config('general', gpu_type='A100')

        # CS cluster ignores gpu_type and uses simple gpu:1
        >>> config = get_cluster_config('priority')  # Uses lab partition
    """
    cluster = get_cluster_type()

    if cluster == 'cs':
        return _get_cs_config(partition_type)
    else:  # bio
        return _get_bio_config(partition_type, gpu_type)


def _get_cs_config(partition_type: str) -> ClusterSlurmConfig:
    """Get CS cluster configuration."""
    if partition_type == 'priority':
        partition = os.getenv('GNN_CS_PRIORITY_PARTITION', 'gpu-yourlab')
        account = os.getenv('GNN_CS_ACCOUNT', 'gpu-research')

        return ClusterSlurmConfig(
            partition=partition,
            account=account,
            qos=None,  # CS doesn't use QOS
            gpu_spec='gpu:1'  # Simple GPU spec on CS
        )

    else:  # general
        partition = os.getenv('GNN_CS_GENERAL_PARTITION', 'killable')

        return ClusterSlurmConfig(
            partition=partition,
            account=None,  # killable doesn't need account
            qos=None,
            gpu_spec='gpu:1'
        )


def _get_bio_config(partition_type: str, gpu_type: Optional[str]) -> ClusterSlurmConfig:
    """Get Bio cluster configuration."""
    if partition_type == 'priority':
        partition = os.getenv('GNN_BIO_PRIORITY_PARTITION', 'your-lab-partition')
        account = os.getenv('GNN_BIO_PRIORITY_ACCOUNT', 'your-account')
        qos = os.getenv('GNN_BIO_PRIORITY_QOS', 'owner')

        # For priority partition, don't specify GPU type (use any available in pool)
        gpu_spec = 'gpu:1'

    else:  # general
        partition = os.getenv('GNN_BIO_GENERAL_PARTITION', 'gpu-general-pool')
        account = None  # general pool doesn't use account
        qos = os.getenv('GNN_BIO_GENERAL_QOS', 'public')

        # For general partition, specify GPU type if provided
        if gpu_type:
            gpu_spec = f'gpu:{gpu_type}:1'
        else:
            gpu_spec = 'gpu:A100:1'  # Default to A100 for general pool

    return ClusterSlurmConfig(
        partition=partition,
        account=account,
        qos=qos,
        gpu_spec=gpu_spec
    )


def print_cluster_info():
    """Print current cluster configuration (for debugging)."""
    cluster = get_cluster_type()

    print(f"\n{'='*70}")
    print(f"CLUSTER CONFIGURATION")
    print(f"{'='*70}")
    print(f"Active cluster: {cluster.upper()}")
    print()

    print("Priority partition:")
    priority = get_cluster_config('priority')
    print(f"  {priority.to_sbatch_lines().replace(chr(10), chr(10) + '  ')}")
    print()

    print("General partition:")
    general = get_cluster_config('general')
    print(f"  {general.to_sbatch_lines().replace(chr(10), chr(10) + '  ')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test/debug mode
    print_cluster_info()
