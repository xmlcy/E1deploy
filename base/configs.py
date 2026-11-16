"""Consolidated robot configs. kp/kd/effort are zero — sim2real scripts override them at every step."""

import numpy as np


class _BaseConfig:
    """Shared defaults."""
    grpc_channel = "localhost"
    # grpc_channel = '192.168.254.100'

    @classmethod
    def _zeros(cls, n):
        return np.zeros(n, dtype=np.float32)


# ── E1 robot variants ──────────────────────────────────────────────

class E1_21dof(_BaseConfig):
    """E1 21DOF: 13 leg + 8 arm"""
    num_leg_actions = 13
    num_arm_actions = 8

    dof_stiffness = _BaseConfig._zeros(13)
    dof_damping   = _BaseConfig._zeros(13)
    effort_limit  = _BaseConfig._zeros(13)


class E1_13dof(_BaseConfig):
    """E1 13DOF: 13 leg, no arms"""
    num_leg_actions = 13
    num_arm_actions = 0

    dof_stiffness = _BaseConfig._zeros(13)
    dof_damping   = _BaseConfig._zeros(13)
    effort_limit  = _BaseConfig._zeros(13)


class E1_12dof(_BaseConfig):
    """E1 12DOF: 12 leg, no arms (leg-only, no waist)"""
    num_leg_actions = 12
    num_arm_actions = 0

    dof_stiffness = _BaseConfig._zeros(12)
    dof_damping   = _BaseConfig._zeros(12)
    effort_limit  = _BaseConfig._zeros(12)


# ── Other robots ────────────────────────────────────────────────────

class X3(_BaseConfig):
    """X3 robot: 14 leg + 10 arm"""
    num_leg_actions = 14
    num_arm_actions = 10

    dof_stiffness = _BaseConfig._zeros(14)
    dof_damping   = _BaseConfig._zeros(14)
    effort_limit  = _BaseConfig._zeros(14)
    grpc_channel = "192.168.55.21"


class DOG(_BaseConfig):
    """DOG robot: 8 leg + 8 arm"""
    num_leg_actions = 8
    num_arm_actions = 8

    dof_stiffness = _BaseConfig._zeros(8)
    dof_damping   = _BaseConfig._zeros(8)
    effort_limit  = _BaseConfig._zeros(8)


class Generic(_BaseConfig):
    """Generic 12 leg + 10 arm"""
    num_leg_actions = 12
    num_arm_actions = 10

    dof_stiffness = _BaseConfig._zeros(12)
    dof_damping   = _BaseConfig._zeros(12)
    effort_limit  = _BaseConfig._zeros(12)
