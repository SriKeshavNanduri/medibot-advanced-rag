"""Canonical role / collection access mapping.

This module is the single source of truth for RBAC. It is deliberately free of
heavy imports (no langchain, no docling, no torch) so that both ``chunking.py``
(the ingestion path) and the FastAPI service can import it cheaply.

"Collection" here means a *document category* -- one of the folders under
``mediassist_data`` -- not a Qdrant collection. There is exactly one Qdrant
collection, named by the ``COLLECTION_NAME`` env var.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """The five roles recognised by the system."""

    DOCTOR = "doctor"
    NURSE = "nurse"
    BILLING_EXECUTIVE = "billing_executive"
    TECHNICIAN = "technician"
    ADMIN = "admin"


#: collection (mediassist_data folder) -> roles allowed to read it
ACCESS_MAPPING: dict[str, list[str]] = {
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "doctor", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
    "general": ["doctor", "nurse", "billing_executive", "technician", "admin"],
}


def get_access_based_on_collection(collection_name) -> list[str]:
    """
    Retrieves the list of roles permitted to read a specific document collection.
    
    Args:
        collection_name (str): The name of the document collection (e.g., "clinical", "billing").
    Returns:
        list[str]: A list of role names that have access to the specified collection.
                   Returns an empty list if the collection name is unknown.
    """
    """Roles permitted to read a collection. Unknown collection -> no roles."""
    return list(ACCESS_MAPPING.get(str(collection_name).strip().lower(), []))


def get_collections_for_role(role) -> list[str]:
    """
    Retrieves the document collections that are readable by a specific role.
    This is the inverse mapping of `ACCESS_MAPPING`.
    
    Args:
        role (str): The role name (e.g., "doctor", "admin").
    Returns:
        list[str]: A sorted list of collection names accessible to the specified role.
    """
    """The inverse of :data:`ACCESS_MAPPING`: collections readable by a role."""
    wanted = str(role).strip().lower()
    return sorted(
        collection
        for collection, roles in ACCESS_MAPPING.items()
        if wanted in roles
    )


def known_roles() -> list[str]:
    """
    Returns a sorted list of all unique role names defined in the `ACCESS_MAPPING`.
    
    Returns:
        list[str]: A sorted list of all known roles in the system.
    """
    """Every role that appears anywhere in the mapping."""
    return sorted(["doctor", "nurse", "billing_executive", "technician", "admin"])
