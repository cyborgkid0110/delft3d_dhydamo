"""HyDAMO DAMO2.2 identifiers and small schema helpers.

The template builder needs to produce the "housekeeping" fields that every
HyDAMO object carries and that the HyDAMO Validatie Module checks during
syntax validation:

* ``globalid``   -- a GUID, unique per object, matching the DAMO2.2 pattern
                    ``{8-4-4-4-12}`` (braces optional).
* ``nen3610id``  -- a unique NEN3610 identifier string.
* ``objectid``   -- a unique integer id.

Foreign keys between objects (for example ``kunstwerkopening.stuwid`` pointing
at ``stuw.globalid``) are plain ``globalid`` strings, so the builders share the
GUIDs they generate here. Keeping the id logic in one place means the rest of
the tool never has to think about the exact GUID format.
"""

from __future__ import annotations

import uuid
from itertools import count

# EPSG:28992 (Amersfoort / RD New) -- the Dutch national grid all HyDAMO data
# lives in. The introduction notebook and the validator both assume it.
CRS_RD = "EPSG:28992"

# The DAMO2.2 GUID fields are written *with* braces in the reference datasets,
# so we do the same. The validator's pattern accepts braces optionally.
_object_id_counter = count(1)


def new_globalid() -> str:
    """Return a fresh brace-wrapped GUID, e.g. ``{0f8e...-...}``."""
    return "{" + str(uuid.uuid4()) + "}"


def new_objectid() -> int:
    """Return the next unique integer ``objectid`` (starts at 1)."""
    return next(_object_id_counter)


def nen3610id(objecttype: str, code: str) -> str:
    """Build a deterministic ``nen3610id`` from an object type and code.

    The exact namespace does not matter for validation as long as the value is
    unique and non-empty; a readable ``NL.WBHCODE.<type>.<code>`` keeps the
    template legible.
    """
    return f"NL.template.{objecttype}.{code}"


def reset_ids() -> None:
    """Reset the ``objectid`` counter.

    Only relevant when the generator is called more than once in the same
    Python process (for example from a test); a fresh CLI run starts at 1
    anyway.
    """
    global _object_id_counter
    _object_id_counter = count(1)
