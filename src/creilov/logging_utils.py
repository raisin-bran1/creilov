"""Targeted logging setup for the sequence-only workflow."""

import logging


class _UnusedStructureMirrors(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.levelno == logging.WARNING
            and message.startswith(
                (
                    "Environment variable CCD_MIRROR_PATH not set.",
                    "Environment variable PDB_MIRROR_PATH not set.",
                )
            )
        )


def quiet_unused_structure_mirrors() -> None:
    """Hide only the two unused mirror-path notices, before importing ProteinGen.

    Leave the variables unset and preserve all other AtomWorks warnings/errors.
    """
    logger = logging.getLogger("atomworks.constants")
    if not any(isinstance(item, _UnusedStructureMirrors) for item in logger.filters):
        logger.addFilter(_UnusedStructureMirrors())
