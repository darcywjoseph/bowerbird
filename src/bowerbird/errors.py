"""Exceptions raised by bowerbird."""


class DatasetError(Exception):
    """A dataset could not be found or does not have the expected layout."""


class ScorerError(Exception):
    """A scorer was asked for by an unknown name or cannot score the data."""
