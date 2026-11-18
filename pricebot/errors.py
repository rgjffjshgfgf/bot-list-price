"""Errors the bot shows to the user. Kept apart from the heavy modules: the bot process loads only this."""


class UserError(Exception):
    """An error whose message (Persian) can be shown to the user as is."""
