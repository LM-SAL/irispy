"""Private helpers for ndcube sequence interoperability."""


def get_sequence_common_axis(sequence):
    """
    Return a sequence common axis.

    ndcube does not expose a public accessor for this value, so irispy keeps
    the private attribute access in one place until upstream provides one.
    """
    return getattr(sequence, "_common_axis", None)
