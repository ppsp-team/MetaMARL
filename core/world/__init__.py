"""Shared blackboard of a bilevel run.

``base`` defines the ``World`` Ray actor and ``context`` the payloads it
stores (``MechanismContext``, ``EnvStepContext``) and their ``Context``
envelope. This ``__init__`` re-exports nothing; import from the submodules
directly.
"""
