"""Bilevel fishery example: an ES regulator over a quota + subsidy space, APPO fishers inside.

The package holds the regulated environment (``regulated_env_shaefer``), the
mechanism vector and its parameter space (``mechanism_v1``), the regulator
environment that scores mechanism candidates (``regulator_env``), the fitness
context exchanged through the World actor (``contexts``), the metric schema
and the reporting queries, ``bilevel`` as the untyped historical entry point
and ``debug`` as the runnable entry point. Nothing is re-exported from here.
"""
