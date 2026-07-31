"""Interest-rate risk: scenarios, sensitivities and scenario P&L.

``scenarios.py`` holds the single shift primitive — effective duration,
key-rate duration and PCA duration are the same computation with a different
``Scenario``.

``ladder.py`` is the other coordinate system: risk against the quoted
instruments rather than against zero rates, which is what gets hedged.
"""
