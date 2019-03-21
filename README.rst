=====
EVNRG
=====


.. image:: https://img.shields.io/pypi/v/evnrg.svg
        :target: https://pypi.python.org/pypi/evnrg

.. image:: https://img.shields.io/travis/accurrently/evnrg.svg
        :target: https://travis-ci.org/accurrently/evnrg

.. image:: https://readthedocs.org/projects/evnrg/badge/?version=latest
        :target: https://evnrg.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/accurrently/evnrg/shield.svg
     :target: https://pyup.io/repos/github/accurrently/evnrg/
     :alt: Updates



EVNRG is an EV electrical demand simulation package that takes in trip data and turns it into useful energy data given a set of assumptions.

This project was inspired by an earlier effort to simulate electrical demand from a theoretical electrified fleet replacement.
This tool is designed to track energy usage across an arbitrarily large fleet, with an arbitrary number of EVSE, and rules 
that govern when charging occurs, and what type of queuing logic to use.


* Documentation: https://evnrg.readthedocs.io.


Features
--------

EVNRG takes in scenario definitions and interval travel data in order to generate arrays (Pandas Series) of:

* Electrical demand (per EVSE bank)
* Fuel consumption (per vehicle)
* Battery state (per vehicle)
* Deferred distance (per vehicle)
* EVSE occupancy (per EVSE bank)

License
-------
This package is Free software under the Apache Software License 2.0. See LICENSE for more information.

Credits
-------

Initially written by Alex Campbell as part of a master's of science thesis project at the University of California, Davis `Energy Graduate Group`_.
This projet falls under the auspices of the UC Davis `Institute of Transportation Studies (ITS)`_ and 
the UC Davis `Plug-in Hybrid and Electric Vehicle Research Center`_. This project was funded by the `Office of Naval Research NEPTUNE`_ program.

.. image:: https://github.com/accurrently/evnrg/raw/master/images/ITS-logo.jpg

.. image:: https://github.com/accurrently/evnrg/raw/master/images/ONR-logo.png
   :width: 50pt

.. image:: https://github.com/accurrently/evnrg/raw/master/images/PHEV-logo.png
   :width: 50pt

This package's skeleton was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`Energy Graduate Group`: https://energy.ucdavis.edu
.. _`Plug-in Hybrid and Electric Vehicle Research Center`: https://phev.ucdavis.edu
.. _`Institute of Transportation Studies (ITS)`: https://its.ucdavis.edu
.. _`Office of Naval Research NEPTUNE`: https://www.onr.navy.mil/en/Science-Technology/Departments/Code-33/All-Programs/333-sea-platforms-weapons/Neptune
