"""
Real CPCB (Central Pollution Control Board) surface water quality readings.

Source: India's National Water Data Portal (nwdp.nwic.gov.in), Ministry of
Jal Shakti - dataset "Surface Water Quality (Manual - Chemical Parameters),
CPCB", Tamil Nadu resource, 2021-2025.
  https://nwdp.nwic.gov.in/dataset/surface-water-quality-manual-chemical-parameters-cpcb

All 35 readings below were transcribed directly from that source CSV - these
are real government-published measurements, not synthetic data. Only
stations with both Dissolved Oxygen (DO) and pH reported are included, all
from January 2021 (05-01-2021 to 27-01-2021) to keep a reasonably consistent
temporal snapshot for a spatial validation.

Turbidity was NOT available for these stations/dates in the source data, so
this real-data validation uses two parameters (DO, pH), not three. The
hazard-weighting term is also not used here - it would need real downstream
population/infrastructure data, which this dataset does not include. This is
an honest scoping limit, not an oversight - see real_data_validation.py.
"""

# (station name, latitude, longitude, dissolved_oxygen_mg_L, pH, date)
REAL_STATIONS = [
    ("Amravati 1km D/S Palani", 10.541244, 77.368656, 6.2, 7.10, "2021-01-11"),
    ("Bahour Lake", 11.819442, 79.729208, 5.2, 7.37, "2021-01-27"),
    ("Bhavani at Bhavani Sagar", 11.485294, 77.134603, 6.0, 7.30, "2021-01-21"),
    ("Bhavani at Bhavani", 11.454094, 77.679567, 6.3, 7.90, "2021-01-21"),
    ("Bhavani D/S Bhavanisagar Sathyamangalam", 11.503881, 77.247036, 6.3, 7.50, "2021-01-21"),
    ("Bhavani D/S Kalingarayan Canal B10", 11.438223, 77.681462, 5.9, 8.50, "2021-01-21"),
    ("Bhavani at Pathirakaliamman Koil", 11.469259, 77.296165, 6.2, 7.50, "2021-01-21"),
    ("Bhavani at Sirumugai", 11.319948, 77.011429, 6.4, 7.60, "2021-01-21"),
    ("Bhavani U/S Kalingarayan Canal B5", 11.442818, 77.676045, 6.2, 8.00, "2021-01-21"),
    ("Cauvery 1km D/S Bhavani Confluence", 11.445663, 77.677626, 6.2, 8.50, "2021-01-21"),
    ("Cauvery at Erode near Chirapalayam", 11.335795, 77.753762, 6.4, 8.20, "2021-01-22"),
    ("Cauvery at Komarapalayam Namakal", 11.423503, 77.685261, 6.2, 7.50, "2021-01-12"),
    ("Cauvery at Kumbakonam Thanjavur", 10.968137, 79.378248, 6.8, 7.70, "2021-01-12"),
    ("Cauvery at Mayiladuthurai Nagapattinam", 11.104894, 79.651871, 5.8, 7.50, "2021-01-05"),
    ("Cauvery at Mettur", 11.796255, 77.806971, 6.5, 8.10, "2021-01-20"),
    ("Cauvery at Mohanur near Pattaipalayam", 11.064392, 78.120358, 6.7, 8.10, "2021-01-12"),
    ("Cauvery at Musiri", 10.955345, 78.443949, 6.0, 7.90, "2021-01-12"),
    ("Cauvery at Pallippalayam", 11.368215, 77.741107, 6.5, 7.90, "2021-01-12"),
    ("Cauvery at Pettaivaithalai Trichy", 10.906389, 78.489697, 6.4, 7.80, "2021-01-12"),
    ("Cauvery at Pitchavaram", 11.418828, 79.772375, 6.2, 6.80, "2021-01-07"),
    ("Cauvery at Pugalur Karur", 10.974467, 78.180075, 6.4, 7.50, "2021-01-07"),
    ("Cauvery at Thanjavur", 11.340000, 79.240000, 6.7, 7.90, "2021-01-12"),
    ("Cauvery at Thirumukkudal", 11.050024, 78.120645, 6.6, 7.70, "2021-01-07"),
    ("Cauvery at Tiruchirappalli D/S", 10.766809, 78.665272, 6.4, 7.80, "2021-01-12"),
    ("Cauvery at Tiruchirappalli U/S", 10.833263, 78.718224, 6.6, 7.70, "2021-01-12"),
    ("Cauvery at Trichy Grand Anaicut", 10.830327, 78.811770, 6.6, 7.90, "2021-01-12"),
    ("Cauvery at Urrachikottai Erode", 11.354178, 77.750436, 6.0, 8.10, "2021-01-21"),
    ("Cauvery at Vairapalayam Namakal", 11.361583, 77.742411, 6.5, 7.70, "2021-01-12"),
    ("Cauvery at Velore near Kattipalayam", 11.180000, 78.010000, 6.6, 8.10, "2021-01-12"),
    ("Kodaikanal Lake", 10.233619, 77.485653, 6.7, 7.50, "2021-01-22"),
    ("Palar at Vaniyambadi", 12.679231, 78.595417, 5.2, 7.00, "2021-01-20"),
    ("Poondi Lake", 13.169731, 79.888236, 6.0, 7.10, "2021-01-06"),
    ("Porur Lake", 13.033945, 80.148344, 6.0, 7.70, "2021-01-06"),
    ("Pulicate Lake", 13.427023, 80.292679, 6.0, 7.20, "2021-01-06"),
    ("Redd Hills", 13.193092, 80.172867, 6.0, 7.20, "2021-01-06"),
]

SOURCE_URL = "https://nwdp.nwic.gov.in/dataset/surface-water-quality-manual-chemical-parameters-cpcb"
SOURCE_NOTE = (
    "CPCB Surface Water Quality (Manual - Chemical Parameters), Tamil Nadu, "
    "January 2021 readings (n=35 stations), via India's National Water Data "
    "Portal (Ministry of Jal Shakti). Real government-published data, not synthetic."
)
