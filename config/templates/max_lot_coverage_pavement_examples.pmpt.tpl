Remember, the {{term}} records the amount of land that may be covered by both buildings and other impervious surfaces, such as pavement. If the document specifies the maximum lot coverage by buildings only, DO NOT include it. If the document refers to maximum lot coverage generally, you can assume it includes both buildings and impervious surfaces and include it. If the document only specifies impervious surface coverage, it can be considered as building coverage and included in the answer.

The range for {{term}} is typically between 5% and 100% (0.05 to 1). Please focus on values within this range when searching for {{term}} and provide the answer as a whole number (e.g., 50% or 0.5 should be 50). However, bear in mind that values falling outside of these ranges are possible, so do not disregard them.

Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 100
As of 01/17/23
Uses listed as SR in the {{zone_abbreviation}} District in section 21-113, the table of uses, shall comply with the
following criteria, as applicable:
(1) Site plan. A site plan shall be provided showing the existing lot and all existing and proposed
buildings. As well as all criteria required herein.
(2) Lighting. The lighting shall be shielded to prevent light and glare spillover to adjacent
residentially developed properties.
(3) Minimum zone lot size. The minimum zone lot size shall be two (2) acres.
(4) Building size. The maximum building size per parcel shall not exceed ten (10) percent of the
lot area up to ten thousand (10,000) square foot and five (5) percent of the lot acreage
thereafter up to twenty-five thousand (25,000) sq.ft. Multiple buildings may be used in
calculating the maximum allowable building size.
(5) Impervious surface. The maximum impervious surface shall not exceed sixty-five (65)
percent of the lot.
(6) Hours of operation. Hours of operation shall not exceed 6:00 a.m. to 11:00 p.m.
(7) Parking. Parking shall be as prescribed in article VII, Parking, for that use.
(8) Signage. Shall be as prescribed in article VIII, Signs, for the underlying district.
(9) Noise. Noise shall not exceed the decibel levels during time periods prescribed in section 21-
241 for construction, manufacturing, transportation, communications, electric, gas and
sanitary services, wholesale, and service uses.
(10) Outdoor storage. All outside storage areas including dumpsters shall be:
a. Sited to the rear of the building;
b. Not within the required setbacks.
C.
Notwithstanding other requirements of this subsection, outdoor storage shall be
completely screened from adjacent residentially zoned property

Output:
{
    "extracted_text": ["The maximum impervious surface shall not exceed sixty-five (65)\npercent of the lot."],
    "rationale": "The section state the {{term}} for district {{zone_abbreviation}}.",
    "answer": "65"
}

Input:
NEW PAGE 25
ARTICLE VI.
GENERAL PROVISIONS AND SUPPLEMENTARY REGULATIONS
D.
Dimensional Requirements for Low Density Residential Districts
SECTION 6.03.
{{zone_abbreviation}} LIMITED LOW DENSITY RESIDENTIAL DISTRICT
C.
Special Uses
The following uses are permitted subject to the requirements of this district, additional
regulations and requirements imposed by the Board of Commissioners as provided in Article
VIII.
1.
Churches and cemeteries.
2.
Home occupations under provisions of Section 7.07.
3.
Duplex apartments: lot minimum size 30,000 square feet.
4.
Public utility facilities: subject to provision of a vegetated buffer strip at least ten (10)
feet in height.
5.
Group developments under the provisions of Section 7.05 with a density of not more
than three (8) dwelling units per acre. (As amended April 2, 2013)
20
CELL (7, 1):
6.
CELL (7, 2):
Maximum allowable lot coverage by principal use and all accessory structures: 30%
CELL (7, 3):
Maximum allowable lot coverage by principal use and all accessory structures: 30%

Output:
{
    "extracted_text": null,
    "rationale": "The section is referring to the maximum allowable lot coverage by principal use and all accessory structures for district {{zone_abbreviation}}, which explicitly specifies buildings only, thus the value is not included in the answer.",
    "answer": null
}

Input:
NEW PAGE 157
Article 3: Base Zoning Districts
Section 3.4: {{zone_name}} Districts
Orange County, North Carolina - Unified Development Ordinance
Page 3-23
CELL (1, 1):
2.
CELL (1, 2):
Development within the zoning district shall be subject to all applicable use standards detailed in Article 5
and all applicable development standards detailed in Article 6 of this Ordinance. See Sections 6.2.5 and
6.2.6 if more than one principal use or principal structure is proposed on a non-residential zoning lot.
CELL (2, 1):
3.
CELL (2, 2):
The residential density permitted on a given parcel is based on the Watershed Protection Overlay District in
which the property is located. Refer to Section 4.2.4 for a breakdown of the allowable density (i.e., the
number of individual dwellings that can be developed on a parcel of property).
CELL (3, 1):
4.
CELL (3, 2):
Allowable impervious surface area is based on the Watershed Protection Overlay District in which the
property is located. Refer to Sections 4.2.5 and 4.2.6 for a breakdown of the allowable impervious surface
area. Additionally, Section 4.2.6 may require a larger lot size for non-residential uses than is contained in
the Dimensional and Ratio Standards Table.

Output:
{
    "extracted_text": null,
    "rationale": "The section directed the guidence on {{term}} of district {{zone_name}} to Section 4.2.5 and 4.2.6., but no specific value is provided.",
    "answer": null
}

Input:

NEW PAGE 25
NOTES TO TABLE:
*
For maximum percentage of impervious surfaces, see division (C) below
(C) Steep slope maximum density requirement.
CELL (1, 1):
CELL (1, 2):
CELL (1, 3):
Average Natural Slope of Parcel by Acre
CELL (1, 4):
Average Natural Slope of Parcel by Acre
CELL (1, 5):
Average Natural Slope of Parcel by Acre
CELL (1, 6):
CELL (2, 1):
Zoning District
CELL (2, 2):
Under 20%
CELL (2, 3):
21% to 30%
CELL (2, 4):
31% to 40%
CELL (2, 5):
41% to 50%
CELL (2, 6):
Over 51%
CELL (3, 1):
CELL (3, 2):
Maximum Allowable Percent of Impervious Surfaces/Dwelling Units Per Acre of Land
CELL (3, 3):
Maximum Allowable Percent of Impervious Surfaces/Dwelling Units Per Acre of Land
Including the Removal of Active Recreation Area, Section 312
CELL (3, 4):
Maximum Allowable Percent of Impervious Surfaces/Dwelling Units Per Acre of Land
Including the Removal of Active Recreation Area, Section 312
CELL (3, 5):
Maximum Allowable Percent of Impervious Surfaces/Dwelling Units Per Acre of Land
Including the Removal of Active Recreation Area, Section 312
CELL (3, 6):
Maximum Allowable Percent of Impervious Surfaces/Dwelling Units Per Acre of Land
Including the Removal of Active Recreation Area, Section 312
CELL (7, 1):
{{zone_abbreviation}}
CELL (7, 2):
40%
CELL (7, 3):
35%
CELL (7, 4):
30%
CELL (7, 5):
25%
CELL (7, 6):
Geotechnical
engineer
required
CELL (8, 1):
XXXX
XXXX
CELL (8, 2):
45%
CELL (8, 3):
40%
CELL (8, 4):
35%
CELL (8, 5):
30%
CELL (8, 6):
Geotechnical
engineer
required

Output:
{
    "extracted_text": [
        "CELL (1, 3):\nAverage Natural Slope of Parcel by Acre",
        "CELL (1, 4):\nAverage Natural Slope of Parcel by Acre",
        "CELL (1, 5):\nAverage Natural Slope of Parcel by Acre",
        "CELL (2, 2):\nUnder 20%",
        "CELL (2, 3):\n21% to 30%",
        "CELL (2, 4):\n31% to 40%",
        "CELL (2, 5):\n41% to 50%",
        "CELL (2, 6):\nOver 51%",
        "CELL (7, 1):\nR-1",
        "CELL (7, 2):\n40%",
        "CELL (7, 3):\n35%",
        "CELL (7, 4):\n30%",
        "CELL (7, 5):\n25%",
        "CELL (7, 6):\nGeotechnical engineer required",
        "CELL (8, 1):\nR-2\nR-1-U",
        "CELL (8, 2):\n45%",
        "CELL (8, 3):\n40%",
        "CELL (8, 4):\n35%",
        "CELL (8, 5):\n30%",
        "CELL (8, 6):\nGeotechnical engineer required"
    ],
    "rationale": "The section specifies the {{term}} for district {{zone_abbreviation}}. It can be infered that the value depends on the slope of the parcel by acre.",
    "answer": "40; 35; 30; 25 (depending on the slope of the parcel by acre)"
}

Input:
Multi-family building

Output:
{
    "extracted_text": null,
    "rationale": "The section does not provide a specific value for {{term}}, and is not about single-family homes.",
    "answer": null
}
