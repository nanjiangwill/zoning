The range for {{term}} is typically between 25 and 500 ft. Please focus on values within this range when searching for {{term}} and provide the answer as a whole number with unit (e.g., 50 ft). However, bear in mind that values falling outside of these ranges are possible, so do not disregard them.

Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 98
ARTICLE 7 - DIMENSIONAL STANDARDS
7.2
Dimensional Standards for Lots and Principal Structures
The following tables establish the minimum dimensional standards for lots, including size,
width, depth, setbacks, height and building coverage:
Table 7-1 Minimum Lot Dimensions for Single Family Residential Zoning Districts
Table 7-2 Project Area and Density Standards for the Multifamily Residential District
Table 7-3 Minimum Lot Dimensions for Nonresidential Zoning Districts
Table 7-4 Setbacks, Height and Building Coverage Requirements
P Indicates that prevailing setbacks are required.
VILLAGE OF LAKE PARK UNIFIED DEVELOPMENT ORDINANCE
DISTRICT
CELL (1, 2):
XXXX
CELL (1, 3):
XXXX
CELL (1, 4):
XXXX
CELL (1, 5):
XXXX
CELL (1, 6):
{{zone_abbreviation}}
CELL (1, 7):
XXXX
CELL (1, 8):
XXXX
CELL (1, 9):
XXXX
CELL (1, 10):
XXXX
CELL (1, 11):
XXXX
CELL (1, 12):
|
CELL (6, 1):
Maximum Height (feet)
CELL (6, 2):
35
CELL (6, 3):
35
CELL (6, 4):
35
CELL (6, 5):
35
CELL (6, 6):
35
CELL (6, 7):
35
CELL (6, 8):
35
CELL (6, 9):
35
CELL (6, 10):
35
CELL (6, 11):
35
CELL (6, 12):
35

Output:
{
    "extracted_text": [["CELL (6, 6):\n35", 98]],
    "rationale": "The section titled Maximum Height (feet) says the answer for district, {{zone_abbreviation}}, explicitly.",
    "answer": "35 ft"
}

Input:
NEW PAGE 66
(B) Table: Permitted Uses by Zoning District.
CELL (1, 1):
Sign
District
CELL (1, 2):
XXXX
CELL (1, 3):
XXXX
CELL (1, 4):
XXXX
CELL (1, 5):
XXXX
CELL (1, 6):
XXXX
CELL (1, 7):
XXXX
CELL (1, 8):
XXXX
CELL (1, 9):
XXXX
CELL (1, 10):
XXXX
CELL (1, 11):
XXXX
CELL (1, 12):
XXXX
CELL (1, 13):
{{zone_abbreviation}}
CELL (2, 1):
Outdoor
advertising
CELL (2, 2):
Not
Permitted
CELL (2, 3):
Not
Permitted
CELL (2, 4):
Not
Permitted
CELL (2, 5):
Not
Permitted
CELL (2, 6):
Not
Permitted d
CELL (2, 7):
Not
Permitted
CELL (2, 8):
Not
Permitted
CELL (2, 9):
Not
Permitted
CELL (2, 10):
Not
Permitted
CELL (2, 11):
Not
Permitted
CELL (2, 12):
Not
Permitted
CELL (2, 13):
Not
Permitted
CELL (3, 1):
Marquees
CELL (3, 2):
Not
Permitted
CELL (3, 3):
Not
Permitted
CELL (3, 4):
6' max
height
CELL (3, 5):
6' max
height
CELL (3, 6):
Not
Permitted d
CELL (3, 7):
6' max
height

Output:
{
    "extracted_text": null,
    "rationale": "The section is about the maximum height of flags in different zoning districts, not the maximum height of buildings.",
    "answer": null
}

Input:
Section 5. - {{zone_name}} ({{zone_abbreviation}}).
5.1. Purpose. The requirements set forth in this district are extended to provide for the proper
development of areas in the Town of Indian Beach which, due to their location, natural
features and access, have an extremely high potential for both permanent and tourist types of
residential development.
5.2. Uses Permitted.
(a) Single-family unattached dwellings.
(b) Two-family attached dwelling (duplex).
(c) Townhouses, apartments and condominiums in accordance with the Town of Indian Beach
Group Housing Project Ordinance.
(d) Public utility buildings and facilities only upon submission of architectural rendering of
such building and facilities.
35 of 78
4/13/24, 12:40

NEW PAGE 36
Indian Beach, NC Code of Ordinances
requirements set forth in Article V, Section 5.8.
(k) Mobile homes for the limited purpose of contractor's temporary field construction offices,
contractor's temporary construction warehouse facilities, temporary sales offices, and
temporary offices and housing for security personnel. Mobile homes under the specific
limitation of this subparagraph shall be permitted only after a building permit has been
issued for a permitted or special use within the RR District, and the mobile homes shall be
maintained upon such building site until the occupancy permit is issued for the
development represented by the building permit, or until the said building permit has
expired, at which time the mobile home must be removed from the RR district.
5.3. Dimensional Requirements for Permitted Uses.
(a) Minimum lot area:
(i) Hotels, motels and accessory uses in accordance with Article V, Section 5.8 hereinafter.
(ii) Detached single-family dwellings - 15,000 square feet; however, if the lot is served by
Public Sewer and a Public Water System, the minimum lot size is 10,000 square feet.
Editor's note- [This subsection as amended by Ord. of 12-9-2010, § I.]
(iii) Two-family dwellings (duplex) - 20,000 square feet.
(b) Maximum building lot coverage - 35 percent
(c) (i) Maximum building height - 100 feet
(ii) Any building with any floor of thirty (30) feet or more in height must have exterior fire
escapes, or fire proof interior stairways if approved by the North Carolina Department
of Insurance, extending from the ground to each floor at thirty (30) feet or above.
Output:
{
    "extracted_text": [["Maximum building height - 100 feet", 36]],
    "rationale": "The section explicitly states the {{term}} for district, {{zone_abbreviation}}.",
    "answer": "100 ft"
}

Input:
NEW PAGE 100
Sec. 21-66. General criteria for uses listed SR in the {{zone_abbreviation}} District in section 21-113.
Uses listed as SR in the Al District in section 21-113, the table of uses, shall comply with
the following criteria, as applicable:
(1) Site plan. A site plan shall be provided showing the existing lot, existing and proposed
100

NEW PAGE 101
As of 01/17/23
buildings, and criteria required herein.
(2) Lighting. Any outdoor or building mounted lighting shall be shielded or directed downward to
prevent upward illumination that may create interference with airport operations.
(3) Building material. No glare-producing material shall be used as exterior siding or as roofing
on any building.
(4) Building height. The maximum height for any building or structure not associated with
administration or operation(s) of the Mid-Carolina Regional Airport shall be limited to the
lesser of the Airport Zoning Overlay (AZO) or thirty-five (35') feet
(5) Parking. Parking shall be as prescribed in Article VII, Parking, for that use.

Output:
{
    "extracted_text": [["The maximum height for any building or structure not associated with\nadministration or operation(s) of the Mid-Carolina Regional Airport shall be limited to the\nlesser of the Airport Zoning Overlay (AZO) or thirty-five (35') feet", 101]],
    "rationale": "The section explicitly states the maximum height of any building in the {{zone_abbreviation}} district.",
    "answer": "35 ft"
}

Input:
Multi-family building

Output:
{
    "extracted_text": null,
    "rationale": "The section does not provide a specific value for {{term}}, and is not about single-family homes.",
    "answer": null
}
