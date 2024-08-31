Remember, min unit size (min lot size per dwelling unit) is not min lot size.

Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 74
Article 4 Zoning Districts
4.5 Commercial Zoning Districts
4.5.4.
Special Standards in the {{zone_abbreviation}} Zoning District
D.
Open Space
Open space requirements for the residential portion of a multifamily or mixed commercial-residential
project shall be the same and conform to the requirements for multifamily developments in the MR-
4-12
CELL (1, 1):
Use
CELL (1, 2):
District
CELL (1, 3):
Lot Width (ft.)
CELL (1, 4):
Minimum
Lot Width (ft.)
CELL (1, 5):
Lot Area per dwelling
unit (s.f.)
CELL (1, 6):
Minimum
Lot Area per dwelling
unit (s.f.)
CELL (1, 7):
CELL (1, 8):
Yard
(ft.)
CELL (1, 9):
CELL (2, 1):
CELL (2, 2):
CELL (2, 3):
With
Water and
Sewer
CELL (2, 4):
Without
Water and
Sewer
CELL (2, 5):
With
Water and
Sewer
CELL (2, 6):
Without
Water and
Sewer
CELL (2, 7):
Front
CELL (2, 8):
Rear
CELL (2, 9):
Side
CELL (3, 1):
All residential, except
CELL (3, 2):
{{zone_abbreviation}}
CELL (3, 3):
60
CELL (3, 4):
60
CELL (3, 5):
6,000
CELL (3, 6):
10,000
CELL (3, 7):
25³
CELL (3, 8):
63
CELL (3, 9):
51,2
CELL (4, 1):
multifamily
CELL (4, 2):
XXXX
CELL (4, 3):
75
CELL (4, 4):
75
CELL (4, 5):
7,500
CELL (4, 6):
10,000
CELL (4, 7):
253
CELL (4, 8):
253
CELL (4, 9):
102
CELL (5, 1):
Multifamily/ Mixed
CELL (5, 2):
{{zone_abbreviation}}
CELL (5, 3):
100
CELL (5, 4):
150
CELL (5, 5):
3,200
CELL (5, 6):
15,000
CELL (5, 7):
Per District
CELL (5, 8):
Per District
CELL (5, 9):
CELL (6, 1):
commercial- residential
CELL (6, 2):
XXXX
CELL (6, 3):
100
CELL (6, 4):
100
CELL (6, 5):
3,200
CELL (6, 6):
7,000
CELL (6, 7):
CELL (6, 8):
CELL (6, 9):

Output:
{
    "extracted_text": [
        "CELL (2, 5):\nWith\nWater and\nSewer",
        "CELL (3, 5):\n6,000",
        "CELL (2, 6):\nWithout\nWater and\nSewer",
        "CELL (3, 6):\n10,000"
    ],
    "rationale": "For district {{zone_abbreviation}}, the section provides the information for {{term}}. It can be inferred that the {{term}} depends on whether the area has water and sewer.",
    "answer": "6,000 sq ft (With Water and Sewer), 10,000 sq ft (Without Water and Sewer)"
}

Input:
NEW PAGE 48
$2.4 General Use District Standards
Article 2. Zoning Districts
Cluster Subdivision Standards
Cluster Residential Subdivision
CELL (1, 1):
{{zone_abbreviation}}
CELL (1, 2):
Single- family
Detached
CELL (1, 3):
Zero
Lot Line
CELL (1, 4):
Alley-loaded
CELL (1, 5):
Two-family
CELL (1, 6):
Townhouse
CELL (1, 7):
Multi-family
CELL (2, 1):
Use
CELL (2, 2):
Permitted
CELL (2, 3):
Permitted
CELL (2, 4):
Permitted
CELL (2, 5):
Not Permitted
CELL (2, 6):
Not Permitted
CELL (2, 7):
Not Permitted
CELL (6, 1):
Lot (min.)
Lot area (sq. ft.)
Lot width (ft.)
Water/sewer, public
CELL (6, 2):
43,560
100
Required
CELL (6, 3):
43,560
100
Required
CELL (6, 4):
43,560
100
Required
CELL (6, 5):
CELL (6, 6):
CELL (6, 7):
CELL (7, 1):
Yards (min. ft.)
Road yard
Side yard (interior)
Side yard (total)
Side yard (road)
Rear yard
CELL (7, 2):
15
5
10
10
15
CELL (7, 3):
15
0
10
10
15
CELL (7, 4):
10
5
10
10
15
CELL (7, 5):
CELL (7, 6):
CELL (7, 7):

Output:
{
    "extracted_text": null
    "rationale": "The section mentions the minimum lot size for {{zone_abbreviation}}, but no {{term}} is provided.",
    "answer": null
}

Input:
NEW PAGE 66
SEC. 9-4-200.4 {{zone_abbreviation}} {{zone_name}} STANDARDS.
(A) General district standards.
(1) Single entity.
(a) Each Mixed Use Institutional (MUI) district must be under the control of a single entity and have a controlling governmental interest or be a hospital, college or university.
(b) Development of properties within the MUI may be accomplished or carried out by either the single entity or in collaboration with a private development partner.
(2) MUI developments may consist of one, or several, lots. They may also occur in phases.
(3) District dimensional standards.
(a) Lot area (net). All uses: none.
(b) Lot width (at the MBL). All uses: none.
(c) Public street setback: 0 feet minimum.
(d) Side setback: 0 feet minimum.
(e) Rear setback: 0 feet minimum.
(f) Height: 5 stories or 70 feet
(4) District density standards.
(a) Minimum habitable (mechanically conditioned) floor area per unit:
1. One bedroom unit: 400 square feet.
2. Two or more bedroom unit: 500 square feet.
(b) Minimum parking: One space per unit.
(5) Multiple principal uses may be allowed on a single lot within an MUI.
(6) Residential uses may not exceed 25% of the total building square footage of an MUI. In the event of a phased development, this ratio will be enforced for each specific phase.

Output:
{
    "extracted_text": [
        "1. One bedroom unit: 400 square feet.",
        "2. Two or more bedroom unit: 500 square feet."
    ],
    "rationale": "The section provides the information for {{term}} of the district {{zone_abbreviation}}. It can be inferred that the {{term}} is dependent on the unit's number of bedrooms.",
    "answer": "400 sq ft (One bedroom unit), 500 sq ft (Two or more bedroom unit)"
}

Input:
Multi-family building

Output:
{
    "extracted_text": null,
    "rationale": "The section does not provide a specific value for {{term}}, and is not about single-family homes.",
    "answer": null
}
