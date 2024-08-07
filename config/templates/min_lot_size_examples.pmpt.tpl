Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 11

{{zone_abbreviation}} Zone

CELL (2, 1):
Field
CELL (2, 2):
Value
CELL (3, 1):
{{term}}
CELL (3, 2):
123456 sq ft
CELL (4, 1):
Apartment Area
CELL (4, 2):
10,000


Output:
{
    "extracted_text": ["CELL (3, 2):\n123456 sq ft"],
    "rationale": "The cell that corresponds to the value for {{term}} in this table has this answer.",
    "answer": "123456"
}

Input:
NEW PAGE 32

Section 6.3 Industrial Area & Dimensional Requirements
1 Public Sewer or Public Water
2 Neither Public Sewer nor Public Water
3 For proposed warehouse/manufacturing uses 35%, for all other uses 25%, subject to
Commission's authority to permit greater coverage based on landscape, architectural design
and site development elements that exceed minimum standard of the Regulation
4
Shall be in addition to requirements of 8.1.2 Landscaping and Buffers
6-4

CELL (1, 1):
CELL (1, 2):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 3):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 4):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 5):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 6):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 7):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 8):
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 9):
CELL (1, 10):
CELL (2, 1):
Zone
CELL (2, 2):
Min
Area
CELL (2, 3):
Min
Width
CELL (2, 4):
Min
Depth
CELL (2, 5):
Max
Stories
CELL (2, 6):
Max
Height
CELL (2, 7):
Max % Lot
Coverage
CELL (2, 8):
Min
Front
Yard
CELL (2, 9):
Min
Side
Yard
CELL (2, 10):
Min
Rear
Yard
CELL (3, 1):
I
CELL (3, 2):
20,000
sq. ft.
CELL (3, 3):
100'
CELL (3, 4):
100'
CELL (3, 5):
2 1/2
CELL (3, 6):
50'
CELL (3, 7):
N/A
CELL (3, 8):
25'
CELL (3, 9):
20'
CELL (3, 10):
20'
CELL (4, 1):
{{zone_abbreviation}}
CELL (4, 2):
40,000
sq. ft. 1
CELL (4, 3):
150'
CELL (4, 4):
150'
CELL (4, 5):
2 1/2
CELL (4, 6):
50'
CELL (4, 7):
25%³
CELL (4, 8):
50'
CELL (4, 9):
20'
CELL (4, 10):
20'
CELL (5, 1):
{{zone_abbreviation}}
CELL (5, 2):
60,000
sq. ft. 2
CELL (5, 3):
200'
CELL (5, 4):
200'
CELL (5, 5):
2 1/2
CELL (5, 6):
50'
CELL (5, 7):
25%³
CELL (5, 8):
50'
CELL (5, 9):
20'
CELL (5, 10):
20'4

Output:
{
    "extracted_text": [
        "1 Public Sewer or Public Water",
        "2 Neither Public Sewer nor Public Water",
        "CELL (4, 2): \n40,000\nsq. ft.",
        "CELL (5, 2): \n60,000\nsq. ft."
    ],
    "rationale": "From this page we can infer that the value is conditional on the presence of a public sewer or water system, and there are two different values for the current zone, depending on that.",
    "answer": "40,000 sq ft (if public water or sewer); 60,000 sq ft (otherwise)"
}

Input:
NEW PAGE 66

{{zone_abbreviation}} Zone

{{term}} is 123 sq ft, side length is 10 ft

NEW PAGE 67

{{zone_abbreviation}} Zone - Senior Active Overlay

{{term}} is 1523 sq ft

DKEWKWKDS Zone

{{term}} is 20,000 sq ft

Output:
{
    "extracted_text": ["{{term}} is 123 sq ft"],
    "rationale": "The section titled {{zone_abbreviation}} says the answer explicitly. We ignore sections that specify regulations for overlay districts within this district.",
    "answer": "123 sq ft"
}

Input:
NEW PAGE 47

specified in Schedule A. The total floor area of all buildings and other structures on any lot, excluding basements, shall not exceed the percentage of the lot as specified in Schedule A.

7.6 Minimum Floor Area - Dwelling: Each dwelling shall have a minimum floor area on the ground floor as specified in Schedule A.

7.7 Corner Lots: On any corner lot, the required minimum street setback specified in Schedule A shall be applied to each street line.

SCHEDULE A
AREA, LOCATION AND BULK STANDARDS Line Standards Districts
A
B
C

{{term}} (in square feet)
8,400
5,500
6,300

Output:
{
    "extracted_text": ["6,300"],
    "rationale": "The section {{term}} or its close synonyms says the answer explicitly.",
    "answer": "6,300 sq ft"
}


Input:
NEW PAGE 84

SECTION 9A - Housing Opportunity Development (This Section Repealed - August 15, 2014)
SECTION 9B - Housing Opportunity Development District (Effective July 1, 2014)

Housing Opportunity Development:
9B.1 Intent and Purpose:
This Housing Opportunity Development ("HOD") District is intended to increase in the Town of Bethany the supply of housing that is within the economic means of moderate income households. The HOD District is a separate and independent zoning district and shall replace, not supplement, the existing zoning district of any property to which it is applied.

9B.2 Permitted Uses:
Principal Uses: Residential homes with a maximum of three dwelling units per building as part of a Housing Opportunity Development.
Accessory Uses:
Accessory buildings, structures, and uses.
Agricultural or farming uses.
Common or community buildings for residents and guests (not for residential purposes or temporary/transient occupancy).

9B.3 Eligible Areas:
Criteria for HOD Application:
The total combined property shall be at least (30) acres in size.
Frontage on Old Amity Road and Meyers Road.
Owned by the same person or entity for at least three years prior to application.

Output:
{
    "extracted_text": [The total combined property shall be at least (30) acres in size.],
    "rationale": "The section state the {{term}}",
    "answer": "30 acres"
}


Input:
Multi-family building

Output:
null
