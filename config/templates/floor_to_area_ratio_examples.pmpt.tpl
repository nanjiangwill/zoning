Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 90
{{zone_name}} ({{zone_abbreviation}})

NEW PAGE 82
(d) General Requirements.
1. Along existing streets, new buildings shall create a transition in spacing,
mass, scale, and street frontage relationship from existing buildings to
buildings in the Transit-Oriented Employment district.
a. New buildings are expected to exceed the scale and volume of
existing buildings, but shall demonstrate compatibility by varying
the massing of buildings to reduce perceived scale and volume.
The definition of massing in Article 12 illustrates the application
of design techniques to reduce the visual perception of size and
integrate larger buildings with pre-existing smaller buildings.
b. Subsection 3.2.14(d)(1) shall not apply to development on
parcels where multifamily structures are an allowable use and
the development contains two or more affordable housing units
for families or individuals with incomes below eighty percent
(80%) of the area median income.
2. On new streets, allowable building and lot types will establish the
development pattern.
(e) Design Provisions.
1. Every building shall share a frontage line with a street, or urban open
Page 82

NEW PAGE 83
space; lots fronting directly onto an urban open space (i.e., without
intervening street) shall be provided rear alley access.
2. New construction favors general office uses, with accessory retail,
personal services, restaurant, and similar uses located at street level and
residential uses permitted on third and higher floors.
3. Notwithstanding the height restrictions of Article 4, Building and Lot
Types, new buildings in the Transit Oriented Employment district are
limited to seven stories or 80 feet in height, whichever is greater.
Minimum building height is 26 feet, measured at the leave line.
4. Minimum permitted Floor Area Ratio (FAR) is .35; preferred FAR will
range from .5 to 1.5.

Output:
{
    "extracted_text": ["4. Minimum permitted Floor Area Ratio (FAR) is .35; preferred FAR will\nrange from .5 to 1.5."],
    "rationale": "The section state the {{term}} for the district, {{zone_abbreviation}}.",
    "answer": "0.35"
}

Input:
NEW PAGE 199
Section 9.406. Urban Residential Districts; area, yard and height regulations.
(1)
{{zone_abbreviation}}: Dimensional requirements for the {{zone_abbreviation}} district are listed below:
9-49
CELL (1, 1):
CELL (1, 2):
CELL (2, 1):
Minimum lot area (square feet)5
CELL (2, 2):
3,000
CELL (3, 1):
Minimum side yard (feet)4
CELL (3, 2):
5
CELL (4, 1):
Minimum setback (feet)
CELL (4, 2):
14 from back of existing or proposed curb,
whichever is greater
CELL (5, 1):
Minimum rear yard (feet)4
CELL (5, 2):
10
CELL (6, 1):
Maximum floor area ratio¹
CELL (6, 2):
0.25
CELL (7, 1):
Maximum height (feet)
CELL (7, 2):
See Tables Below
CELL (8, 1):
Minimum lot width (feet)
CELL (8, 2):
20

Output:
{
    "extracted_text": ["CELL (6, 2):\n0.25"],
    "rationale": "The table is specified the requirement for {{zone_abbreviation}}, and the cell that corresponds to the value for {{term}} in this table has this answer.",
    "answer": "0.25"
}

Input:
NEW PAGE 25
Flag - Any fabric or bunting containing colors, patterns, or symbols used as a symbol of a
government or other entity or organization.
Flashing Sign - A sign, the illumination of which is not kept constant in intensity at all times
when in use and which exhibits marked changes in lighting effects.
Flexible Space ("Flex Space") - A building or structure containing, under a common roof,
two or more uses permitted under Article 5, Zoning Districts, Permitted Uses, and
Dimensional Requirements, within the zoning district in which the Flexible Space is located.
Flood Plain - Any land area susceptible to being inundated by water from any source.
Floor Area (gross) - The sum of the gross horizontal areas of the several floors of a building
measured from the centerline of a wall separating two buildings, but not including interior
parking spaces, loading spaces for motor vehicles.
Floor Area (net) - The total of all floor areas of a building, excluding stairwells and elevator
shafts, utility and equipment rooms, restrooms, interior vehicular parking or loading, and
basements when not used for human habitation or service to the public.
Floor Area Ratio (FAR) - A relationship determined by dividing the gross floor area of all
buildings on a lot by the area of that lot.
Fraternities/Sororities Residential - A building or structure occupied and maintained for
residential uses exclusively for college or university students who are members of a social,
honorary, or professional organization which is chartered by a national, fraternal or sororal
order which is so recognized by the university, college or other institutions
Freestanding Sign - Any sign supported by structures or supports that are placed on, or
anchored in, the ground and that are independent from any building or other structure. A
Article 3 - Definitions of Terms
Page 18 of 51
Amended 2023-3-27

Output:
{
    "extracted_text": null,
    "rationale": "The section explains the definition of the term {{term}}, but does not provide a specific value for district, {{zone_abbreviation}}.",
    "answer": null
}

Input:
NEW PAGE 169
Article 3: Base Zoning Districts
{{zone_abbreviation}}
DIMENSIONAL AND RATIO STANDARDS
{{zone_name}}
Lot Size, min., per use
20,000
(square feet)
PURPOSE
Lot Width, min. (feet)
NR [1]
100
The purpose of the Medium Industrial-2 (I-2) District is to provide
locations for enterprises engaged in manufacturing, processing,
creating, repairing, renovating, painting, cleaning, and
Front Setback from
NR
50
assembling of goods, merchandise or equipment. Performance
ROW, min. (feet)
standards will be used to ensure the absence of adverse impact
beyond the lot boundaries of the use.
Side Setback, min.
None [2]
APPLICABILITY
(feet)
This district will usually be applied where the following
conditions exist:
Rear Setback, min.
(feet)
None [2]
1. Site is located within areas designated by the adopted
Comprehensive Plan as a Commercial-Industrial Transition
Activity Node.
Height, max. (feet)
45 [3]
2. Water and sewer mains exist at the site or be made available
as part of the development process.
Floor Area Ratio, max.
.65
DIMENSIONAL STANDARDS NOTES:
Required Open Space
[1] R = residential, NR = non-residential
.40
Ratio, min.
[2] Required side and rear setbacks adjacent to residentially
zoned land shall be equal to the required side or rear setback of
the adjacent residential district.
Required
[3] Two feet of additional height shall be allowed for one foot
Pedestrian/Landscape
.05

Output:
{
    "extracted_text": [".65"],
    "rationale": "The section state the {{term}} for the district, {{zone_abbreviation}}.",
    "answer": "0.65"
}


Input:
Multi-family building

Output:
{
    "extracted_text": null,
    "rationale": "The section does not provide a specific value for {{term}}, and is not about single-family homes.",
    "answer": null
}
