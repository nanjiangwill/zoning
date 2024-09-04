The range for {{term}} is typically between 1 and 20 (per dwelling unit, square feet of floor area, etc.). Please focus on values within this range when searching for {{term}} and provide the answer as a decimal value with unit (e.g., 1 parking space per unit plus 1 guest parking space for every 4 units should be calculated as "1.25 per unit"). However, bear in mind that values falling outside of these ranges are possible, so do not disregard them.

Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 165
165/167
CELL (1, 1):
Restaurants (with drive through service)
CELL (1, 2):
One (1) space for each four (4) seating
accommodations, plus one (1) space for each
two (2) employees on the shift of largest
employment; and Three (3) stacking spaces
for each drive through window.
CELL (2, 1):
School, Elementary and Middle (both public
and private):
CELL (2, 2):
One (1) parking space for each classroom and
administrative office, plus one (1) parking
space for each employee and one (1) large
space for each bus.
CELL (3, 1):
School, High School (both public and private)
CELL (3, 2):
One (1) parking space for each fifteen (15)
students for which the building was designed,
plus one (1) parking space for each classroom
and administrative office, plus one (1) parking
space for each employee, plus one (1) large
space for each bus.
CELL (4, 1):
Shopping Centers, large and small
CELL (4, 2):
One (1) space for each three hundred (300)
square feet of gross floor area.
CELL (5, 1):
Single Family, Duplex, Condominiums,
Manufactured Homes, and similar residential
CELL (5, 2):
Two (2) spaces per dwelling unit
CELL (6, 1):
Terminals, Bus
CELL (6, 2):
One (1) space for each employee and one (1)
space for each bus loading ramp and track.
CELL (7, 1):
Stadiums, Theaters, and similar uses
involving the assembling of persons:
CELL (7, 2):
One (1) parking space for each four (4) seats
in the largest assembly room. One (1) seat
equals two (2) feet of bench length.

Output:
{
    "extracted_text": [["CELL (5, 2):\nTwo (2) spaces per dwelling unit", 165]],
    "rationale": "The cell corresponding to single-family contains the information for {{term}}.",
    "answer": "2 per dwelling unit"
}

Input:
NEW PAGE 191
CHAPTER 10:
PARKING
10.3 Required Vehicle and Bicycle Parking
(a) All square footage calculations are gross interior floor area with the exception of a Restaurant/Bar use which can
include both interior and exterior gross dining floor area for square footage calculations.
(b) Required bicycle parking spaces are based on the indicated minimum percentage of vehicle parking spaces provided.
A single "inverted U" bicycle parking rack will count as two (2) bicycle parking spaces. The minimum number of
bicycle parking spaces per use, when required, is two (2) or one rack and the maximum number of required bicycle
spaces shall be 20 or 10 racks.
(c) Bicycle parking is required for multi-family dwellings of only more than 4 units per building
(d) Garage parking shall not count towards residential parking requirements, except for homes with two car garages,
which may count 1 garage space towards parking requirements.
SALISBURY, NC LAND DEVELOPMENT ORDINANCE
10-3
ADOPTED DECEMBER 18, 2007; EFFECTIVE JANUARY 1, 2008
AMENDED 5/6/08, ORD.2008-17;4/3/18, ORD.2018-16; 10/2/18, ORD.2018-48; 6/18/19,
ORD.2019-40;5/17/22, ORD.2022-37;2/21/23,ORD.2023-15;5/16/23,ORD.2023-31
CELL (1, 1):
Use Type
CELL (1, 2):
Vehicle Parking Spaces
CELL (1, 3):
Vehicle Parking Spaces
CELL (1, 4):
Bicycle
CELL (2, 1):
CELL (2, 2):
Minimum
Required(a)
CELL (2, 3):
Maximum
Permitted(a)
CELL (2, 4):
Parking
Spaces(b)
CELL (3, 1):
Residential
CELL (3, 2):
1 per bedroom up to 2
per unit
CELL (3, 3):
CELL (3, 4):
5% (c)
CELL (4, 1):
Lodging
CELL (4, 2):
1 per room or suite
CELL (4, 3):
CELL (4, 4):
2%
CELL (5, 1):
General Office /
Business or Personal
Service
CELL (5, 2):
2 per 1000 ft2
CELL (5, 3):
5 per 1000 ft2
CELL (5, 4):
5%

Output:
{
    "extracted_text": [["CELL (3, 2):\n1 per bedroom up to 2 per unit", 191]],
    "rationale": "The cell corresponding to residential contains the information for {{term}}.",
    "answer": "1 per bedroom, 2 per unit"
}

Input:
NEW PAGE 130
8.1. Off-Street Parking Requirements.
With the exception of Section 8.6, there shall be provided at the time of the erection of any
building, or at the time any principal building is enlarged or increased in capacity by adding dwelling
units, guest rooms, seats, or floor area; or before conversion from one (1) type of use or occupancy to
another, permanent off-street parking space in the amount specified by this section. Such parking
spaces may be provided in a parking garage or parking lot constructed in accordance with Section 8.2.
C.
Minimum Off-Street Parking Requirements. The following off-street parking spaces shall be
required:
Created: 2024-05-20 14:11:49 [EST]
(Supp. No. 31)
Page 130 of 250
CELL (1, 1):
Classification
CELL (1, 2):
Off-Street Parking Requirement
(Any fraction space e.g., 47.3 shall be considered
the next whole number, e.g., 48)
CELL (2, 1):
Residential:
CELL (2, 2):
CELL (3, 1):
Housing designed for and used by the elderly
CELL (3, 2):
1 space per 2 dwelling units
CELL (4, 1):
Incidental home occupations
CELL (4, 2):
1 space per addition to the residential
requirement
CELL (5, 1):
Multi-Family residences including townhouses
CELL (5, 2):
2 spaces per dwelling unit
CELL (6, 1):
Congregate care
CELL (6, 2):
1 space per 2 beds 1 space per 2 Dwelling Units
CELL (7, 1):
Single-family and two-family residences (may be
in a single drive with one car behind the other)
CELL (7, 2):
2 spaces per Dwelling Unit
CELL (8, 1):
Commercial and Industrial:
CELL (8, 2):
CELL (9, 1):
Auto service stations and/or repair shops
CELL (9, 2):
3 spaces per service bay, plus 1 space per
wrecker or service vehicle and 2 spaces per gas
dispenser
CELL (10, 1):
Auto sales
CELL (10, 2):
3 spaces plus 1 space per 400 square feet of
building area devoted to sales
CELL (11, 1):
Bank and consumer financial services
CELL (11, 2):
1 space per 200 square feet of gross floor area
CELL (12, 1):
Barber & beauty shops and other similar personal
services
CELL (12, 2):
2 spaces per operator
CELL (13, 1):
Car washes
CELL (13, 2):
3 spaces per service bay
CELL (14, 1):
Delivery, ambulance, taxi, and other similar
services
CELL (14, 2):
1 space per vehicle, plus 1 space for each
employee
CELL (15, 1):
Drive-through services such as banks, automobile
service stations, dry cleaners, car washes and
similar uses (in addition to Use Requirements)
CELL (15, 2):
Stacking for 4 vehicles at each bay window or
lane

Output:
{
    "extracted_text": [["CELL (7, 2):\n2 spaces per Dwelling Unit", 130]],
    "rationale": "The cell corresponding to single-family contains the information for {{term}}.",
    "answer": "2 per dwelling unit"

Input:
Multi-family building

Output:
{
    "extracted_text": null
    "rationale": "The section does not provide a specific value for {{term}}, and is not about single-family homes."
    "answer": null
}
