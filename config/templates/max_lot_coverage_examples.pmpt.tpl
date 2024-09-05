Remember, the {{term}} refers solely to the maximum lot coverage of buildings for primary use and all accessory structures. Impervious surface coverage, such as pavement, is not included in this value. If the document specifies that impervious surface is included or simply refers to maximum lot coverage generally, which may contain other impervious surfaces, DO NOT include it.

The range for {{term}} is typically between 5% and 100% (0.05 to 1). Please focus on values within this range when searching for {{term}} and provide the answer as a whole number (e.g., 50% or 0.5 should be 50). However, bear in mind that values falling outside of these ranges are possible, so do not disregard them.

Here are several examples that you can use as references.
# Examples

Input:
NEW PAGE 125
New Bern, NC Code of Ordinances
about:blank
(Ord. No. 1996-13, § 5, 2-13-96)
Section 15-181. - Residential density.
(a)
Subject to the provisions of section 15-190, every lot used for single-family detached residential purposes shall have at least the number of square feet indicated as
the minimum permissible in the zone where the use is located, according to section 15-180 (Minimum lot size requirements).
(b)
Subject to section 15-182, every lot developed for single family-attached residential purposes shall have the number of square feet per dwelling unit indicated in the
following table.
CELL (1, 1):
Zone
CELL (1, 2):
Minimum Square feet per lot
CELL (2, 1):
{{zone_abbreviation}}
CELL (2, 2):
2,500

NEW PAGE 126
New Bern, NC Code of Ordinances
about:blank
(c) Subject to section 15-182 (Residential density bonuses), every lot developed for duplex or multifamily residential purposes shall have the number of square feet per
dwelling unit indicated on the following table. In determining the number of dwelling units permissible on a tract of land, the fractions shall be rounded to the
nearest whole number.
126 of 294
4/22/24, 19:55
CELL (1, 1):
XXXX
CELL (1, 2):
2,000
CELL (2, 1):
XXXX
CELL (2, 2):
1,500
CELL (3, 1):
XXXX
CELL (3, 2):
No minimum, maximum 75% lot coverage
CELL (1, 1):
Zone
CELL (1, 2):
Minimum Square feet per Dwelling Unit, Multifamily and Duplex
CELL (2, 1):
XXXX
CELL (2, 2):
5 acres first unit; 20,000 each additional unit
CELL (3, 1):
XXXX
CELL (3, 2):
XXXX
CELL (4, 1):
XXXX
CELL (4, 2):
20,000 first unit; 10,000 each additional unit
CELL (5, 1):
{{zone_abbreviation}}
CELL (5, 2):
10,000 first unit: 5,000 each additional unit

NEW PAGE 127
New Bern, NC Code of Ordinances
about:blank
(d) Notwithstanding subsections (b) and (c) of this section, the total ground area covered by the principal building and all accessory buildings shall not exceed 30 percent
of the total lot area, except in the case of single-family attached which shall have a maximum lot coverage of 75% of the lot area.
(Ord. No. 1996-13, § 6, 2-13-96; Ord. No. 2006-12, § 1, 4-25-06; Ord. No. 16-047, §§ 59, 60, 9-13-16)

Output:
{
    "extracted_text": ["in the case of single-family attached which shall have a maximum lot coverage of 75% of the lot area."],
    "rationale": "It can be inferred that the {{term}} used in this section refers to total ground area covered by the principal building and all accessory building, satisfying our definition. And we focus on single-family for residential area, thus the {{term}} is 75%.",
    "answer": "75"
}

Input:
NEW PAGE 187
CHARLOTTE CODE
PART. : MULTI-FAMILY DISTRICTS
- All other buildings, including
30
30
30
30
30
planned multi-family developments
(except as provided for in
Section 9.303(19)(f))
(Petition No. 2010-073 § 9.305(1)(e1)(e2), 12/20/10)
(Petition No. 2014-088 § 9.305(1)(e1)(e2), 10/20/2014)
(i)
Maximum building coverage for
See Table 9.305(1)(i)
detached dwellings only
(Petition No. 2007-70, § 9.305(i), 06/18/07)
9 37
CELL (1, 1):
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
CELL (2, 1):
(e2) Minimum setback from right-of-way
along local and collector
CELL (2, 2):
CELL (2, 3):
CELL (2, 4):
CELL (2, 5):
CELL (2, 6):
CELL (3, 1):
streets (feet) 3, 10,11,12,13,14
- Detached, duplex, triplex and
quadraplex dwellings
CELL (3, 2):
17
CELL (3, 3):
17
CELL (3, 4):
17
CELL (3, 5):
17
CELL (3, 6):
17
CELL (4, 1):

NEW PAGE 188
CHARLOTTE CODE
PART 3 : MULTI-FAMILY DISTRICTS
Table 9.305(1)(i)
Maximum Building Coverage for Detached Dwellings
(Petition No. 2007-70, § 9.305(i), 06/18/07)
(j)
Maximum height (feet) 7
(Petition No. 2007-70, § 9.305(j), 06/18/07)
See Tables Below
(Petition No. 2011-038 § 9.305(1)(j), 07/18/2011)
Table 9.305(1)(j)(A)
9-38
CELL (1, 1):
Single Family Lot Size
(Sq. Ft.)
CELL (1, 2):
Maximum Building Coverage
(%)
CELL (2, 1):
Up to 4,000
CELL (2, 2):
50
CELL (3, 1):
4,001-6,500
CELL (3, 2):
45
CELL (4, 1):
6,501-8,500
CELL (4, 2):
40
CELL (5, 1):
8,501-15,000
CELL (5, 2):
35
CELL (6, 1):
15,001 or greater
CELL (6, 2):
30

Output:
{
    "extracted_text": [
        "CELL (2, 1):\nUp to 4,000",
        "CELL (2, 2):\n50",
        "CELL (3, 1):\n4,001-6,500",
        "CELL (3, 2):\n45",
        "CELL (4, 1):\n6,501-8,500",
        "CELL (4, 2):\n40",
        "CELL (5, 1):\n8,501-15,000",
        "CELL (5, 2):\n35",
        "CELL (6, 1):\n15,001 or greater",
        "CELL (6, 2):\n30"
    ],
    "rationale": "The cell specifies the requirement of {{term}} for the district, {{zone_abbreviation}}. It can be inferred that the requirement for {{term}} follows the instruction on Table 9.305(1)(i), which depends on Single Family Lot Size.",
    "answer": "50; 45; 40; 35; 30 (depending on the Single Family Lot Size)"
}

Input:
NEW PAGE 42
Brunswick County, NC Code of Ordinances
about:blank
4.2 - GROUPING OF DISTRICTS
4.2.1. Where the phase "residential district" is used in this Ordinance, the phrase shall be construed to include the following districts:
A. {{zone_abbreviation}} Low Density Residential;
B. XXXX Medium Density Residential;
C. XXXX High Density Residential;
4.3.1. Planned Development
A. Intent
The intent of a Planned Development (PD) is to promote quality development by providing flexibility in the mixture of uses and in meeting
dimensional and other requirements of this Ordinance. A PD utilizes exceptional design and best management practices that result in
development that is aesthetically pleasing, promotes environmental sensitivity and makes more efficient use of the land, resulting in increased
open space.
B. Planned Development Approval
Planned Development projects shall be approved in accordance with the Planned Development approval process found in Section 3.3.3 and the
site plan requirements as outlined in Article 3.
C. Development Intensity
The building area coverage and number of dwelling units in a project utilizing the PD development standards shall be calculated as follows:
1. The building area coverage shall be the dimensional standards of the applicable zoning District (i.e., R-7500, C-LD, etc.). However, the
developed area may be increased as a result of utilizing exceptional design and/or best management practices as provided in Section 6.1,
Design Flexibility. The extent of the allowable increase will be determined on a case-by-case basis by the Planning Director (minor site plans)
or the Planning Board (major site plans) in relation to the extent of the successful use of exceptional design and best management practices
in the project site plan.

Output:
{
    "extracted_text": null,
    "rationale": "The section does refer to building area coverage, following our definition of {{term}}, but it not provide a specific number for {{zone_abbreviation}}, thus the value is not included in the answer.",
    "answer": null

}

Input:
NEW PAGE 61
Table 4-2-1 Table of Density and Dimensional Requirements
CELL (1, 1):
Zoning District
CELL (1, 2):
Minimum Lot Area (Sq Ft.)
CELL (1, 3):
Minimum LotWidth (Ft) **
CELL (1, 4):
Front Yard Setback (Ft.)
CELL (1, 5): Side Yard Setback (Ft.)
CELL (1, 6):
Rear Yard Setback (Ft.)
CELL (1, 7):
Maximum Building Height (Ft)
CELL (1, 8):
Maximum Lot Coverage
CELL (1, 9):
Development Standards
CELL (2, 1):
{{zone_abbreviation}}
CELL (2, 2):
CELL (2, 3):
CELL (2, 4):
CELL (2, 5):
CELL (2, 6):
CELL (2, 7):
CELL (2, 8):
CELL (2, 9):
CELL (3, 1):
Single-family dwelling
CELL (3, 2):
20,000*
CELL (3, 3):
85
CELL (3, 4):
30
CELL (3, 5):
10 a
CELL (3, 6):
25 g
CELL (3, 7):
40
CELL (3, 8):
40%
CELL (3, 9):
CELL (4, 1):
XXXX
CELL (4, 2):
CELL (4, 3):
CELL (4, 4):
CELL (4, 5):
CELL (4, 6):
CELL (4, 7):
CELL (4, 8):
CELL (4, 9):
CELL (5, 1):
Single-family dwelling
CELL (5, 2):
15,000
CELL (5, 3):
75
CELL (5, 4):
30
CELL (5, 5):
10 a
CELL (5, 6):
25 g
CELL (5, 7):
35
CELL (5, 8):
40%
CELL (5, 9):
CELL (6, 1):
XXXX
CELL (6, 2):
CELL (6, 3):
CELL (6, 4):
CELL (6, 5):
CELL (6, 6):
CELL (6, 7):
CELL (6, 8):
CELL (6, 9):
CELL (7, 1):
Single-family dwelling
CELL (7, 2):
12,000
CELL (7, 3):
65
CELL (7, 4):
25
CELL (7, 5):
10 a
CELL (7, 6):
25 g
CELL (7, 7):
35
CELL (7, 8):
40%
CELL (7, 9):
CELL (8, 1):
XXXX
CELL (8, 2):
CELL (8, 3):
CELL (8, 4):
CELL (8, 5):
CELL (8, 6):
CELL (8, 7):
CELL (8, 8):
CELL (8, 9):

Output:
{
    "extracted_text": null,
    "rationale": "The section does not specify the maximum lot coverage is for buildings only, thus the value is not included in the answer.",
    "answer": null
}

Input:
Multi-family building

Output:
{
    "extracted_text": null,
    "rationale": "The section does not provide a specific value for {{term}}, and is not about single-family homes.",
    "answer": null
}
