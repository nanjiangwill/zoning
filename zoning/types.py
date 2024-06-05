from dataclasses import dataclass

@dataclass
class District:
    full_name: str
    short_name: str

@dataclass
class PageSearchOutput:
    text: str
    page_number: int
    highlight: list[str]
    score: float
    query: str

@dataclass
class ExtractionOutput:
    extracted_text: list[str]
    rationale: str
    answer: str | None

    def __str__(self):
        return f"ExtractionOutput(extracted_text={self.extracted_text}, rationale={self.rationale}, answer={self.answer})"

# @dataclass
# class ExtractionOutput2:
#     district_explanation: str
#     district: str
#     term_explanation: str
#     term: str
#     explanation: str
#     answer: str | None

#     def __str__(self):
#         return f"ExtractionOutput(extracted_text={self.extracted_text}, rationale={self.rationale}, answer={self.answer})"

@dataclass
class LookupOutput:
    output: ExtractionOutput | ExtractionOutput2 | None
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """The set of pages, in descending order or relevance, used to produce the
    result."""

    def __str__(self):
        return f"LookupOutput(output={self.output}, search_pages=[...], search_pages_expanded={self.search_pages_expanded})"

    def to_dict(self):
        return json.loads(self.model_dump_json())

