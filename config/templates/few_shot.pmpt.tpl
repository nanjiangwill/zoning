# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{zone_name}}" and with an
abbreviated name "{{zone_abbreviation}}".

You are looking to find the value for "{{term}}", which also goes by the
following other names: {{synonyms}}. Only output values that are seen in the
input and do not guess! Output MUST be valid JSON, and should follow the schema
detailed below. Ensure that the field "extracted_text" does not span multiple
lines and that it is a real substring of the input. You CANNOT make up a value
for "extracted_text", and it MUST be a substring! "extracted_text" will be used
in the python statement `extracted_text in input` and if that returns False, the
universe will be destroyed! If you cannot extract reasonable text, then you
should not return an answer. For {{term}} in residential districts, we are only
interested in the answer as it pertains to single-family homes.

# Schema
{
    "extracted_text": list[str], // The verbatim text from which the result was extracted. ONLY USE VALUES EXTRACTED DIRECTLY FROM THE TEXT. Make sure to include "\n" and any type of special characters.
    "rationale": str, // A string containing a natural language explanation for the following answer
    "answer": str // The value of {{term}} extracted from the text. Answer must include units and must be normalized, e.g. (sqr. ft. becomes sq ft)
}

{% include term + "_examples.pmpt.tpl" %}
