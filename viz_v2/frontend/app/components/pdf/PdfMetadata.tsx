import { Box, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import api from "../../services/api";

interface PdfMetadataProps {
    state: string;
    evalTerm: string;
    district: string;
}

const PdfMetadata = ({ state, evalTerm, district }: PdfMetadataProps) => {
    const [metadata, setMetadata] = useState<any>(null);

    useEffect(() => {
        if (state && evalTerm && district) {
            api.get(`/pdf_metadata?state=${state}&eval_term=${evalTerm}&place=${district}`).then((response) => {
                setMetadata(response.data);
            });
        }
    }, [state, evalTerm, district]);

    if (!metadata) {
        return <Text>Select a state, eval term, and district to view metadata</Text>;
    }

    return (
        <Box p={4} bg="white" shadow="md" borderRadius="md" height="180px">
            <Text fontWeight="bold">Evaluation Term: {metadata.eval_term}</Text>
            <Text>District Full Name: {metadata.district_full_name}</Text>
            <Text>District Short Name: {metadata.district_short_name}</Text>
            <Text>Value: {metadata.norm}</Text>
            <Text>Rationale: {metadata.rationale}</Text>
        </Box>
    );
};

export default PdfMetadata;
