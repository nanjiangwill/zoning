'use client';

import {Flex, Box} from "@chakra-ui/react";
import {useState} from "react";
import Sidebar from "./Sidebar";
import PdfMetadata from "./components/pdf/PdfMetadata";
import PdfViewer from "./components/pdf/PdfViewer";

export default function Home() {
    const [state, setState] = useState<string>("");
    const [evalTerm, setEvalTerm] = useState<string>("");
    const [district, setDistrict] = useState<string>("");

    const handleSelectionChange = (selectedState: string, selectedEvalTerm: string, selectedDistrict: string) => {
        setState(selectedState);
        setEvalTerm(selectedEvalTerm);
        setDistrict(selectedDistrict);
    };

    return (
        <Flex height="100vh">
            <Box w="250px" bg="gray.100" p={2}>
                <Sidebar onSelectionChange={handleSelectionChange}/>
            </Box>
            <Box flex="1" bg="gray.50" p={"10px"}>
                <PdfMetadata state={state} evalTerm={evalTerm} district={district}/>
                <PdfViewer state={state} evalTerm={evalTerm} district={district}/>
            </Box>
        </Flex>
    );
}
