import { Box, VStack } from "@chakra-ui/react";
import { useState } from "react";
import StateSelect from "./components/sidebar/StateSelect";
import EvalTermSelect from "./components/sidebar/EvalTermSelect";
import DistrictSelect from "./components/sidebar/DistrictSelect";

interface SidebarProps {
    onSelectionChange: (state: string, evalTerm: string, district: string) => void;
}

const Sidebar = ({ onSelectionChange }: SidebarProps) => {
    const [state, setState] = useState<string>("");
    const [evalTerm, setEvalTerm] = useState<string>("");
    const [district, setDistrict] = useState<string>("");

    const handleStateChange = (selectedState: string) => {
        setState(selectedState);
        setEvalTerm("");
        setDistrict("");
        onSelectionChange(selectedState, "", "");
    };

    const handleEvalTermChange = (selectedEvalTerm: string) => {
        setEvalTerm(selectedEvalTerm);
        setDistrict("");
        onSelectionChange(state, selectedEvalTerm, "");
    };

    const handleDistrictChange = (selectedDistrict: string) => {
        setDistrict(selectedDistrict);
        onSelectionChange(state, evalTerm, selectedDistrict);
    };

    return (
        <VStack spacing={4} align="stretch">
            <StateSelect onStateChange={handleStateChange} />
            <EvalTermSelect state={state} onEvalTermChange={handleEvalTermChange} />
            <DistrictSelect state={state} evalTerm={evalTerm} onDistrictChange={handleDistrictChange} />
        </VStack>
    );
};

export default Sidebar;
