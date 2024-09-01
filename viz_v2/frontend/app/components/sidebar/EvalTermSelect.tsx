import { Select } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import api from "../../services/api";

interface EvalTermSelectProps {
    state: string;
    onEvalTermChange: (evalTerm: string) => void;
}

const EvalTermSelect = ({ state, onEvalTermChange }: EvalTermSelectProps) => {
    const [evalTerms, setEvalTerms] = useState<string[]>([]);

    useEffect(() => {
        if (state) {
            api.get(`/evals?state=${state}`).then((response) => {
                setEvalTerms(response.data);
            });
        }
    }, [state]);

    return (
        <Select placeholder="Select Eval Term" onChange={(e) => onEvalTermChange(e.target.value)}>
            {evalTerms.map((term) => (
                <option key={term} value={term}>
                    {term}
                </option>
            ))}
        </Select>
    );
};

export default EvalTermSelect;
