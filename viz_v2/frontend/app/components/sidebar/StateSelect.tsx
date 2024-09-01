import { Select } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import api from "../../services/api";

interface StateSelectProps {
    onStateChange: (state: string) => void;
}

const StateSelect = ({ onStateChange }: StateSelectProps) => {
    const [states, setStates] = useState<string[]>([]);

    useEffect(() => {
        api.get("/state").then((response) => {
            setStates(response.data);
        });
    }, []);

    return (
        <Select placeholder="Select State" onChange={(e) => onStateChange(e.target.value)}>
            {states.map((state) => (
                <option key={state} value={state}>
                    {state}
                </option>
            ))}
        </Select>
    );
};

export default StateSelect;
