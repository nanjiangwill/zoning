import { Select } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import api from "../../services/api";

interface DistrictSelectProps {
    state: string;
    evalTerm: string;
    onDistrictChange: (district: string) => void;
}

const DistrictSelect = ({ state, evalTerm, onDistrictChange }: DistrictSelectProps) => {
    const [districts, setDistricts] = useState<string[]>([]);

    useEffect(() => {
        if (state && evalTerm) {
            api.get(`/evals/${evalTerm}?state=${state}`).then((response) => {
                setDistricts(response.data);
            });
        }
    }, [state, evalTerm]);

    return (
        <Select placeholder="Select District" onChange={(e) => onDistrictChange(e.target.value)}>
            {districts.map((district) => (
                <option key={district} value={district}>
                    {district}
                </option>
            ))}
        </Select>
    );
};

export default DistrictSelect;
