import api from "../../services/api";
import {ScrollMode, SpecialZoomLevel, Viewer, Worker,} from '@react-pdf-viewer/core';
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';
import {defaultLayoutPlugin} from "@react-pdf-viewer/default-layout";

interface PdfViewerProps {
    state: string;
    evalTerm: string;
    district: string;
}

const PdfViewer = ({ state, evalTerm, district }: PdfViewerProps) => {
    if (!state || !evalTerm || !district) {
        return null;
    }

    const pdfUrl = `${api.defaults.baseURL}/pdf_file?state=${state}&eval_term=${evalTerm}&place=${district}`;
    const pdfjsVersion = '3.11.174';
    const defaultLayoutPluginInstance = defaultLayoutPlugin();

    return (
        <Worker workerUrl={`https://unpkg.com/pdfjs-dist@${pdfjsVersion}/build/pdf.worker.min.js`}>
            <div style={{ width: 'calc(100vw - 270px)', height: 'calc(100vh - 200px)' }}>
                <Viewer
                    fileUrl={pdfUrl}
                    scrollMode={ScrollMode.Horizontal}
                    defaultScale={SpecialZoomLevel.PageFit}
                    plugins={[
                        defaultLayoutPluginInstance
                    ]}
                />
            </div>
        </Worker>
    );
};

export default PdfViewer;
