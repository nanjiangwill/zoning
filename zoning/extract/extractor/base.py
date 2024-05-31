class Extractor:
    def __init__(self, extractor_config):
        self.config = extractor_config
        self.name = extractor_config.name
        self.output_path = extractor_config.output_path

    def extract(self, pdf_path):
        pass

    def post_extract(self):
        pass
