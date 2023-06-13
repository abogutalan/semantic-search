import requests
from bs4 import BeautifulSoup
import pandas as pd

class SparkNLPScraper:
    def __init__(self, url="https://sparknlp.org/docs/en/annotators"):
        self.url = url
        self.titles = []
        self.texts = []
        self.links = []

    def scrape_website(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        divs = soup.find_all('div', class_=['h3-box tabs-python-scala-box', 'tabs-box tabs-new'])

        for div in divs:
            self._extract_info_from_div(div)

    def _extract_info_from_div(self, div):
        content = ''
        h2 = div.find('h2')

        if h2 is not None:
            self.titles.append(h2['id'])
            self.links.append("https://sparknlp.org/docs/en/annotators#" + h2['id'])
            content += self._extract_text(div)
        if content != '':
            self.texts.append(content)

    def _extract_text(self, div):
        content = ''
        tag_types = ['p', 'table', 'details']

        for tag_type in tag_types:
            tags = div.find_all(tag_type)
            for tag in tags:
                if tag.text not in content:
                    content += tag.text

        return content

    def to_dataframe(self):
        return pd.DataFrame({
            "Title": self.titles,
            "Text": self.texts,
            "Link": self.links
        }).reset_index()

    def to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)


# Run the main function
if __name__ == "__main__":
    # Use the class to scrape the website and save the data to a CSV
    scraper = SparkNLPScraper()
    scraper.scrape_website()
    scraper.to_csv('annotators.csv')
