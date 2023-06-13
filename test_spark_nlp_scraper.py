import unittest
from unittest.mock import patch
from spark_nlp_scraper import SparkNLPScraper

class TestSparkNLPScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = SparkNLPScraper()

    @patch('requests.get')
    def test_scrape_website(self, mock_get):
        mock_get.return_value.content = b'<div class="h3-box tabs-python-scala-box"><h2 id="title">Title</h2><p>Text</p></div>'
        self.scraper.scrape_website()
        self.assertEqual(self.scraper.titles, ['title'])
        self.assertEqual(self.scraper.links, ['https://sparknlp.org/docs/en/annotators#title'])
        self.assertEqual(self.scraper.texts, ['Text'])

    def test_to_dataframe(self):
        self.scraper.titles = ['title1', 'title2']
        self.scraper.texts = ['text1', 'text2']
        self.scraper.links = ['link1', 'link2']
        df = self.scraper.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df['Title']), ['title1', 'title2'])
        self.assertEqual(list(df['Text']), ['text1', 'text2'])
        self.assertEqual(list(df['Link']), ['link1', 'link2'])

    @patch('pandas.DataFrame.to_csv')
    def test_to_csv(self, mock_to_csv):
        self.scraper.to_csv('filename')
        mock_to_csv.assert_called_once()

if __name__ == '__main__':
    unittest.main()
