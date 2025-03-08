standardized_prompt_template = """Given the title and content of a healthcare news article, analyze whether the claims align with plausible scenarios and whether the article maintains internal consistency. Check for misleading or unclear statements, and conclude whether the news is real or fake.
News Title:
{TITLE}
News Content:
{NEWS}
Conclusion: """

instruction = "You are a helpful assistant in misinformation detection within healthcare news article. You only output 'Fake' word for fake news and 'Real' word for correct/real news."