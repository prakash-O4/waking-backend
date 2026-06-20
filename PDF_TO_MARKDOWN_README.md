# PDF to Markdown Converter

A robust PDF to Markdown converter that uses **Azure AI Document Intelligence** to extract text and tables from PDF files while preserving Nepali Unicode characters.

## Features

- **Nepali Language Support**: Preserves Nepali (देवनागरी) Unicode characters accurately
- **Table Extraction**: Converts PDF tables to Markdown table syntax
- **Intelligent Layout Analysis**: Recognizes document structure (headings, paragraphs, footers, etc.)
- **Batch Processing**: Convert multiple PDFs at once
- **Cloud-Based AI**: Uses Azure's advanced Document Intelligence service
- **UTF-8 Encoding**: Ensures all characters are properly preserved

## Prerequisites

1. **Azure Account**: You need an Azure subscription
2. **Azure Document Intelligence Resource**: Create a Document Intelligence resource in Azure Portal
3. **Python 3.8+**: Required for running the script

## Setup Instructions

### 1. Create Azure Document Intelligence Resource

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new **Document Intelligence** (or **Form Recognizer**) resource
3. After creation, go to **Keys and Endpoint** section
4. Copy the **Endpoint URL** and **Key**

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

The following packages will be installed:
- `azure-ai-documentintelligence==1.0.0b1` - Azure Document Intelligence SDK
- `azure-core==1.30.0` - Azure core libraries
- `python-dotenv` - Environment variable management

### 3. Configure Environment Variables

Add your Azure credentials to the `.env` file:

```bash
# Azure Document Intelligence (for PDF to Markdown conversion)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-resource-name.cognitiveservices.azure.com/"
AZURE_DOCUMENT_INTELLIGENCE_KEY="your-azure-key-here"
```

Replace:
- `your-resource-name` with your actual Azure resource name
- `your-azure-key-here` with your actual API key

## Usage

### Command Line Interface

#### Convert Single PDF

```bash
# Convert a single PDF file
python app/utils/pdf_to_markdown.py input.pdf -o output.md

# Without specifying output (creates input.md)
python app/utils/pdf_to_markdown.py input.pdf
```

#### Batch Convert Multiple PDFs

```bash
# Convert all PDFs in a directory
python app/utils/pdf_to_markdown.py source/ -o processed/ --batch

# This will convert all PDF files from 'source/' directory
# and save Markdown files to 'processed/' directory
```

### Python Script Usage

```python
from app.utils.pdf_to_markdown import PDFToMarkdownConverter

# Initialize converter
converter = PDFToMarkdownConverter()

# Convert single file
markdown_content = converter.convert_pdf_to_markdown(
    pdf_path="documents/nepal_constitution.pdf",
    output_path="output/nepal_constitution.md"
)

# Batch convert
output_files = converter.batch_convert(
    input_dir="source/",
    output_dir="processed/chunks/",
    pattern="*.pdf"
)

print(f"Converted {len(output_files)} files")
```

### Example: Converting Legal Documents

```python
from app.utils.pdf_to_markdown import PDFToMarkdownConverter

# For your legal document processing pipeline
converter = PDFToMarkdownConverter()

# Convert Nepali legal PDFs
legal_docs = [
    "source/nepal_constitution.pdf",
    "source/civil_code.pdf",
    "source/criminal_code.pdf"
]

for doc in legal_docs:
    output_path = doc.replace(".pdf", ".md").replace("source/", "processed/")
    converter.convert_pdf_to_markdown(doc, output_path)
    print(f"✓ Converted: {doc}")
```

## Output Format

The converter generates structured Markdown with:

### Document Structure
```markdown
# Document Analysis
*Extracted using Azure Document Intelligence*
---

## Page 1

### Heading Text
Regular paragraph text in Nepali...

| Header 1 | Header 2 | Header 3 |
| --- | --- | --- |
| Cell 1 | Cell 2 | Cell 3 |
| Data 1 | Data 2 | Data 3 |

## Page 2
...
```

### Content Types Supported

- **Titles**: Formatted as `# Title`
- **Section Headings**: Formatted as `## Heading`
- **Paragraphs**: Regular text with proper line breaks
- **Tables**: Converted to Markdown table syntax
- **Headers/Footers**: Formatted as italics
- **Footnotes**: Formatted as blockquotes

## Integration with Wakilg-Backend

This converter is designed to work seamlessly with your legal document processing pipeline:

```python
from app.utils.pdf_to_markdown import PDFToMarkdownConverter

def process_legal_documents():
    """Process legal PDFs for RAG ingestion"""
    converter = PDFToMarkdownConverter()

    # Convert PDFs to Markdown
    converter.batch_convert(
        input_dir="source/",
        output_dir="app/processed/chunks/"
    )

    # Now the Markdown files can be:
    # 1. Chunked for embedding
    # 2. Ingested into Pinecone
    # 3. Used by the RAG system
```

## Advantages Over Other Methods

### Why Azure Document Intelligence?

1. **Superior Nepali Support**: Better Unicode handling than PyPDF2, pdfplumber, or OCR
2. **Table Detection**: Automatically identifies and preserves table structure
3. **Layout Analysis**: Understands document hierarchy (headings, paragraphs, etc.)
4. **Cloud-Based**: No need to manage complex OCR models locally
5. **Pre-trained**: Works out-of-box without training on Nepali text

### Comparison with Alternatives

| Feature | Azure DI | PyPDF2 | pdfplumber | Tesseract OCR |
|---------|----------|---------|------------|---------------|
| Nepali Character Accuracy | ✓✓✓ | ✓ | ✓✓ | ✓✓ |
| Table Extraction | ✓✓✓ | ✗ | ✓✓ | ✗ |
| Layout Analysis | ✓✓✓ | ✗ | ✓ | ✗ |
| No Local Setup | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✗ |
| Scanned PDF Support | ✓✓✓ | ✗ | ✗ | ✓✓✓ |

## Troubleshooting

### Common Issues

#### 1. Authentication Error
```
Error: Azure Document Intelligence credentials not found
```
**Solution**: Ensure `.env` file contains valid `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENT_INTELLIGENCE_KEY`

#### 2. Character Encoding Issues
```
UnicodeDecodeError or mojibake characters
```
**Solution**: The script uses UTF-8 encoding by default. Ensure your terminal supports UTF-8:
```bash
export LANG=en_US.UTF-8
```

#### 3. Rate Limiting
```
Error: Too many requests
```
**Solution**: Azure has rate limits. For batch processing, add delays:
```python
import time
for pdf in pdfs:
    converter.convert_pdf_to_markdown(pdf, output)
    time.sleep(1)  # Wait 1 second between requests
```

## Pricing

Azure Document Intelligence pricing (as of 2024):
- **Free Tier**: 500 pages/month
- **Standard**: ~$1.50 per 1000 pages

For the Wakilg-Backend project processing legal documents, the free tier should be sufficient for development.

## Advanced Configuration

### Custom Locale

By default, the converter uses Nepali locale (`ne`). You can customize:

```python
# In pdf_to_markdown.py, modify line ~60:
poller = self.client.begin_analyze_document(
    model_id="prebuilt-layout",
    analyze_request=pdf_content,
    content_type="application/pdf",
    locale="ne"  # Change to "hi" for Hindi, "en" for English, etc.
)
```

### Custom Model

For specialized document types, you can train a custom model:

```python
# Use custom model instead of prebuilt
poller = self.client.begin_analyze_document(
    model_id="your-custom-model-id",  # Instead of "prebuilt-layout"
    analyze_request=pdf_content,
    content_type="application/pdf"
)
```

## Example Workflow

### Full Legal Document Processing Pipeline

```bash
# 1. Place PDF files in source directory
ls source/
# nepal_constitution.pdf
# civil_code_2074.pdf
# criminal_code_2074.pdf

# 2. Batch convert to Markdown
python app/utils/pdf_to_markdown.py source/ -o app/processed/chunks/ --batch

# 3. Verify output
ls app/processed/chunks/
# nepal_constitution.md
# civil_code_2074.md
# criminal_code_2074.md

# 4. Now these can be chunked and ingested into Pinecone for RAG
```

## Support

For issues or questions:
1. Check the [Azure Document Intelligence documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
2. Review the script's inline comments
3. Test with a small PDF first before batch processing

## License

This script is part of the Wakilg-Backend project.

---

**Note**: Always test the converter with sample PDFs first to ensure character accuracy meets your requirements. Nepali Unicode should be preserved perfectly, but always verify the output quality.
