# Architecture: Contact Extraction System

This document details the design of the robust contact information extraction pipeline, a key feature of the LLM Server.

## Overview

The primary goal of the contact extraction system is to reliably parse structured contact information from images (like business cards) using large language models. A key challenge with LLMs is that their output format is not always guaranteed; they can sometimes fail to produce the expected structured JSON.

To address this, the system is built around a **dual-path processing logic** that ensures maximum resilience. It first attempts to get a perfectly structured Pydantic object directly from the model. If that fails for any reason, it gracefully falls back to a secondary path that parses the model's raw text output.

## Key Components

### 1. The `ExtractContact` Pydantic Model

This is the canonical data structure for all extracted contact information. It provides a well-defined, type-safe schema for the data. Using a Pydantic model ensures that the final output of the API is always consistent and predictable.

It is defined in `llm_server/core/modules.py`.

**Simplified Structure:**

```json
{
  "name": {
    "prefix": "Dr.",
    "given_name": "Jane",
    "family_name": "Doe",
    "suffix": "PhD"
  },
  "work": {
    "job_title": "Lead Scientist",
    "organization_name": "Research Inc."
  },
  "contact": {
    "phone_numbers": [
      { "label": "work", "value": "+1-123-456-7890" },
      { "label": "mobile", "value": "+1-987-654-3210" }
    ],
    "email_addresses": [
      { "label": "work", "value": "j.doe@research.inc" }
    ],
    "postal_addresses": [
      { "label": "work", "value": { "street": "123 Lab Dr", "city": "Scienceville", "state": "CA", "postal_code": "90210" } }
    ],
    "social_profiles": [
      { "service": "linkedin", "username": "janedoe-sci", "url_string": "https://linkedin.com/in/janedoe-sci" }
    ]
  },
  "notes": "Met at the 2025 AI conference."
}
```

### 2. Dual-Path Processing Logic

The core of the system's resilience lies in the ContactExtractorProcessor (`llm_server/core/output_processors.py`). It orchestrates the two processing paths.

**Path 1: The Happy Path (Structured Output)**

- An image is sent to the LLM using the ContactExtractor DSPy signature, which is a dspy.TypedPredictor.
- This instructs the model to return a JSON object that directly matches the ExtractContact Pydantic model.
- If the model successfully returns a valid, structured response, the processor validates it and passes it on. This is the most efficient and reliable path.

**Path 2: The Fallback Path (Raw Text Parsing)**

- **Trigger**: This path is activated if the model fails to return a valid structured object. This can happen if the model's output is malformed JSON, is missing required fields, or is just plain text.
- **Raw Text Capture**: The ProgramManager is designed to capture the raw text completion from the model's history, even if the structured parsing fails.
- **Manual Parsing**: The ContactExtractorProcessor takes this raw text and uses a series of robust regular expressions to find and parse the different sections of the contact information (name, work, contact, etc.).
- **Object Construction**: It then uses the parsed data to construct an ExtractContact object.

This ensures that even in cases of model failure, we can still salvage and structure the data, preventing a total failure of the request.

### Visual Flow

```
[Image Input]
      |
      v
[ContactExtractor DSPy Signature] -> (Attempts to get structured Pydantic object)
      |
      +--------------------------------+
      |                                |
      v                                v
[Is output a valid            [Is output invalid or raw text?]
 ExtractContact object?]                 |
      |                                |
  (YES) -> [Path 1: Success]           |
      |                                |
      v                                v
[Return structured object]     (YES) -> [Path 2: Fallback Activated]
                                         |
                                         v
                                     [Parse raw text from model history]
                                         |
                                         v
                                     [Construct ExtractContact object manually]
                                         |
                                         v
                                     [Return structured object]
```

## Implications for Developers

**Resilience by Default**: The `/v1/extract-contact` endpoint is highly resilient. A model's failure to generate perfect JSON will not typically result in a failed API call.

**Debugging**: If you encounter an extraction issue, check the application logs. A message like "Falling back to manual parsing for raw text output" indicates that Path 2 was triggered. This can help you identify if a prompt needs to be improved to generate better structured output.

**Improving the System**: The primary way to improve the system is by enhancing the ContactExtractor signature and prompt to make the "Happy Path" succeed more often. The fallback path is a safety net, not the primary mechanism.