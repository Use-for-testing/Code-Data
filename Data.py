import os
import hashlib
import logging
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fetch_stack_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face authentication
HF_TOKEN = "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"
try:
    login(token=HF_TOKEN)
    logger.info("Successfully authenticated with Hugging Face")
except Exception as e:
    logger.error(f"Failed to authenticate with Hugging Face: {e}")
    exit(1)

# Define desired languages
DESIRED_LANGUAGES = [
    "Swift", "Python", "Lua", "C", "C++", "Objective-C", "C#",
    "Ruby", "JavaScript", "TypeScript", "Luau"
]

# Language mapping for normalization
LANGUAGE_MAPPING = {
    "Cpp": "C++", 
    "JavaScript": "JavaScript", 
    "TypeScript": "TypeScript",
    "ObjectiveC": "Objective-C", 
    "CSharp": "C#"
}

# File extensions for each language
LANGUAGE_EXTENSIONS = {
    "Swift": [".swift"],
    "Python": [".py", ".pyx", ".pyw"],
    "Lua": [".lua"],
    "C": [".c", ".h"],
    "C++": [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"],
    "Objective-C": [".m", ".mm"],
    "C#": [".cs"],
    "Ruby": [".rb"],
    "JavaScript": [".js", ".jsx", ".mjs"],
    "TypeScript": [".ts", ".tsx"],
    "Luau": [".luau"]
}

# Output directory for code files
OUTPUT_BASE_DIR = "code_by_language"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Create subdirectories for each language
for lang in DESIRED_LANGUAGES:
    lang_dir = os.path.join(OUTPUT_BASE_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    logger.info(f"Created directory: {lang_dir}")

# Limit number of files per language
MAX_FILES_PER_LANGUAGE = 10000

# Track number of files saved per language
files_saved = {lang: 0 for lang in DESIRED_LANGUAGES}

def get_safe_filename(path, content, lang):
    """Generate a unique, safe filename based on the file path and content."""
    try:
        unique_str = f"{path}_{content[:100]}"  # Use first 100 chars of content
        hash_object = hashlib.md5(unique_str.encode())
        base_name = hash_object.hexdigest()
        # Use appropriate extension based on language
        ext_map = {
            "Swift": ".swift", "Python": ".py", "Lua": ".lua", "C": ".c",
            "C++": ".cpp", "Objective-C": ".m", "C#": ".cs", "Ruby": ".rb",
            "JavaScript": ".js", "TypeScript": ".ts", "Luau": ".luau"
        }
        ext = ext_map.get(lang, os.path.splitext(path)[1] if "." in path else ".txt")
        return f"{base_name}{ext}"
    except Exception as e:
        logger.error(f"Error generating safe filename for {path}: {e}")
        return f"{hashlib.md5(content.encode()).hexdigest()}.txt"

def create_sample_files():
    """Create sample files for each language when dataset access fails."""
    logger.info("Creating sample files for each language...")
    
    # Sample code snippets for each language
    samples = {
        "Python": [
            "def hello_world():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello_world()",
            "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\n    def greet(self):\n        return f'Hello, my name is {self.name} and I am {self.age} years old.'"
        ],
        "JavaScript": [
            "function helloWorld() {\n    console.log('Hello, World!');\n}\n\nhelloWorld();",
            "class Person {\n    constructor(name, age) {\n        this.name = name;\n        this.age = age;\n    }\n\n    greet() {\n        return `Hello, my name is ${this.name} and I am ${this.age} years old.`;\n    }\n}"
        ],
        "TypeScript": [
            "function helloWorld(): void {\n    console.log('Hello, World!');\n}\n\nhelloWorld();",
            "class Person {\n    name: string;\n    age: number;\n\n    constructor(name: string, age: number) {\n        this.name = name;\n        this.age = age;\n    }\n\n    greet(): string {\n        return `Hello, my name is ${this.name} and I am ${this.age} years old.`;\n    }\n}"
        ],
        "C++": [
            "#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}",
            "#include <string>\n#include <iostream>\n\nclass Person {\nprivate:\n    std::string name;\n    int age;\n\npublic:\n    Person(std::string name, int age) : name(name), age(age) {}\n\n    std::string greet() {\n        return \"Hello, my name is \" + name + \" and I am \" + std::to_string(age) + \" years old.\";\n    }\n};"
        ],
        "C": [
            "#include <stdio.h>\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}",
            "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\ntypedef struct {\n    char* name;\n    int age;\n} Person;\n\nPerson* create_person(const char* name, int age) {\n    Person* p = malloc(sizeof(Person));\n    p->name = strdup(name);\n    p->age = age;\n    return p;\n}"
        ],
        "Swift": [
            "func helloWorld() {\n    print(\"Hello, World!\")\n}\n\nhelloWorld()",
            "class Person {\n    let name: String\n    let age: Int\n\n    init(name: String, age: Int) {\n        self.name = name\n        self.age = age\n    }\n\n    func greet() -> String {\n        return \"Hello, my name is \\(name) and I am \\(age) years old.\"\n    }\n}"
        ],
        "Ruby": [
            "def hello_world\n  puts 'Hello, World!'\nend\n\nhello_world",
            "class Person\n  attr_reader :name, :age\n\n  def initialize(name, age)\n    @name = name\n    @age = age\n  end\n\n  def greet\n    \"Hello, my name is #{@name} and I am #{@age} years old.\"\n  end\nend"
        ],
        "C#": [
            "using System;\n\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello, World!\");\n    }\n}",
            "using System;\n\npublic class Person {\n    public string Name { get; }\n    public int Age { get; }\n\n    public Person(string name, int age) {\n        Name = name;\n        Age = age;\n    }\n\n    public string Greet() {\n        return $\"Hello, my name is {Name} and I am {Age} years old.\";\n    }\n}"
        ],
        "Lua": [
            "function hello_world()\n    print(\"Hello, World!\")\nend\n\nhello_world()",
            "Person = {}\nPerson.__index = Person\n\nfunction Person.new(name, age)\n    local self = setmetatable({}, Person)\n    self.name = name\n    self.age = age\n    return self\nend\n\nfunction Person:greet()\n    return string.format(\"Hello, my name is %s and I am %d years old.\", self.name, self.age)\nend"
        ],
        "Objective-C": [
            "#import <Foundation/Foundation.h>\n\nint main(int argc, const char * argv[]) {\n    @autoreleasepool {\n        NSLog(@\"Hello, World!\");\n    }\n    return 0;\n}",
            "#import <Foundation/Foundation.h>\n\n@interface Person : NSObject\n\n@property (nonatomic, strong) NSString *name;\n@property (nonatomic, assign) NSInteger age;\n\n- (instancetype)initWithName:(NSString *)name age:(NSInteger)age;\n- (NSString *)greet;\n\n@end\n\n@implementation Person\n\n- (instancetype)initWithName:(NSString *)name age:(NSInteger)age {\n    self = [super init];\n    if (self) {\n        _name = name;\n        _age = age;\n    }\n    return self;\n}\n\n- (NSString *)greet {\n    return [NSString stringWithFormat:@\"Hello, my name is %@ and I am %ld years old.\", _name, (long)_age];\n}\n\n@end"
        ],
        "Luau": [
            "local function helloWorld()\n    print(\"Hello, World!\")\nend\n\nhelloWorld()",
            "local Person = {}\nPerson.__index = Person\n\nfunction Person.new(name: string, age: number)\n    local self = setmetatable({}, Person)\n    self.name = name\n    self.age = age\n    return self\nend\n\nfunction Person:greet(): string\n    return string.format(\"Hello, my name is %s and I am %d years old.\", self.name, self.age)\nend\n\nreturn Person"
        ]
    }
    
    # Create sample files for each language
    for language, code_samples in samples.items():
        if language in DESIRED_LANGUAGES and files_saved[language] < MAX_FILES_PER_LANGUAGE:
            # Get file extension for the language
            ext = LANGUAGE_EXTENSIONS.get(language, [".txt"])[0]
            
            # Create sample files
            for i, code in enumerate(code_samples):
                safe_filename = f"sample_{i+1}_{hashlib.md5(code.encode()).hexdigest()[:8]}{ext}"
                output_path = os.path.join(OUTPUT_BASE_DIR, language, safe_filename)
                
                # Skip if file already exists
                if os.path.exists(output_path):
                    continue
                
                # Save file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                files_saved[language] += 1
                logger.info(f"Created sample file: {output_path} (Total: {files_saved[language]})")

# Verify dataset availability
try:
    logger.info("Checking dataset availability...")
    # Attempt to download a small file from the dataset to verify access
    hf_hub_download(
        repo_id="bigcode/the-stack-v2",
        filename="README.md",
        repo_type="dataset",
        token=HF_TOKEN
    )
    logger.info("Dataset access verified: Successfully accessed bigcode/the-stack-v2")
except Exception as e:
    logger.error(f"Cannot access dataset: {e}")
    logger.warning("Will create sample files instead.")
    create_sample_files()
    exit(1)

# Check network connectivity
try:
    response = requests.get("https://huggingface.co", timeout=5)
    if response.status_code != 200:
        logger.error("Network issue: Cannot reach Hugging Face servers")
        exit(1)
    logger.info("Network check passed: Hugging Face servers reachable")
except requests.RequestException as e:
    logger.error(f"Network check failed: {e}")
    exit(1)

# Load dataset in streaming mode
try:
    ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True, token=HF_TOKEN)
    logger.info("Successfully loaded dataset in streaming mode")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    logger.warning("Creating sample files instead...")
    create_sample_files()
    exit(1)

# Iterate over dataset and filter by language
try:
    for file in tqdm(ds, desc="Processing files", unit="file"):
        # Skip if all languages have reached their file limit
        if all(files_saved[lang] >= MAX_FILES_PER_LANGUAGE for lang in DESIRED_LANGUAGES):
            logger.info("All languages reached file limit; stopping")
            break

        language = file.get("lang")
        content = file.get("content")
        path = file.get("path", "unknown")

        # Handle Luau fallback for Roblox-related Lua files
        if language == "Lua" and any(keyword in path.lower() for keyword in ["roblox", ".luau"]):
            language = "Luau"

        # Normalize language names to match DESIRED_LANGUAGES
        language = LANGUAGE_MAPPING.get(language, language)

        if language in DESIRED_LANGUAGES and files_saved[language] < MAX_FILES_PER_LANGUAGE:
            if not content or not path or not content.strip():
                logger.warning(f"Skipping file: missing or empty content/path")
                continue

            # Generate a safe filename
            safe_filename = get_safe_filename(path, content, language)
            output_path = os.path.join(OUTPUT_BASE_DIR, language, safe_filename)

            # Check for duplicate content
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if os.path.exists(output_path):
                logger.warning(f"Skipping duplicate file: {safe_filename}")
                continue

            # Save file
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                files_saved[language] += 1
                logger.info(f"Saved {safe_filename} to {language}/ (Total: {files_saved[language]})")
            except (OSError, UnicodeEncodeError) as e:
                logger.warning(f"Failed to save {safe_filename} to {language}/: {e}")

        # Log progress periodically
        if sum(files_saved.values()) % 100 == 0:
            logger.info(f"Files saved per language: {files_saved}")

except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    # If we failed to process the dataset, create sample files
    if all(count == 0 for count in files_saved.values()):
        logger.warning("No files processed from dataset. Creating sample files...")
        create_sample_files()
    exit(1)

# Final summary
logger.info("\nFinal count of files saved per language:")
for lang, count in files_saved.items():
    logger.info(f"{lang}: {count} files")
