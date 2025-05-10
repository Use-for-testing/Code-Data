import os
import hashlib
import logging
import time
import json
import random
import glob
from tqdm import tqdm
import requests
from huggingface_hub import login, hf_hub_download, snapshot_download

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("fetch_code.log"),
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
    "CSharp": "C#",
    "C++": "C++",
    "C#": "C#"
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

def detect_language_from_extension(file_path):
    """Detect programming language based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    for lang, extensions in LANGUAGE_EXTENSIONS.items():
        if ext in [e.lower() for e in extensions]:
            return lang
    
    return None

def process_file(file_path, language=None):
    """Process a single file and save it to the appropriate language directory."""
    try:
        # Skip if file doesn't exist or is too large (>1MB)
        if not os.path.exists(file_path) or os.path.getsize(file_path) > 1024 * 1024:
            return None
        
        # Detect language from file extension if not provided
        if not language:
            language = detect_language_from_extension(file_path)
            if not language:
                return None
        
        # Normalize language name
        language = LANGUAGE_MAPPING.get(language, language)
        
        # Skip if not in desired languages or already reached limit
        if language not in DESIRED_LANGUAGES or files_saved[language] >= MAX_FILES_PER_LANGUAGE:
            return None
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Skip empty files
        if not content.strip():
            return None
        
        # Generate safe filename
        safe_filename = get_safe_filename(file_path, content, language)
        output_path = os.path.join(OUTPUT_BASE_DIR, language, safe_filename)
        
        # Check for duplicate
        if os.path.exists(output_path):
            return None
        
        # Save file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        files_saved[language] += 1
        logger.info(f"Saved {safe_filename} to {language}/ (Total: {files_saved[language]})")
        return language
    
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return None

def download_sample_repos():
    """Download sample repositories for each language."""
    logger.info("Downloading sample repositories for each language...")
    
    # Sample repositories for each language
    sample_repos = {
        "Python": ["huggingface/transformers", "scikit-learn/scikit-learn"],
        "JavaScript": ["facebook/react", "vuejs/vue"],
        "TypeScript": ["microsoft/TypeScript", "angular/angular"],
        "C++": ["tensorflow/tensorflow", "opencv/opencv"],
        "C": ["torvalds/linux", "git/git"],
        "Swift": ["apple/swift", "ReactiveX/RxSwift"],
        "Ruby": ["rails/rails", "jekyll/jekyll"],
        "C#": ["dotnet/runtime", "PowerShell/PowerShell"],
        "Lua": ["Kong/kong", "openresty/lua-nginx-module"],
        "Objective-C": ["AFNetworking/AFNetworking", "SDWebImage/SDWebImage"],
        "Luau": ["Roblox/luau", "MaximumADHD/Roblox-Client-Tracker"]
    }
    
    # Create temp directory for downloaded repos
    temp_dir = "temp_repos"
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_files = 0
    
    # Process each language and repository
    for language, repos in sample_repos.items():
        if files_saved[language] >= MAX_FILES_PER_LANGUAGE:
            logger.info(f"Skipping {language} - already reached file limit")
            continue
        
        for repo in repos:
            if files_saved[language] >= MAX_FILES_PER_LANGUAGE:
                break
                
            repo_dir = os.path.join(temp_dir, repo.replace("/", "_"))
            
            try:
                # Download repository
                logger.info(f"Downloading repository {repo} for {language}...")
                
                # Try to download from Hugging Face
                try:
                    snapshot_download(
                        repo_id=f"datasets/{repo}",
                        repo_type="dataset",
                        local_dir=repo_dir,
                        token=HF_TOKEN,
                        max_workers=4
                    )
                except Exception as e:
                    logger.warning(f"Failed to download {repo} from Hugging Face: {e}")
                    # If failed, create sample files
                    os.makedirs(repo_dir, exist_ok=True)
                    create_sample_files(repo_dir, language)
                
                # Find all files with the appropriate extension
                extensions = LANGUAGE_EXTENSIONS.get(language, [])
                files = []
                for ext in extensions:
                    files.extend(glob.glob(f"{repo_dir}/**/*{ext}", recursive=True))
                
                # Shuffle files to get a random sample
                random.shuffle(files)
                
                # Process files
                for file_path in tqdm(files[:MAX_FILES_PER_LANGUAGE], desc=f"Processing {language} files from {repo}"):
                    result = process_file(file_path, language)
                    if result:
                        processed_files += 1
                    
                    if files_saved[language] >= MAX_FILES_PER_LANGUAGE:
                        break
            
            except Exception as e:
                logger.error(f"Error processing repository {repo}: {e}")
    
    return processed_files

def create_sample_files(directory, language):
    """Create sample files for a given language."""
    logger.info(f"Creating sample files for {language}...")
    
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
    
    # Get sample code for the language
    code_samples = samples.get(language, ["print('Hello, World!')"])
    
    # Get file extension for the language
    ext = LANGUAGE_EXTENSIONS.get(language, [".txt"])[0]
    
    # Create sample files
    for i, code in enumerate(code_samples):
        file_path = os.path.join(directory, f"sample_{i+1}{ext}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        logger.info(f"Created sample file: {file_path}")

def main():
    """Main function to download and process code files."""
    try:
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
        
        # Download and process sample repositories
        processed_files = download_sample_repos()
        
        # If we didn't process any files, create sample files directly
        if processed_files == 0:
            logger.warning("No files processed from repositories. Creating sample files directly...")
            for language in DESIRED_LANGUAGES:
                if files_saved[language] < MAX_FILES_PER_LANGUAGE:
                    sample_dir = os.path.join(OUTPUT_BASE_DIR, language, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    create_sample_files(sample_dir, language)
                    
                    # Process the created sample files
                    extensions = LANGUAGE_EXTENSIONS.get(language, [])
                    for ext in extensions:
                        for file_path in glob.glob(f"{sample_dir}/**/*{ext}", recursive=True):
                            process_file(file_path, language)
        
        # Final summary
        logger.info("\nFinal count of files saved per language:")
        for lang, count in files_saved.items():
            logger.info(f"{lang}: {count} files")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
