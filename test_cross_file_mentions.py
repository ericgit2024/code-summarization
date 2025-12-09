"""
Test if your model mentions file sources for cross-file dependencies.
"""

from src.model.inference import InferencePipeline

# Example with cross-file dependency
code_with_cross_file = """
def create_post(user_id, title, content):
    # This calls a function from user_service.py
    user = UserService.get_user(user_id)
    
    if not user or not user.is_active:
        logger.error(f"Invalid user: {user_id}")
        return None
    
    # This calls a function from id_generator.py
    post_id = generate_unique_id()
    
    post = Post(id=post_id, title=title, content=content, author=user)
    posts_collection.append(post)
    
    return post
"""

print("="*60)
print("CODE WITH CROSS-FILE DEPENDENCIES:")
print("="*60)
print(code_with_cross_file)

# Initialize pipeline
pipeline = InferencePipeline(model_dir='gemma_lora_finetuned')

# Generate summary (Normal mode)
print("\n" + "="*60)
print("GENERATING SUMMARY (Normal Mode)...")
print("="*60)
summary_normal = pipeline.summarize(code=code_with_cross_file)

print("\n" + "="*60)
print("GENERATED SUMMARY:")
print("="*60)
print(summary_normal)

# Check what's mentioned
print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

checks = {
    "Mentions UserService.get_user()": "UserService.get_user" in summary_normal,
    "Mentions generate_unique_id()": "generate_unique_id" in summary_normal,
    "Mentions 'user_service.py'": "user_service.py" in summary_normal,
    "Mentions 'id_generator.py'": "id_generator.py" in summary_normal,
    "Mentions 'from user_service'": "from user_service" in summary_normal.lower(),
    "Mentions any file source": any(x in summary_normal.lower() for x in ['.py', 'file', 'module', 'from'])
}

for check, result in checks.items():
    status = "✅" if result else "❌"
    print(f"{status} {check}: {result}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

if checks["Mentions any file source"]:
    print("✅ Your model DOES mention file sources!")
else:
    print("❌ Your model does NOT mention file sources.")
    print("   It only mentions function names, not which files they come from.")
