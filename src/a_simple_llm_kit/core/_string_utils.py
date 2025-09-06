def simple_to_camel(snake_str: str) -> str:
    """Basic snake_case to camelCase without acronym detection"""
    components = snake_str.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])
