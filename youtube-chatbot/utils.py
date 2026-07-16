from urllib.parse import urlparse, parse_qs

def extracting_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get('v',[None])[0]
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")
    return None

def format_timestamp(seconds):
    seconds = int(seconds)

    minutes, remaining_seconds = divmod(seconds, 60)

    return f"{minutes}:{remaining_seconds:02d}"