import urllib.request
import http.client
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL to send the request to")
    args = parser.parse_args()

    url = args.url

    
    response:http.client.HTTPResponse = urllib.request.urlopen(url)
    response_content:str = response.read().decode('utf-8')
    response_code:int = response.status
    response.close()

    if response_code != 200:
        print("Error: ", response_code)
    else:
        print("Annotated File:", response_content)

if __name__ == "__main__":
    main()