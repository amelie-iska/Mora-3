"""Finite original-source range worker with no proxy/auth discovery."""
import urllib.request
from data_sources_fetch_ranges import main
if __name__=="__main__":
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    main()
