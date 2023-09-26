from torrentp import TorrentDownloader

# Function to download a torrent using a magnet link
def download_torrent_or_magnet(magnet_link, save_path):
    torrent_file = TorrentDownloader(magnet_link, save_path)
    torrent_file.start_download()
    print("Download complete!")