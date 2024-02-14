#!/bin/bash

sudo apt update
sudo apt install transmission-cli

# Academic Torrents URL for Synth90k
TORRENT_URL="https://academictorrents.com/download/3d0b4f09080703d2a9c6be50715b46389fdb3af1.torrent"

# Download the torrent file using wget
wget -O academic.torrent "$TORRENT_URL"

# Start downloading the torrent file
transmission-cli -w "$(pwd)" academic.torrent

rm academic.torrent

tar -xvf mjsynth.tar.gz

rm mjsynth.tar.gz
