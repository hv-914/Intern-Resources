#!/bin/bash

# Installing Adobe AIR 32.0.0.116
echo "Installing Adobe AIR 32.0.0.116..."
hdiutil attach AdobeAIR-32.0.0.116.dmg
open /Volumes/"Adobe AIR"/"Adobe AIR Installer.app"
read -p "Press Enter once Adobe AIR 32.0.0.116 has been installed..."
hdiutil detach /Volumes/"Adobe AIR"

sudo xattr -r -d com.apple.quarantine /Library/Frameworks/Adobe\ AIR.framework

# Installing Scratch2
echo "Installing Scratch 2..."
cp -R Scratch2/Scratch2.app /Applications/
cp -R PhiroDemo.app /Applications/

open -a "PhiroDemo.app"
open -a "Scratch2"