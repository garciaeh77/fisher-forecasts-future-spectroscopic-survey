#!/bin/zsh
# Double-click this file to export followup_talk.pptx to Keynote (.key) and PDF.
# The first time, macOS may ask you to allow Terminal to control Keynote:
# System Settings → Privacy & Security → Automation → Terminal → Keynote.

set -e
DOCS="$(cd "$(dirname "$0")" && pwd)"
PPTX="$DOCS/followup_talk.pptx"
KEY="$DOCS/followup_talk.key"
PDF="$DOCS/followup_talk_keynote.pdf"

osascript <<APPLESCRIPT
set pptxPath to POSIX file "$PPTX"
set keyPath to POSIX file "$KEY"
set pdfPath to POSIX file "$PDF"

tell application "Keynote"
	activate
	set theDoc to open pptxPath
	delay 4
	save theDoc in keyPath
	delay 1
	export theDoc to pdfPath as PDF
	delay 1
	close theDoc saving no
end tell
APPLESCRIPT

echo "Wrote:"
ls -lh "$KEY" "$PDF"
read -k 1 "?Press any key to close…"
