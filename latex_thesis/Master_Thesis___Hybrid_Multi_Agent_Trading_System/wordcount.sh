#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
texcount -total -sum 0*.tex Lists.tex Nomenclature.tex 2>/dev/null | grep -E "^Words"
