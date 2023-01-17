#!/bin/bash -ex

git lfs uninstall
git -c http.sslVerify=false push -f "$1" "HEAD:refs/heads/${2}"
