#!/usr/bin/bash

ENV_FILE=$1
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Error: .env file not found."
    exit 1
fi
export $(grep -vE "^(#.*|\s*)$" "${ENV_FILE}")


