# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import slack_sdk
import sys


# Get token from arguments and initialise a slack client
slack_bot_token = sys.argv[1]
client=slack_sdk.WebClient(token=slack_bot_token)

# Lookup the users slack profile from the github email (=slack email)
github_email = sys.argv[2]
graphcore_username = github_email.split("@")[0]
response=client.users_lookupByEmail(email=f"{graphcore_username}@graphcore.ai")

# Output the slack ID
print(response['user']['id'])
