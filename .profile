# -*- shell-script -*-
# TACC startup script: ~/.profile version 2.1  -- 12/17/2013


# This file is sourced on login shells but only if ~/.bash_profile
# or ~/.bash_login do not exist! This file is not sourced on
# non-login interactive shells.

# We recommend you place in your ~/.bashrc everything you wish to be
# available in both login and non-login interactive shells. Remember
# that any commands that are placed here will be unavailable in VNC
# and other interactive non-login sessions.

# Note that ~/.bashrc is not automatically sourced on login shells.
# By sourcing here, we insure that login and non-login shells have
# consistent environments.

if [ -f $HOME/.bashrc ]; then
  source $HOME/.bashrc
fi
