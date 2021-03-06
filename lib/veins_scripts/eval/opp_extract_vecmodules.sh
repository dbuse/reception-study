#!/bin/bash
#
# Copyright (C) 2019 Dominik S. Buse <buse@ccs-labs.org>
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

# extract all modules names from an OMNeT++ vector
#
# Works like a unix filter and can read the input vector from a file name or stdin.

set -e

FNAME="$1"
if [[ -z "$1" ]]; then
    FNAME="-"
fi

cat $FNAME | grep '^vector' | cut -d " " -f 3 | sort | uniq
