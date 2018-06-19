#!/bin/bash

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

source ~/.bashrc
cd ${TRAVIS_BUILD_DIR}
or_die mkdir travis-build

cd travis-build

if [[ "$DO_BUILD" == "yes" ]] ; then
    or_die cmake ../
    or_die make VERBOSE=1
fi

exit 0