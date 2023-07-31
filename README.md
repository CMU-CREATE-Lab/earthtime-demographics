To check out and run:

    Pick a destination directory, e.g. "iter002"

    git clone --recursive https://github.com/CMU-CREATE-Lab/earthtime-demographics.git iter002
    cd iter002
    # Be sure to have python3.11 or newer installed
    /usr/bin/python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

    # Provide symlinks to columncache, expression_cache and various prototile directories.  (Below assumes placement in ~rsargent/uwsgi on hal15)
    ln -s ../dotmaptiles-data/server/{columncache,prototiles,prototiles002,prototiles003,prototiles.2020,expression_cache} .

    # Create portnum.txt with the port you want to serve from, e.g. 5052 for iter002 on hal15
    
    # Run run.sh
    ./run.sh

    # Try tests from http://localhost:<portnum>

To configure apache to proxy from this iteration:

    Edit /etc/apache2/sites-enabled/dotmaptiles.conf and change port number in RewriteRule

To automatically run, do this from /etc/cron.d/dotmaptiles:

    */1 * * * * rsargent run-one /home/rsargent/uwsgi/iter002/run.sh

