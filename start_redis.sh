if ! command -v redis-server &> /dev/null
then
    echo "'redis-server' command not found. Attempting to install..."
    apt-get install -y redis-server
else
    echo "redis-server is already installed."
fi

if ! python -c "import redis" &> /dev/null
then
    echo "Python 'redis' library not found. Attempting to install with pip..."
    pip install redis
else
    echo "Python 'redis' library is already installed."
fi

redis-server --bind $MASTER_ADDR --port 9969 --save "" &
