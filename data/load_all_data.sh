DATASETS="Art Books Dolls Laundry Moebius Reindeer"
for NAME in ${DATASETS}; do
    ./load_data.sh ${NAME};
done
