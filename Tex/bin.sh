for file in ./*
do
    find . -name '*.aux' -delete
    find . -name '*.bcf' -delete
    find . -name '*.out' -delete
    find . -name '*.log' -delete
    find . -name '*.xml' -delete
    find . -name '*.gz' -delete
done