for file in ./*
do
    find . -name '*.aux' -delete
    find . -name '*.bcf' -delete
    find . -name '*.out' -delete
    find . -name '*.log' -delete
    find . -name '*.xml' -delete
    find . -name '*.gz' -delete
    find . -name '*.bbl' -delete
    find . -name '*.fls' -delete
    find . -name '*.fdb_latexmk' -delete
    find . -name '*.blg' -delete
    find . -name '*.xdv' -delete
    find . -name '*.toc' -delete
done

git add .
git commit -m "Added New contents"
git push origin main
