package : 
	rm -rf dist 
	python3 setup.py sdist
	twine upload dist/*

git : 
	git add .
	git commit -m "commit"
	git push origin main