all:
	echo "Building permutohedral_encoding"
	# python3 -m pip install -v --user --editable ./
	python3 -m pip install -v --user ./

clean:
	python3 -m pip uninstall permutohedral_encoding
	rm -rf build dist *.egg-info build permutohedral_encoding*.so 
