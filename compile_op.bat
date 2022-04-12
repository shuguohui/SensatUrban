cd utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd utils/cpp_wrappers/cpp_subsampling
python setup.py build_ext --inplace
cd ../../../