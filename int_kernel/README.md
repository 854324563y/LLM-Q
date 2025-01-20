# compile
source environment.sh  
原地编译  
python setup.py build_ext --inplace  
或直接  
python setup.py install


## tests
```python
python -m pytest tests/test_w8a8_gemm.py -v
python -m pytest tests/test_s8t_s8n_f16t_gemm.py -v

python -m pytest tests/test_w8a8o16_linear.py -s
pytest -m pytest tests/test_w4a4o16_linear.py -v
```