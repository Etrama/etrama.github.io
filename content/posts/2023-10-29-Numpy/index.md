---
title: "Numpy Exercises 1-50"
date: 2023-10-29T00:41:13-04:00
draft: false
author: "Kaushik Moudgalya"
showtoc: true
tags: ["Numpy", "Machine Learning", "Linear Algebra", "GitGud"]
math: true
---
[View this post in a Jupyter Notebook](https://github.com/Etrama/gitgud-with-numpy/blob/master/1-50_Numpy_exercises_etrama.ipynb)

# First 50 numpy exercises

This is a set of exercises collected by [Rougier](https://github.com/rougier/numpy-100).

All credits to Rougier for curating this list. I am simply trying to solve it for practice and hoping it serves as a reference for others. I am surprised I didn't come across it before.

This is intended to serve as a stepping stone to becoming a better Data Scientist / Machine Learning Researcher. I came across this list on Neel Nanda's page titled ["A Barebones Guide to Mechanistic Interpretability Prerequisites"](https://www.neelnanda.io/mechanistic-interpretability/prereqs) in the hopes of being a better researcher and maybe someday working on mechanistic interpretability.

All the warnings and errors shown in the notebook are intentional.

## 1. Import the numpy package under the name `np` (★☆☆)


```python
import numpy as np
```

## 2. Print the numpy version and the configuration (★☆☆)


```python
print(f"{np.__version__=}")
print(np.show_config())
```

    np.__version__='1.26.0'
    Build Dependencies:
      blas:
        detection method: pkgconfig
        found: true
        include directory: C:/Users/kaush/anaconda3/envs/stats/Library/include
        lib directory: C:/Users/kaush/anaconda3/envs/stats/Library/lib
        name: mkl-sdl
        pc file directory: C:\b\abs_9fu2cs2527\croot\numpy_and_numpy_base_1695830496596\_h_env\Library\lib\pkgconfig
        version: '2023.1'
      lapack:
        detection method: pkgconfig
        found: true
        include directory: C:/Users/kaush/anaconda3/envs/stats/Library/include
        lib directory: C:/Users/kaush/anaconda3/envs/stats/Library/lib
        name: mkl-sdl
        pc file directory: C:\b\abs_9fu2cs2527\croot\numpy_and_numpy_base_1695830496596\_h_env\Library\lib\pkgconfig
        version: '2023.1'
    Compilers:
      c:
        commands: cl.exe
        linker: link
        name: msvc
        version: 19.29.30152
      c++:
        commands: cl.exe
        linker: link
        name: msvc
        version: 19.29.30152
      cython:
        commands: cython
        linker: cython
        name: cython
        version: 3.0.0
    Machine Information:
      build:
        cpu: x86_64
        endian: little
        family: x86_64
        system: windows
      host:
        cpu: x86_64
        endian: little
        family: x86_64
        system: windows
    Python Information:
      path: C:\b\abs_9fu2cs2527\croot\numpy_and_numpy_base_1695830496596\_h_env\python.exe
      version: '3.11'
    SIMD Extensions:
      baseline:
      - SSE
      - SSE2
      - SSE3
      found:
      - SSSE3
      - SSE41
      - POPCNT
      - SSE42
      - AVX
      - F16C
      - FMA3
      - AVX2
      not found:
      - AVX512F
      - AVX512CD
      - AVX512_SKX
      - AVX512_CLX
      - AVX512_CNL
      - AVX512_ICL
    
    None
    

## 3. Create a null vector of size 10 (★☆☆)


```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
null_vector = np.zeros(10)
null_vector

```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



I prefer the non-print version of the output.  
The print version loses the commas and the array(*) notation.

## 4. How to find the memory size of any array (★☆☆)


```python
null_vector = np.zeros(10)
print(f"{null_vector.size = }")
print(f"{null_vector.itemsize = }")
print(f"Total size of null_vector = {null_vector.size * null_vector.itemsize} bytes")
```

    null_vector.size = 10
    null_vector.itemsize = 8
    Total size of null_vector = 80 bytes
    

## 5. How to get the documentation of the numpy add function from the command line? (★☆☆)

There are 3 ways to do this, the first two ways with the "?" only apply to notebooks I believe, but I could be wrong.  
This might be the best thing you learn from this notebook if you weren't aware of it before.


```python
np.add?
```

    [1;31mCall signature:[0m  [0mnp[0m[1;33m.[0m[0madd[0m[1;33m([0m[1;33m*[0m[0margs[0m[1;33m,[0m [1;33m**[0m[0mkwargs[0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mType:[0m            ufunc
    [1;31mString form:[0m     <ufunc 'add'>
    [1;31mFile:[0m            c:\users\kaush\anaconda3\envs\stats\lib\site-packages\numpy\__init__.py
    [1;31mDocstring:[0m      
    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    
    The ``+`` operator can be used as a shorthand for ``np.add`` on ndarrays.
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> x1 + x2
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])
    [1;31mClass docstring:[0m
    Functions that operate element by element on whole arrays.
    
    To see the documentation for a specific ufunc, use `info`.  For
    example, ``np.info(np.sin)``.  Because ufuncs are written in C
    (for speed) and linked into Python with NumPy's ufunc facility,
    Python's help() function finds this page whenever help() is called
    on a ufunc.
    
    A detailed explanation of ufuncs can be found in the docs for :ref:`ufuncs`.
    
    **Calling ufuncs:** ``op(*x[, out], where=True, **kwargs)``
    
    Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.
    
    The broadcasting rules are:
    
    * Dimensions of length 1 may be prepended to either array.
    * Arrays may be repeated along dimensions of length 1.
    
    Parameters
    ----------
    *x : array_like
        Input arrays.
    out : ndarray, None, or tuple of ndarray and None, optional
        Alternate array object(s) in which to put the result; if provided, it
        must have a shape that the inputs broadcast to. A tuple of arrays
        (possible only as a keyword argument) must have length equal to the
        number of outputs; use None for uninitialized outputs to be
        allocated by the ufunc.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    r : ndarray or tuple of ndarray
        `r` will have the shape that the arrays in `x` broadcast to; if `out` is
        provided, it will be returned. If not, `r` will be allocated and
        may contain uninitialized values. If the function has more than one
        output, then the result will be a tuple of arrays.

Instead of the command used above you can also use "np.add??" with two questions marks ot get the same results.
And finally, if you need a non-question mark method, "np.info(np.add)" will do the trick.

## 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)


```python
null_vector = np.zeros(10)
null_vector[4] = 1
null_vector
```




    array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])



## 7. Create a vector with values ranging from 10 to 49 (★☆☆)

I'm not sure if the vector should contain random values from 10 to 40 or sequential values from 10 to 49. So we'll do both.

### Sequential Value Version


```python
seq_vec = np.arange(10,50)
seq_vec
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
           44, 45, 46, 47, 48, 49])



### Random Integer Version


```python
rand_vec = np.random.randint(low=10,high=50,size=20)
# remember that the low is inclusive and the high is exclusive
rand_vec
```




    array([37, 47, 30, 44, 40, 47, 18, 10, 20, 41, 26, 39, 21, 39, 29, 20, 46,
           36, 36, 20])



## 8. Reverse a vector (first element becomes last) (★☆☆)


```python
seq_vec = np.arange(0,9)
seq_vec[::-1]
```




    array([8, 7, 6, 5, 4, 3, 2, 1, 0])



### Let's see what happens with a matrix:


```python
seq_vec = np.arange(0,9)
seq_vec = seq_vec.reshape(3, 3)
seq_vec
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
seq_vec[::-1]
```




    array([[6, 7, 8],
           [3, 4, 5],
           [0, 1, 2]])



### Finally, let's see what happens with a tensor for good measure:


```python
tensor = np.random.randint(low=0,high=10,size=20)
tensor
```




    array([2, 4, 6, 0, 1, 3, 0, 7, 2, 5, 7, 3, 9, 4, 4, 9, 2, 0, 7, 6])




```python
tensor = tensor.reshape(2, 5, 2)
tensor
```




    array([[[2, 4],
            [6, 0],
            [1, 3],
            [0, 7],
            [2, 5]],
    
           [[7, 3],
            [9, 4],
            [4, 9],
            [2, 0],
            [7, 6]]])




```python
tensor[::-1]
```




    array([[[7, 3],
            [9, 4],
            [4, 9],
            [2, 0],
            [7, 6]],
    
           [[2, 4],
            [6, 0],
            [1, 3],
            [0, 7],
            [2, 5]]])



So we see that for a vector we reverse the elements, for a matrix we reverse the row order and for a tensor we reverse the matrix order.

## 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)


```python
matrix = np.arange(0,9).reshape(3,3)
matrix
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])



## 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)


```python
vector = np.array([1,2,0,0,4,0])
vector.nonzero()
```




    (array([0, 1, 4], dtype=int64),)




```python
vector[vector.nonzero()]
```




    array([1, 2, 4])



## 11. Create a 3x3 identity matrix (★☆☆)


```python
identity_matrix = np.eye(3)
identity_matrix
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])



## 12. Create a 3x3x3 array with random values (★☆☆)

### Typical:


```python
random_int_tensor = np.random.randint(low=0,high=10,size=(3,3,3))
random_int_tensor
```




    array([[[1, 7, 6],
            [0, 5, 8],
            [1, 1, 4]],
    
           [[9, 4, 6],
            [3, 0, 0],
            [7, 6, 9]],
    
           [[9, 6, 4],
            [4, 9, 2],
            [0, 5, 1]]])



### A variant:


```python
random_tensor = np.random.random((3,3,3))
random_tensor
```




    array([[[0.10881079, 0.82777305, 0.73024425],
            [0.41001143, 0.6864714 , 0.8286755 ],
            [0.45317059, 0.731864  , 0.24504508]],
    
           [[0.27105589, 0.31979774, 0.53269988],
            [0.74576895, 0.50118312, 0.91641389],
            [0.47912516, 0.88949762, 0.97725423]],
    
           [[0.18685545, 0.84190864, 0.59051282],
            [0.50079992, 0.10817144, 0.95391992],
            [0.03638185, 0.52208639, 0.48453625]]])



## 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)


```python
random_array = np.random.random((10,10))
random_array
```




    array([[0.54719157, 0.83132803, 0.43281279, 0.19023279, 0.53131627,
            0.72643636, 0.68634597, 0.61568373, 0.04141836, 0.87535182],
           [0.56288263, 0.89685912, 0.14898017, 0.14752818, 0.40686519,
            0.22272535, 0.97574528, 0.37510763, 0.22641714, 0.69185797],
           [0.42784303, 0.41151925, 0.32606156, 0.0714769 , 0.77423661,
            0.74242478, 0.5406395 , 0.221898  , 0.50075341, 0.2534813 ],
           [0.34410587, 0.24116138, 0.18034516, 0.09613779, 0.42510876,
            0.65994404, 0.80072605, 0.09166011, 0.61156817, 0.92289304],
           [0.24209721, 0.91770228, 0.19088493, 0.60953766, 0.57709861,
            0.85253104, 0.06680409, 0.30503953, 0.2193305 , 0.77070331],
           [0.75137254, 0.02115599, 0.91557485, 0.80031674, 0.29321583,
            0.16392821, 0.27879865, 0.06015219, 0.20942407, 0.13387668],
           [0.48156959, 0.55267476, 0.66147292, 0.49684291, 0.23053481,
            0.42214255, 0.46477693, 0.84837187, 0.17167265, 0.53384572],
           [0.16045123, 0.37403666, 0.82160964, 0.36511628, 0.84468648,
            0.54006019, 0.86844253, 0.90649856, 0.94265222, 0.33791684],
           [0.77838936, 0.66922879, 0.81474053, 0.04284387, 0.30508264,
            0.77516171, 0.735498  , 0.62239045, 0.64329189, 0.26323616],
           [0.79270077, 0.28751881, 0.68060313, 0.33527429, 0.07126371,
            0.89227643, 0.74099821, 0.1413614 , 0.83575371, 0.35286316]])




```python
print(f"{random_array.min() = }")
print(f"{random_array.max() = }")
```

    random_array.min() = 0.021155988089256672
    random_array.max() = 0.9757452816746033
    

## 14. Create a random vector of size 30 and find the mean value (★☆☆)


```python
random_vector = np.random.random((30,))
random_vector
```




    array([0.77884844, 0.074565  , 0.86744892, 0.52180949, 0.92067033,
           0.15437312, 0.5693557 , 0.27752174, 0.13960521, 0.88391854,
           0.07697968, 0.32042306, 0.01562025, 0.11639959, 0.64780853,
           0.57181685, 0.27859389, 0.55697004, 0.13823586, 0.58121217,
           0.22869387, 0.28077958, 0.12044934, 0.27486188, 0.94946611,
           0.07677409, 0.23081254, 0.52773675, 0.06160974, 0.28575782])




```python
random_vector.mean()
```




    0.38430393805805496



## 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)


```python
random_vector = np.random.randint(low=1, high=2, size=(15,)).reshape(5,3)
random_vector[1:-1,1:-1] = 0 # this notation is super useful.
random_vector
```




    array([[1, 1, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 1, 1]])



Let's do a few more for practice


```python
random_vector = np.random.randint(low=1, high=2, size=(20,)).reshape(4,5)
random_vector[1:-1,1:-1] =  "0" # funny, stringified numbers work but characters do not
# random_vector[1:-1,1:-1] = "f"
random_vector
```




    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])



- Clearly, we need to have more than 2 columns because there is no middle column with just 2 columns.
- Same idea with rows, we need to have more than 2 rows because there is no middle row with just 2 rows
- Combining these ideas together, the reshape dims should be (3,3) or greater based on the input size


```python
random_vector = np.random.randint(low=1, high=2, size=(30,)).reshape(5,6)
random_vector[1:-1,1:-1] = 0 #it's nice how this part is the same for the same idea
random_vector
```




    array([[1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1]])



## 16. How to add a border (filled with 0's) around an existing array? (★☆☆)


```python
random_vector = np.random.randint(low=1, high=2, size=(20,)).reshape(4,5)
np.pad(random_vector, pad_width=1, mode='constant', constant_values=0)
```




    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])



## 17. What is the result of the following expression? (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```

We'll do these one by one, with a comment guessing the answer before actually running the code. Here we go:

Should be nan, you can't multiply 0 and "not a number"


```python
0 * np.nan 
```




    nan



Should be false, one not a number need not equal another not a number


```python
np.nan == np.nan 
```




    False



Should be false, the comparison is pointless


```python
np.inf > np.nan
```




    False



nan, nan, nan


```python
np.nan - np.nan
```




    nan



true, nan is in "nan set"


```python
np.nan in set([np.nan]) 
```




    True



False, floating point precision is not exact


```python
0.3 == 3 * 0.1
```




    False




```python
3 * 0.1
```




    0.30000000000000004



## 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

Technically this is right I guess? They don't say anything about the rest of the matrix values ;)


```python
matrix = np.tril(np.arange(1,6), k=-1)
matrix
```




    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [1, 2, 0, 0, 0],
           [1, 2, 3, 0, 0],
           [1, 2, 3, 4, 0]])



The actual answer is:


```python
matrix = np.diag(np.arange(1,5), k=-1)
matrix
```




    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 2, 0, 0, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 4, 0]])



Just some practice:


```python
np.diag(np.arange(1,5), k=0)
```




    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])




```python
np.diag(np.arange(1,5), k=1)
```




    array([[0, 1, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 3, 0],
           [0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0]])



## 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)


```python
matrix = np.zeros((8,8))
matrix[0::2,0::2] = 1
matrix[1::2,1::2] = 1
matrix
```




    array([[1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.]])



## 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

I had to look this up:


```python
matrix = np.random.random((6,7,8))
np.unravel_index(99, (6,7,8))
```




    (1, 5, 3)




```python
matrix[1,5,3]
```




    0.9383448132311133



## 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

An okay solution:


```python
x = np.tile(np.array([1,0]), (1,4))
y = np.tile(np.array([0,1]), (1,4))
matrix = np.vstack((x,y))
matrix = np.tile(matrix, (4,1))
matrix
```




    array([[1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1]])



A better solution:


```python
lego_matrix = np.array([[1,0],[0,1]])
np.tile(lego_matrix, (4,4))
```




    array([[1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1]])



## 22. Normalize a 5x5 random matrix (★☆☆)

Considering columns as features:


```python
matrix = np.random.random((5,5))
means = matrix.mean(axis=0)
stds = matrix.std(axis=0)
stnd_matrix = (matrix - means)/stds
stnd_matrix
```




    array([[-1.86104626,  0.89601576, -0.20306801,  1.42171603, -1.39310597],
           [ 1.02475131, -1.22849526, -0.92514335,  0.90187193,  1.50611269],
           [-0.02190145, -1.21569425, -0.80098574, -0.4950345 , -0.16300854],
           [ 0.67164855,  0.69785806,  1.85913594, -0.56179446, -0.59267962],
           [ 0.18654785,  0.85031569,  0.07006115, -1.266759  ,  0.64268143]])



considering the entire collection of 25 points as a sample  
which will have a sample mean and a sample standard deviation


```python
matrix = np.random.random((5,5))

stnd_matrix = (matrix - matrix.mean())/matrix.std()
stnd_matrix
```




    array([[ 0.97126576, -1.25866243,  1.1132708 ,  0.97516659,  1.32779957],
           [-0.75447592, -0.00851028, -0.54713304, -1.57301718,  0.10986079],
           [ 0.36812637,  1.18438609, -1.2378124 ,  0.70977168,  1.17780931],
           [-0.39664462,  1.37723218,  1.02383698, -0.60006811, -0.60826184],
           [ 0.26182278, -1.52569149, -1.76727619,  0.47042362, -0.79321902]])



## 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

![Numpy docs on ubyte](./23.png)


```python
# https://numpy.org/doc/stable/reference/arrays.dtypes.html
# https://stackoverflow.com/questions/2350072/custom-data-types-in-numpy-arrays
color = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
```




    dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])



## 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

What is a "real matrix product"? assuming dot product


```python
matrix1 = np.random.random((5,3))
matrix2 = np.random.random((3,2))
matrix1.dot(matrix2)
# my bad, I guess there is no cross product for matrices, it only exists for vectors
```




    array([[0.94993761, 0.64650895],
           [1.07557273, 0.7285574 ],
           [0.64810463, 0.5250493 ],
           [0.65178681, 0.48073052],
           [1.03281097, 0.82463669]])



## 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)


```python
array_1d = np.arange(0,10)
array_1d [(array_1d > 3) & (array_1d < 8)] = 0
array_1d
```




    array([0, 1, 2, 3, 0, 0, 0, 0, 8, 9])



## 26. What is the output of the following script? (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

The cell below should add 0,1,2,3,4 and then subtract 1  
i.e. 0+1+2+3+4 - 1 = 10 - 1 = 9


```python
sum(range(5),-1)
```




    10



The cell below performs the sum along the last axis of the input  
i.e. sums all elements in the array. This is the numpy version of the sum function.


```python
from numpy import *
sum(range(5),-1)

```




    10



## 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```


```python
z = np.random.randint(low=0,high=10,size=(10,))
z
```




    array([1, 1, 3, 0, 9, 7, 2, 6, 8, 6])



I don't know the answer, so let's try it out


```python
z**z
```




    array([        1,         1,        27,         1, 387420489,    823543,
                   4,     46656,  16777216,     46656])



I think this is done elementwise.  
But isn't it weird that 0**0 is 1?


```python
2 << z >> 2
```




    array([  1,   1,   4,   0, 256,  64,   2,  32, 128,  32], dtype=int32)



Funny how we don't get the original array after  
left shifting and right shifting by the same amount


```python
z < -z

```




    array([False, False, False, False, False, False, False, False, False,
           False])



Well duh  
If you're worried about the 0  
It's fine because the condition says less than  
not less than or equal to


```python
z
```




    array([1, 1, 3, 0, 9, 7, 2, 6, 8, 6])




```python
1j*z
# one Jay Z ? ;)
```




    array([0.+1.j, 0.+1.j, 0.+3.j, 0.+0.j, 0.+9.j, 0.+7.j, 0.+2.j, 0.+6.j,
           0.+8.j, 0.+6.j])




```python
z  / 1 / 1
```




    array([1., 1., 3., 0., 9., 7., 2., 6., 8., 6.])



Funny, converts the array to floats  


```python
z < z > z
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    c:\Users\kaush\Desktop\python_projects_new\gitgud-with-numpy\100_Numpy_exercises_etrama.ipynb Cell 96 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kaush/Desktop/python_projects_new/gitgud-with-numpy/100_Numpy_exercises_etrama.ipynb#Y516sZmlsZQ%3D%3D?line=0'>1</a> z < z > z
    

    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()



```python
z < z
```




    array([False, False, False, False, False, False, False, False, False,
           False])




```python
z > z
```




    array([False, False, False, False, False, False, False, False, False,
           False])



## 28. What are the result of the following expressions? (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

I don't know, so let's find out


```python
np.array(0) / np.array(0)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\2670511084.py:2: RuntimeWarning: invalid value encountered in divide
      np.array(0) / np.array(0)
    




    nan




```python
np.array(0) // np.array(0)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\2018018105.py:1: RuntimeWarning: divide by zero encountered in floor_divide
      np.array(0) // np.array(0)
    




    0




```python
np.array([np.nan]).astype(int).astype(float)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\699728972.py:1: RuntimeWarning: invalid value encountered in cast
      np.array([np.nan]).astype(int).astype(float)
    




    array([-2.14748365e+09])



Say what now?


```python
np.array([nan])
```




    array([nan])




```python
np.array([nan]).astype(int)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\3805164281.py:1: RuntimeWarning: invalid value encountered in cast
      np.array([nan]).astype(int)
    




    array([-2147483648])




```python
np.array([nan]).astype(int).astype(float)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\2776176757.py:1: RuntimeWarning: invalid value encountered in cast
      np.array([nan]).astype(int).astype(float)
    




    array([-2.14748365e+09])



I guess nan is initialized as a random LARGE number.

The questions are getting tougher from this point onwards, so just assume that everything was done with the help our friend Google.

## 29. How to round away from zero a float array ? (★☆☆)

I had no idea, what this meant and had to check Rougier's solution:


```python
Z = np.random.uniform(-10,+10,10)
Z
```




    array([-3.0068558 ,  2.38833   , -1.76744513, -1.07521525, -7.18468193,
           -0.23104748, -7.87887182,  6.20622653,  6.0051356 , -9.94051071])




```python
print(np.copysign(np.ceil(np.abs(Z)), Z))
```

    [ 4. 10. -3. -4. -1.  6. 10. -4. -3. -1.]
    


```python
# More readable but less efficient
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```

    [ 4. 10. -3. -4. -1.  6. 10. -4. -3. -1.]
    

## 30. How to find common values between two arrays? (★☆☆)

Typically, I would do this:


```python
array1 = np.arange(0,10)
array2 = np.arange(5,15)
```


```python
array1
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
array2
```




    array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
set(array1).intersection(set(array2))
```




    {5, 6, 7, 8, 9}



But, I guess there is a more numpy way to do it, which we'll look up.


```python
np.intersect1d(array1, array2)
```




    array([5, 6, 7, 8, 9])



The arrays are treated as "flat arrays" to find common elements if they are not already 1D


```python
matrix1 = np.array([[1,2],[3,4]])
matrix2 = np.array([[3,4],[5,6]])
np.intersect1d(matrix1, matrix2)
```




    array([3, 4])



## 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

Typically, I would do this


```python
import warnings
warnings.filterwarnings('ignore')
```

No more divide by zero warnings:


```python
np.array(0) / np.array(0)
```




    nan



The more numpy way is, Rougier's solutions themsevles have the best answer:


```python
# Author: Rougier
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)
```


```python
# Equivalently with a context manager
with np.errstate(all="ignore"):
    np.arange(3) / 0
```

## 32. Is the following expressions true? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

What even is emath? Apparently, it finds the complex square root.  
Idk, if sqrt can handle square roots of negative numbers, I would guess not.  
Otherwise, why would we have emath?  
Expression should be false / lead to an error.  


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```




    False




```python
np.emath.sqrt(-1)
```




    1j




```python
np.sqrt(-1)
```




    nan



## 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

This I had to look up.


```python
today = np.datetime64("today")
yesterday = today - np.timedelta64(1)
tomorrow = today + np.timedelta64(1)
```


```python
yesterday, today, tomorrow
```




    (numpy.datetime64('2023-10-24'),
     numpy.datetime64('2023-10-25'),
     numpy.datetime64('2023-10-26'))



## 34. How to get all the dates corresponding to the month of July 2016? (★★☆)


```python
# https://www.geeksforgeeks.org/display-all-the-dates-for-a-particular-month-using-numpy/
np.arange("2016-07","2016-08",dtype='datetime64[D]')
```




    array(['2016-07-01', '2016-07-02', '2016-07-03', '2016-07-04',
           '2016-07-05', '2016-07-06', '2016-07-07', '2016-07-08',
           '2016-07-09', '2016-07-10', '2016-07-11', '2016-07-12',
           '2016-07-13', '2016-07-14', '2016-07-15', '2016-07-16',
           '2016-07-17', '2016-07-18', '2016-07-19', '2016-07-20',
           '2016-07-21', '2016-07-22', '2016-07-23', '2016-07-24',
           '2016-07-25', '2016-07-26', '2016-07-27', '2016-07-28',
           '2016-07-29', '2016-07-30', '2016-07-31'], dtype='datetime64[D]')



## 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

If we write the equation like we have below, it seems quite trivial to do this. Am I missing something?

$$
 (\mathrm{A} + \mathrm{B})(-\mathrm{A}/2)
$$


```python
A = np.random.random((5,5))
B = np.random.random((5,5))
np.dot(A+B, -A/2)
```




    array([[-1.29422991, -0.95662934, -1.19734799, -0.84429619, -1.18800801],
           [-1.64869539, -1.19190915, -1.563335  , -1.15549011, -1.54080264],
           [-1.91574953, -1.42081103, -2.11914374, -1.23216746, -1.4935421 ],
           [-0.98203675, -0.74057567, -1.0951566 , -0.96027426, -0.98819992],
           [-1.29275308, -0.77859687, -1.23129806, -0.85070669, -1.05444337]])



I completely missed the point. The keyword is to do it "in place".


```python
np.dot(A+B, -A/2,out=A)
```




    array([[-1.29422991, -0.95662934, -1.19734799, -0.84429619, -1.18800801],
           [-1.64869539, -1.19190915, -1.563335  , -1.15549011, -1.54080264],
           [-1.91574953, -1.42081103, -2.11914374, -1.23216746, -1.4935421 ],
           [-0.98203675, -0.74057567, -1.0951566 , -0.96027426, -0.98819992],
           [-1.29275308, -0.77859687, -1.23129806, -0.85070669, -1.05444337]])



So we need to use the out keyword and not the assignment operator.


```python
A
```




    array([[-1.29422991, -0.95662934, -1.19734799, -0.84429619, -1.18800801],
           [-1.64869539, -1.19190915, -1.563335  , -1.15549011, -1.54080264],
           [-1.91574953, -1.42081103, -2.11914374, -1.23216746, -1.4935421 ],
           [-0.98203675, -0.74057567, -1.0951566 , -0.96027426, -0.98819992],
           [-1.29275308, -0.77859687, -1.23129806, -0.85070669, -1.05444337]])



Rougier's solution is kinda neat too:


```python
A = np.random.random((5,5))
B = np.random.random((5,5))
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(B,A)
```




    array([[-0.31509915, -0.0700177 , -0.27793745, -0.47833879, -0.59615885],
           [-0.09988082, -0.06581093, -0.13581843, -0.73121121, -0.63472097],
           [-0.40544299, -0.03960959, -0.23190091, -0.25586552, -0.58855885],
           [-0.21526681, -0.27484055, -0.01746818, -0.03661305, -0.12157086],
           [-0.12278767, -0.0547132 , -0.72528912, -0.11565302, -0.28357008]])



## 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)


```python
a = np.random.uniform(low=0,high=1.0,size=(10))*100
```

The array "a" contains random positive integers.

Method 1:


```python
a
```




    array([22.11421753, 51.86133863, 18.51164469, 58.50565343, 68.14403282,
           48.08201432, 28.72401268, 38.50121503, 83.77582821, 80.23137176])




```python
a.astype(int)
```




    array([22, 51, 18, 58, 68, 48, 28, 38, 83, 80])



Method 2:


```python
# https://stackoverflow.com/questions/6681743/splitting-a-number-into-the-integer-and-decimal-parts
a-a%1
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



Method 3:


```python
# this is venturing out of numpy but oh well
# https://stackoverflow.com/questions/6681743/splitting-a-number-into-the-integer-and-decimal-parts
list_a = a.tolist()
for idx, val in enumerate(list_a):
    list_a[idx] = str(val).split('.')[0]
np.array(list_a)
```




    array(['22', '51', '18', '58', '68', '48', '28', '38', '83', '80'],
          dtype='<U2')



Method 4:


```python
a // 1
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



Thanks to Rougier, we hav a couple more:

Method 5:


```python
np.floor(a)
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



Method 6:


```python
np.trunc(a)
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



## 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)


```python
vector = np.arange(0,5)
np.vstack((vector, vector, vector, vector, vector))
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])



Rougier's solution is again more elegant:


```python
Z = np.zeros((5,5))
Z += np.arange(5)
Z
```




    array([[0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.]])



Another elegant solution by Rougier using tile:


```python
np.tile(np.arange(0,5),(5,1))
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])



## 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

A generator function called "jenny":


```python
def jenny():
    for i in range(1,11):
        yield i
```


```python
for i in jenny():
    print(i)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    


```python
a = np.array([])
for i in jenny():
    a  = np.hstack((a,i))
a
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



Yet again, Rougier delivers:


```python
np.fromiter(jenny(), dtype=float, count=-1)
# count -1 means that we use all the values from the iterator
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



It seems like I need to think less mechanically and more "numpy"ly.

## 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)


```python
np.linspace(0,1,12)[1:-1]
```




    array([0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545,
           0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091])



Rougier:


```python
np.linspace(0,1,11,endpoint=False)[1:]
```




    array([0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545,
           0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091])



## 40. Create a random vector of size 10 and sort it (★★☆)


```python
random_vector = np.random.randint(low=0, high=10, size=(10,))
random_vector
```




    array([3, 6, 5, 3, 8, 8, 1, 6, 9, 0])



    Not in place:


```python
sorted(random_vector)
```




    [0, 1, 3, 3, 5, 6, 6, 8, 8, 9]



In place:


```python
random_vector.sort()
random_vector
```




    array([0, 1, 3, 3, 5, 6, 6, 8, 8, 9])



## 41. How to sum a small array faster than np.sum? (★★☆)

No idea. Looking up.


```python
#https://stackoverflow.com/questions/10922231/pythons-sum-vs-numpys-numpy-sum
array = np.arange(0,10)
```

Sum is supposed to be a tiny bit faster for small arrays.

![Speed comparison of different sum functions](./41.png)


```python
%time
np.sum(array)
```

    CPU times: total: 0 ns
    Wall time: 0 ns
    




    45




```python
%time
sum(array)
```

    CPU times: total: 0 ns
    Wall time: 0 ns
    




    45



The code was not able to capture the time difference.


```python
import time
time1 = time.time()
np.sum(array)
time2 = time.time()
print(f"Time taken: {time2-time1}")
```

    Time taken: 0.0010027885437011719
    


```python
import time
time1 = time.time()
sum(array)
time2 = time.time()
print(f"Time taken: {time2-time1}")
```

    Time taken: 0.0
    

To be rigorous, we should probably run the same code like 10 times and see if sum is consistently faster than np.sum() for small arrays. But this is good enough for me.

## 42. Consider two random array A and B, check if they are equal (★★☆)


```python
A = np.random.random((10,10))
B = np.random.random((10,10))
np.all(A==B)
```




    False




```python
A = np.random.randint(low=0, high=10, size=(10,10))
B = A.copy()
np.all(A==B)
```




    True




```python
B[0,0] = 100
np.all(A==B)
```




    False



Rougier's solution is more numpy like:


```python
A = np.random.randint(low=0, high=10, size=(10,10))
B = A.copy()
np.allclose(A,B)
```




    True




```python
np.array_equal(A,B)
```




    True



## 43. Make an array immutable (read-only) (★★☆)


```python
# https://stackoverflow.com/questions/5541324/immutable-numpy-array
a = np.random.random((5,5))
a
```




    array([[0.8227012 , 0.12601625, 0.42540863, 0.97627089, 0.63388622],
           [0.75687311, 0.45598492, 0.10625496, 0.58104675, 0.09436362],
           [0.16486368, 0.13800275, 0.21366973, 0.76358301, 0.52054695],
           [0.08156548, 0.09853984, 0.17364485, 0.00972851, 0.74932949],
           [0.4892877 , 0.273489  , 0.73034423, 0.20120066, 0.16257508]])




```python
a.setflags(write=False)
```


```python
a[0, 0] = 100
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    c:\Users\kaush\Desktop\python_projects_new\gitgud-with-numpy\100_Numpy_exercises_etrama.ipynb Cell 243 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kaush/Desktop/python_projects_new/gitgud-with-numpy/100_Numpy_exercises_etrama.ipynb#Z1016sZmlsZQ%3D%3D?line=0'>1</a> a[0, 0] = 100
    

    ValueError: assignment destination is read-only


## 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)


```python
cartesian_coords = np.random.randint(low=0, high=10, size=(10,2))
cartesian_coords
```




    array([[9, 5],
           [6, 0],
           [9, 7],
           [0, 1],
           [3, 2],
           [7, 6],
           [5, 5],
           [6, 1],
           [5, 4],
           [5, 1]])



Cartesian to polar coordinates:
$$
r = \sqrt{x^2 + y^2} 
$$
$$
\theta = \arctan(y/x)
$$


```python
#https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def polar(x, y) -> tuple:
  """returns rho, theta (degrees)"""
  return np.hypot(x, y), np.degrees(np.arctan2(y, x))
```


```python
polar(3,4)
```




    (5.0, 53.13010235415598)




```python
polar(cartesian_coords[0][0], cartesian_coords[0][1])
```




    (10.295630140987, 29.054604099077146)




```python
polar(*cartesian_coords[0])
```




    (10.295630140987, 29.054604099077146)




```python
polar(cartesian_coords[:,0], cartesian_coords[:,1])
```




    (array([10.29563014,  6.        , 11.40175425,  1.        ,  3.60555128,
             9.21954446,  7.07106781,  6.08276253,  6.40312424,  5.09901951]),
     array([29.0546041 ,  0.        , 37.87498365, 90.        , 33.69006753,
            40.60129465, 45.        ,  9.46232221, 38.65980825, 11.30993247]))



## 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)


```python
random_vector = np.random.random((10,))
random_vector
```




    array([0.74971807, 0.90937433, 0.63791424, 0.07883616, 0.92606344,
           0.57586556, 0.84885824, 0.64565938, 0.85861547, 0.29818429])




```python
np.max(random_vector), np.argmax(random_vector)
```




    (0.9260634367924555, 4)




```python
random_vector[np.argmax(random_vector)] = 0
random_vector
```




    array([0.74971807, 0.90937433, 0.63791424, 0.07883616, 0.        ,
           0.57586556, 0.84885824, 0.64565938, 0.85861547, 0.29818429])



## 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)


```python
#https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)
np.meshgrid(x, y)
```




    [array([[0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ]]),
     array([[0.        , 0.        , 0.        , 0.        , 0.        ,
             0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,
             0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111],
            [0.22222222, 0.22222222, 0.22222222, 0.22222222, 0.22222222,
             0.22222222, 0.22222222, 0.22222222, 0.22222222, 0.22222222],
            [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
             0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333],
            [0.44444444, 0.44444444, 0.44444444, 0.44444444, 0.44444444,
             0.44444444, 0.44444444, 0.44444444, 0.44444444, 0.44444444],
            [0.55555556, 0.55555556, 0.55555556, 0.55555556, 0.55555556,
             0.55555556, 0.55555556, 0.55555556, 0.55555556, 0.55555556],
            [0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667,
             0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
            [0.77777778, 0.77777778, 0.77777778, 0.77777778, 0.77777778,
             0.77777778, 0.77777778, 0.77777778, 0.77777778, 0.77777778],
            [0.88888889, 0.88888889, 0.88888889, 0.88888889, 0.88888889,
             0.88888889, 0.88888889, 0.88888889, 0.88888889, 0.88888889],
            [1.        , 1.        , 1.        , 1.        , 1.        ,
             1.        , 1.        , 1.        , 1.        , 1.        ]])]



Rougier demonstrates a way with custom dtypes:


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
Z
```




    array([[(0.  , 0.  ), (0.25, 0.  ), (0.5 , 0.  ), (0.75, 0.  ),
            (1.  , 0.  )],
           [(0.  , 0.25), (0.25, 0.25), (0.5 , 0.25), (0.75, 0.25),
            (1.  , 0.25)],
           [(0.  , 0.5 ), (0.25, 0.5 ), (0.5 , 0.5 ), (0.75, 0.5 ),
            (1.  , 0.5 )],
           [(0.  , 0.75), (0.25, 0.75), (0.5 , 0.75), (0.75, 0.75),
            (1.  , 0.75)],
           [(0.  , 1.  ), (0.25, 1.  ), (0.5 , 1.  ), (0.75, 1.  ),
            (1.  , 1.  )]], dtype=[('x', '<f8'), ('y', '<f8')])



## 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

From Wikipedia:

![Wikipedia entry about the Cauchy matrix](./47.png)

I had an idea about how to do this but it was completely wrong. So here's an analysis of Rougier's solution:


```python
X = np.arange(8)
```


```python
Y = X + 0.5
```

Some [docs](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html) on np.ufunc.outer:

![Screenshot of docs on np.ufunc.outer](./47_pic2.png)


```python
C = 1.0 / np.subtract.outer(X, Y)
C
```




    array([[-2.        , -0.66666667, -0.4       , -0.28571429, -0.22222222,
            -0.18181818, -0.15384615, -0.13333333],
           [ 2.        , -2.        , -0.66666667, -0.4       , -0.28571429,
            -0.22222222, -0.18181818, -0.15384615],
           [ 0.66666667,  2.        , -2.        , -0.66666667, -0.4       ,
            -0.28571429, -0.22222222, -0.18181818],
           [ 0.4       ,  0.66666667,  2.        , -2.        , -0.66666667,
            -0.4       , -0.28571429, -0.22222222],
           [ 0.28571429,  0.4       ,  0.66666667,  2.        , -2.        ,
            -0.66666667, -0.4       , -0.28571429],
           [ 0.22222222,  0.28571429,  0.4       ,  0.66666667,  2.        ,
            -2.        , -0.66666667, -0.4       ],
           [ 0.18181818,  0.22222222,  0.28571429,  0.4       ,  0.66666667,
             2.        , -2.        , -0.66666667],
           [ 0.15384615,  0.18181818,  0.22222222,  0.28571429,  0.4       ,
             0.66666667,  2.        , -2.        ]])




```python
np.linalg.det(C)
```




    3638.1636371179666



## 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)


```python
#https://numpy.org/doc/stable/reference/arrays.scalars.html
#https://stackoverflow.com/questions/21968643/what-is-a-scalar-in-numpy
np.ScalarType
```




    (int,
     float,
     complex,
     bool,
     bytes,
     str,
     memoryview,
     numpy.bool_,
     numpy.complex64,
     numpy.complex128,
     numpy.clongdouble,
     numpy.float16,
     numpy.float32,
     numpy.float64,
     numpy.longdouble,
     numpy.int8,
     numpy.int16,
     numpy.int32,
     numpy.intc,
     numpy.int64,
     numpy.datetime64,
     numpy.timedelta64,
     numpy.object_,
     numpy.bytes_,
     numpy.str_,
     numpy.uint8,
     numpy.uint16,
     numpy.uint32,
     numpy.uintc,
     numpy.uint64,
     numpy.void)




```python
# https://stackoverflow.com/questions/21968643/what-is-a-scalar-in-numpy
np.iinfo(int).min
```




    -2147483648




```python
np.iinfo(int).max
```




    2147483647



Making this a little more readable:


```python
# https://stackoverflow.com/questions/1823058/how-to-print-a-number-using-commas-as-thousands-separators
f"{np.iinfo(int).min:,}"
```




    '-2,147,483,648'




```python
f"{np.iinfo(int).max:,}"
```




    '2,147,483,647'



Rougier's solution also mentions finfo, which has the same usage as iinfo.
iinfo : integer info
finfo : floating info
The iinfo function won't work on floating points and vice-versa.

Net-net, use the iinfo function along with min and max attributes.

## 49. How to print all the values of an array? (★★☆)

Here's a little spoiler where we reduce the nubmer of elements printed to illustrate the concept:


```python
np.set_printoptions(edgeitems=3, infstr='inf',
linewidth=75, nanstr='nan', precision=4,
suppress=False, threshold=10, formatter=None)
```


```python
x = np.random.random((2,25))
x
```




    array([[0.7326, 0.3885, 0.7054, ..., 0.7952, 0.9176, 0.1371],
           [0.5355, 0.9166, 0.6616, ..., 0.6557, 0.2873, 0.4325]])




```python
# https://www.tutorialspoint.com/print-full-numpy-array-without-truncation
print(np.array2string(x, threshold = np.inf))
```

    [[0.7326 0.3885 0.7054 0.1701 0.0089 0.4658 0.6829 0.3698 0.839  0.0291
      0.0325 0.6899 0.4933 0.0129 0.8781 0.8731 0.2675 0.3508 0.8044 0.8538
      0.2541 0.206  0.7952 0.9176 0.1371]
     [0.5355 0.9166 0.6616 0.8649 0.8627 0.1209 0.9403 0.5591 0.4984 0.6451
      0.0474 0.7181 0.2877 0.6242 0.0809 0.479  0.6387 0.0954 0.9675 0.2337
      0.9417 0.3076 0.6557 0.2873 0.4325]]
    

There's also the option to set it for all arrays using the below command:


```python
# https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
np.set_printoptions(threshold = np.inf)
```


```python
x
```




    array([[0.7326, 0.3885, 0.7054, 0.1701, 0.0089, 0.4658, 0.6829, 0.3698,
            0.839 , 0.0291, 0.0325, 0.6899, 0.4933, 0.0129, 0.8781, 0.8731,
            0.2675, 0.3508, 0.8044, 0.8538, 0.2541, 0.206 , 0.7952, 0.9176,
            0.1371],
           [0.5355, 0.9166, 0.6616, 0.8649, 0.8627, 0.1209, 0.9403, 0.5591,
            0.4984, 0.6451, 0.0474, 0.7181, 0.2877, 0.6242, 0.0809, 0.479 ,
            0.6387, 0.0954, 0.9675, 0.2337, 0.9417, 0.3076, 0.6557, 0.2873,
            0.4325]])



But, we don't want to do that for all arrays, it would be a hassle. To go back to the default settings:


```python
np.set_printoptions(edgeitems=3, infstr='inf',
linewidth=75, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)
```


```python
x
```




    array([[0.73264177, 0.38851353, 0.70537431, 0.17011432, 0.0088696 ,
            0.4658418 , 0.68294654, 0.36981174, 0.83903836, 0.02906772,
            0.03249127, 0.68991214, 0.49333107, 0.012852  , 0.87812564,
            0.87311948, 0.26750855, 0.35083639, 0.80438231, 0.85384351,
            0.2540569 , 0.20598551, 0.79517393, 0.9176368 , 0.13711803],
           [0.53550674, 0.91657382, 0.66164627, 0.86486562, 0.86269119,
            0.12085891, 0.94030286, 0.55912791, 0.49837007, 0.64507481,
            0.04736146, 0.71807128, 0.28765902, 0.62424607, 0.08093324,
            0.47902631, 0.63871016, 0.0953806 , 0.96748215, 0.23368196,
            0.94166781, 0.30755876, 0.65565653, 0.28733728, 0.43249023]])



We can also use the following inside a context manager so that it doesn't affect the rest of the code. Notice the difference, "set_printoptions" vs "printoptions":


```python

with np.printoptions(threshold=np.inf):
    print(np.random.random((10,10)))
```

    [[0.7618326  0.79573013 0.9563873  0.13062328 0.6602825  0.3379972
      0.22284173 0.30150536 0.71444143 0.16917473]
     [0.54641232 0.57468012 0.88355025 0.64471134 0.46182536 0.29883407
      0.94852017 0.03651475 0.43126739 0.4678156 ]
     [0.77952997 0.08597148 0.06126495 0.67021934 0.65303227 0.22346408
      0.25561434 0.43481986 0.44254066 0.97158417]
     [0.5437536  0.39894178 0.90191363 0.74202053 0.15575024 0.25749774
      0.47767359 0.75187067 0.06711609 0.21855438]
     [0.21772281 0.87867921 0.58686406 0.37310854 0.27342194 0.31111413
      0.69686658 0.37541849 0.37762563 0.68973765]
     [0.22268774 0.84661048 0.35911794 0.92515828 0.10762659 0.35037844
      0.70698563 0.74384673 0.70447225 0.61398061]
     [0.14093381 0.64291759 0.604174   0.10825487 0.00536428 0.85580352
      0.46131248 0.83283808 0.45928353 0.07981769]
     [0.67254472 0.16499803 0.77563766 0.98068295 0.3047522  0.84885608
      0.97191411 0.90478863 0.9906626  0.84091089]
     [0.68697503 0.1870671  0.1869576  0.29526836 0.75266941 0.97917746
      0.27951043 0.97247457 0.6009091  0.50003127]
     [0.38038589 0.34540133 0.2697606  0.52336444 0.02582581 0.5614044
      0.33716516 0.6774765  0.35029573 0.99183216]]
    

## 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

I didn't really understand the question to be honest, but once I looked at Rougier's solution, everything made sense.


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```

    94
    
