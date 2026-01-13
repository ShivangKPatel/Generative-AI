import numpy as np

def basic_arrays():
    print("=== Basic Arrays ===")
    # 1D array
    a = np.array([1, 2, 3, 4])
    print("a:", a, "shape:", a.shape, "dtype:", a.dtype)

    # 2D array
    b = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    print("b:\n", b[0][2])
    print("shape:", b.shape, "ndim:", b.ndim, "dtype:", b.dtype)

    # Special arrays
    zeros = np.zeros((2, 3))
    ones = np.ones((3, 3))
    arange = np.arange(0, 10, 2)
    linspace = np.linspace(0, 1, 5)

    print("zeros:\n", zeros)
    print("ones:\n", ones)
    print("arange:", arange)
    print("linspace:", linspace)

    # Reshape
    c = np.arange(12)
    c_reshaped = c.reshape(3, 4)
    print("c:", c)
    print("c reshaped to 3x4:\n", c_reshaped)


def indexing_and_slicing():
    print("\n=== Indexing and Slicing ===")
    x = np.arange(10)
    print("x:", x)
    print("x[0]:", x[0])
    print("x[-1]:", x[-1])
    print("x[2:7]:", x[2:7])
    print("x[::2]:", x[::2])

    m = np.arange(1, 13).reshape(3, 4)
    print("m:\n", m)
    print("m[0, 0]:", m[0, 0])
    print("m[1, 2]:", m[1, 2])
    print("first row:", m[0, :])
    print("second column:", m[:, 1])
    print("submatrix (rows 0-1, cols 1-2):\n", m[0:2, 1:3])


def broadcasting_examples():
    print("\n=== Broadcasting ===")
    # Same shape
    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])
    print("a:", a)
    print("b:", b)
    print("a + b:", a + b)

    # Scalar and array
    print("a + 5:", a + 5)
    print("a * 2:", a * 2)

    # Broadcasting with 2D and 1D arrays
    m = np.arange(12).reshape(3, 4)
    row_vec = np.array([1, 0, -1, 2])
    col_vec = np.array([[1], [2], [3]])

    print("m:\n", m)
    print("row_vec:", row_vec)
    print("col_vec:\n", col_vec)

    # Add row_vec to each row of m
    print("m + row_vec:\n", m + row_vec)

    # Multiply col_vec with each column of m
    print("m * col_vec:\n", m * col_vec)


def basic_operations():
    print("\n=== Basic Operations (ufuncs, reductions) ===")
    x = np.array([1, 2, 3, 4, 5])
    print("x:", x)
    print("x ** 2:", x ** 2)
    print("np.sqrt(x):", np.sqrt(x))
    print("np.exp(x):", np.exp(x))

    print("sum:", np.sum(x))
    print("mean:", np.mean(x))
    print("std:", np.std(x))
    print("min:", np.min(x), "max:", np.max(x))

    m = np.arange(1, 13).reshape(3, 4)
    print("m:\n", m)
    print("sum over rows (axis=0):", np.sum(m, axis=0))
    print("sum over cols (axis=1):", np.sum(m, axis=1))


def linear_algebra_ops():
    print("\n=== Linear Algebra Operations ===")
    # Matrix multiplication
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    print("A:\n", A)
    print("B:\n", B)

    # Elementwise vs matrix multiplication
    print("A * B (elementwise):\n", A * B)
    print("A @ B (matrix product):\n", A @ B)
    print("np.dot(A, B):\n", np.dot(A, B))

    # Transpose
    print("A.T:\n", A.T)

    # Inverse (for invertible square matrices)
    A_inv = np.linalg.inv(A)
    print("A_inv:\n", A_inv)
    print("A @ A_inv (should be identity):\n", A @ A_inv)

    # Determinant
    det_A = np.linalg.det(A)
    print("det(A):", det_A)

    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(A)
    print("eigenvalues of A:", eigvals)
    print("eigenvectors of A:\n", eigvecs)

    # Solve linear system Ax = b
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)
    print("Solve Ax = b with b = [1, 2]")
    print("x:", x)
    print("Check A @ x:", A @ x)


def main():
    np.set_printoptions(precision=3, suppress=True)

    basic_arrays()
    indexing_and_slicing()
    broadcasting_examples()
    basic_operations()
    linear_algebra_ops()


if __name__ == "__main__":
    main()