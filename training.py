import numpy
import cvxpy

def get_nearest_positive_semidefinite(X):
    eigvals, eigvecs = numpy.linalg.eigh(X)
    eigvals[eigvals < 0] = 0
    return eigvecs @ numpy.diag(eigvals) @ eigvecs.T

def get_joint_analysis_mtxs(ld, af, se, beta, n):
    D = 2*af*(1 - af)*n
    N = n/(D*se**2) - beta**2/se**2 + 1
    N = numpy.repeat(N, beta.size).reshape(-1, N.size)
    N = numpy.minimum(N, N.T)
    B = ld * numpy.outer(numpy.sqrt(D/n), numpy.sqrt(D/n)) * N
    D = D * numpy.diag(N)/n
    c = D * beta
    c = c.reshape(-1, 1)
    B = get_nearest_positive_semidefinite(B)
    return B, c

def train_predictor_lasso(ld, af, se, beta, n, lambda_):
    B, c = get_joint_analysis_mtxs(ld, af, se, beta, n)
    b = cvxpy.Variable((beta.size, 1))
    loss = cvxpy.quad_form(b, cvxpy.psd_wrap(B)) - 2 * b.T @ c
    regularization = lambda_ * cvxpy.norm1(b)
    objective = cvxpy.Minimize(loss + regularization)
    constraints = []
    problem = cvxpy.Problem(objective, constraints)

    problem.solve()
    return b.value
