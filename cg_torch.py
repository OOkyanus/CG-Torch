import torch

def conjugateGradient(A,x_0,b,inner_product,scaling_op,max_nof_iterations,th=0.0001):
    r = b - A(x_0)
    p = r
    x = x_0

    inner_product_rr = inner_product(r, r)
    if torch.all(inner_product_rr==0):
        return x

    for k in range(max_nof_iterations):
        inner_product_rr=inner_product(r,r)

        alpha=inner_product_rr/inner_product(p,A(p))

        alpha_p=scaling_op(alpha,p)

        th_compare=inner_product(alpha_p,alpha_p)/inner_product(x,x)

        if torch.all(th_compare < th**2):
            x = x + alpha_p
            return x

        x = x + alpha_p

        r_next = r-scaling_op(alpha,A(p))

        beta= inner_product(r_next,r_next)/inner_product_rr

        p=r_next + scaling_op(beta,p)
        r=r_next

    #warnings.warn("CG iterations did not converged.\n"
    #              "The given termination threshold is not satisfied. Max. iterations had reached")
    return x
