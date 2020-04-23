import torch

def apply_dropout(m):
    """Apply Dropout at test time"""
    if type(m) == torch.nn.Dropout2d:
        m.train()

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def bdiag_prod(M1, M2):
    """Computes matrix-matrix product of
    two block matrices containing only 
    diagonal submatrices.
    Input shapes: (N2, P, N1), (N1, P, N3)
    Output shape: (N2, P, N3)"""

    assert M1.size(2)==M2.size(0), "Dimension 2 of M1 does not match dimension 0 of M2"
    assert len(M1.size())==3 and len(M2.size())==3, "Expected input tensors to be of shape (N, P, N)"

    M = torch.einsum('ijk,kjl->ijl', M1, M2)

    return M

def blockwise_diag_inv(A, B, C, D, A_inv=None):
    #TODO: make this more efficient
    I = torch.zeros_like(A)
    for i in range(A.size(0)):
        I[i,:,i] = torch.ones(A.size(1)).to(A.device)
    if A_inv is None:
        A_inv = 1.0 / A
    aux = D - bdiag_prod(C, bdiag_prod(A_inv, B))
    aux_inv = 1.0 / aux
    A_aux = bdiag_prod(aux_inv, bdiag_prod(C, A_inv))
    A_aux_2 = I + bdiag_prod(B, A_aux)
    A_r = bdiag_prod(A_inv, A_aux_2)
    B_aux = bdiag_prod(B, aux_inv)
    B_r = - bdiag_prod(A_inv, B_aux)
    C_r = B_r.transpose(0, 2)
    D_r = aux_inv
    return A_r, B_r, C_r, D_r

def _cat_diag_blocks(A11, A12, A21, A22, N):
    A = torch.cat((torch.cat((A11, A12), 2), torch.cat((A21, A22), 2)), 0).view(N, -1, N)
    return A

def bmat_inv_logdet(M):
    """Given a block matrix composed 
    only of diagonal submatrices,
    returns its inverse and log determinant
    at cost of O(N^2 P).
    Input shape: (N, P, N)
    Output shapes: (N, P, N), scalar"""

    assert len(M.size())==3 and M.size(0)==M.size(2), "Expected input tensor to be of shape (N, P, N)"

    N = M.size(0)
    A_inv = None

    if N == 1:
        A_inv = 1. / M

    logdet = M[0,:,0].log().sum()

    for j in range(1, N):
        A = M[:j,:,:j]
        B = M[:j,:,j:(j+1)]
        C = M[j:(j+1),:,:j]
        D = M[j:(j+1),:,j:(j+1)]
        A11_inv, A12_inv, A21_inv, A22_inv = blockwise_diag_inv(A, B, C, D, A_inv=A_inv)
        A_inv = _cat_diag_blocks(A11_inv, A12_inv, A21_inv, A22_inv, j + 1)
        logdet += - A22_inv.flatten().log().sum()

    return A_inv, logdet

if __name__ == "__main__":
    from time import time

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    device = torch.device("cuda")

    N = 100
    P = 10
    S = 1000

    F = torch.randn((N, P, S)).to(device)

    #(N, P, N) symmetric
    M = torch.einsum('ijk,mjk->ijm', F, F) / S + 1e-1 * torch.ones((N, P, N)).to(device)

    tic = time()
    M_inv_block, logdet = bmat_inv_logdet(M)
    toc = time() - tic

    print('Time elapsed (s): ', round(toc, 5))