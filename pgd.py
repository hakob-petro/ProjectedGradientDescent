# Standard modules
from typing import Callable, Iterable, Literal, Union

# Third-party modules
import torch
from torch.optim import Optimizer


class PGD(Optimizer):
    """
    Projected Gradient Descent optimizer.

    :param parameters (Union[Iterable, dict]): an iterable of :class:`torch.Tensor` s or
        :class:`dict` s. Specifies what Tensors should be optimized.
        Reference: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.step
    :param lr (float): learning rate of PGD algorithm. If not specified,
        then will be used auto search of best lr in each step.
        Reference: https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf, section 4.2.
    :param projector (Callable): a callable projection function, that returns projected p vector for given x vector
        from certain convex set. If not specified, then wil be used projection on union simplexes.
    :param nboundupdate (int): Used only if callable is not specifeied.
    :param reltol (float): Used only if callable is not specifeied.
    :param abstol (float): Used only if callable is not specifeied.
    :param maxiters (int): maximum number of iterations to convergence.
    :param method (Literal['fast', 'default']): 'fast' accelrated algorithm can be used if projector not specified.
    """

    def __init__(self, parameters: Union[Iterable, dict], lr: float, projector: Callable,
                 nboundupdate: int = 100, reltol: float = 1e-4, abstol: float = 0.0, maxiters: int = 1e7,
                 method: Literal['fast', 'default'] = 'default'):
        if parameters is None:
            raise ValueError("No parameters that need to be oprimized.")
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if method not in Literal['fast', 'default']:
            raise ValueError("Past wrong algorithm method.")

        defaults = {
            "lr": lr,
            "lower_bound": float("inf"),
            "projector": projector,
            "nboundupdate": nboundupdate,
            "reltol": reltol,
            "abstol": abstol,
            "maxiters": maxiters,
            "method": method
        }

        super().__init__(parameters, defaults)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Pass closure funciton!")
        else:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for params in group['params']:
                if params.grad is None:
                    continue
                orig_grad = params.grad.data
                if orig_grad.is_sparse:
                    raise RuntimeError("Couldn't handle sparsed arrays.")

                state = self.state[params]
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1

                projected = PGD.project_simplex(params)
                # loss = closure()

                if group["method"] == "fast":
                    y = projected.clone().detach()
                    loss = closure()
                    grad = loss.grad.data
                    s = 1.0 / torch.norm(grad)
                    while True:
                        y_new = PGD.project_simplex(y - s * grad)
                        # abs important as values close to machine precision
                        # might become negative in fft convolution screwing
                        # up cost calculations
                        loss_new = closure()
                        grad_new = loss_new.grad.data
                        if loss_new < loss + torch.mm(y_new - y, grad.T) + \
                                0.5 * torch.norm(y_new - y) ** 2 / s:
                            break
                        s *= 0.8
                    s /= 3
                else:
                    loss = closure()

                if (state["step"] % group["nboundupdate"] == 0) or (state["step"] == 0):
                    if group["method"] == 'fast':
                        loss = closure()
                    else:
                        i = torch.argmin(loss.grad.data)
                        group["lower_bound"] = max(
                            (group["lower_bound"], loss - torch.sum(projected * loss.grad.data) + loss.grad.data[i]))
                    gap = loss - group["lower_bound"]
                    if (group["lower_bound"] > 0 and gap / group["lower_bound"] < group["reltol"])\
                            or gap < group["abstol"]:
                        break

                if group["method"] == 'fast':
                    loss = closure()
                    projected, projected_old = PGD.project_simplex(y - s * loss.grad.data), projected
                    y = projected + state["step"] / (state["step"] + 3.0) * (projected - projected_old)
                else:
                    # see e.g section 4.2 in http://www.web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
                    s = 1.0 / torch.norm(grad)
                    z = mproject(p - s * grad)
                    fnew, gradnew = mfun(z, *args)
                    while fnew > f + np.dot(z - p, grad.T) + \
                            0.5 * np.linalg.norm(z - p) ** 2 / s:
                        s *= 0.5
                        z = mproject(p - s * grad)
                        fnew, gradnew = mfun(z, *args)
                    p = z
                    f, grad = fnew, gradnew
        return loss

    @staticmethod
    def project_simplex(x: torch.Tensor) -> torch.Tensor:
        """
        Take a vector x (with possible nonnegative entries and non-normalized)
            and project it onto the unit simplex.

        :param x: vector need to be projected on simplex.
        :type x: torch.Tensor
        :return torch.Tensor: projected vector
        """
        x_sorted, _ = torch.sort(x)[::-1]
        # entries need to sum up to 1 (unit simplex)
        sum_ = 1.0
        astar = -1
        lambda_a = (torch.cumsum(x_sorted, dim=0) - sum_) / torch.arange(1.0, len(x_sorted) + 1.0)
        for i in range(len(lambda_a) - 1):
            if lambda_a[i] >= x_sorted[i + 1]:
                astar = i
                break
        p = torch.maximum(x - lambda_a[astar], torch.zeros_like(x))
        return p
