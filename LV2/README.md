# Assignment 1
$$\textrm{Maximise } V(p)=p(1-p)$$
# Assignment 2
# Assignment 3
$$\textrm{Maximize }L(p,q,u) = pq + u(p+q-1)$$
# Assignment 4
The problem is to minimize the Lagrangian of the form
$$\textrm{Maximize }L(p,q,u) = pq + u_1(p+q-1) + u_2(-p) + u_3(p-1) + u_4(-q) + u_5(q-1)$$
Which is a relaxation of the constrained minimization problem
$$\begin{aligned}
\textrm{Maximize } f(x) &= pq\\ p + q &= 1\\ p &\geq 0\\ p &\leq 1\\ q &\geq 0\\ p &\leq 1 \end{aligned}$$
which is equivalent to
$$\begin{align}
\textrm{Maximize } f(x) &= pq\\ g_1(p,q) = p + q - 1 &= 0\\ g_2(p,q) = -p  &\leq 0\\ g_3(p,q) = p-1 &\leq 0\\ g_4(p,q) = -q  &\leq 0\\ g_5(p,q) = q-1 &\leq 0 \end{align}$$