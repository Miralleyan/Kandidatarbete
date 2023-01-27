# Assignment 1
$$\textrm{Maximise } V(p)=p(1-p)$$
# Assignment 2
# Assignment 3
$$\textrm{Maximize }L(p,q,u) = pq + u(p+q-1)$$
# Assignment 4
The problem is to minimize the Lagrangian of the form
$$\textrm{Maximize }L(p,q,u) = pq + u_1(p+q-1) + u_2(-p) + u_3(p-1) + u_4(-q) + u_5(q-1)$$
Which is a relaxation of the constrained minimization problem

$$\begin{align}
\textrm{Maximize } f(p,q) &= pq\\
p + q &= 1\\
0 \leq p &\leq 1\\
0 \leq q &\leq 1
\end{align}
$$

which is equivalent to

$$\begin{align} 
\textrm{Maximize } f(p,q) &= pq &\\
g_1(p,q) &= \hspace{0.8em} p + q - 1 &= 0\\
g_2(p,q) &= -p  &\leq 0\\
g_3(p,q) &= \hspace{0.8em} p\hspace{1.6em} -1 &\leq 0\\
g_4(p,q) &= \hspace{1.6em}-q  &\leq 0\\
g_5(p,q) &= \hspace{2.4em} q-1 &\leq 0
\end{align}
$$