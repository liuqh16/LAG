# 导弹建模

### 导弹运动与动力学模型

惯性坐标系下，导弹运动学方程为：

$$
\begin{cases}
\dot{x}(t)=v(t)\cos{\theta(t)}\cos{\varphi(t)} \\
\dot{y}(t)=v(t)\cos{\theta(t)}\sin{\varphi(t)} \\
\dot{z}(t)=v(t)\sin{\theta(t)}
\end{cases}
$$

式中, $(x,y,z)$ 为导弹在惯性坐标系下的坐标： $(v,\theta,\varphi)$ 为导弹的速度，航迹俯仰角和航迹偏航角，均为飞行时间 $t$ 的函数。根据导弹的飞行分为主动段和被动段，在惯性系中，导弹主动段所受力主要包括：推力 $T(t)$, 重力 $G=m(t)g$, 气动阻力 $D(t)$ (在非惯性系下还需要考虑表视力 $C$ )，因此在弹道坐标系下，导弹的质点动力学方程为：

$$
\begin{cases}
\dot{v}(t)=g (n_{x}(t)-\sin{\theta(t)}) \\
\dot{\varphi}(t)=\frac{g}{v(t)}n_{y}(t)\cos{\theta(t)} \\
\dot{\theta}(t)=\frac{g}{v(t)}(n_{z}(t)-\cos{\theta(t)})
\end{cases}
$$

其中, $n_{x}(t)=\frac{T(t)-D(t)}{m(t)g}$, 为速度方向过载, $n_{y}, n_{z}$ 为导弹在偏航方向和俯仰方向的侧向控制过载通，通过过比例导引法计算得出； $m(t)$ 为导弹当前质量，$g$ 为重力加速度常数, 取 $g=9.81m/s^2$ 。

$T(t)$ 为导弹推力，其大小满足：

$$
T(t) = gI_{\text{sp}}\dot{m}(t)
$$

式中， $I_{\text{sp}}$ 为比冲（假设为常量）， $\dot{m}(t)$ 为燃料质量燃烧率。

$D(t)$ 为气动阻力，其大小满足：

$$
D(t) = \frac{1}{2}c_D(v(t))\cdot S\cdot \rho(z(t))\cdot v^2(t)
$$

式中， $v(t)$ 为导弹在 $t$ 时刻的速度大小， $z(t)$ 是导弹在 $t$ 时刻的水平高度， $S$ 为计算面积（定义为与速度正交的弹体截面积）， $c_D(v)$ 是阻力系数，它是速度大小 $v$ 的函数，可近似认为是常数，取 $c_D(v)\equiv 0.1$ 。 $\rho(z)$ 是空气密度，它是高度 $z$ 的函数，其大小满足近似关系式： $\rho(z)=\rho_0e^{-z/k}$ ，式中 $\rho_0=1.225kg/m^3,k=9300m$ 。

对于导弹的计算面积 $S$ ，其大小近似满足：

$$
S = \pi(\frac{d}{2})^2+(\sin^2(\Delta\theta)+\sin^2(\Delta\varphi))dL
$$

式中， $d$ 为导弹直径， $L$ 为导弹半径， $\Delta\theta,\Delta\varphi$ 为导弹姿态和航迹在俯仰角、偏航角上的差值，可近似认为是航迹俯仰角、偏航角 $\theta,\varphi$ 在 $\Delta t$ 时间内的变化量，即 $\Delta\theta=\dot{\theta}\Delta t$, $\Delta\varphi=\dot{\varphi}\Delta t$。
 
### 导弹导引控制模型

导弹导引采用比例导引律。假设在相互垂直的两个控制平面内导引系数均为 $K=3$ ，偏航和俯仰方向的两个侧向控制过载定义为：

$$
\begin{cases}
n_{y}=\frac{Kv}{g}\cos{\theta}\dot{\beta} \\
n_{z}=\frac{Kv}{g}\dot{\varepsilon}+\cos{\theta}
\end{cases}
$$

式中， $\beta,\varepsilon$ 分别为视线偏角与视线倾角， $\dot{\beta},\dot{\varepsilon}$ 分别为视线偏角和视线倾角随时间变化的导数。

$$
\begin{cases}
\beta=arctan(r_y/r_x) \\
\varepsilon=arctan(r_z/\sqrt{r_x^2+r_y^2})
\end{cases}
$$

视线矢量即为距离矢量 $\vec{r}$ ，有 $r_x=x_t-x_m,r_y=y_t-y_m,r_z=z_t-z_m$, $x_t,y_t,z_t$ 为目标机位置，模值定义为 $R=\sqrt{r_x^2+r_y^2+r_z^2}$ 。视线偏角和视线倾角及其随时间的导数公式定义为:

$$
\begin{cases}
\dot{\beta}=(\dot{r_y}r_x-r_y\dot{r_x})/(r_x^2+r_y^2) \\
\dot{\varepsilon}=\frac{(r_x^2+r_y^2)\dot{r_z}-r_z(\dot{r_x}r_x+\dot{r_y}r_y)}{R^2\sqrt{r_x^2+r_y^2}}
\end{cases}
$$
