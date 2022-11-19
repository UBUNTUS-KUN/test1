#################################################################
## 調和ポテンシャル中の電子の固有関数　初期値の与え方
#################################################################
import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#図全体
fig = plt.figure(figsize=(15, 8))
#全体設定
plt.rcParams['font.family'] = 'Times New Roman' #フォント
plt.rcParams['font.size'] = 24 #フォントサイズ
plt.rcParams["mathtext.fontset"] = 'cm' #数式用フォント
#カラーリストの取得
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#################################################################
## 物理定数
#################################################################
#プランク定数
h = 6.62607015 * 1.0E-34
hbar = h / (2.0 * np.pi)
#電子の質量
me = 9.1093837015 * 1.0E-31
#電子ボルト
eV = 1.602176634 * 1.0E-19
#ナノメートル
nm = 1E-9
#虚数単位
I = 0.0 + 1.0j


#################################################################
## 規格化エルミート多項式
#################################################################
def NormalizedHermitePolynomial( n, x ):
	H0 = (1.0 / np.pi)**(1.0/4.0) * np.exp( - x**2 / 2.0 ) 
	H1 = (4.0 / np.pi)**(1.0/4.0) * np.exp( - x**2 / 2.0 )  * x
	if(n==0): return H0
	if(n==1): return H1
	for m in range(2, n+1):
		H2 = np.sqrt( 2.0 / m ) * x * H1 -  np.sqrt( (m - 1) / m )  * H0
		H0 = H1
		H1 = H2
	return H2

#################################################################
## 物理系の設定
#################################################################
#量子井戸の幅（m）
omega = 1.0 * 1.0E+15
A = np.sqrt( me * omega / hbar)
#固有関数
def verphi(n, x):
	barX = A * x
	return np.sqrt(A) * NormalizedHermitePolynomial( n, barX )
#固有エネルギー（eV）
def Energy(n):
	return (n + 1.0/2.0) * hbar * omega

#状態数
NS = 100

#################################################################
## 初期分布：ガウス分布
#################################################################
sigma = 1.0 * 10**10
x0 = 1.0 * nm
def verphi0(x):
	return ( sigma**2 / np.pi )**(1.0/4.0) * np.exp( - 1.0/2.0 * sigma**2 * (x - x0)**2 )

#被積分関数
def integral_orthonomality(x, n):
	return verphi(n, x) * verphi0(x)

L = 7 * nm

#描画範囲
x_min = -L/2
x_max = L/2
cn = []
for n in range( NS + 1 ):
	#ガウス・ルジャンドル積分
	result = integrate.quad(
		integral_orthonomality, #被積分関数
		x_min, x_max,           #積分区間の下端と上端
		args = ( n )           #被積分関数へ渡す引数
	)
	cn.append( result[0] )	
	#ターミナルへ出力
	print( "(" + str(n) + ")  " + str( result[0]) )

#################################################################
## 波動関数
#################################################################
def psi(x, t):
	_psi = 0.0 + 0.0j
	for n in range(NS):
		_psi += cn[n] * verphi(n, x) * np.exp( - I * ( n + 1.0/2.0) * omega * t )
	return _psi	
#########################################################################
# 波動関数 アニメーション
#########################################################################
#計算時間の幅
ts = 0
te = 100

#基底状態の周期
T0 = 2.0 * np.pi * hbar / Energy(0)
print( "基底状態の周期：" + str(T0) + " [s]" )

#時間間隔
dt = T0 / (te - ts + 1)

plt.title( u"調和ポテンシャル中の波動関数", fontsize=20, fontname="Yu Gothic", fontweight=1000)
plt.xlabel(r"$x\, [{\rm nm}]$", fontsize=20)

L = 7 * nm
#アニメーション作成用
ims = []

#描画範囲
x_min = -L/2
x_max = L/2
#描画区間数
NX = 500
#座標点配列の生成
x = np.linspace(x_min, x_max, NX)

#罫線の描画
plt.grid(which = "major", axis = "x", alpha = 0.2, linestyle = "-", linewidth = 1)
plt.grid(which = "major", axis = "y", alpha = 0.2, linestyle = "-", linewidth = 1)

#描画範囲を設定
plt.xlim([-3.5, 3.5])
plt.ylim([-1.5, 6.5])

#調和ポテンシャルの概形
yE = 1.0 / 2.0 * me * omega**2 * x**2 /eV /5.0

ims = []
#各時刻における波動関数の計算
for tn in range(ts, te):
	#実時間の取得
	t = dt * tn
	#波動関数の計算
	ys = psi(x, t).real / np.sqrt(A) * 2+ yE

	#波動関数の表示
	img  = plt.plot( x/nm, yE, linestyle='dotted', color="black", linewidth = 1 )
	img += plt.plot( x/nm, ys, colors[0], linestyle='solid', linewidth = 3 )
	
	#アニメーションに追加
	ims.append( img )

#余白の調整
#plt.subplots_adjust(left=0.15, right=0.90, bottom=0.1, top=0.99)
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.10, top = 0.95)

#アニメーションの生成
ani = animation.ArtistAnimation(fig, ims, interval=10)
#アニメーションの保存
#ani.save("output.html", writer=animation.HTMLWriter())
#ani.save('anim.gif', writer='pillow')
#ani.save("output.mp4", writer="ffmpeg", dpi=300)
#グラフの表示
plt.show()
