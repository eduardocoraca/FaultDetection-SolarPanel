1
×»
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
*
Erf
x"T
y"T"
Ttype:
2
¿
ExtractImagePatches
images"T
patches"T"
ksizes	list(int)(0"
strides	list(int)(0"
rates	list(int)(0"
Ttype:
2	
""
paddingstring:
SAMEVALID
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8Üí,
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1@* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:1@*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:@*
dtype0

layer_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer_normalization_12/gamma

0layer_normalization_12/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_12/gamma*
_output_shapes
:@*
dtype0

layer_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_12/beta

/layer_normalization_12/beta/Read/ReadVariableOpReadVariableOplayer_normalization_12/beta*
_output_shapes
:@*
dtype0

layer_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer_normalization_13/gamma

0layer_normalization_13/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_13/gamma*
_output_shapes
:@*
dtype0

layer_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_13/beta

/layer_normalization_13/beta/Read/ReadVariableOpReadVariableOplayer_normalization_13/beta*
_output_shapes
:@*
dtype0
{
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_22/kernel
t
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes
:	@*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	@*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0

layer_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer_normalization_14/gamma

0layer_normalization_14/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_14/gamma*
_output_shapes
:@*
dtype0

layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_14/beta

/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOplayer_normalization_14/beta*
_output_shapes
:@*
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
©
&patch_encoder_7/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¥@*7
shared_name(&patch_encoder_7/embedding_3/embeddings
¢
:patch_encoder_7/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp&patch_encoder_7/embedding_3/embeddings*
_output_shapes
:	¥@*
dtype0
¦
#multi_head_attention_6/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_6/query/kernel

7multi_head_attention_6/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/query/kernel*"
_output_shapes
:@@*
dtype0

!multi_head_attention_6/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_6/query/bias

5multi_head_attention_6/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/query/bias*
_output_shapes

:@*
dtype0
¢
!multi_head_attention_6/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention_6/key/kernel

5multi_head_attention_6/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/key/kernel*"
_output_shapes
:@@*
dtype0

multi_head_attention_6/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_6/key/bias

3multi_head_attention_6/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_6/key/bias*
_output_shapes

:@*
dtype0
¦
#multi_head_attention_6/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_6/value/kernel

7multi_head_attention_6/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/value/kernel*"
_output_shapes
:@@*
dtype0

!multi_head_attention_6/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_6/value/bias

5multi_head_attention_6/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/value/bias*
_output_shapes

:@*
dtype0
¼
.multi_head_attention_6/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.multi_head_attention_6/attention_output/kernel
µ
Bmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_6/attention_output/kernel*"
_output_shapes
:@@*
dtype0
°
,multi_head_attention_6/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention_6/attention_output/bias
©
@multi_head_attention_6/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_6/attention_output/bias*
_output_shapes
:@*
dtype0
¦
#multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_7/query/kernel

7multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_7/query/kernel*"
_output_shapes
:@@*
dtype0

!multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_7/query/bias

5multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_7/query/bias*
_output_shapes

:@*
dtype0
¢
!multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention_7/key/kernel

5multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_7/key/kernel*"
_output_shapes
:@@*
dtype0

multi_head_attention_7/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_7/key/bias

3multi_head_attention_7/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_7/key/bias*
_output_shapes

:@*
dtype0
¦
#multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_7/value/kernel

7multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_7/value/kernel*"
_output_shapes
:@@*
dtype0

!multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_7/value/bias

5multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_7/value/bias*
_output_shapes

:@*
dtype0
¼
.multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.multi_head_attention_7/attention_output/kernel
µ
Bmulti_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_7/attention_output/kernel*"
_output_shapes
:@@*
dtype0
°
,multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention_7/attention_output/bias
©
@multi_head_attention_7/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_7/attention_output/bias*
_output_shapes
:@*
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
l

Variable_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
j
ConstConst*&
_output_shapes
:*
dtype0*%
valueB*s£OC
l
Const_1Const*&
_output_shapes
:*
dtype0*%
valueB*¼sD

NoOpNoOp

Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ã
value¸B´ B¬
Ø
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
­
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer-4
	variables
regularization_losses
trainable_variables
 	keras_api
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
j
3position_embedding
4	variables
5regularization_losses
6trainable_variables
7	keras_api
q
8axis
	9gamma
:beta
;	variables
<regularization_losses
=trainable_variables
>	keras_api
»
?_query_dense
@
_key_dense
A_value_dense
B_softmax
C_dropout_layer
D_output_dense
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
q
Maxis
	Ngamma
Obeta
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
h

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
R
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
q
laxis
	mgamma
nbeta
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
»
s_query_dense
t
_key_dense
u_value_dense
v_softmax
w_dropout_layer
x_output_dense
y	variables
zregularization_losses
{trainable_variables
|	keras_api

}0
~1
2
%3
&4
5
96
:7
8
9
10
11
12
13
14
15
N16
O17
T18
U19
^20
_21
m22
n23
24
25
26
27
28
29
30
31
 
ï
%0
&1
2
93
:4
5
6
7
8
9
10
11
12
N13
O14
T15
U16
^17
_18
m19
n20
21
22
23
24
25
26
27
28
²
 layer_regularization_losses
layer_metrics
layers
	variables
regularization_losses
trainable_variables
non_trainable_variables
metrics
 
ª

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
}mean
}
adapt_mean
~variance
~adapt_variance
	count
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
a
	_rng
 	variables
¡regularization_losses
¢trainable_variables
£	keras_api
a
	¤_rng
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
a
	©_rng
ª	variables
«regularization_losses
¬trainable_variables
­	keras_api

}0
~1
2
 
 
²
 ®layer_regularization_losses
¯layer_metrics
°layers
	variables
regularization_losses
trainable_variables
±non_trainable_variables
²metrics
 
 
 
²
 ³layer_regularization_losses
´layer_metrics
µlayers
!	variables
"regularization_losses
#trainable_variables
¶non_trainable_variables
·metrics
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
²
 ¸layer_regularization_losses
¹layer_metrics
ºlayers
'	variables
(regularization_losses
)trainable_variables
»non_trainable_variables
¼metrics
 
 
 
²
 ½layer_regularization_losses
¾layer_metrics
¿layers
+	variables
,regularization_losses
-trainable_variables
Ànon_trainable_variables
Ámetrics
 
 
 
²
 Âlayer_regularization_losses
Ãlayer_metrics
Älayers
/	variables
0regularization_losses
1trainable_variables
Ånon_trainable_variables
Æmetrics
g

embeddings
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api

0
 

0
²
 Ëlayer_regularization_losses
Ìlayer_metrics
Ílayers
4	variables
5regularization_losses
6trainable_variables
Înon_trainable_variables
Ïmetrics
 
ge
VARIABLE_VALUElayer_normalization_12/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_12/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
²
 Ðlayer_regularization_losses
Ñlayer_metrics
Òlayers
;	variables
<regularization_losses
=trainable_variables
Ónon_trainable_variables
Ômetrics
¡
Õpartial_output_shape
Öfull_output_shape
kernel
	bias
×	variables
Øregularization_losses
Ùtrainable_variables
Ú	keras_api
¡
Ûpartial_output_shape
Üfull_output_shape
kernel
	bias
Ý	variables
Þregularization_losses
ßtrainable_variables
à	keras_api
¡
ápartial_output_shape
âfull_output_shape
kernel
	bias
ã	variables
äregularization_losses
åtrainable_variables
æ	keras_api
V
ç	variables
èregularization_losses
étrainable_variables
ê	keras_api
V
ë	variables
ìregularization_losses
ítrainable_variables
î	keras_api
¡
ïpartial_output_shape
ðfull_output_shape
kernel
	bias
ñ	variables
òregularization_losses
ótrainable_variables
ô	keras_api
@
0
1
2
3
4
5
6
7
 
@
0
1
2
3
4
5
6
7
²
 õlayer_regularization_losses
ölayer_metrics
÷layers
E	variables
Fregularization_losses
Gtrainable_variables
ønon_trainable_variables
ùmetrics
 
 
 
²
 úlayer_regularization_losses
ûlayer_metrics
ülayers
I	variables
Jregularization_losses
Ktrainable_variables
ýnon_trainable_variables
þmetrics
 
ge
VARIABLE_VALUElayer_normalization_13/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_13/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
²
 ÿlayer_regularization_losses
layer_metrics
layers
P	variables
Qregularization_losses
Rtrainable_variables
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
²
 layer_regularization_losses
layer_metrics
layers
V	variables
Wregularization_losses
Xtrainable_variables
non_trainable_variables
metrics
 
 
 
²
 layer_regularization_losses
layer_metrics
layers
Z	variables
[regularization_losses
\trainable_variables
non_trainable_variables
metrics
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
²
 layer_regularization_losses
layer_metrics
layers
`	variables
aregularization_losses
btrainable_variables
non_trainable_variables
metrics
 
 
 
²
 layer_regularization_losses
layer_metrics
layers
d	variables
eregularization_losses
ftrainable_variables
non_trainable_variables
metrics
 
 
 
²
 layer_regularization_losses
layer_metrics
layers
h	variables
iregularization_losses
jtrainable_variables
non_trainable_variables
metrics
 
ge
VARIABLE_VALUElayer_normalization_14/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_14/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE

m0
n1
 

m0
n1
²
 layer_regularization_losses
layer_metrics
layers
o	variables
pregularization_losses
qtrainable_variables
 non_trainable_variables
¡metrics
¡
¢partial_output_shape
£full_output_shape
kernel
	bias
¤	variables
¥regularization_losses
¦trainable_variables
§	keras_api
¡
¨partial_output_shape
©full_output_shape
kernel
	bias
ª	variables
«regularization_losses
¬trainable_variables
­	keras_api
¡
®partial_output_shape
¯full_output_shape
kernel
	bias
°	variables
±regularization_losses
²trainable_variables
³	keras_api
V
´	variables
µregularization_losses
¶trainable_variables
·	keras_api
V
¸	variables
¹regularization_losses
ºtrainable_variables
»	keras_api
¡
¼partial_output_shape
½full_output_shape
kernel
	bias
¾	variables
¿regularization_losses
Àtrainable_variables
Á	keras_api
@
0
1
2
3
4
5
6
7
 
@
0
1
2
3
4
5
6
7
²
 Âlayer_regularization_losses
Ãlayer_metrics
Älayers
y	variables
zregularization_losses
{trainable_variables
Ånon_trainable_variables
Æmetrics
@>
VARIABLE_VALUEmean&variables/0/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEvariance&variables/1/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUEcount&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&patch_encoder_7/embedding_3/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#multi_head_attention_6/query/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!multi_head_attention_6/query/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!multi_head_attention_6/key/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmulti_head_attention_6/key/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#multi_head_attention_6/value/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!multi_head_attention_6/value/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.multi_head_attention_6/attention_output/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,multi_head_attention_6/attention_output/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#multi_head_attention_7/query/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!multi_head_attention_7/query/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!multi_head_attention_7/key/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmulti_head_attention_7/key/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#multi_head_attention_7/value/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!multi_head_attention_7/value/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.multi_head_attention_7/attention_output/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,multi_head_attention_7/attention_output/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17

}0
~1
2
 
 
 
 
 
 
 
 
 
µ
 Çlayer_regularization_losses
Èlayer_metrics
Élayers
	variables
regularization_losses
trainable_variables
Ênon_trainable_variables
Ëmetrics

Ì
_state_var
 
 
 
µ
 Ílayer_regularization_losses
Îlayer_metrics
Ïlayers
 	variables
¡regularization_losses
¢trainable_variables
Ðnon_trainable_variables
Ñmetrics

Ò
_state_var
 
 
 
µ
 Ólayer_regularization_losses
Ôlayer_metrics
Õlayers
¥	variables
¦regularization_losses
§trainable_variables
Önon_trainable_variables
×metrics

Ø
_state_var
 
 
 
µ
 Ùlayer_regularization_losses
Úlayer_metrics
Ûlayers
ª	variables
«regularization_losses
¬trainable_variables
Ünon_trainable_variables
Ýmetrics
 
 
#
0
1
2
3
4

}0
~1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
 

0
µ
 Þlayer_regularization_losses
ßlayer_metrics
àlayers
Ç	variables
Èregularization_losses
Étrainable_variables
ánon_trainable_variables
âmetrics
 
 

30
 
 
 
 
 
 
 
 
 

0
1
 

0
1
µ
 ãlayer_regularization_losses
älayer_metrics
ålayers
×	variables
Øregularization_losses
Ùtrainable_variables
ænon_trainable_variables
çmetrics
 
 

0
1
 

0
1
µ
 èlayer_regularization_losses
élayer_metrics
êlayers
Ý	variables
Þregularization_losses
ßtrainable_variables
ënon_trainable_variables
ìmetrics
 
 

0
1
 

0
1
µ
 ílayer_regularization_losses
îlayer_metrics
ïlayers
ã	variables
äregularization_losses
åtrainable_variables
ðnon_trainable_variables
ñmetrics
 
 
 
µ
 òlayer_regularization_losses
ólayer_metrics
ôlayers
ç	variables
èregularization_losses
étrainable_variables
õnon_trainable_variables
ömetrics
 
 
 
µ
 ÷layer_regularization_losses
ølayer_metrics
ùlayers
ë	variables
ìregularization_losses
ítrainable_variables
únon_trainable_variables
ûmetrics
 
 

0
1
 

0
1
µ
 ülayer_regularization_losses
ýlayer_metrics
þlayers
ñ	variables
òregularization_losses
ótrainable_variables
ÿnon_trainable_variables
metrics
 
 
*
?0
@1
A2
B3
C4
D5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 

0
1
µ
 layer_regularization_losses
layer_metrics
layers
¤	variables
¥regularization_losses
¦trainable_variables
non_trainable_variables
metrics
 
 

0
1
 

0
1
µ
 layer_regularization_losses
layer_metrics
layers
ª	variables
«regularization_losses
¬trainable_variables
non_trainable_variables
metrics
 
 

0
1
 

0
1
µ
 layer_regularization_losses
layer_metrics
layers
°	variables
±regularization_losses
²trainable_variables
non_trainable_variables
metrics
 
 
 
µ
 layer_regularization_losses
layer_metrics
layers
´	variables
µregularization_losses
¶trainable_variables
non_trainable_variables
metrics
 
 
 
µ
 layer_regularization_losses
layer_metrics
layers
¸	variables
¹regularization_losses
ºtrainable_variables
non_trainable_variables
metrics
 
 

0
1
 

0
1
µ
 layer_regularization_losses
layer_metrics
layers
¾	variables
¿regularization_losses
Àtrainable_variables
non_trainable_variables
metrics
 
 
*
s0
t1
u2
v3
w4
x5
 
 
 
 
 
 
 
ec
VARIABLE_VALUEVariableGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ge
VARIABLE_VALUE
Variable_1Glayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ge
VARIABLE_VALUE
Variable_2Glayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_4Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
©

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4ConstConst_1dense_21/kerneldense_21/bias&patch_encoder_7/embedding_3/embeddingslayer_normalization_12/gammalayer_normalization_12/beta#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/biaslayer_normalization_13/gammalayer_normalization_13/betadense_22/kerneldense_22/biasdense_23/kerneldense_23/biaslayer_normalization_14/gammalayer_normalization_14/beta#multi_head_attention_7/query/kernel!multi_head_attention_7/query/bias!multi_head_attention_7/key/kernelmulti_head_attention_7/key/bias#multi_head_attention_7/value/kernel!multi_head_attention_7/value/bias.multi_head_attention_7/attention_output/kernel,multi_head_attention_7/attention_output/bias*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1120466
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp0layer_normalization_12/gamma/Read/ReadVariableOp/layer_normalization_12/beta/Read/ReadVariableOp0layer_normalization_13/gamma/Read/ReadVariableOp/layer_normalization_13/beta/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp0layer_normalization_14/gamma/Read/ReadVariableOp/layer_normalization_14/beta/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp:patch_encoder_7/embedding_3/embeddings/Read/ReadVariableOp7multi_head_attention_6/query/kernel/Read/ReadVariableOp5multi_head_attention_6/query/bias/Read/ReadVariableOp5multi_head_attention_6/key/kernel/Read/ReadVariableOp3multi_head_attention_6/key/bias/Read/ReadVariableOp7multi_head_attention_6/value/kernel/Read/ReadVariableOp5multi_head_attention_6/value/bias/Read/ReadVariableOpBmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_6/attention_output/bias/Read/ReadVariableOp7multi_head_attention_7/query/kernel/Read/ReadVariableOp5multi_head_attention_7/query/bias/Read/ReadVariableOp5multi_head_attention_7/key/kernel/Read/ReadVariableOp3multi_head_attention_7/key/bias/Read/ReadVariableOp7multi_head_attention_7/value/kernel/Read/ReadVariableOp5multi_head_attention_7/value/bias/Read/ReadVariableOpBmulti_head_attention_7/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_7/attention_output/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpConst_2*0
Tin)
'2%				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1122888


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biaslayer_normalization_12/gammalayer_normalization_12/betalayer_normalization_13/gammalayer_normalization_13/betadense_22/kerneldense_22/biasdense_23/kerneldense_23/biaslayer_normalization_14/gammalayer_normalization_14/betameanvariancecount&patch_encoder_7/embedding_3/embeddings#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/bias#multi_head_attention_7/query/kernel!multi_head_attention_7/query/bias!multi_head_attention_7/key/kernelmulti_head_attention_7/key/bias#multi_head_attention_7/value/kernel!multi_head_attention_7/value/bias.multi_head_attention_7/attention_output/kernel,multi_head_attention_7/attention_output/biasVariable
Variable_1
Variable_2*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1123003øó*
¥
j
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1122483

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
ð
Â
E__inference_model_11_layer_call_and_return_conditional_losses_1120850

inputs+
'data_augmentation_normalization_3_sub_y,
(data_augmentation_normalization_3_sqrt_x<
*dense_21_tensordot_readvariableop_resource:1@6
(dense_21_biasadd_readvariableop_resource:@G
4patch_encoder_7_embedding_3_embedding_lookup_1120657:	¥@J
<layer_normalization_12_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_12_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_6_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_query_add_readvariableop_resource:@V
@multi_head_attention_6_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_6_key_add_readvariableop_resource:@X
Bmulti_head_attention_6_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_value_add_readvariableop_resource:@c
Mmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_6_attention_output_add_readvariableop_resource:@J
<layer_normalization_13_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_13_batchnorm_readvariableop_resource:@=
*dense_22_tensordot_readvariableop_resource:	@7
(dense_22_biasadd_readvariableop_resource:	=
*dense_23_tensordot_readvariableop_resource:	@6
(dense_23_biasadd_readvariableop_resource:@J
<layer_normalization_14_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_14_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_7_query_add_readvariableop_resource:@V
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_7_key_add_readvariableop_resource:@X
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_7_value_add_readvariableop_resource:@c
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_7_attention_output_add_readvariableop_resource:@
identity¢dense_21/BiasAdd/ReadVariableOp¢!dense_21/Tensordot/ReadVariableOp¢dense_22/BiasAdd/ReadVariableOp¢!dense_22/Tensordot/ReadVariableOp¢dense_23/BiasAdd/ReadVariableOp¢!dense_23/Tensordot/ReadVariableOp¢/layer_normalization_12/batchnorm/ReadVariableOp¢3layer_normalization_12/batchnorm/mul/ReadVariableOp¢/layer_normalization_13/batchnorm/ReadVariableOp¢3layer_normalization_13/batchnorm/mul/ReadVariableOp¢/layer_normalization_14/batchnorm/ReadVariableOp¢3layer_normalization_14/batchnorm/mul/ReadVariableOp¢:multi_head_attention_6/attention_output/add/ReadVariableOp¢Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_6/key/add/ReadVariableOp¢7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/query/add/ReadVariableOp¢9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/value/add/ReadVariableOp¢9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp¢:multi_head_attention_7/attention_output/add/ReadVariableOp¢Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_7/key/add/ReadVariableOp¢7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/query/add/ReadVariableOp¢9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/value/add/ReadVariableOp¢9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢,patch_encoder_7/embedding_3/embedding_lookupÂ
%data_augmentation/normalization_3/subSubinputs'data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%data_augmentation/normalization_3/sub³
&data_augmentation/normalization_3/SqrtSqrt(data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:2(
&data_augmentation/normalization_3/Sqrt
+data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32-
+data_augmentation/normalization_3/Maximum/yô
)data_augmentation/normalization_3/MaximumMaximum*data_augmentation/normalization_3/Sqrt:y:04data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2+
)data_augmentation/normalization_3/Maximum÷
)data_augmentation/normalization_3/truedivRealDiv)data_augmentation/normalization_3/sub:z:0-data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)data_augmentation/normalization_3/truediv¥
(data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   2*
(data_augmentation/resizing_3/resize/size±
2data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear-data_augmentation/normalization_3/truediv:z:01data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(24
2data_augmentation/resizing_3/resize/ResizeBilinear¤
patches_7/ExtractImagePatchesExtractImagePatchesCdata_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksizes
*
paddingVALID*
rates
*
strides
2
patches_7/ExtractImagePatches
patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ¤  1   2
patches_7/Reshape/shape³
patches_7/ReshapeReshape'patches_7/ExtractImagePatches:patches:0 patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
patches_7/Reshape±
!dense_21/Tensordot/ReadVariableOpReadVariableOp*dense_21_tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02#
!dense_21/Tensordot/ReadVariableOp|
dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_21/Tensordot/axes
dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_21/Tensordot/free~
dense_21/Tensordot/ShapeShapepatches_7/Reshape:output:0*
T0*
_output_shapes
:2
dense_21/Tensordot/Shape
 dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/GatherV2/axisþ
dense_21/Tensordot/GatherV2GatherV2!dense_21/Tensordot/Shape:output:0 dense_21/Tensordot/free:output:0)dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_21/Tensordot/GatherV2
"dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_21/Tensordot/GatherV2_1/axis
dense_21/Tensordot/GatherV2_1GatherV2!dense_21/Tensordot/Shape:output:0 dense_21/Tensordot/axes:output:0+dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_21/Tensordot/GatherV2_1~
dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_21/Tensordot/Const¤
dense_21/Tensordot/ProdProd$dense_21/Tensordot/GatherV2:output:0!dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prod
dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_21/Tensordot/Const_1¬
dense_21/Tensordot/Prod_1Prod&dense_21/Tensordot/GatherV2_1:output:0#dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prod_1
dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_21/Tensordot/concat/axisÝ
dense_21/Tensordot/concatConcatV2 dense_21/Tensordot/free:output:0 dense_21/Tensordot/axes:output:0'dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat°
dense_21/Tensordot/stackPack dense_21/Tensordot/Prod:output:0"dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/stackÀ
dense_21/Tensordot/transpose	Transposepatches_7/Reshape:output:0"dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
dense_21/Tensordot/transposeÃ
dense_21/Tensordot/ReshapeReshape dense_21/Tensordot/transpose:y:0!dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_21/Tensordot/ReshapeÂ
dense_21/Tensordot/MatMulMatMul#dense_21/Tensordot/Reshape:output:0)dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_21/Tensordot/MatMul
dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_21/Tensordot/Const_2
 dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/concat_1/axisê
dense_21/Tensordot/concat_1ConcatV2$dense_21/Tensordot/GatherV2:output:0#dense_21/Tensordot/Const_2:output:0)dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat_1µ
dense_21/TensordotReshape#dense_21/Tensordot/MatMul:product:0$dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
dense_21/Tensordot§
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_21/BiasAdd/ReadVariableOp¬
dense_21/BiasAddBiasAdddense_21/Tensordot:output:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
dense_21/BiasAdd
lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2!
lambda_3/Mean/reduction_indices
lambda_3/MeanMeandense_21/BiasAdd:output:0(lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lambda_3/Mean
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2
lambda_3/Reshape/shape
lambda_3/ReshapeReshapelambda_3/Mean:output:0lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lambda_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisÒ
concatenate_3/concatConcatV2lambda_3/Reshape:output:0dense_21/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
concatenate_3/concat|
patch_encoder_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
patch_encoder_7/range/start}
patch_encoder_7/range/limitConst*
_output_shapes
: *
dtype0*
value
B :¥2
patch_encoder_7/range/limit|
patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder_7/range/deltaÆ
patch_encoder_7/rangeRange$patch_encoder_7/range/start:output:0$patch_encoder_7/range/limit:output:0$patch_encoder_7/range/delta:output:0*
_output_shapes	
:¥2
patch_encoder_7/rangeù
,patch_encoder_7/embedding_3/embedding_lookupResourceGather4patch_encoder_7_embedding_3_embedding_lookup_1120657patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1120657*
_output_shapes
:	¥@*
dtype02.
,patch_encoder_7/embedding_3/embedding_lookupÒ
5patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity5patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1120657*
_output_shapes
:	¥@27
5patch_encoder_7/embedding_3/embedding_lookup/Identityè
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1Identity>patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	¥@29
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1Ë
patch_encoder_7/addAddV2concatenate_3/concat:output:0@patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
patch_encoder_7/add¸
5layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_12/moments/mean/reduction_indicesó
#layer_normalization_12/moments/meanMeanpatch_encoder_7/add:z:0>layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2%
#layer_normalization_12/moments/meanÏ
+layer_normalization_12/moments/StopGradientStopGradient,layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2-
+layer_normalization_12/moments/StopGradientÿ
0layer_normalization_12/moments/SquaredDifferenceSquaredDifferencepatch_encoder_7/add:z:04layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
0layer_normalization_12/moments/SquaredDifferenceÀ
9layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_12/moments/variance/reduction_indices
'layer_normalization_12/moments/varianceMean4layer_normalization_12/moments/SquaredDifference:z:0Blayer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2)
'layer_normalization_12/moments/variance
&layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_12/batchnorm/add/yï
$layer_normalization_12/batchnorm/addAddV20layer_normalization_12/moments/variance:output:0/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2&
$layer_normalization_12/batchnorm/addº
&layer_normalization_12/batchnorm/RsqrtRsqrt(layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2(
&layer_normalization_12/batchnorm/Rsqrtã
3layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_12/batchnorm/mul/ReadVariableOpó
$layer_normalization_12/batchnorm/mulMul*layer_normalization_12/batchnorm/Rsqrt:y:0;layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_12/batchnorm/mulÑ
&layer_normalization_12/batchnorm/mul_1Mulpatch_encoder_7/add:z:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_12/batchnorm/mul_1æ
&layer_normalization_12/batchnorm/mul_2Mul,layer_normalization_12/moments/mean:output:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_12/batchnorm/mul_2×
/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_12/batchnorm/ReadVariableOpï
$layer_normalization_12/batchnorm/subSub7layer_normalization_12/batchnorm/ReadVariableOp:value:0*layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_12/batchnorm/subæ
&layer_normalization_12/batchnorm/add_1AddV2*layer_normalization_12/batchnorm/mul_1:z:0(layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_12/batchnorm/add_1ý
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp²
*multi_head_attention_6/query/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_6/query/einsum/EinsumÛ
/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp8multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/query/add/ReadVariableOpö
 multi_head_attention_6/query/addAddV23multi_head_attention_6/query/einsum/Einsum:output:07multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_6/query/add÷
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¬
(multi_head_attention_6/key/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0?multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2*
(multi_head_attention_6/key/einsum/EinsumÕ
-multi_head_attention_6/key/add/ReadVariableOpReadVariableOp6multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_6/key/add/ReadVariableOpî
multi_head_attention_6/key/addAddV21multi_head_attention_6/key/einsum/Einsum:output:05multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2 
multi_head_attention_6/key/addý
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp²
*multi_head_attention_6/value/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_6/value/einsum/EinsumÛ
/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp8multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/value/add/ReadVariableOpö
 multi_head_attention_6/value/addAddV23multi_head_attention_6/value/einsum/Einsum:output:07multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_6/value/add
multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_6/Mul/yÇ
multi_head_attention_6/MulMul$multi_head_attention_6/query/add:z:0%multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
multi_head_attention_6/Mulþ
$multi_head_attention_6/einsum/EinsumEinsum"multi_head_attention_6/key/add:z:0multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2&
$multi_head_attention_6/einsum/EinsumÆ
&multi_head_attention_6/softmax/SoftmaxSoftmax-multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2(
&multi_head_attention_6/softmax/SoftmaxÌ
'multi_head_attention_6/dropout/IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2)
'multi_head_attention_6/dropout/Identity
&multi_head_attention_6/einsum_1/EinsumEinsum0multi_head_attention_6/dropout/Identity:output:0$multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2(
&multi_head_attention_6/einsum_1/Einsum
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_6/attention_output/einsum/EinsumEinsum/multi_head_attention_6/einsum_1/Einsum:output:0Lmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe27
5multi_head_attention_6/attention_output/einsum/Einsumø
:multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_6/attention_output/add/ReadVariableOp
+multi_head_attention_6/attention_output/addAddV2>multi_head_attention_6/attention_output/einsum/Einsum:output:0Bmulti_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2-
+multi_head_attention_6/attention_output/add¢

add_12/addAddV2/multi_head_attention_6/attention_output/add:z:0patch_encoder_7/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

add_12/add¸
5layer_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_13/moments/mean/reduction_indicesê
#layer_normalization_13/moments/meanMeanadd_12/add:z:0>layer_normalization_13/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2%
#layer_normalization_13/moments/meanÏ
+layer_normalization_13/moments/StopGradientStopGradient,layer_normalization_13/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2-
+layer_normalization_13/moments/StopGradientö
0layer_normalization_13/moments/SquaredDifferenceSquaredDifferenceadd_12/add:z:04layer_normalization_13/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
0layer_normalization_13/moments/SquaredDifferenceÀ
9layer_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_13/moments/variance/reduction_indices
'layer_normalization_13/moments/varianceMean4layer_normalization_13/moments/SquaredDifference:z:0Blayer_normalization_13/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2)
'layer_normalization_13/moments/variance
&layer_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_13/batchnorm/add/yï
$layer_normalization_13/batchnorm/addAddV20layer_normalization_13/moments/variance:output:0/layer_normalization_13/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2&
$layer_normalization_13/batchnorm/addº
&layer_normalization_13/batchnorm/RsqrtRsqrt(layer_normalization_13/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2(
&layer_normalization_13/batchnorm/Rsqrtã
3layer_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_13/batchnorm/mul/ReadVariableOpó
$layer_normalization_13/batchnorm/mulMul*layer_normalization_13/batchnorm/Rsqrt:y:0;layer_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_13/batchnorm/mulÈ
&layer_normalization_13/batchnorm/mul_1Muladd_12/add:z:0(layer_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_13/batchnorm/mul_1æ
&layer_normalization_13/batchnorm/mul_2Mul,layer_normalization_13/moments/mean:output:0(layer_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_13/batchnorm/mul_2×
/layer_normalization_13/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_13/batchnorm/ReadVariableOpï
$layer_normalization_13/batchnorm/subSub7layer_normalization_13/batchnorm/ReadVariableOp:value:0*layer_normalization_13/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_13/batchnorm/subæ
&layer_normalization_13/batchnorm/add_1AddV2*layer_normalization_13/batchnorm/mul_1:z:0(layer_normalization_13/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_13/batchnorm/add_1²
!dense_22/Tensordot/ReadVariableOpReadVariableOp*dense_22_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02#
!dense_22/Tensordot/ReadVariableOp|
dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_22/Tensordot/axes
dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_22/Tensordot/free
dense_22/Tensordot/ShapeShape*layer_normalization_13/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_22/Tensordot/Shape
 dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/GatherV2/axisþ
dense_22/Tensordot/GatherV2GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/free:output:0)dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2
"dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_22/Tensordot/GatherV2_1/axis
dense_22/Tensordot/GatherV2_1GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/axes:output:0+dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2_1~
dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const¤
dense_22/Tensordot/ProdProd$dense_22/Tensordot/GatherV2:output:0!dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod
dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const_1¬
dense_22/Tensordot/Prod_1Prod&dense_22/Tensordot/GatherV2_1:output:0#dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod_1
dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_22/Tensordot/concat/axisÝ
dense_22/Tensordot/concatConcatV2 dense_22/Tensordot/free:output:0 dense_22/Tensordot/axes:output:0'dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat°
dense_22/Tensordot/stackPack dense_22/Tensordot/Prod:output:0"dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/stackÐ
dense_22/Tensordot/transpose	Transpose*layer_normalization_13/batchnorm/add_1:z:0"dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_22/Tensordot/transposeÃ
dense_22/Tensordot/ReshapeReshape dense_22/Tensordot/transpose:y:0!dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_22/Tensordot/ReshapeÃ
dense_22/Tensordot/MatMulMatMul#dense_22/Tensordot/Reshape:output:0)dense_22/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_22/Tensordot/MatMul
dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_22/Tensordot/Const_2
 dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/concat_1/axisê
dense_22/Tensordot/concat_1ConcatV2$dense_22/Tensordot/GatherV2:output:0#dense_22/Tensordot/Const_2:output:0)dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat_1¶
dense_22/TensordotReshape#dense_22/Tensordot/MatMul:product:0$dense_22/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Tensordot¨
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp­
dense_22/BiasAddBiasAdddense_22/Tensordot:output:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/BiasAddo
dense_22/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_22/Gelu/mul/x
dense_22/Gelu/mulMuldense_22/Gelu/mul/x:output:0dense_22/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/mulq
dense_22/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
dense_22/Gelu/Cast/x«
dense_22/Gelu/truedivRealDivdense_22/BiasAdd:output:0dense_22/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/truediv
dense_22/Gelu/ErfErfdense_22/Gelu/truediv:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/Erfo
dense_22/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_22/Gelu/add/x
dense_22/Gelu/addAddV2dense_22/Gelu/add/x:output:0dense_22/Gelu/Erf:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/add
dense_22/Gelu/mul_1Muldense_22/Gelu/mul:z:0dense_22/Gelu/add:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/mul_1
dropout_18/IdentityIdentitydense_22/Gelu/mul_1:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout_18/Identity²
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axes
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/free
dense_23/Tensordot/ShapeShapedropout_18/Identity:output:0*
T0*
_output_shapes
:2
dense_23/Tensordot/Shape
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axisþ
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axis
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const¤
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1¬
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axisÝ
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat°
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stackÃ
dense_23/Tensordot/transpose	Transposedropout_18/Identity:output:0"dense_23/Tensordot/concat:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_23/Tensordot/transposeÃ
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_23/Tensordot/ReshapeÂ
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_23/Tensordot/MatMul
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_23/Tensordot/Const_2
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axisê
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1µ
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Tensordot§
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp¬
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/BiasAddo
dense_23/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_23/Gelu/mul/x
dense_23/Gelu/mulMuldense_23/Gelu/mul/x:output:0dense_23/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/mulq
dense_23/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
dense_23/Gelu/Cast/xª
dense_23/Gelu/truedivRealDivdense_23/BiasAdd:output:0dense_23/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/truediv
dense_23/Gelu/ErfErfdense_23/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/Erfo
dense_23/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_23/Gelu/add/x
dense_23/Gelu/addAddV2dense_23/Gelu/add/x:output:0dense_23/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/add
dense_23/Gelu/mul_1Muldense_23/Gelu/mul:z:0dense_23/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/mul_1
dropout_19/IdentityIdentitydense_23/Gelu/mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout_19/Identity

add_13/addAddV2dropout_19/Identity:output:0add_12/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

add_13/add¸
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesê
#layer_normalization_14/moments/meanMeanadd_13/add:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2%
#layer_normalization_14/moments/meanÏ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2-
+layer_normalization_14/moments/StopGradientö
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd_13/add:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
0layer_normalization_14/moments/SquaredDifferenceÀ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yï
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2&
$layer_normalization_14/batchnorm/addº
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2(
&layer_normalization_14/batchnorm/Rsqrtã
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpó
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_14/batchnorm/mulÈ
&layer_normalization_14/batchnorm/mul_1Muladd_13/add:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_14/batchnorm/mul_1æ
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_14/batchnorm/mul_2×
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpï
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_14/batchnorm/subæ
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_14/batchnorm/add_1ý
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp²
*multi_head_attention_7/query/einsum/EinsumEinsum*layer_normalization_14/batchnorm/add_1:z:0Amulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumÛ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_7/query/add/ReadVariableOpö
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_7/query/add÷
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¬
(multi_head_attention_7/key/einsum/EinsumEinsum*layer_normalization_14/batchnorm/add_1:z:0?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumÕ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpî
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2 
multi_head_attention_7/key/addý
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp²
*multi_head_attention_7/value/einsum/EinsumEinsum*layer_normalization_14/batchnorm/add_1:z:0Amulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumÛ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_7/value/add/ReadVariableOpö
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_7/Mul/yÇ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
multi_head_attention_7/Mulþ
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumÆ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2(
&multi_head_attention_7/softmax/SoftmaxÌ
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2)
'multi_head_attention_7/dropout/Identity
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumø
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2-
+multi_head_attention_7/attention_output/add
IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp"^dense_21/Tensordot/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp"^dense_22/Tensordot/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp0^layer_normalization_12/batchnorm/ReadVariableOp4^layer_normalization_12/batchnorm/mul/ReadVariableOp0^layer_normalization_13/batchnorm/ReadVariableOp4^layer_normalization_13/batchnorm/mul/ReadVariableOp0^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2F
!dense_21/Tensordot/ReadVariableOp!dense_21/Tensordot/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2F
!dense_22/Tensordot/ReadVariableOp!dense_22/Tensordot/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2b
/layer_normalization_12/batchnorm/ReadVariableOp/layer_normalization_12/batchnorm/ReadVariableOp2j
3layer_normalization_12/batchnorm/mul/ReadVariableOp3layer_normalization_12/batchnorm/mul/ReadVariableOp2b
/layer_normalization_13/batchnorm/ReadVariableOp/layer_normalization_13/batchnorm/ReadVariableOp2j
3layer_normalization_13/batchnorm/mul/ReadVariableOp3layer_normalization_13/batchnorm/mul/ReadVariableOp2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_6/key/add/ReadVariableOp-multi_head_attention_6/key/add/ReadVariableOp2r
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/query/add/ReadVariableOp/multi_head_attention_6/query/add/ReadVariableOp2v
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/value/add/ReadVariableOp/multi_head_attention_6/value/add/ReadVariableOp2v
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,patch_encoder_7/embedding_3/embedding_lookup,patch_encoder_7/embedding_3/embedding_lookup:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
0

S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1122339	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmax
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/Identity¹
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue

¢
*__inference_model_11_layer_call_fn_1120606

inputs
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:1@
	unknown_5:@
	unknown_6:	¥@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:	@

unknown_20:	

unknown_21:	@

unknown_22:@

unknown_23:@

unknown_24:@ 

unknown_25:@@

unknown_26:@ 

unknown_27:@@

unknown_28:@ 

unknown_29:@@

unknown_30:@ 

unknown_31:@@

unknown_32:@
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*?
_read_only_resource_inputs!
	
 !"*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_11_layer_call_and_return_conditional_losses_11200772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ö
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_1122212

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
¬§
ç
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118792

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip¢!stateful_uniform_1/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1¡
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¼
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub³
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniformz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1§
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform_1/shapeu
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2
stateful_uniform_1/max~
stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform_1/Const¡
stateful_uniform_1/ProdProd!stateful_uniform_1/shape:output:0!stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/Prodx
stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/Cast/x
stateful_uniform_1/Cast_1Cast stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform_1/Cast_1
!stateful_uniform_1/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource"stateful_uniform_1/Cast/x:output:0stateful_uniform_1/Cast_1:y:0 ^stateful_uniform/RngReadAndSkip*
_output_shapes
:2#
!stateful_uniform_1/RngReadAndSkip
&stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&stateful_uniform_1/strided_slice/stack
(stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_1
(stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_2Ú
 stateful_uniform_1/strided_sliceStridedSlice)stateful_uniform_1/RngReadAndSkip:value:0/stateful_uniform_1/strided_slice/stack:output:01stateful_uniform_1/strided_slice/stack_1:output:01stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2"
 stateful_uniform_1/strided_slice
stateful_uniform_1/BitcastBitcast)stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcast
(stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice_1/stack¢
*stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_1¢
*stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_2Ò
"stateful_uniform_1/strided_slice_1StridedSlice)stateful_uniform_1/RngReadAndSkip:value:01stateful_uniform_1/strided_slice_1/stack:output:03stateful_uniform_1/strided_slice_1/stack_1:output:03stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2$
"stateful_uniform_1/strided_slice_1¥
stateful_uniform_1/Bitcast_1Bitcast+stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcast_1¤
/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :21
/stateful_uniform_1/StatelessRandomUniformV2/algÈ
+stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2!stateful_uniform_1/shape:output:0%stateful_uniform_1/Bitcast_1:output:0#stateful_uniform_1/Bitcast:output:08stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stateful_uniform_1/StatelessRandomUniformV2
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub»
stateful_uniform_1/mulMul4stateful_uniform_1/StatelessRandomUniformV2:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform_1/mul 
stateful_uniform_1AddV2stateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2stateful_uniform_1:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shape
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stack
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ª
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zoom_matrix/strided_slicek
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub/yr
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/subs
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv/y
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stack
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ñ
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_1o
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub_1/x£
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_1
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub_2/yv
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/sub_2w
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv_1/y
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stack
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ñ
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_2o
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub_3/x£
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_3
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mul_1
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stack
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ñ
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_3z
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/packed/1³
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros/packedw
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros/Const¥
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1¹
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_1/packed{
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_1/Const­
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_1
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stack
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ñ
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_4~
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/packed/1¹
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_2/packed{
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_2/Const­
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisá
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_valueÄ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity
NoOpNoOp ^stateful_uniform/RngReadAndSkip"^stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip2F
!stateful_uniform_1/RngReadAndSkip!stateful_uniform_1/RngReadAndSkip:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
¡
f
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1122409

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
Ã9

S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1122382	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const¨
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÖ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yè
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/GreaterEqual¡
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Cast¤
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mul_1¹
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue


*__inference_dense_23_layer_call_fn_1122147

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_11194032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs

ý
8__inference_multi_head_attention_6_layer_call_fn_1121918	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity

identity_1¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_11192592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
£
¡
8__inference_layer_normalization_14_layer_call_fn_1122233

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_11194462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
¬§
ç
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1122743

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip¢!stateful_uniform_1/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1¡
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¼
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub³
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniformz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1§
stateful_uniform_1/shapePackstrided_slice:output:0#stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform_1/shapeu
stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2
stateful_uniform_1/max~
stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform_1/Const¡
stateful_uniform_1/ProdProd!stateful_uniform_1/shape:output:0!stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/Prodx
stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/Cast/x
stateful_uniform_1/Cast_1Cast stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform_1/Cast_1
!stateful_uniform_1/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource"stateful_uniform_1/Cast/x:output:0stateful_uniform_1/Cast_1:y:0 ^stateful_uniform/RngReadAndSkip*
_output_shapes
:2#
!stateful_uniform_1/RngReadAndSkip
&stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&stateful_uniform_1/strided_slice/stack
(stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_1
(stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_2Ú
 stateful_uniform_1/strided_sliceStridedSlice)stateful_uniform_1/RngReadAndSkip:value:0/stateful_uniform_1/strided_slice/stack:output:01stateful_uniform_1/strided_slice/stack_1:output:01stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2"
 stateful_uniform_1/strided_slice
stateful_uniform_1/BitcastBitcast)stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcast
(stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice_1/stack¢
*stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_1¢
*stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_2Ò
"stateful_uniform_1/strided_slice_1StridedSlice)stateful_uniform_1/RngReadAndSkip:value:01stateful_uniform_1/strided_slice_1/stack:output:03stateful_uniform_1/strided_slice_1/stack_1:output:03stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2$
"stateful_uniform_1/strided_slice_1¥
stateful_uniform_1/Bitcast_1Bitcast+stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcast_1¤
/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :21
/stateful_uniform_1/StatelessRandomUniformV2/algÈ
+stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2!stateful_uniform_1/shape:output:0%stateful_uniform_1/Bitcast_1:output:0#stateful_uniform_1/Bitcast:output:08stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stateful_uniform_1/StatelessRandomUniformV2
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub»
stateful_uniform_1/mulMul4stateful_uniform_1/StatelessRandomUniformV2:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform_1/mul 
stateful_uniform_1AddV2stateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2stateful_uniform_1:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shape
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stack
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ª
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zoom_matrix/strided_slicek
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub/yr
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/subs
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv/y
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stack
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ñ
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_1o
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub_1/x£
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_1
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub_2/yv
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/sub_2w
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv_1/y
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stack
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ñ
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_2o
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
zoom_matrix/sub_3/x£
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_3
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mul_1
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stack
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ñ
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_3z
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/packed/1³
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros/packedw
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros/Const¥
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1¹
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_1/packed{
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_1/Const­
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_1
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stack
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ñ
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_4~
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/packed/1¹
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_2/packed{
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_2/Const­
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisá
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_valueÄ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity
NoOpNoOp ^stateful_uniform/RngReadAndSkip"^stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip2F
!stateful_uniform_1/RngReadAndSkip!stateful_uniform_1/RngReadAndSkip:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs


Ó
3__inference_data_augmentation_layer_call_fn_1121436

inputs
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11190372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
ï'
þ
E__inference_dense_22_layer_call_and_return_conditional_losses_1122111

inputs4
!tensordot_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xz
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
Gelu/Cast/x
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/mul_1o
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
æ'
ý
E__inference_dense_23_layer_call_and_return_conditional_losses_1119403

inputs4
!tensordot_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xy
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
Gelu/Cast/x
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/mul_1n
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs


*__inference_dense_22_layer_call_fn_1122073

inputs
unknown:	@
	unknown_0:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_11193522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
Ø
Ç
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1122601

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ ¾2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ >2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¸
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¯
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_valueÈ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
þ

1__inference_patch_encoder_7_layer_call_fn_1121849

projection
unknown:	¥@
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCall
projectionunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11191912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
$
_user_specified_name
projection
é
m
C__inference_add_13_layer_call_and_return_conditional_losses_1119422

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs:TP
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
ü
b
F__inference_patches_7_layer_call_and_return_conditional_losses_1121764

images
identityÓ
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksizes
*
paddingVALID*
rates
*
strides
2
ExtractImagePatchess
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ¤  1   2
Reshape/shape
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameimages
0

S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1119488	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmax
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/Identity¹
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
¼
e
,__inference_dropout_19_layer_call_fn_1122195

inputs
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_11197022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
÷
O
3__inference_random_rotation_3_layer_call_fn_1122472

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11186422
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
£
¡
8__inference_layer_normalization_13_layer_call_fn_1122042

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_11193082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
M

 __inference__traced_save_1122888
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop;
7savev2_layer_normalization_12_gamma_read_readvariableop:
6savev2_layer_normalization_12_beta_read_readvariableop;
7savev2_layer_normalization_13_gamma_read_readvariableop:
6savev2_layer_normalization_13_beta_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop;
7savev2_layer_normalization_14_gamma_read_readvariableop:
6savev2_layer_normalization_14_beta_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	E
Asavev2_patch_encoder_7_embedding_3_embeddings_read_readvariableopB
>savev2_multi_head_attention_6_query_kernel_read_readvariableop@
<savev2_multi_head_attention_6_query_bias_read_readvariableop@
<savev2_multi_head_attention_6_key_kernel_read_readvariableop>
:savev2_multi_head_attention_6_key_bias_read_readvariableopB
>savev2_multi_head_attention_6_value_kernel_read_readvariableop@
<savev2_multi_head_attention_6_value_bias_read_readvariableopM
Isavev2_multi_head_attention_6_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_6_attention_output_bias_read_readvariableopB
>savev2_multi_head_attention_7_query_kernel_read_readvariableop@
<savev2_multi_head_attention_7_query_bias_read_readvariableop@
<savev2_multi_head_attention_7_key_kernel_read_readvariableop>
:savev2_multi_head_attention_7_key_bias_read_readvariableopB
>savev2_multi_head_attention_7_value_kernel_read_readvariableop@
<savev2_multi_head_attention_7_value_bias_read_readvariableopM
Isavev2_multi_head_attention_7_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_7_attention_output_bias_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	)
%savev2_variable_2_read_readvariableop	
savev2_const_2

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename»
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Í
valueÃBÀ$B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÐ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesæ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop7savev2_layer_normalization_12_gamma_read_readvariableop6savev2_layer_normalization_12_beta_read_readvariableop7savev2_layer_normalization_13_gamma_read_readvariableop6savev2_layer_normalization_13_beta_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop7savev2_layer_normalization_14_gamma_read_readvariableop6savev2_layer_normalization_14_beta_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableopAsavev2_patch_encoder_7_embedding_3_embeddings_read_readvariableop>savev2_multi_head_attention_6_query_kernel_read_readvariableop<savev2_multi_head_attention_6_query_bias_read_readvariableop<savev2_multi_head_attention_6_key_kernel_read_readvariableop:savev2_multi_head_attention_6_key_bias_read_readvariableop>savev2_multi_head_attention_6_value_kernel_read_readvariableop<savev2_multi_head_attention_6_value_bias_read_readvariableopIsavev2_multi_head_attention_6_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_6_attention_output_bias_read_readvariableop>savev2_multi_head_attention_7_query_kernel_read_readvariableop<savev2_multi_head_attention_7_query_bias_read_readvariableop<savev2_multi_head_attention_7_key_kernel_read_readvariableop:savev2_multi_head_attention_7_key_bias_read_readvariableop>savev2_multi_head_attention_7_value_kernel_read_readvariableop<savev2_multi_head_attention_7_value_bias_read_readvariableopIsavev2_multi_head_attention_7_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_7_attention_output_bias_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$				2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Ó
_input_shapesÁ
¾: :1@:@:@:@:@:@:	@::	@:@:@:@::: :	¥@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:1@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::%	!

_output_shapes
:	@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	¥@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
::$

_output_shapes
: 

e
G__inference_dropout_19_layer_call_and_return_conditional_losses_1119414

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs

e
G__inference_dropout_19_layer_call_and_return_conditional_losses_1122200

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
Ã9

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1122021	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const¨
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÖ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yè
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/GreaterEqual¡
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Cast¤
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mul_1¹
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
Üf

J__inference_random_flip_3_layer_call_and_return_conditional_losses_1118994

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_like
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceÔ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi25
3stateless_random_flip_left_right/control_dependency¼
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¶
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¨
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceñ
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/max
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgÊ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÓ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=stateless_random_flip_left_right/stateless_random_uniform/mul¸
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9stateless_random_flip_left_right/stateless_random_uniform¦
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1¦
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2¦
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shape
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stateless_random_flip_left_right/ReshapeÆ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/Round¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2,
*stateless_random_flip_left_right/ReverseV2ï
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2&
$stateless_random_flip_left_right/mul
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&stateless_random_flip_left_right/sub/xê
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/subú
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2(
&stateless_random_flip_left_right/mul_1æ
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2&
$stateless_random_flip_left_right/add
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity¤
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs

£
*__inference_model_11_layer_call_fn_1120221
input_4
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:1@
	unknown_5:@
	unknown_6:	¥@
	unknown_7:@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:	@

unknown_20:	

unknown_21:	@

unknown_22:@

unknown_23:@

unknown_24:@ 

unknown_25:@@

unknown_26:@ 

unknown_27:@@

unknown_28:@ 

unknown_29:@@

unknown_30:@ 

unknown_31:@@

unknown_32:@
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*?
_read_only_resource_inputs!
	
 !"*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_11_layer_call_and_return_conditional_losses_11200772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ó
F
*__inference_lambda_3_layer_call_fn_1121813

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11198982
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1119898

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs
Î
¢
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119101
normalization_3_input
normalization_3_sub_y
normalization_3_sqrt_x#
random_flip_3_1119091:	'
random_rotation_3_1119094:	#
random_zoom_3_1119097:	
identity¢%random_flip_3/StatefulPartitionedCall¢)random_rotation_3/StatefulPartitionedCall¢%random_zoom_3/StatefulPartitionedCall
normalization_3/subSubnormalization_3_inputnormalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/sub}
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*&
_output_shapes
:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_3/Maximum/y¬
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum¯
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/truedivü
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11186302
resizing_3/PartitionedCall½
%random_flip_3/StatefulPartitionedCallStatefulPartitionedCall#resizing_3/PartitionedCall:output:0random_flip_3_1119091*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11189942'
%random_flip_3/StatefulPartitionedCallØ
)random_rotation_3/StatefulPartitionedCallStatefulPartitionedCall.random_flip_3/StatefulPartitionedCall:output:0random_rotation_3_1119094*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11189232+
)random_rotation_3/StatefulPartitionedCallÌ
%random_zoom_3/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_3/StatefulPartitionedCall:output:0random_zoom_3_1119097*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11187922'
%random_zoom_3/StatefulPartitionedCall
IdentityIdentity.random_zoom_3/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

IdentityÊ
NoOpNoOp&^random_flip_3/StatefulPartitionedCall*^random_rotation_3/StatefulPartitionedCall&^random_zoom_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ::: : : 2N
%random_flip_3/StatefulPartitionedCall%random_flip_3/StatefulPartitionedCall2V
)random_rotation_3/StatefulPartitionedCall)random_rotation_3/StatefulPartitionedCall2N
%random_zoom_3/StatefulPartitionedCall%random_zoom_3/StatefulPartitionedCall:h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:


S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_1122064

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesÀ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
«
k
3__inference_data_augmentation_layer_call_fn_1121421

inputs
unknown
	unknown_0
identityî
PartitionedCallPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11186512
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
0

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1119259	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmax
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/Identity¹
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
æ'
ý
E__inference_dense_23_layer_call_and_return_conditional_losses_1122185

inputs4
!tensordot_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xy
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
Gelu/Cast/x
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Gelu/mul_1n
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs
ï
K
/__inference_random_zoom_3_layer_call_fn_1122606

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11186482
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs

ý
8__inference_multi_head_attention_7_layer_call_fn_1122303	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity

identity_1¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_11196442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
Ø
Ç
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1118923

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ ¾2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ >2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¸
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¯
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_valueÈ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
ç
¡
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1121449

inputs
normalization_3_sub_y
normalization_3_sqrt_x
identity
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/sub}
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*&
_output_shapes
:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_3/Maximum/y¬
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum¯
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/truediv
resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   2
resizing_3/resize/sizeé
 resizing_3/resize/ResizeBilinearResizeBilinearnormalization_3/truediv:z:0resizing_3/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(2"
 resizing_3/resize/ResizeBilinear
IdentityIdentity1resizing_3/resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
òÏ
Ù$
E__inference_model_11_layer_call_and_return_conditional_losses_1121412

inputs+
'data_augmentation_normalization_3_sub_y,
(data_augmentation_normalization_3_sqrt_x_
Qdata_augmentation_random_flip_3_stateful_uniform_full_int_rngreadandskip_resource:	Z
Ldata_augmentation_random_rotation_3_stateful_uniform_rngreadandskip_resource:	V
Hdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resource:	<
*dense_21_tensordot_readvariableop_resource:1@6
(dense_21_biasadd_readvariableop_resource:@G
4patch_encoder_7_embedding_3_embedding_lookup_1121191:	¥@J
<layer_normalization_12_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_12_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_6_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_query_add_readvariableop_resource:@V
@multi_head_attention_6_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_6_key_add_readvariableop_resource:@X
Bmulti_head_attention_6_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_value_add_readvariableop_resource:@c
Mmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_6_attention_output_add_readvariableop_resource:@J
<layer_normalization_13_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_13_batchnorm_readvariableop_resource:@=
*dense_22_tensordot_readvariableop_resource:	@7
(dense_22_biasadd_readvariableop_resource:	=
*dense_23_tensordot_readvariableop_resource:	@6
(dense_23_biasadd_readvariableop_resource:@J
<layer_normalization_14_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_14_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_7_query_add_readvariableop_resource:@V
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_7_key_add_readvariableop_resource:@X
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_7_value_add_readvariableop_resource:@c
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_7_attention_output_add_readvariableop_resource:@
identity¢Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip¢odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip¢?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip¢Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip¢dense_21/BiasAdd/ReadVariableOp¢!dense_21/Tensordot/ReadVariableOp¢dense_22/BiasAdd/ReadVariableOp¢!dense_22/Tensordot/ReadVariableOp¢dense_23/BiasAdd/ReadVariableOp¢!dense_23/Tensordot/ReadVariableOp¢/layer_normalization_12/batchnorm/ReadVariableOp¢3layer_normalization_12/batchnorm/mul/ReadVariableOp¢/layer_normalization_13/batchnorm/ReadVariableOp¢3layer_normalization_13/batchnorm/mul/ReadVariableOp¢/layer_normalization_14/batchnorm/ReadVariableOp¢3layer_normalization_14/batchnorm/mul/ReadVariableOp¢:multi_head_attention_6/attention_output/add/ReadVariableOp¢Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_6/key/add/ReadVariableOp¢7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/query/add/ReadVariableOp¢9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/value/add/ReadVariableOp¢9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp¢:multi_head_attention_7/attention_output/add/ReadVariableOp¢Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_7/key/add/ReadVariableOp¢7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/query/add/ReadVariableOp¢9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_7/value/add/ReadVariableOp¢9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢,patch_encoder_7/embedding_3/embedding_lookupÂ
%data_augmentation/normalization_3/subSubinputs'data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%data_augmentation/normalization_3/sub³
&data_augmentation/normalization_3/SqrtSqrt(data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:2(
&data_augmentation/normalization_3/Sqrt
+data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32-
+data_augmentation/normalization_3/Maximum/yô
)data_augmentation/normalization_3/MaximumMaximum*data_augmentation/normalization_3/Sqrt:y:04data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2+
)data_augmentation/normalization_3/Maximum÷
)data_augmentation/normalization_3/truedivRealDiv)data_augmentation/normalization_3/sub:z:0-data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)data_augmentation/normalization_3/truediv¥
(data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   2*
(data_augmentation/resizing_3/resize/size±
2data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear-data_augmentation/normalization_3/truediv:z:01data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(24
2data_augmentation/resizing_3/resize/ResizeBilinearÌ
?data_augmentation/random_flip_3/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2A
?data_augmentation/random_flip_3/stateful_uniform_full_int/shapeÌ
?data_augmentation/random_flip_3/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?data_augmentation/random_flip_3/stateful_uniform_full_int/Const½
>data_augmentation/random_flip_3/stateful_uniform_full_int/ProdProdHdata_augmentation/random_flip_3/stateful_uniform_full_int/shape:output:0Hdata_augmentation/random_flip_3/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2@
>data_augmentation/random_flip_3/stateful_uniform_full_int/ProdÆ
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2B
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast/x
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast_1CastGdata_augmentation/random_flip_3/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast_1¦
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipQdata_augmentation_random_flip_3_stateful_uniform_full_int_rngreadandskip_resourceIdata_augmentation/random_flip_3/stateful_uniform_full_int/Cast/x:output:0Ddata_augmentation/random_flip_3/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2J
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipè
Mdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stackì
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_1ì
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_2Ä
Gdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_sliceStridedSlicePdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Vdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack:output:0Xdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_1:output:0Xdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2I
Gdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice
Adata_augmentation/random_flip_3/stateful_uniform_full_int/BitcastBitcastPdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02C
Adata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcastì
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stackð
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1ð
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2¼
Idata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1StridedSlicePdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Xdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack:output:0Zdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Zdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2K
Idata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1
Cdata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast_1BitcastRdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02E
Cdata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast_1À
=data_augmentation/random_flip_3/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=data_augmentation/random_flip_3/stateful_uniform_full_int/algî
9data_augmentation/random_flip_3/stateful_uniform_full_intStatelessRandomUniformFullIntV2Hdata_augmentation/random_flip_3/stateful_uniform_full_int/shape:output:0Ldata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast_1:output:0Jdata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast:output:0Fdata_augmentation/random_flip_3/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2;
9data_augmentation/random_flip_3/stateful_uniform_full_int¢
*data_augmentation/random_flip_3/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2,
*data_augmentation/random_flip_3/zeros_like
%data_augmentation/random_flip_3/stackPackBdata_augmentation/random_flip_3/stateful_uniform_full_int:output:03data_augmentation/random_flip_3/zeros_like:output:0*
N*
T0	*
_output_shapes

:2'
%data_augmentation/random_flip_3/stack»
3data_augmentation/random_flip_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3data_augmentation/random_flip_3/strided_slice/stack¿
5data_augmentation/random_flip_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5data_augmentation/random_flip_3/strided_slice/stack_1¿
5data_augmentation/random_flip_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5data_augmentation/random_flip_3/strided_slice/stack_2È
-data_augmentation/random_flip_3/strided_sliceStridedSlice.data_augmentation/random_flip_3/stack:output:0<data_augmentation/random_flip_3/strided_slice/stack:output:0>data_augmentation/random_flip_3/strided_slice/stack_1:output:0>data_augmentation/random_flip_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2/
-data_augmentation/random_flip_3/strided_sliceý
Sdata_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependencyIdentityCdata_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
T0*E
_class;
97loc:@data_augmentation/resizing_3/resize/ResizeBilinear*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2U
Sdata_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/ShapeShape\data_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/Shapeö
Tdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stackú
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_1ú
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_2è
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_sliceStridedSliceOdata_augmentation/random_flip_3/stateless_random_flip_left_right/Shape:output:0]data_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack:output:0_data_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_1:output:0_data_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2P
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_sliceÑ
_data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shapePackWdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2a
_data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shape
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/maxê
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter6data_augmentation/random_flip_3/strided_slice:output:0* 
_output_shapes
::2x
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgw^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2q
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg
rdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2hdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0|data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0udata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
rdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subSubfdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/max:output:0fdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subÓ
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mulMul{data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0adata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mul¸
Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniformAddV2adata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0fdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniformæ
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2R
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/1æ
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2R
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/2æ
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2R
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/3À
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shapePackWdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice:output:0Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/1:output:0Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/2:output:0Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2P
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape
Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReshapeReshape]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform:z:0Wdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape¦
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/RoundRoundQdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/Roundì
Odata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2/axis
Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2	ReverseV2\data_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency:output:0Xdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2L
Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2ï
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/mulMulJdata_augmentation/random_flip_3/stateless_random_flip_left_right/Round:y:0Sdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2F
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/mulÕ
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub/xê
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/subSubOdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub/x:output:0Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/subú
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul_1MulHdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub:z:0\data_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul_1æ
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/addAddV2Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul:z:0Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2F
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/addÎ
)data_augmentation/random_rotation_3/ShapeShapeHdata_augmentation/random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2+
)data_augmentation/random_rotation_3/Shape¼
7data_augmentation/random_rotation_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7data_augmentation/random_rotation_3/strided_slice/stackÀ
9data_augmentation/random_rotation_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9data_augmentation/random_rotation_3/strided_slice/stack_1À
9data_augmentation/random_rotation_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9data_augmentation/random_rotation_3/strided_slice/stack_2º
1data_augmentation/random_rotation_3/strided_sliceStridedSlice2data_augmentation/random_rotation_3/Shape:output:0@data_augmentation/random_rotation_3/strided_slice/stack:output:0Bdata_augmentation/random_rotation_3/strided_slice/stack_1:output:0Bdata_augmentation/random_rotation_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1data_augmentation/random_rotation_3/strided_sliceÉ
9data_augmentation/random_rotation_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/strided_slice_1/stackÍ
;data_augmentation/random_rotation_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2=
;data_augmentation/random_rotation_3/strided_slice_1/stack_1Ä
;data_augmentation/random_rotation_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;data_augmentation/random_rotation_3/strided_slice_1/stack_2Ä
3data_augmentation/random_rotation_3/strided_slice_1StridedSlice2data_augmentation/random_rotation_3/Shape:output:0Bdata_augmentation/random_rotation_3/strided_slice_1/stack:output:0Ddata_augmentation/random_rotation_3/strided_slice_1/stack_1:output:0Ddata_augmentation/random_rotation_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3data_augmentation/random_rotation_3/strided_slice_1Ê
(data_augmentation/random_rotation_3/CastCast<data_augmentation/random_rotation_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(data_augmentation/random_rotation_3/CastÉ
9data_augmentation/random_rotation_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/strided_slice_2/stackÍ
;data_augmentation/random_rotation_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2=
;data_augmentation/random_rotation_3/strided_slice_2/stack_1Ä
;data_augmentation/random_rotation_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;data_augmentation/random_rotation_3/strided_slice_2/stack_2Ä
3data_augmentation/random_rotation_3/strided_slice_2StridedSlice2data_augmentation/random_rotation_3/Shape:output:0Bdata_augmentation/random_rotation_3/strided_slice_2/stack:output:0Ddata_augmentation/random_rotation_3/strided_slice_2/stack_1:output:0Ddata_augmentation/random_rotation_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3data_augmentation/random_rotation_3/strided_slice_2Î
*data_augmentation/random_rotation_3/Cast_1Cast<data_augmentation/random_rotation_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*data_augmentation/random_rotation_3/Cast_1ê
:data_augmentation/random_rotation_3/stateful_uniform/shapePack:data_augmentation/random_rotation_3/strided_slice:output:0*
N*
T0*
_output_shapes
:2<
:data_augmentation/random_rotation_3/stateful_uniform/shape¹
8data_augmentation/random_rotation_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ ¾2:
8data_augmentation/random_rotation_3/stateful_uniform/min¹
8data_augmentation/random_rotation_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ >2:
8data_augmentation/random_rotation_3/stateful_uniform/maxÂ
:data_augmentation/random_rotation_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:data_augmentation/random_rotation_3/stateful_uniform/Const©
9data_augmentation/random_rotation_3/stateful_uniform/ProdProdCdata_augmentation/random_rotation_3/stateful_uniform/shape:output:0Cdata_augmentation/random_rotation_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/stateful_uniform/Prod¼
;data_augmentation/random_rotation_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2=
;data_augmentation/random_rotation_3/stateful_uniform/Cast/xö
;data_augmentation/random_rotation_3/stateful_uniform/Cast_1CastBdata_augmentation/random_rotation_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;data_augmentation/random_rotation_3/stateful_uniform/Cast_1
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipRngReadAndSkipLdata_augmentation_random_rotation_3_stateful_uniform_rngreadandskip_resourceDdata_augmentation/random_rotation_3/stateful_uniform/Cast/x:output:0?data_augmentation/random_rotation_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:2E
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipÞ
Hdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stackâ
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_1â
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_2¦
Bdata_augmentation/random_rotation_3/stateful_uniform/strided_sliceStridedSliceKdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip:value:0Qdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack:output:0Sdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_1:output:0Sdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2D
Bdata_augmentation/random_rotation_3/stateful_uniform/strided_slice
<data_augmentation/random_rotation_3/stateful_uniform/BitcastBitcastKdata_augmentation/random_rotation_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02>
<data_augmentation/random_rotation_3/stateful_uniform/Bitcastâ
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stackæ
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_1æ
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_2
Ddata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1StridedSliceKdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip:value:0Sdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack:output:0Udata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_1:output:0Udata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2F
Ddata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1
>data_augmentation/random_rotation_3/stateful_uniform/Bitcast_1BitcastMdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02@
>data_augmentation/random_rotation_3/stateful_uniform/Bitcast_1è
Qdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2S
Qdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2/alg
Mdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Cdata_augmentation/random_rotation_3/stateful_uniform/shape:output:0Gdata_augmentation/random_rotation_3/stateful_uniform/Bitcast_1:output:0Edata_augmentation/random_rotation_3/stateful_uniform/Bitcast:output:0Zdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
Mdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2¢
8data_augmentation/random_rotation_3/stateful_uniform/subSubAdata_augmentation/random_rotation_3/stateful_uniform/max:output:0Adata_augmentation/random_rotation_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2:
8data_augmentation/random_rotation_3/stateful_uniform/sub¿
8data_augmentation/random_rotation_3/stateful_uniform/mulMulVdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2:output:0<data_augmentation/random_rotation_3/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8data_augmentation/random_rotation_3/stateful_uniform/mul¤
4data_augmentation/random_rotation_3/stateful_uniformAddV2<data_augmentation/random_rotation_3/stateful_uniform/mul:z:0Adata_augmentation/random_rotation_3/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4data_augmentation/random_rotation_3/stateful_uniform»
9data_augmentation/random_rotation_3/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9data_augmentation/random_rotation_3/rotation_matrix/sub/y
7data_augmentation/random_rotation_3/rotation_matrix/subSub.data_augmentation/random_rotation_3/Cast_1:y:0Bdata_augmentation/random_rotation_3/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 29
7data_augmentation/random_rotation_3/rotation_matrix/subá
7data_augmentation/random_rotation_3/rotation_matrix/CosCos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7data_augmentation/random_rotation_3/rotation_matrix/Cos¿
;data_augmentation/random_rotation_3/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_1/y
9data_augmentation/random_rotation_3/rotation_matrix/sub_1Sub.data_augmentation/random_rotation_3/Cast_1:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_1£
7data_augmentation/random_rotation_3/rotation_matrix/mulMul;data_augmentation/random_rotation_3/rotation_matrix/Cos:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7data_augmentation/random_rotation_3/rotation_matrix/mulá
7data_augmentation/random_rotation_3/rotation_matrix/SinSin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7data_augmentation/random_rotation_3/rotation_matrix/Sin¿
;data_augmentation/random_rotation_3/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_2/y
9data_augmentation/random_rotation_3/rotation_matrix/sub_2Sub,data_augmentation/random_rotation_3/Cast:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_2§
9data_augmentation/random_rotation_3/rotation_matrix/mul_1Mul;data_augmentation/random_rotation_3/rotation_matrix/Sin:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/mul_1§
9data_augmentation/random_rotation_3/rotation_matrix/sub_3Sub;data_augmentation/random_rotation_3/rotation_matrix/mul:z:0=data_augmentation/random_rotation_3/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_3§
9data_augmentation/random_rotation_3/rotation_matrix/sub_4Sub;data_augmentation/random_rotation_3/rotation_matrix/sub:z:0=data_augmentation/random_rotation_3/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_4Ã
=data_augmentation/random_rotation_3/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2?
=data_augmentation/random_rotation_3/rotation_matrix/truediv/yº
;data_augmentation/random_rotation_3/rotation_matrix/truedivRealDiv=data_augmentation/random_rotation_3/rotation_matrix/sub_4:z:0Fdata_augmentation/random_rotation_3/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;data_augmentation/random_rotation_3/rotation_matrix/truediv¿
;data_augmentation/random_rotation_3/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_5/y
9data_augmentation/random_rotation_3/rotation_matrix/sub_5Sub,data_augmentation/random_rotation_3/Cast:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_5å
9data_augmentation/random_rotation_3/rotation_matrix/Sin_1Sin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/Sin_1¿
;data_augmentation/random_rotation_3/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_6/y
9data_augmentation/random_rotation_3/rotation_matrix/sub_6Sub.data_augmentation/random_rotation_3/Cast_1:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_6©
9data_augmentation/random_rotation_3/rotation_matrix/mul_2Mul=data_augmentation/random_rotation_3/rotation_matrix/Sin_1:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/mul_2å
9data_augmentation/random_rotation_3/rotation_matrix/Cos_1Cos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/Cos_1¿
;data_augmentation/random_rotation_3/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_7/y
9data_augmentation/random_rotation_3/rotation_matrix/sub_7Sub,data_augmentation/random_rotation_3/Cast:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_7©
9data_augmentation/random_rotation_3/rotation_matrix/mul_3Mul=data_augmentation/random_rotation_3/rotation_matrix/Cos_1:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/mul_3§
7data_augmentation/random_rotation_3/rotation_matrix/addAddV2=data_augmentation/random_rotation_3/rotation_matrix/mul_2:z:0=data_augmentation/random_rotation_3/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7data_augmentation/random_rotation_3/rotation_matrix/add§
9data_augmentation/random_rotation_3/rotation_matrix/sub_8Sub=data_augmentation/random_rotation_3/rotation_matrix/sub_5:z:0;data_augmentation/random_rotation_3/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_8Ç
?data_augmentation/random_rotation_3/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2A
?data_augmentation/random_rotation_3/rotation_matrix/truediv_1/yÀ
=data_augmentation/random_rotation_3/rotation_matrix/truediv_1RealDiv=data_augmentation/random_rotation_3/rotation_matrix/sub_8:z:0Hdata_augmentation/random_rotation_3/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=data_augmentation/random_rotation_3/rotation_matrix/truediv_1Þ
9data_augmentation/random_rotation_3/rotation_matrix/ShapeShape8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*
_output_shapes
:2;
9data_augmentation/random_rotation_3/rotation_matrix/ShapeÜ
Gdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stackà
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_1à
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_2
Adata_augmentation/random_rotation_3/rotation_matrix/strided_sliceStridedSliceBdata_augmentation/random_rotation_3/rotation_matrix/Shape:output:0Pdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack:output:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_1:output:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Adata_augmentation/random_rotation_3/rotation_matrix/strided_sliceå
9data_augmentation/random_rotation_3/rotation_matrix/Cos_2Cos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/Cos_2ç
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stackë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_1ë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_2Ï
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Cos_2:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1å
9data_augmentation/random_rotation_3/rotation_matrix/Sin_2Sin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/Sin_2ç
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stackë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_1ë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_2Ï
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Sin_2:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2ù
7data_augmentation/random_rotation_3/rotation_matrix/NegNegLdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7data_augmentation/random_rotation_3/rotation_matrix/Negç
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stackë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_1ë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_2Ñ
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3StridedSlice?data_augmentation/random_rotation_3/rotation_matrix/truediv:z:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3å
9data_augmentation/random_rotation_3/rotation_matrix/Sin_3Sin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/Sin_3ç
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stackë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_1ë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_2Ï
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Sin_3:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4å
9data_augmentation/random_rotation_3/rotation_matrix/Cos_3Cos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/Cos_3ç
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stackë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_1ë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_2Ï
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Cos_3:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5ç
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stackë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_1ë
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_2Ó
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6StridedSliceAdata_augmentation/random_rotation_3/rotation_matrix/truediv_1:z:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6Ê
Bdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed/1Ó
@data_augmentation/random_rotation_3/rotation_matrix/zeros/packedPackJdata_augmentation/random_rotation_3/rotation_matrix/strided_slice:output:0Kdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@data_augmentation/random_rotation_3/rotation_matrix/zeros/packedÇ
?data_augmentation/random_rotation_3/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?data_augmentation/random_rotation_3/rotation_matrix/zeros/ConstÅ
9data_augmentation/random_rotation_3/rotation_matrix/zerosFillIdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed:output:0Hdata_augmentation/random_rotation_3/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9data_augmentation/random_rotation_3/rotation_matrix/zerosÄ
?data_augmentation/random_rotation_3/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2A
?data_augmentation/random_rotation_3/rotation_matrix/concat/axis
:data_augmentation/random_rotation_3/rotation_matrix/concatConcatV2Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1:output:0;data_augmentation/random_rotation_3/rotation_matrix/Neg:y:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3:output:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4:output:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5:output:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6:output:0Bdata_augmentation/random_rotation_3/rotation_matrix/zeros:output:0Hdata_augmentation/random_rotation_3/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:data_augmentation/random_rotation_3/rotation_matrix/concatâ
3data_augmentation/random_rotation_3/transform/ShapeShapeHdata_augmentation/random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:25
3data_augmentation/random_rotation_3/transform/ShapeÐ
Adata_augmentation/random_rotation_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Adata_augmentation/random_rotation_3/transform/strided_slice/stackÔ
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_1Ô
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_2â
;data_augmentation/random_rotation_3/transform/strided_sliceStridedSlice<data_augmentation/random_rotation_3/transform/Shape:output:0Jdata_augmentation/random_rotation_3/transform/strided_slice/stack:output:0Ldata_augmentation/random_rotation_3/transform/strided_slice/stack_1:output:0Ldata_augmentation/random_rotation_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2=
;data_augmentation/random_rotation_3/transform/strided_slice¹
8data_augmentation/random_rotation_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8data_augmentation/random_rotation_3/transform/fill_value¾
Hdata_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/add:z:0Cdata_augmentation/random_rotation_3/rotation_matrix/concat:output:0Ddata_augmentation/random_rotation_3/transform/strided_slice:output:0Adata_augmentation/random_rotation_3/transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2J
Hdata_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3Û
%data_augmentation/random_zoom_3/ShapeShape]data_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2'
%data_augmentation/random_zoom_3/Shape´
3data_augmentation/random_zoom_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3data_augmentation/random_zoom_3/strided_slice/stack¸
5data_augmentation/random_zoom_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5data_augmentation/random_zoom_3/strided_slice/stack_1¸
5data_augmentation/random_zoom_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5data_augmentation/random_zoom_3/strided_slice/stack_2¢
-data_augmentation/random_zoom_3/strided_sliceStridedSlice.data_augmentation/random_zoom_3/Shape:output:0<data_augmentation/random_zoom_3/strided_slice/stack:output:0>data_augmentation/random_zoom_3/strided_slice/stack_1:output:0>data_augmentation/random_zoom_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-data_augmentation/random_zoom_3/strided_sliceÁ
5data_augmentation/random_zoom_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ27
5data_augmentation/random_zoom_3/strided_slice_1/stackÅ
7data_augmentation/random_zoom_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ29
7data_augmentation/random_zoom_3/strided_slice_1/stack_1¼
7data_augmentation/random_zoom_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7data_augmentation/random_zoom_3/strided_slice_1/stack_2¬
/data_augmentation/random_zoom_3/strided_slice_1StridedSlice.data_augmentation/random_zoom_3/Shape:output:0>data_augmentation/random_zoom_3/strided_slice_1/stack:output:0@data_augmentation/random_zoom_3/strided_slice_1/stack_1:output:0@data_augmentation/random_zoom_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/data_augmentation/random_zoom_3/strided_slice_1¾
$data_augmentation/random_zoom_3/CastCast8data_augmentation/random_zoom_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$data_augmentation/random_zoom_3/CastÁ
5data_augmentation/random_zoom_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ27
5data_augmentation/random_zoom_3/strided_slice_2/stackÅ
7data_augmentation/random_zoom_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ29
7data_augmentation/random_zoom_3/strided_slice_2/stack_1¼
7data_augmentation/random_zoom_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7data_augmentation/random_zoom_3/strided_slice_2/stack_2¬
/data_augmentation/random_zoom_3/strided_slice_2StridedSlice.data_augmentation/random_zoom_3/Shape:output:0>data_augmentation/random_zoom_3/strided_slice_2/stack:output:0@data_augmentation/random_zoom_3/strided_slice_2/stack_1:output:0@data_augmentation/random_zoom_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/data_augmentation/random_zoom_3/strided_slice_2Â
&data_augmentation/random_zoom_3/Cast_1Cast8data_augmentation/random_zoom_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&data_augmentation/random_zoom_3/Cast_1¶
8data_augmentation/random_zoom_3/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8data_augmentation/random_zoom_3/stateful_uniform/shape/1¡
6data_augmentation/random_zoom_3/stateful_uniform/shapePack6data_augmentation/random_zoom_3/strided_slice:output:0Adata_augmentation/random_zoom_3/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:28
6data_augmentation/random_zoom_3/stateful_uniform/shape±
4data_augmentation/random_zoom_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?26
4data_augmentation/random_zoom_3/stateful_uniform/min±
4data_augmentation/random_zoom_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?26
4data_augmentation/random_zoom_3/stateful_uniform/maxº
6data_augmentation/random_zoom_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6data_augmentation/random_zoom_3/stateful_uniform/Const
5data_augmentation/random_zoom_3/stateful_uniform/ProdProd?data_augmentation/random_zoom_3/stateful_uniform/shape:output:0?data_augmentation/random_zoom_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 27
5data_augmentation/random_zoom_3/stateful_uniform/Prod´
7data_augmentation/random_zoom_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :29
7data_augmentation/random_zoom_3/stateful_uniform/Cast/xê
7data_augmentation/random_zoom_3/stateful_uniform/Cast_1Cast>data_augmentation/random_zoom_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 29
7data_augmentation/random_zoom_3/stateful_uniform/Cast_1ù
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipRngReadAndSkipHdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resource@data_augmentation/random_zoom_3/stateful_uniform/Cast/x:output:0;data_augmentation/random_zoom_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:2A
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipÖ
Ddata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Ddata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stackÚ
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_1Ú
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_2
>data_augmentation/random_zoom_3/stateful_uniform/strided_sliceStridedSliceGdata_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip:value:0Mdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack:output:0Odata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_1:output:0Odata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2@
>data_augmentation/random_zoom_3/stateful_uniform/strided_sliceù
8data_augmentation/random_zoom_3/stateful_uniform/BitcastBitcastGdata_augmentation/random_zoom_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02:
8data_augmentation/random_zoom_3/stateful_uniform/BitcastÚ
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stackÞ
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_1Þ
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_2
@data_augmentation/random_zoom_3/stateful_uniform/strided_slice_1StridedSliceGdata_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip:value:0Odata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack:output:0Qdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_1:output:0Qdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2B
@data_augmentation/random_zoom_3/stateful_uniform/strided_slice_1ÿ
:data_augmentation/random_zoom_3/stateful_uniform/Bitcast_1BitcastIdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02<
:data_augmentation/random_zoom_3/stateful_uniform/Bitcast_1à
Mdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2O
Mdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algü
Idata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2?data_augmentation/random_zoom_3/stateful_uniform/shape:output:0Cdata_augmentation/random_zoom_3/stateful_uniform/Bitcast_1:output:0Adata_augmentation/random_zoom_3/stateful_uniform/Bitcast:output:0Vdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Idata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2
4data_augmentation/random_zoom_3/stateful_uniform/subSub=data_augmentation/random_zoom_3/stateful_uniform/max:output:0=data_augmentation/random_zoom_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 26
4data_augmentation/random_zoom_3/stateful_uniform/sub³
4data_augmentation/random_zoom_3/stateful_uniform/mulMulRdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2:output:08data_augmentation/random_zoom_3/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4data_augmentation/random_zoom_3/stateful_uniform/mul
0data_augmentation/random_zoom_3/stateful_uniformAddV28data_augmentation/random_zoom_3/stateful_uniform/mul:z:0=data_augmentation/random_zoom_3/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0data_augmentation/random_zoom_3/stateful_uniformº
:data_augmentation/random_zoom_3/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:data_augmentation/random_zoom_3/stateful_uniform_1/shape/1§
8data_augmentation/random_zoom_3/stateful_uniform_1/shapePack6data_augmentation/random_zoom_3/strided_slice:output:0Cdata_augmentation/random_zoom_3/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8data_augmentation/random_zoom_3/stateful_uniform_1/shapeµ
6data_augmentation/random_zoom_3/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?28
6data_augmentation/random_zoom_3/stateful_uniform_1/minµ
6data_augmentation/random_zoom_3/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?28
6data_augmentation/random_zoom_3/stateful_uniform_1/max¾
8data_augmentation/random_zoom_3/stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8data_augmentation/random_zoom_3/stateful_uniform_1/Const¡
7data_augmentation/random_zoom_3/stateful_uniform_1/ProdProdAdata_augmentation/random_zoom_3/stateful_uniform_1/shape:output:0Adata_augmentation/random_zoom_3/stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 29
7data_augmentation/random_zoom_3/stateful_uniform_1/Prod¸
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2;
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast/xð
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast_1Cast@data_augmentation/random_zoom_3/stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast_1Ã
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipRngReadAndSkipHdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resourceBdata_augmentation/random_zoom_3/stateful_uniform_1/Cast/x:output:0=data_augmentation/random_zoom_3/stateful_uniform_1/Cast_1:y:0@^data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip*
_output_shapes
:2C
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipÚ
Fdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stackÞ
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_1Þ
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_2
@data_augmentation/random_zoom_3/stateful_uniform_1/strided_sliceStridedSliceIdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0Odata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack:output:0Qdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_1:output:0Qdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2B
@data_augmentation/random_zoom_3/stateful_uniform_1/strided_sliceÿ
:data_augmentation/random_zoom_3/stateful_uniform_1/BitcastBitcastIdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02<
:data_augmentation/random_zoom_3/stateful_uniform_1/BitcastÞ
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stackâ
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1â
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2
Bdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1StridedSliceIdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0Qdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack:output:0Sdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1:output:0Sdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2D
Bdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1
<data_augmentation/random_zoom_3/stateful_uniform_1/Bitcast_1BitcastKdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02>
<data_augmentation/random_zoom_3/stateful_uniform_1/Bitcast_1ä
Odata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2Q
Odata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/alg
Kdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2Adata_augmentation/random_zoom_3/stateful_uniform_1/shape:output:0Edata_augmentation/random_zoom_3/stateful_uniform_1/Bitcast_1:output:0Cdata_augmentation/random_zoom_3/stateful_uniform_1/Bitcast:output:0Xdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
Kdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2
6data_augmentation/random_zoom_3/stateful_uniform_1/subSub?data_augmentation/random_zoom_3/stateful_uniform_1/max:output:0?data_augmentation/random_zoom_3/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 28
6data_augmentation/random_zoom_3/stateful_uniform_1/sub»
6data_augmentation/random_zoom_3/stateful_uniform_1/mulMulTdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2:output:0:data_augmentation/random_zoom_3/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6data_augmentation/random_zoom_3/stateful_uniform_1/mul 
2data_augmentation/random_zoom_3/stateful_uniform_1AddV2:data_augmentation/random_zoom_3/stateful_uniform_1/mul:z:0?data_augmentation/random_zoom_3/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2data_augmentation/random_zoom_3/stateful_uniform_1
+data_augmentation/random_zoom_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+data_augmentation/random_zoom_3/concat/axis»
&data_augmentation/random_zoom_3/concatConcatV26data_augmentation/random_zoom_3/stateful_uniform_1:z:04data_augmentation/random_zoom_3/stateful_uniform:z:04data_augmentation/random_zoom_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&data_augmentation/random_zoom_3/concatÅ
1data_augmentation/random_zoom_3/zoom_matrix/ShapeShape/data_augmentation/random_zoom_3/concat:output:0*
T0*
_output_shapes
:23
1data_augmentation/random_zoom_3/zoom_matrix/ShapeÌ
?data_augmentation/random_zoom_3/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?data_augmentation/random_zoom_3/zoom_matrix/strided_slice/stackÐ
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_1Ð
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_2ê
9data_augmentation/random_zoom_3/zoom_matrix/strided_sliceStridedSlice:data_augmentation/random_zoom_3/zoom_matrix/Shape:output:0Hdata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_1:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9data_augmentation/random_zoom_3/zoom_matrix/strided_slice«
1data_augmentation/random_zoom_3/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1data_augmentation/random_zoom_3/zoom_matrix/sub/yò
/data_augmentation/random_zoom_3/zoom_matrix/subSub*data_augmentation/random_zoom_3/Cast_1:y:0:data_augmentation/random_zoom_3/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 21
/data_augmentation/random_zoom_3/zoom_matrix/sub³
5data_augmentation/random_zoom_3/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @27
5data_augmentation/random_zoom_3/zoom_matrix/truediv/y
3data_augmentation/random_zoom_3/zoom_matrix/truedivRealDiv3data_augmentation/random_zoom_3/zoom_matrix/sub:z:0>data_augmentation/random_zoom_3/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 25
3data_augmentation/random_zoom_3/zoom_matrix/truedivÛ
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stackß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_1ß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_2±
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_1StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_1¯
3data_augmentation/random_zoom_3/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3data_augmentation/random_zoom_3/zoom_matrix/sub_1/x£
1data_augmentation/random_zoom_3/zoom_matrix/sub_1Sub<data_augmentation/random_zoom_3/zoom_matrix/sub_1/x:output:0Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1data_augmentation/random_zoom_3/zoom_matrix/sub_1
/data_augmentation/random_zoom_3/zoom_matrix/mulMul7data_augmentation/random_zoom_3/zoom_matrix/truediv:z:05data_augmentation/random_zoom_3/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/data_augmentation/random_zoom_3/zoom_matrix/mul¯
3data_augmentation/random_zoom_3/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3data_augmentation/random_zoom_3/zoom_matrix/sub_2/yö
1data_augmentation/random_zoom_3/zoom_matrix/sub_2Sub(data_augmentation/random_zoom_3/Cast:y:0<data_augmentation/random_zoom_3/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 23
1data_augmentation/random_zoom_3/zoom_matrix/sub_2·
7data_augmentation/random_zoom_3/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @29
7data_augmentation/random_zoom_3/zoom_matrix/truediv_1/y
5data_augmentation/random_zoom_3/zoom_matrix/truediv_1RealDiv5data_augmentation/random_zoom_3/zoom_matrix/sub_2:z:0@data_augmentation/random_zoom_3/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 27
5data_augmentation/random_zoom_3/zoom_matrix/truediv_1Û
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stackß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_1ß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_2±
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_2StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_2¯
3data_augmentation/random_zoom_3/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3data_augmentation/random_zoom_3/zoom_matrix/sub_3/x£
1data_augmentation/random_zoom_3/zoom_matrix/sub_3Sub<data_augmentation/random_zoom_3/zoom_matrix/sub_3/x:output:0Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1data_augmentation/random_zoom_3/zoom_matrix/sub_3
1data_augmentation/random_zoom_3/zoom_matrix/mul_1Mul9data_augmentation/random_zoom_3/zoom_matrix/truediv_1:z:05data_augmentation/random_zoom_3/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1data_augmentation/random_zoom_3/zoom_matrix/mul_1Û
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stackß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_1ß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_2±
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_3StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_3º
:data_augmentation/random_zoom_3/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:data_augmentation/random_zoom_3/zoom_matrix/zeros/packed/1³
8data_augmentation/random_zoom_3/zoom_matrix/zeros/packedPackBdata_augmentation/random_zoom_3/zoom_matrix/strided_slice:output:0Cdata_augmentation/random_zoom_3/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2:
8data_augmentation/random_zoom_3/zoom_matrix/zeros/packed·
7data_augmentation/random_zoom_3/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7data_augmentation/random_zoom_3/zoom_matrix/zeros/Const¥
1data_augmentation/random_zoom_3/zoom_matrix/zerosFillAdata_augmentation/random_zoom_3/zoom_matrix/zeros/packed:output:0@data_augmentation/random_zoom_3/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1data_augmentation/random_zoom_3/zoom_matrix/zeros¾
<data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed/1¹
:data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packedPackBdata_augmentation/random_zoom_3/zoom_matrix/strided_slice:output:0Edata_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2<
:data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed»
9data_augmentation/random_zoom_3/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9data_augmentation/random_zoom_3/zoom_matrix/zeros_1/Const­
3data_augmentation/random_zoom_3/zoom_matrix/zeros_1FillCdata_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed:output:0Bdata_augmentation/random_zoom_3/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3data_augmentation/random_zoom_3/zoom_matrix/zeros_1Û
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stackß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_1ß
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_2±
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_4StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_4¾
<data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed/1¹
:data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packedPackBdata_augmentation/random_zoom_3/zoom_matrix/strided_slice:output:0Edata_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2<
:data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed»
9data_augmentation/random_zoom_3/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9data_augmentation/random_zoom_3/zoom_matrix/zeros_2/Const­
3data_augmentation/random_zoom_3/zoom_matrix/zeros_2FillCdata_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed:output:0Bdata_augmentation/random_zoom_3/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3data_augmentation/random_zoom_3/zoom_matrix/zeros_2´
7data_augmentation/random_zoom_3/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :29
7data_augmentation/random_zoom_3/zoom_matrix/concat/axis¡
2data_augmentation/random_zoom_3/zoom_matrix/concatConcatV2Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3:output:0:data_augmentation/random_zoom_3/zoom_matrix/zeros:output:03data_augmentation/random_zoom_3/zoom_matrix/mul:z:0<data_augmentation/random_zoom_3/zoom_matrix/zeros_1:output:0Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4:output:05data_augmentation/random_zoom_3/zoom_matrix/mul_1:z:0<data_augmentation/random_zoom_3/zoom_matrix/zeros_2:output:0@data_augmentation/random_zoom_3/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2data_augmentation/random_zoom_3/zoom_matrix/concatï
/data_augmentation/random_zoom_3/transform/ShapeShape]data_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:21
/data_augmentation/random_zoom_3/transform/ShapeÈ
=data_augmentation/random_zoom_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=data_augmentation/random_zoom_3/transform/strided_slice/stackÌ
?data_augmentation/random_zoom_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?data_augmentation/random_zoom_3/transform/strided_slice/stack_1Ì
?data_augmentation/random_zoom_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?data_augmentation/random_zoom_3/transform/strided_slice/stack_2Ê
7data_augmentation/random_zoom_3/transform/strided_sliceStridedSlice8data_augmentation/random_zoom_3/transform/Shape:output:0Fdata_augmentation/random_zoom_3/transform/strided_slice/stack:output:0Hdata_augmentation/random_zoom_3/transform/strided_slice/stack_1:output:0Hdata_augmentation/random_zoom_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:29
7data_augmentation/random_zoom_3/transform/strided_slice±
4data_augmentation/random_zoom_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4data_augmentation/random_zoom_3/transform/fill_value»
Ddata_augmentation/random_zoom_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3]data_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0;data_augmentation/random_zoom_3/zoom_matrix/concat:output:0@data_augmentation/random_zoom_3/transform/strided_slice:output:0=data_augmentation/random_zoom_3/transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2F
Ddata_augmentation/random_zoom_3/transform/ImageProjectiveTransformV3º
patches_7/ExtractImagePatchesExtractImagePatchesYdata_augmentation/random_zoom_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksizes
*
paddingVALID*
rates
*
strides
2
patches_7/ExtractImagePatches
patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ¤  1   2
patches_7/Reshape/shape³
patches_7/ReshapeReshape'patches_7/ExtractImagePatches:patches:0 patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
patches_7/Reshape±
!dense_21/Tensordot/ReadVariableOpReadVariableOp*dense_21_tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02#
!dense_21/Tensordot/ReadVariableOp|
dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_21/Tensordot/axes
dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_21/Tensordot/free~
dense_21/Tensordot/ShapeShapepatches_7/Reshape:output:0*
T0*
_output_shapes
:2
dense_21/Tensordot/Shape
 dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/GatherV2/axisþ
dense_21/Tensordot/GatherV2GatherV2!dense_21/Tensordot/Shape:output:0 dense_21/Tensordot/free:output:0)dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_21/Tensordot/GatherV2
"dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_21/Tensordot/GatherV2_1/axis
dense_21/Tensordot/GatherV2_1GatherV2!dense_21/Tensordot/Shape:output:0 dense_21/Tensordot/axes:output:0+dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_21/Tensordot/GatherV2_1~
dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_21/Tensordot/Const¤
dense_21/Tensordot/ProdProd$dense_21/Tensordot/GatherV2:output:0!dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prod
dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_21/Tensordot/Const_1¬
dense_21/Tensordot/Prod_1Prod&dense_21/Tensordot/GatherV2_1:output:0#dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prod_1
dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_21/Tensordot/concat/axisÝ
dense_21/Tensordot/concatConcatV2 dense_21/Tensordot/free:output:0 dense_21/Tensordot/axes:output:0'dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat°
dense_21/Tensordot/stackPack dense_21/Tensordot/Prod:output:0"dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/stackÀ
dense_21/Tensordot/transpose	Transposepatches_7/Reshape:output:0"dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
dense_21/Tensordot/transposeÃ
dense_21/Tensordot/ReshapeReshape dense_21/Tensordot/transpose:y:0!dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_21/Tensordot/ReshapeÂ
dense_21/Tensordot/MatMulMatMul#dense_21/Tensordot/Reshape:output:0)dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_21/Tensordot/MatMul
dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_21/Tensordot/Const_2
 dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/concat_1/axisê
dense_21/Tensordot/concat_1ConcatV2$dense_21/Tensordot/GatherV2:output:0#dense_21/Tensordot/Const_2:output:0)dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat_1µ
dense_21/TensordotReshape#dense_21/Tensordot/MatMul:product:0$dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
dense_21/Tensordot§
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_21/BiasAdd/ReadVariableOp¬
dense_21/BiasAddBiasAdddense_21/Tensordot:output:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
dense_21/BiasAdd
lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2!
lambda_3/Mean/reduction_indices
lambda_3/MeanMeandense_21/BiasAdd:output:0(lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lambda_3/Mean
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2
lambda_3/Reshape/shape
lambda_3/ReshapeReshapelambda_3/Mean:output:0lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
lambda_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisÒ
concatenate_3/concatConcatV2lambda_3/Reshape:output:0dense_21/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
concatenate_3/concat|
patch_encoder_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
patch_encoder_7/range/start}
patch_encoder_7/range/limitConst*
_output_shapes
: *
dtype0*
value
B :¥2
patch_encoder_7/range/limit|
patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder_7/range/deltaÆ
patch_encoder_7/rangeRange$patch_encoder_7/range/start:output:0$patch_encoder_7/range/limit:output:0$patch_encoder_7/range/delta:output:0*
_output_shapes	
:¥2
patch_encoder_7/rangeù
,patch_encoder_7/embedding_3/embedding_lookupResourceGather4patch_encoder_7_embedding_3_embedding_lookup_1121191patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1121191*
_output_shapes
:	¥@*
dtype02.
,patch_encoder_7/embedding_3/embedding_lookupÒ
5patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity5patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1121191*
_output_shapes
:	¥@27
5patch_encoder_7/embedding_3/embedding_lookup/Identityè
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1Identity>patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	¥@29
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1Ë
patch_encoder_7/addAddV2concatenate_3/concat:output:0@patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
patch_encoder_7/add¸
5layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_12/moments/mean/reduction_indicesó
#layer_normalization_12/moments/meanMeanpatch_encoder_7/add:z:0>layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2%
#layer_normalization_12/moments/meanÏ
+layer_normalization_12/moments/StopGradientStopGradient,layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2-
+layer_normalization_12/moments/StopGradientÿ
0layer_normalization_12/moments/SquaredDifferenceSquaredDifferencepatch_encoder_7/add:z:04layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
0layer_normalization_12/moments/SquaredDifferenceÀ
9layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_12/moments/variance/reduction_indices
'layer_normalization_12/moments/varianceMean4layer_normalization_12/moments/SquaredDifference:z:0Blayer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2)
'layer_normalization_12/moments/variance
&layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_12/batchnorm/add/yï
$layer_normalization_12/batchnorm/addAddV20layer_normalization_12/moments/variance:output:0/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2&
$layer_normalization_12/batchnorm/addº
&layer_normalization_12/batchnorm/RsqrtRsqrt(layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2(
&layer_normalization_12/batchnorm/Rsqrtã
3layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_12/batchnorm/mul/ReadVariableOpó
$layer_normalization_12/batchnorm/mulMul*layer_normalization_12/batchnorm/Rsqrt:y:0;layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_12/batchnorm/mulÑ
&layer_normalization_12/batchnorm/mul_1Mulpatch_encoder_7/add:z:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_12/batchnorm/mul_1æ
&layer_normalization_12/batchnorm/mul_2Mul,layer_normalization_12/moments/mean:output:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_12/batchnorm/mul_2×
/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_12/batchnorm/ReadVariableOpï
$layer_normalization_12/batchnorm/subSub7layer_normalization_12/batchnorm/ReadVariableOp:value:0*layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_12/batchnorm/subæ
&layer_normalization_12/batchnorm/add_1AddV2*layer_normalization_12/batchnorm/mul_1:z:0(layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_12/batchnorm/add_1ý
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp²
*multi_head_attention_6/query/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_6/query/einsum/EinsumÛ
/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp8multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/query/add/ReadVariableOpö
 multi_head_attention_6/query/addAddV23multi_head_attention_6/query/einsum/Einsum:output:07multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_6/query/add÷
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¬
(multi_head_attention_6/key/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0?multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2*
(multi_head_attention_6/key/einsum/EinsumÕ
-multi_head_attention_6/key/add/ReadVariableOpReadVariableOp6multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_6/key/add/ReadVariableOpî
multi_head_attention_6/key/addAddV21multi_head_attention_6/key/einsum/Einsum:output:05multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2 
multi_head_attention_6/key/addý
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp²
*multi_head_attention_6/value/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_6/value/einsum/EinsumÛ
/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp8multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/value/add/ReadVariableOpö
 multi_head_attention_6/value/addAddV23multi_head_attention_6/value/einsum/Einsum:output:07multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_6/value/add
multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_6/Mul/yÇ
multi_head_attention_6/MulMul$multi_head_attention_6/query/add:z:0%multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
multi_head_attention_6/Mulþ
$multi_head_attention_6/einsum/EinsumEinsum"multi_head_attention_6/key/add:z:0multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2&
$multi_head_attention_6/einsum/EinsumÆ
&multi_head_attention_6/softmax/SoftmaxSoftmax-multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2(
&multi_head_attention_6/softmax/Softmax¡
,multi_head_attention_6/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,multi_head_attention_6/dropout/dropout/Const
*multi_head_attention_6/dropout/dropout/MulMul0multi_head_attention_6/softmax/Softmax:softmax:05multi_head_attention_6/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2,
*multi_head_attention_6/dropout/dropout/Mul¼
,multi_head_attention_6/dropout/dropout/ShapeShape0multi_head_attention_6/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_6/dropout/dropout/Shape
Cmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_6/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
dtype02E
Cmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_6/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5multi_head_attention_6/dropout/dropout/GreaterEqual/yÄ
3multi_head_attention_6/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_6/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥25
3multi_head_attention_6/dropout/dropout/GreaterEqualæ
+multi_head_attention_6/dropout/dropout/CastCast7multi_head_attention_6/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2-
+multi_head_attention_6/dropout/dropout/Cast
,multi_head_attention_6/dropout/dropout/Mul_1Mul.multi_head_attention_6/dropout/dropout/Mul:z:0/multi_head_attention_6/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2.
,multi_head_attention_6/dropout/dropout/Mul_1
&multi_head_attention_6/einsum_1/EinsumEinsum0multi_head_attention_6/dropout/dropout/Mul_1:z:0$multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2(
&multi_head_attention_6/einsum_1/Einsum
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_6/attention_output/einsum/EinsumEinsum/multi_head_attention_6/einsum_1/Einsum:output:0Lmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe27
5multi_head_attention_6/attention_output/einsum/Einsumø
:multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_6/attention_output/add/ReadVariableOp
+multi_head_attention_6/attention_output/addAddV2>multi_head_attention_6/attention_output/einsum/Einsum:output:0Bmulti_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2-
+multi_head_attention_6/attention_output/add¢

add_12/addAddV2/multi_head_attention_6/attention_output/add:z:0patch_encoder_7/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

add_12/add¸
5layer_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_13/moments/mean/reduction_indicesê
#layer_normalization_13/moments/meanMeanadd_12/add:z:0>layer_normalization_13/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2%
#layer_normalization_13/moments/meanÏ
+layer_normalization_13/moments/StopGradientStopGradient,layer_normalization_13/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2-
+layer_normalization_13/moments/StopGradientö
0layer_normalization_13/moments/SquaredDifferenceSquaredDifferenceadd_12/add:z:04layer_normalization_13/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
0layer_normalization_13/moments/SquaredDifferenceÀ
9layer_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_13/moments/variance/reduction_indices
'layer_normalization_13/moments/varianceMean4layer_normalization_13/moments/SquaredDifference:z:0Blayer_normalization_13/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2)
'layer_normalization_13/moments/variance
&layer_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_13/batchnorm/add/yï
$layer_normalization_13/batchnorm/addAddV20layer_normalization_13/moments/variance:output:0/layer_normalization_13/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2&
$layer_normalization_13/batchnorm/addº
&layer_normalization_13/batchnorm/RsqrtRsqrt(layer_normalization_13/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2(
&layer_normalization_13/batchnorm/Rsqrtã
3layer_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_13/batchnorm/mul/ReadVariableOpó
$layer_normalization_13/batchnorm/mulMul*layer_normalization_13/batchnorm/Rsqrt:y:0;layer_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_13/batchnorm/mulÈ
&layer_normalization_13/batchnorm/mul_1Muladd_12/add:z:0(layer_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_13/batchnorm/mul_1æ
&layer_normalization_13/batchnorm/mul_2Mul,layer_normalization_13/moments/mean:output:0(layer_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_13/batchnorm/mul_2×
/layer_normalization_13/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_13/batchnorm/ReadVariableOpï
$layer_normalization_13/batchnorm/subSub7layer_normalization_13/batchnorm/ReadVariableOp:value:0*layer_normalization_13/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_13/batchnorm/subæ
&layer_normalization_13/batchnorm/add_1AddV2*layer_normalization_13/batchnorm/mul_1:z:0(layer_normalization_13/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_13/batchnorm/add_1²
!dense_22/Tensordot/ReadVariableOpReadVariableOp*dense_22_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02#
!dense_22/Tensordot/ReadVariableOp|
dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_22/Tensordot/axes
dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_22/Tensordot/free
dense_22/Tensordot/ShapeShape*layer_normalization_13/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_22/Tensordot/Shape
 dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/GatherV2/axisþ
dense_22/Tensordot/GatherV2GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/free:output:0)dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2
"dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_22/Tensordot/GatherV2_1/axis
dense_22/Tensordot/GatherV2_1GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/axes:output:0+dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2_1~
dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const¤
dense_22/Tensordot/ProdProd$dense_22/Tensordot/GatherV2:output:0!dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod
dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const_1¬
dense_22/Tensordot/Prod_1Prod&dense_22/Tensordot/GatherV2_1:output:0#dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod_1
dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_22/Tensordot/concat/axisÝ
dense_22/Tensordot/concatConcatV2 dense_22/Tensordot/free:output:0 dense_22/Tensordot/axes:output:0'dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat°
dense_22/Tensordot/stackPack dense_22/Tensordot/Prod:output:0"dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/stackÐ
dense_22/Tensordot/transpose	Transpose*layer_normalization_13/batchnorm/add_1:z:0"dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_22/Tensordot/transposeÃ
dense_22/Tensordot/ReshapeReshape dense_22/Tensordot/transpose:y:0!dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_22/Tensordot/ReshapeÃ
dense_22/Tensordot/MatMulMatMul#dense_22/Tensordot/Reshape:output:0)dense_22/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_22/Tensordot/MatMul
dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_22/Tensordot/Const_2
 dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/concat_1/axisê
dense_22/Tensordot/concat_1ConcatV2$dense_22/Tensordot/GatherV2:output:0#dense_22/Tensordot/Const_2:output:0)dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat_1¶
dense_22/TensordotReshape#dense_22/Tensordot/MatMul:product:0$dense_22/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Tensordot¨
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp­
dense_22/BiasAddBiasAdddense_22/Tensordot:output:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/BiasAddo
dense_22/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_22/Gelu/mul/x
dense_22/Gelu/mulMuldense_22/Gelu/mul/x:output:0dense_22/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/mulq
dense_22/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
dense_22/Gelu/Cast/x«
dense_22/Gelu/truedivRealDivdense_22/BiasAdd:output:0dense_22/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/truediv
dense_22/Gelu/ErfErfdense_22/Gelu/truediv:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/Erfo
dense_22/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_22/Gelu/add/x
dense_22/Gelu/addAddV2dense_22/Gelu/add/x:output:0dense_22/Gelu/Erf:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/add
dense_22/Gelu/mul_1Muldense_22/Gelu/mul:z:0dense_22/Gelu/add:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_22/Gelu/mul_1y
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_18/dropout/Const«
dropout_18/dropout/MulMuldense_22/Gelu/mul_1:z:0!dropout_18/dropout/Const:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout_18/dropout/Mul{
dropout_18/dropout/ShapeShapedense_22/Gelu/mul_1:z:0*
T0*
_output_shapes
:2
dropout_18/dropout/ShapeÛ
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
dtype021
/dropout_18/dropout/random_uniform/RandomUniform
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_18/dropout/GreaterEqual/yð
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2!
dropout_18/dropout/GreaterEqual¦
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout_18/dropout/Cast¬
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout_18/dropout/Mul_1²
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axes
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/free
dense_23/Tensordot/ShapeShapedropout_18/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_23/Tensordot/Shape
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axisþ
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axis
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const¤
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1¬
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axisÝ
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat°
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stackÃ
dense_23/Tensordot/transpose	Transposedropout_18/dropout/Mul_1:z:0"dense_23/Tensordot/concat:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dense_23/Tensordot/transposeÃ
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_23/Tensordot/ReshapeÂ
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_23/Tensordot/MatMul
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_23/Tensordot/Const_2
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axisê
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1µ
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Tensordot§
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp¬
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/BiasAddo
dense_23/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_23/Gelu/mul/x
dense_23/Gelu/mulMuldense_23/Gelu/mul/x:output:0dense_23/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/mulq
dense_23/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
dense_23/Gelu/Cast/xª
dense_23/Gelu/truedivRealDivdense_23/BiasAdd:output:0dense_23/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/truediv
dense_23/Gelu/ErfErfdense_23/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/Erfo
dense_23/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dense_23/Gelu/add/x
dense_23/Gelu/addAddV2dense_23/Gelu/add/x:output:0dense_23/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/add
dense_23/Gelu/mul_1Muldense_23/Gelu/mul:z:0dense_23/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dense_23/Gelu/mul_1y
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_19/dropout/Constª
dropout_19/dropout/MulMuldense_23/Gelu/mul_1:z:0!dropout_19/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout_19/dropout/Mul{
dropout_19/dropout/ShapeShapedense_23/Gelu/mul_1:z:0*
T0*
_output_shapes
:2
dropout_19/dropout/ShapeÚ
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_19/dropout/GreaterEqual/yï
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2!
dropout_19/dropout/GreaterEqual¥
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout_19/dropout/Cast«
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout_19/dropout/Mul_1

add_13/addAddV2dropout_19/dropout/Mul_1:z:0add_12/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

add_13/add¸
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesê
#layer_normalization_14/moments/meanMeanadd_13/add:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2%
#layer_normalization_14/moments/meanÏ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2-
+layer_normalization_14/moments/StopGradientö
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd_13/add:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@22
0layer_normalization_14/moments/SquaredDifferenceÀ
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indices
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2)
'layer_normalization_14/moments/variance
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752(
&layer_normalization_14/batchnorm/add/yï
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2&
$layer_normalization_14/batchnorm/addº
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2(
&layer_normalization_14/batchnorm/Rsqrtã
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpó
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_14/batchnorm/mulÈ
&layer_normalization_14/batchnorm/mul_1Muladd_13/add:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_14/batchnorm/mul_1æ
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_14/batchnorm/mul_2×
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpï
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2&
$layer_normalization_14/batchnorm/subæ
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2(
&layer_normalization_14/batchnorm/add_1ý
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp²
*multi_head_attention_7/query/einsum/EinsumEinsum*layer_normalization_14/batchnorm/add_1:z:0Amulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/EinsumÛ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_7/query/add/ReadVariableOpö
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_7/query/add÷
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¬
(multi_head_attention_7/key/einsum/EinsumEinsum*layer_normalization_14/batchnorm/add_1:z:0?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/EinsumÕ
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpî
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2 
multi_head_attention_7/key/addý
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp²
*multi_head_attention_7/value/einsum/EinsumEinsum*layer_normalization_14/batchnorm/add_1:z:0Amulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/EinsumÛ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_7/value/add/ReadVariableOpö
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2"
 multi_head_attention_7/value/add
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_7/Mul/yÇ
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
multi_head_attention_7/Mulþ
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/EinsumÆ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2(
&multi_head_attention_7/softmax/Softmax¡
,multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2.
,multi_head_attention_7/dropout/dropout/Const
*multi_head_attention_7/dropout/dropout/MulMul0multi_head_attention_7/softmax/Softmax:softmax:05multi_head_attention_7/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2,
*multi_head_attention_7/dropout/dropout/Mul¼
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/Shape
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
dtype02E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=27
5multi_head_attention_7/dropout/dropout/GreaterEqual/yÄ
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥25
3multi_head_attention_7/dropout/dropout/GreaterEqualæ
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2-
+multi_head_attention_7/dropout/dropout/Cast
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2.
,multi_head_attention_7/dropout/dropout/Mul_1
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/Einsum
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpÔ
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumø
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOp
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2-
+multi_head_attention_7/attention_output/add
IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity
NoOpNoOpI^data_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipp^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgw^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterD^data_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip@^data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipB^data_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip ^dense_21/BiasAdd/ReadVariableOp"^dense_21/Tensordot/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp"^dense_22/Tensordot/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp0^layer_normalization_12/batchnorm/ReadVariableOp4^layer_normalization_12/batchnorm/mul/ReadVariableOp0^layer_normalization_13/batchnorm/ReadVariableOp4^layer_normalization_13/batchnorm/mul/ReadVariableOp0^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipHdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip2â
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgodata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2ð
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCountervdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipCdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip2
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip2
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipAdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2F
!dense_21/Tensordot/ReadVariableOp!dense_21/Tensordot/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2F
!dense_22/Tensordot/ReadVariableOp!dense_22/Tensordot/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2b
/layer_normalization_12/batchnorm/ReadVariableOp/layer_normalization_12/batchnorm/ReadVariableOp2j
3layer_normalization_12/batchnorm/mul/ReadVariableOp3layer_normalization_12/batchnorm/mul/ReadVariableOp2b
/layer_normalization_13/batchnorm/ReadVariableOp/layer_normalization_13/batchnorm/ReadVariableOp2j
3layer_normalization_13/batchnorm/mul/ReadVariableOp3layer_normalization_13/batchnorm/mul/ReadVariableOp2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_6/key/add/ReadVariableOp-multi_head_attention_6/key/add/ReadVariableOp2r
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/query/add/ReadVariableOp/multi_head_attention_6/query/add/ReadVariableOp2v
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/value/add/ReadVariableOp/multi_head_attention_6/value/add/ReadVariableOp2v
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,patch_encoder_7/embedding_3/embedding_lookup,patch_encoder_7/embedding_3/embedding_lookup:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Üf

J__inference_random_flip_3_layer_call_and_return_conditional_losses_1122467

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_like
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceÔ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi25
3stateless_random_flip_left_right/control_dependency¼
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¶
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¨
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceñ
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/max
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgÊ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÓ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=stateless_random_flip_left_right/stateless_random_uniform/mul¸
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9stateless_random_flip_left_right/stateless_random_uniform¦
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1¦
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2¦
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shape
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stateless_random_flip_left_right/ReshapeÆ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/Round¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2,
*stateless_random_flip_left_right/ReverseV2ï
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2&
$stateless_random_flip_left_right/mul
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&stateless_random_flip_left_right/sub/xê
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/subú
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2(
&stateless_random_flip_left_right/mul_1æ
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2&
$stateless_random_flip_left_right/add
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity¤
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
à
T
(__inference_add_12_layer_call_fn_1122027
inputs_0
inputs_1
identityÖ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_12_layer_call_and_return_conditional_losses_11192842
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/1
±

â
3__inference_data_augmentation_layer_call_fn_1119065
normalization_3_input
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallnormalization_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11190372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1121821

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs
Ã9

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1119836	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const¨
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÖ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yè
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/GreaterEqual¡
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Cast¤
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mul_1¹
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
ë
H
,__inference_resizing_3_layer_call_fn_1122387

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11186302
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
H
,__inference_dropout_18_layer_call_fn_1122116

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_11193632
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs
ý
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1119175

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
concath
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ¤@:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:TP
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs

ý
8__inference_multi_head_attention_6_layer_call_fn_1121942	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity

identity_1¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_11198362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
^
ñ
E__inference_model_11_layer_call_and_return_conditional_losses_1119508

inputs
data_augmentation_1119108
data_augmentation_1119110"
dense_21_1119153:1@
dense_21_1119155:@*
patch_encoder_7_1119192:	¥@,
layer_normalization_12_1119218:@,
layer_normalization_12_1119220:@4
multi_head_attention_6_1119260:@@0
multi_head_attention_6_1119262:@4
multi_head_attention_6_1119264:@@0
multi_head_attention_6_1119266:@4
multi_head_attention_6_1119268:@@0
multi_head_attention_6_1119270:@4
multi_head_attention_6_1119272:@@,
multi_head_attention_6_1119274:@,
layer_normalization_13_1119309:@,
layer_normalization_13_1119311:@#
dense_22_1119353:	@
dense_22_1119355:	#
dense_23_1119404:	@
dense_23_1119406:@,
layer_normalization_14_1119447:@,
layer_normalization_14_1119449:@4
multi_head_attention_7_1119489:@@0
multi_head_attention_7_1119491:@4
multi_head_attention_7_1119493:@@0
multi_head_attention_7_1119495:@4
multi_head_attention_7_1119497:@@0
multi_head_attention_7_1119499:@4
multi_head_attention_7_1119501:@@,
multi_head_attention_7_1119503:@
identity¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.layer_normalization_13/StatefulPartitionedCall¢.layer_normalization_14/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢.multi_head_attention_7/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCall´
!data_augmentation/PartitionedCallPartitionedCallinputsdata_augmentation_1119108data_augmentation_1119110*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11186512#
!data_augmentation/PartitionedCall
patches_7/PartitionedCallPartitionedCall*data_augmentation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11191202
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1119153dense_21_1119155*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11191522"
 dense_21/StatefulPartitionedCallÿ
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11191662
lambda_3/PartitionedCall³
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11191752
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1119192*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11191912)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1119218layer_normalization_12_1119220*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111921720
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1119260multi_head_attention_6_1119262multi_head_attention_6_1119264multi_head_attention_6_1119266multi_head_attention_6_1119268multi_head_attention_6_1119270multi_head_attention_6_1119272multi_head_attention_6_1119274*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111925920
.multi_head_attention_6/StatefulPartitionedCall»
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:00patch_encoder_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_12_layer_call_and_return_conditional_losses_11192842
add_12/PartitionedCallþ
.layer_normalization_13/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_13_1119309layer_normalization_13_1119311*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_111930820
.layer_normalization_13/StatefulPartitionedCallÑ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:0dense_22_1119353dense_22_1119355*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_11193522"
 dense_22/StatefulPartitionedCall
dropout_18/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_11193632
dropout_18/PartitionedCall¼
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_23_1119404dense_23_1119406*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_11194032"
 dense_23/StatefulPartitionedCall
dropout_19/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_11194142
dropout_19/PartitionedCall
add_13/PartitionedCallPartitionedCall#dropout_19/PartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_13_layer_call_and_return_conditional_losses_11194222
add_13/PartitionedCallþ
.layer_normalization_14/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_14_1119447layer_normalization_14_1119449*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_111944620
.layer_normalization_14/StatefulPartitionedCallº
.multi_head_attention_7/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:07layer_normalization_14/StatefulPartitionedCall:output:0multi_head_attention_7_1119489multi_head_attention_7_1119491multi_head_attention_7_1119493multi_head_attention_7_1119495multi_head_attention_7_1119497multi_head_attention_7_1119499multi_head_attention_7_1119501multi_head_attention_7_1119503*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_111948820
.multi_head_attention_7/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_7/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

IdentityÖ
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^layer_normalization_13/StatefulPartitionedCall/^layer_normalization_14/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall/^multi_head_attention_7/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.layer_normalization_13/StatefulPartitionedCall.layer_normalization_13/StatefulPartitionedCall2`
.layer_normalization_14/StatefulPartitionedCall.layer_normalization_14/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2`
.multi_head_attention_7/StatefulPartitionedCall.multi_head_attention_7/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
õ

/__inference_random_flip_3_layer_call_fn_1122405

inputs
unknown:	
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11189942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
±
¡
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1118651

inputs
normalization_3_sub_y
normalization_3_sqrt_x
identity
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/sub}
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*&
_output_shapes
:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_3/Maximum/y¬
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum¯
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/truedivü
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11186302
resizing_3/PartitionedCall
random_flip_3/PartitionedCallPartitionedCall#resizing_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11186362
random_flip_3/PartitionedCall
!random_rotation_3/PartitionedCallPartitionedCall&random_flip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11186422#
!random_rotation_3/PartitionedCall
random_zoom_3/PartitionedCallPartitionedCall*random_rotation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11186482
random_zoom_3/PartitionedCall
IdentityIdentity&random_zoom_3/PartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:


*__inference_dense_21_layer_call_fn_1121773

inputs
unknown:1@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11191522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤1: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1
 
_user_specified_nameinputs
ï'
þ
E__inference_dense_22_layer_call_and_return_conditional_losses_1119352

inputs4
!tensordot_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xz
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
Gelu/Cast/x
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Gelu/mul_1o
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs

v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1121842
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
concath
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ¤@:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
"
_user_specified_name
inputs/1
ü
b
F__inference_patches_7_layer_call_and_return_conditional_losses_1119120

images
identityÓ
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksizes
*
paddingVALID*
rates
*
strides
2
ExtractImagePatchess
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ¤  1   2
Reshape/shape
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameimages
í 
ü
E__inference_dense_21_layer_call_and_return_conditional_losses_1121803

inputs3
!tensordot_readvariableop_resource:1@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1
 
_user_specified_nameinputs
à
T
(__inference_add_13_layer_call_fn_1122218
inputs_0
inputs_1
identityÖ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_13_layer_call_and_return_conditional_losses_11194222
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/1
 

N__inference_data_augmentation_layer_call_and_return_conditional_losses_1121752

inputs
normalization_3_sub_y
normalization_3_sqrt_xM
?random_flip_3_stateful_uniform_full_int_rngreadandskip_resource:	H
:random_rotation_3_stateful_uniform_rngreadandskip_resource:	D
6random_zoom_3_stateful_uniform_rngreadandskip_resource:	
identity¢6random_flip_3/stateful_uniform_full_int/RngReadAndSkip¢]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢1random_rotation_3/stateful_uniform/RngReadAndSkip¢-random_zoom_3/stateful_uniform/RngReadAndSkip¢/random_zoom_3/stateful_uniform_1/RngReadAndSkip
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/sub}
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*&
_output_shapes
:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_3/Maximum/y¬
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum¯
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/truediv
resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   2
resizing_3/resize/sizeé
 resizing_3/resize/ResizeBilinearResizeBilinearnormalization_3/truediv:z:0resizing_3/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(2"
 resizing_3/resize/ResizeBilinear¨
-random_flip_3/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_flip_3/stateful_uniform_full_int/shape¨
-random_flip_3/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_flip_3/stateful_uniform_full_int/Constõ
,random_flip_3/stateful_uniform_full_int/ProdProd6random_flip_3/stateful_uniform_full_int/shape:output:06random_flip_3/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2.
,random_flip_3/stateful_uniform_full_int/Prod¢
.random_flip_3/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip_3/stateful_uniform_full_int/Cast/xÏ
.random_flip_3/stateful_uniform_full_int/Cast_1Cast5random_flip_3/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.random_flip_3/stateful_uniform_full_int/Cast_1Ì
6random_flip_3/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip?random_flip_3_stateful_uniform_full_int_rngreadandskip_resource7random_flip_3/stateful_uniform_full_int/Cast/x:output:02random_flip_3/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:28
6random_flip_3/stateful_uniform_full_int/RngReadAndSkipÄ
;random_flip_3/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;random_flip_3/stateful_uniform_full_int/strided_slice/stackÈ
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_1È
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_2Ø
5random_flip_3/stateful_uniform_full_int/strided_sliceStridedSlice>random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip_3/stateful_uniform_full_int/strided_slice/stack:output:0Frandom_flip_3/stateful_uniform_full_int/strided_slice/stack_1:output:0Frandom_flip_3/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask27
5random_flip_3/stateful_uniform_full_int/strided_sliceÞ
/random_flip_3/stateful_uniform_full_int/BitcastBitcast>random_flip_3/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type021
/random_flip_3/stateful_uniform_full_int/BitcastÈ
=random_flip_3/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateful_uniform_full_int/strided_slice_1/stackÌ
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1Ì
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2Ð
7random_flip_3/stateful_uniform_full_int/strided_slice_1StridedSlice>random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_flip_3/stateful_uniform_full_int/strided_slice_1/stack:output:0Hrandom_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Hrandom_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:29
7random_flip_3/stateful_uniform_full_int/strided_slice_1ä
1random_flip_3/stateful_uniform_full_int/Bitcast_1Bitcast@random_flip_3/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type023
1random_flip_3/stateful_uniform_full_int/Bitcast_1
+random_flip_3/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_flip_3/stateful_uniform_full_int/alg
'random_flip_3/stateful_uniform_full_intStatelessRandomUniformFullIntV26random_flip_3/stateful_uniform_full_int/shape:output:0:random_flip_3/stateful_uniform_full_int/Bitcast_1:output:08random_flip_3/stateful_uniform_full_int/Bitcast:output:04random_flip_3/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2)
'random_flip_3/stateful_uniform_full_int~
random_flip_3/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip_3/zeros_like¹
random_flip_3/stackPack0random_flip_3/stateful_uniform_full_int:output:0!random_flip_3/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip_3/stack
!random_flip_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_flip_3/strided_slice/stack
#random_flip_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_flip_3/strided_slice/stack_1
#random_flip_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_flip_3/strided_slice/stack_2Ü
random_flip_3/strided_sliceStridedSlicerandom_flip_3/stack:output:0*random_flip_3/strided_slice/stack:output:0,random_flip_3/strided_slice/stack_1:output:0,random_flip_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip_3/strided_sliceµ
Arandom_flip_3/stateless_random_flip_left_right/control_dependencyIdentity1resizing_3/resize/ResizeBilinear:resized_images:0*
T0*3
_class)
'%loc:@resizing_3/resize/ResizeBilinear*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2C
Arandom_flip_3/stateless_random_flip_left_right/control_dependencyæ
4random_flip_3/stateless_random_flip_left_right/ShapeShapeJrandom_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:26
4random_flip_3/stateless_random_flip_left_right/ShapeÒ
Brandom_flip_3/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Brandom_flip_3/stateless_random_flip_left_right/strided_slice/stackÖ
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_1Ö
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_2ü
<random_flip_3/stateless_random_flip_left_right/strided_sliceStridedSlice=random_flip_3/stateless_random_flip_left_right/Shape:output:0Krandom_flip_3/stateless_random_flip_left_right/strided_slice/stack:output:0Mrandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_1:output:0Mrandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<random_flip_3/stateless_random_flip_left_right/strided_slice
Mrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shapePackErandom_flip_3/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2O
Mrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shapeß
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/minß
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/max´
drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip_3/strided_slice:output:0* 
_output_shapes
::2f
drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterÖ
]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlge^random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2_
]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg
`random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Vrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0jrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0crandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
`random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2î
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subSubTrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Trandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/sub
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mulMulirandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Orandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mulð
Grandom_flip_3/stateless_random_flip_left_right/stateless_random_uniformAddV2Orandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Trandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
Grandom_flip_3/stateless_random_flip_left_right/stateless_random_uniformÂ
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/1Â
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/2Â
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2@
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/3Ô
<random_flip_3/stateless_random_flip_left_right/Reshape/shapePackErandom_flip_3/stateless_random_flip_left_right/strided_slice:output:0Grandom_flip_3/stateless_random_flip_left_right/Reshape/shape/1:output:0Grandom_flip_3/stateless_random_flip_left_right/Reshape/shape/2:output:0Grandom_flip_3/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2>
<random_flip_3/stateless_random_flip_left_right/Reshape/shapeÉ
6random_flip_3/stateless_random_flip_left_right/ReshapeReshapeKrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform:z:0Erandom_flip_3/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6random_flip_3/stateless_random_flip_left_right/Reshapeð
4random_flip_3/stateless_random_flip_left_right/RoundRound?random_flip_3/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4random_flip_3/stateless_random_flip_left_right/RoundÈ
=random_flip_3/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateless_random_flip_left_right/ReverseV2/axisÐ
8random_flip_3/stateless_random_flip_left_right/ReverseV2	ReverseV2Jrandom_flip_3/stateless_random_flip_left_right/control_dependency:output:0Frandom_flip_3/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2:
8random_flip_3/stateless_random_flip_left_right/ReverseV2§
2random_flip_3/stateless_random_flip_left_right/mulMul8random_flip_3/stateless_random_flip_left_right/Round:y:0Arandom_flip_3/stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi24
2random_flip_3/stateless_random_flip_left_right/mul±
4random_flip_3/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4random_flip_3/stateless_random_flip_left_right/sub/x¢
2random_flip_3/stateless_random_flip_left_right/subSub=random_flip_3/stateless_random_flip_left_right/sub/x:output:08random_flip_3/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip_3/stateless_random_flip_left_right/sub²
4random_flip_3/stateless_random_flip_left_right/mul_1Mul6random_flip_3/stateless_random_flip_left_right/sub:z:0Jrandom_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi26
4random_flip_3/stateless_random_flip_left_right/mul_1
2random_flip_3/stateless_random_flip_left_right/addAddV26random_flip_3/stateless_random_flip_left_right/mul:z:08random_flip_3/stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi24
2random_flip_3/stateless_random_flip_left_right/add
random_rotation_3/ShapeShape6random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation_3/Shape
%random_rotation_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_3/strided_slice/stack
'random_rotation_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_3/strided_slice/stack_1
'random_rotation_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_3/strided_slice/stack_2Î
random_rotation_3/strided_sliceStridedSlice random_rotation_3/Shape:output:0.random_rotation_3/strided_slice/stack:output:00random_rotation_3/strided_slice/stack_1:output:00random_rotation_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_3/strided_slice¥
'random_rotation_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'random_rotation_3/strided_slice_1/stack©
)random_rotation_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2+
)random_rotation_3/strided_slice_1/stack_1 
)random_rotation_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_3/strided_slice_1/stack_2Ø
!random_rotation_3/strided_slice_1StridedSlice random_rotation_3/Shape:output:00random_rotation_3/strided_slice_1/stack:output:02random_rotation_3/strided_slice_1/stack_1:output:02random_rotation_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_3/strided_slice_1
random_rotation_3/CastCast*random_rotation_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_3/Cast¥
'random_rotation_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2)
'random_rotation_3/strided_slice_2/stack©
)random_rotation_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2+
)random_rotation_3/strided_slice_2/stack_1 
)random_rotation_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_3/strided_slice_2/stack_2Ø
!random_rotation_3/strided_slice_2StridedSlice random_rotation_3/Shape:output:00random_rotation_3/strided_slice_2/stack:output:02random_rotation_3/strided_slice_2/stack_1:output:02random_rotation_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_3/strided_slice_2
random_rotation_3/Cast_1Cast*random_rotation_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_3/Cast_1´
(random_rotation_3/stateful_uniform/shapePack(random_rotation_3/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_3/stateful_uniform/shape
&random_rotation_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ ¾2(
&random_rotation_3/stateful_uniform/min
&random_rotation_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ý­ >2(
&random_rotation_3/stateful_uniform/max
(random_rotation_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(random_rotation_3/stateful_uniform/Constá
'random_rotation_3/stateful_uniform/ProdProd1random_rotation_3/stateful_uniform/shape:output:01random_rotation_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/stateful_uniform/Prod
)random_rotation_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_rotation_3/stateful_uniform/Cast/xÀ
)random_rotation_3/stateful_uniform/Cast_1Cast0random_rotation_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)random_rotation_3/stateful_uniform/Cast_1³
1random_rotation_3/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_3_stateful_uniform_rngreadandskip_resource2random_rotation_3/stateful_uniform/Cast/x:output:0-random_rotation_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:23
1random_rotation_3/stateful_uniform/RngReadAndSkipº
6random_rotation_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_3/stateful_uniform/strided_slice/stack¾
8random_rotation_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_3/stateful_uniform/strided_slice/stack_1¾
8random_rotation_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_3/stateful_uniform/strided_slice/stack_2º
0random_rotation_3/stateful_uniform/strided_sliceStridedSlice9random_rotation_3/stateful_uniform/RngReadAndSkip:value:0?random_rotation_3/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_3/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask22
0random_rotation_3/stateful_uniform/strided_sliceÏ
*random_rotation_3/stateful_uniform/BitcastBitcast9random_rotation_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation_3/stateful_uniform/Bitcast¾
8random_rotation_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_3/stateful_uniform/strided_slice_1/stackÂ
:random_rotation_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_3/stateful_uniform/strided_slice_1/stack_1Â
:random_rotation_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_3/stateful_uniform/strided_slice_1/stack_2²
2random_rotation_3/stateful_uniform/strided_slice_1StridedSlice9random_rotation_3/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_3/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_3/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:24
2random_rotation_3/stateful_uniform/strided_slice_1Õ
,random_rotation_3/stateful_uniform/Bitcast_1Bitcast;random_rotation_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02.
,random_rotation_3/stateful_uniform/Bitcast_1Ä
?random_rotation_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2A
?random_rotation_3/stateful_uniform/StatelessRandomUniformV2/alg¤
;random_rotation_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_3/stateful_uniform/shape:output:05random_rotation_3/stateful_uniform/Bitcast_1:output:03random_rotation_3/stateful_uniform/Bitcast:output:0Hrandom_rotation_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;random_rotation_3/stateful_uniform/StatelessRandomUniformV2Ú
&random_rotation_3/stateful_uniform/subSub/random_rotation_3/stateful_uniform/max:output:0/random_rotation_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_3/stateful_uniform/sub÷
&random_rotation_3/stateful_uniform/mulMulDrandom_rotation_3/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_3/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation_3/stateful_uniform/mulÜ
"random_rotation_3/stateful_uniformAddV2*random_rotation_3/stateful_uniform/mul:z:0/random_rotation_3/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"random_rotation_3/stateful_uniform
'random_rotation_3/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation_3/rotation_matrix/sub/yÆ
%random_rotation_3/rotation_matrix/subSubrandom_rotation_3/Cast_1:y:00random_rotation_3/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_3/rotation_matrix/sub«
%random_rotation_3/rotation_matrix/CosCos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation_3/rotation_matrix/Cos
)random_rotation_3/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_3/rotation_matrix/sub_1/yÌ
'random_rotation_3/rotation_matrix/sub_1Subrandom_rotation_3/Cast_1:y:02random_rotation_3/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_1Û
%random_rotation_3/rotation_matrix/mulMul)random_rotation_3/rotation_matrix/Cos:y:0+random_rotation_3/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation_3/rotation_matrix/mul«
%random_rotation_3/rotation_matrix/SinSin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation_3/rotation_matrix/Sin
)random_rotation_3/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_3/rotation_matrix/sub_2/yÊ
'random_rotation_3/rotation_matrix/sub_2Subrandom_rotation_3/Cast:y:02random_rotation_3/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_2ß
'random_rotation_3/rotation_matrix/mul_1Mul)random_rotation_3/rotation_matrix/Sin:y:0+random_rotation_3/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/mul_1ß
'random_rotation_3/rotation_matrix/sub_3Sub)random_rotation_3/rotation_matrix/mul:z:0+random_rotation_3/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/sub_3ß
'random_rotation_3/rotation_matrix/sub_4Sub)random_rotation_3/rotation_matrix/sub:z:0+random_rotation_3/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/sub_4
+random_rotation_3/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_3/rotation_matrix/truediv/yò
)random_rotation_3/rotation_matrix/truedivRealDiv+random_rotation_3/rotation_matrix/sub_4:z:04random_rotation_3/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation_3/rotation_matrix/truediv
)random_rotation_3/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_3/rotation_matrix/sub_5/yÊ
'random_rotation_3/rotation_matrix/sub_5Subrandom_rotation_3/Cast:y:02random_rotation_3/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_5¯
'random_rotation_3/rotation_matrix/Sin_1Sin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/Sin_1
)random_rotation_3/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_3/rotation_matrix/sub_6/yÌ
'random_rotation_3/rotation_matrix/sub_6Subrandom_rotation_3/Cast_1:y:02random_rotation_3/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_6á
'random_rotation_3/rotation_matrix/mul_2Mul+random_rotation_3/rotation_matrix/Sin_1:y:0+random_rotation_3/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/mul_2¯
'random_rotation_3/rotation_matrix/Cos_1Cos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/Cos_1
)random_rotation_3/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_rotation_3/rotation_matrix/sub_7/yÊ
'random_rotation_3/rotation_matrix/sub_7Subrandom_rotation_3/Cast:y:02random_rotation_3/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_7á
'random_rotation_3/rotation_matrix/mul_3Mul+random_rotation_3/rotation_matrix/Cos_1:y:0+random_rotation_3/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/mul_3ß
%random_rotation_3/rotation_matrix/addAddV2+random_rotation_3/rotation_matrix/mul_2:z:0+random_rotation_3/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation_3/rotation_matrix/addß
'random_rotation_3/rotation_matrix/sub_8Sub+random_rotation_3/rotation_matrix/sub_5:z:0)random_rotation_3/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/sub_8£
-random_rotation_3/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_3/rotation_matrix/truediv_1/yø
+random_rotation_3/rotation_matrix/truediv_1RealDiv+random_rotation_3/rotation_matrix/sub_8:z:06random_rotation_3/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+random_rotation_3/rotation_matrix/truediv_1¨
'random_rotation_3/rotation_matrix/ShapeShape&random_rotation_3/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_3/rotation_matrix/Shape¸
5random_rotation_3/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_3/rotation_matrix/strided_slice/stack¼
7random_rotation_3/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_3/rotation_matrix/strided_slice/stack_1¼
7random_rotation_3/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_3/rotation_matrix/strided_slice/stack_2®
/random_rotation_3/rotation_matrix/strided_sliceStridedSlice0random_rotation_3/rotation_matrix/Shape:output:0>random_rotation_3/rotation_matrix/strided_slice/stack:output:0@random_rotation_3/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_3/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_3/rotation_matrix/strided_slice¯
'random_rotation_3/rotation_matrix/Cos_2Cos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/Cos_2Ã
7random_rotation_3/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_1/stackÇ
9random_rotation_3/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_1/stack_1Ç
9random_rotation_3/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_1/stack_2ã
1random_rotation_3/rotation_matrix/strided_slice_1StridedSlice+random_rotation_3/rotation_matrix/Cos_2:y:0@random_rotation_3/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_1¯
'random_rotation_3/rotation_matrix/Sin_2Sin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/Sin_2Ã
7random_rotation_3/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_2/stackÇ
9random_rotation_3/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_2/stack_1Ç
9random_rotation_3/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_2/stack_2ã
1random_rotation_3/rotation_matrix/strided_slice_2StridedSlice+random_rotation_3/rotation_matrix/Sin_2:y:0@random_rotation_3/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_2Ã
%random_rotation_3/rotation_matrix/NegNeg:random_rotation_3/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation_3/rotation_matrix/NegÃ
7random_rotation_3/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_3/stackÇ
9random_rotation_3/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_3/stack_1Ç
9random_rotation_3/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_3/stack_2å
1random_rotation_3/rotation_matrix/strided_slice_3StridedSlice-random_rotation_3/rotation_matrix/truediv:z:0@random_rotation_3/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_3¯
'random_rotation_3/rotation_matrix/Sin_3Sin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/Sin_3Ã
7random_rotation_3/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_4/stackÇ
9random_rotation_3/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_4/stack_1Ç
9random_rotation_3/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_4/stack_2ã
1random_rotation_3/rotation_matrix/strided_slice_4StridedSlice+random_rotation_3/rotation_matrix/Sin_3:y:0@random_rotation_3/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_4¯
'random_rotation_3/rotation_matrix/Cos_3Cos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/Cos_3Ã
7random_rotation_3/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_5/stackÇ
9random_rotation_3/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_5/stack_1Ç
9random_rotation_3/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_5/stack_2ã
1random_rotation_3/rotation_matrix/strided_slice_5StridedSlice+random_rotation_3/rotation_matrix/Cos_3:y:0@random_rotation_3/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_5Ã
7random_rotation_3/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_6/stackÇ
9random_rotation_3/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_6/stack_1Ç
9random_rotation_3/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_6/stack_2ç
1random_rotation_3/rotation_matrix/strided_slice_6StridedSlice/random_rotation_3/rotation_matrix/truediv_1:z:0@random_rotation_3/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_6¦
0random_rotation_3/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_3/rotation_matrix/zeros/packed/1
.random_rotation_3/rotation_matrix/zeros/packedPack8random_rotation_3/rotation_matrix/strided_slice:output:09random_rotation_3/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_3/rotation_matrix/zeros/packed£
-random_rotation_3/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_3/rotation_matrix/zeros/Constý
'random_rotation_3/rotation_matrix/zerosFill7random_rotation_3/rotation_matrix/zeros/packed:output:06random_rotation_3/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_3/rotation_matrix/zeros 
-random_rotation_3/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_3/rotation_matrix/concat/axisÜ
(random_rotation_3/rotation_matrix/concatConcatV2:random_rotation_3/rotation_matrix/strided_slice_1:output:0)random_rotation_3/rotation_matrix/Neg:y:0:random_rotation_3/rotation_matrix/strided_slice_3:output:0:random_rotation_3/rotation_matrix/strided_slice_4:output:0:random_rotation_3/rotation_matrix/strided_slice_5:output:0:random_rotation_3/rotation_matrix/strided_slice_6:output:00random_rotation_3/rotation_matrix/zeros:output:06random_rotation_3/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_3/rotation_matrix/concat¬
!random_rotation_3/transform/ShapeShape6random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_3/transform/Shape¬
/random_rotation_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_3/transform/strided_slice/stack°
1random_rotation_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_3/transform/strided_slice/stack_1°
1random_rotation_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_3/transform/strided_slice/stack_2ö
)random_rotation_3/transform/strided_sliceStridedSlice*random_rotation_3/transform/Shape:output:08random_rotation_3/transform/strided_slice/stack:output:0:random_rotation_3/transform/strided_slice/stack_1:output:0:random_rotation_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_3/transform/strided_slice
&random_rotation_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_3/transform/fill_valueÒ
6random_rotation_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV36random_flip_3/stateless_random_flip_left_right/add:z:01random_rotation_3/rotation_matrix/concat:output:02random_rotation_3/transform/strided_slice:output:0/random_rotation_3/transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_3/transform/ImageProjectiveTransformV3¥
random_zoom_3/ShapeShapeKrandom_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_3/Shape
!random_zoom_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!random_zoom_3/strided_slice/stack
#random_zoom_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_3/strided_slice/stack_1
#random_zoom_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_3/strided_slice/stack_2¶
random_zoom_3/strided_sliceStridedSlicerandom_zoom_3/Shape:output:0*random_zoom_3/strided_slice/stack:output:0,random_zoom_3/strided_slice/stack_1:output:0,random_zoom_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_3/strided_slice
#random_zoom_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2%
#random_zoom_3/strided_slice_1/stack¡
%random_zoom_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2'
%random_zoom_3/strided_slice_1/stack_1
%random_zoom_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_3/strided_slice_1/stack_2À
random_zoom_3/strided_slice_1StridedSlicerandom_zoom_3/Shape:output:0,random_zoom_3/strided_slice_1/stack:output:0.random_zoom_3/strided_slice_1/stack_1:output:0.random_zoom_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_3/strided_slice_1
random_zoom_3/CastCast&random_zoom_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_3/Cast
#random_zoom_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2%
#random_zoom_3/strided_slice_2/stack¡
%random_zoom_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2'
%random_zoom_3/strided_slice_2/stack_1
%random_zoom_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_3/strided_slice_2/stack_2À
random_zoom_3/strided_slice_2StridedSlicerandom_zoom_3/Shape:output:0,random_zoom_3/strided_slice_2/stack:output:0.random_zoom_3/strided_slice_2/stack_1:output:0.random_zoom_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_3/strided_slice_2
random_zoom_3/Cast_1Cast&random_zoom_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_3/Cast_1
&random_zoom_3/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_3/stateful_uniform/shape/1Ù
$random_zoom_3/stateful_uniform/shapePack$random_zoom_3/strided_slice:output:0/random_zoom_3/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom_3/stateful_uniform/shape
"random_zoom_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2$
"random_zoom_3/stateful_uniform/min
"random_zoom_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2$
"random_zoom_3/stateful_uniform/max
$random_zoom_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$random_zoom_3/stateful_uniform/ConstÑ
#random_zoom_3/stateful_uniform/ProdProd-random_zoom_3/stateful_uniform/shape:output:0-random_zoom_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2%
#random_zoom_3/stateful_uniform/Prod
%random_zoom_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_3/stateful_uniform/Cast/x´
%random_zoom_3/stateful_uniform/Cast_1Cast,random_zoom_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%random_zoom_3/stateful_uniform/Cast_1
-random_zoom_3/stateful_uniform/RngReadAndSkipRngReadAndSkip6random_zoom_3_stateful_uniform_rngreadandskip_resource.random_zoom_3/stateful_uniform/Cast/x:output:0)random_zoom_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:2/
-random_zoom_3/stateful_uniform/RngReadAndSkip²
2random_zoom_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2random_zoom_3/stateful_uniform/strided_slice/stack¶
4random_zoom_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4random_zoom_3/stateful_uniform/strided_slice/stack_1¶
4random_zoom_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4random_zoom_3/stateful_uniform/strided_slice/stack_2¢
,random_zoom_3/stateful_uniform/strided_sliceStridedSlice5random_zoom_3/stateful_uniform/RngReadAndSkip:value:0;random_zoom_3/stateful_uniform/strided_slice/stack:output:0=random_zoom_3/stateful_uniform/strided_slice/stack_1:output:0=random_zoom_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2.
,random_zoom_3/stateful_uniform/strided_sliceÃ
&random_zoom_3/stateful_uniform/BitcastBitcast5random_zoom_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02(
&random_zoom_3/stateful_uniform/Bitcast¶
4random_zoom_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4random_zoom_3/stateful_uniform/strided_slice_1/stackº
6random_zoom_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform/strided_slice_1/stack_1º
6random_zoom_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform/strided_slice_1/stack_2
.random_zoom_3/stateful_uniform/strided_slice_1StridedSlice5random_zoom_3/stateful_uniform/RngReadAndSkip:value:0=random_zoom_3/stateful_uniform/strided_slice_1/stack:output:0?random_zoom_3/stateful_uniform/strided_slice_1/stack_1:output:0?random_zoom_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:20
.random_zoom_3/stateful_uniform/strided_slice_1É
(random_zoom_3/stateful_uniform/Bitcast_1Bitcast7random_zoom_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02*
(random_zoom_3/stateful_uniform/Bitcast_1¼
;random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2=
;random_zoom_3/stateful_uniform/StatelessRandomUniformV2/alg
7random_zoom_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2-random_zoom_3/stateful_uniform/shape:output:01random_zoom_3/stateful_uniform/Bitcast_1:output:0/random_zoom_3/stateful_uniform/Bitcast:output:0Drandom_zoom_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7random_zoom_3/stateful_uniform/StatelessRandomUniformV2Ê
"random_zoom_3/stateful_uniform/subSub+random_zoom_3/stateful_uniform/max:output:0+random_zoom_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_zoom_3/stateful_uniform/subë
"random_zoom_3/stateful_uniform/mulMul@random_zoom_3/stateful_uniform/StatelessRandomUniformV2:output:0&random_zoom_3/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"random_zoom_3/stateful_uniform/mulÐ
random_zoom_3/stateful_uniformAddV2&random_zoom_3/stateful_uniform/mul:z:0+random_zoom_3/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
random_zoom_3/stateful_uniform
(random_zoom_3/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_3/stateful_uniform_1/shape/1ß
&random_zoom_3/stateful_uniform_1/shapePack$random_zoom_3/strided_slice:output:01random_zoom_3/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_3/stateful_uniform_1/shape
$random_zoom_3/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?2&
$random_zoom_3/stateful_uniform_1/min
$random_zoom_3/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?2&
$random_zoom_3/stateful_uniform_1/max
&random_zoom_3/stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_zoom_3/stateful_uniform_1/ConstÙ
%random_zoom_3/stateful_uniform_1/ProdProd/random_zoom_3/stateful_uniform_1/shape:output:0/random_zoom_3/stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 2'
%random_zoom_3/stateful_uniform_1/Prod
'random_zoom_3/stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_3/stateful_uniform_1/Cast/xº
'random_zoom_3/stateful_uniform_1/Cast_1Cast.random_zoom_3/stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'random_zoom_3/stateful_uniform_1/Cast_1×
/random_zoom_3/stateful_uniform_1/RngReadAndSkipRngReadAndSkip6random_zoom_3_stateful_uniform_rngreadandskip_resource0random_zoom_3/stateful_uniform_1/Cast/x:output:0+random_zoom_3/stateful_uniform_1/Cast_1:y:0.^random_zoom_3/stateful_uniform/RngReadAndSkip*
_output_shapes
:21
/random_zoom_3/stateful_uniform_1/RngReadAndSkip¶
4random_zoom_3/stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4random_zoom_3/stateful_uniform_1/strided_slice/stackº
6random_zoom_3/stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform_1/strided_slice/stack_1º
6random_zoom_3/stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform_1/strided_slice/stack_2®
.random_zoom_3/stateful_uniform_1/strided_sliceStridedSlice7random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0=random_zoom_3/stateful_uniform_1/strided_slice/stack:output:0?random_zoom_3/stateful_uniform_1/strided_slice/stack_1:output:0?random_zoom_3/stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask20
.random_zoom_3/stateful_uniform_1/strided_sliceÉ
(random_zoom_3/stateful_uniform_1/BitcastBitcast7random_zoom_3/stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02*
(random_zoom_3/stateful_uniform_1/Bitcastº
6random_zoom_3/stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform_1/strided_slice_1/stack¾
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1¾
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2¦
0random_zoom_3/stateful_uniform_1/strided_slice_1StridedSlice7random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0?random_zoom_3/stateful_uniform_1/strided_slice_1/stack:output:0Arandom_zoom_3/stateful_uniform_1/strided_slice_1/stack_1:output:0Arandom_zoom_3/stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:22
0random_zoom_3/stateful_uniform_1/strided_slice_1Ï
*random_zoom_3/stateful_uniform_1/Bitcast_1Bitcast9random_zoom_3/stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02,
*random_zoom_3/stateful_uniform_1/Bitcast_1À
=random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/alg
9random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2/random_zoom_3/stateful_uniform_1/shape:output:03random_zoom_3/stateful_uniform_1/Bitcast_1:output:01random_zoom_3/stateful_uniform_1/Bitcast:output:0Frandom_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2Ò
$random_zoom_3/stateful_uniform_1/subSub-random_zoom_3/stateful_uniform_1/max:output:0-random_zoom_3/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2&
$random_zoom_3/stateful_uniform_1/subó
$random_zoom_3/stateful_uniform_1/mulMulBrandom_zoom_3/stateful_uniform_1/StatelessRandomUniformV2:output:0(random_zoom_3/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_zoom_3/stateful_uniform_1/mulØ
 random_zoom_3/stateful_uniform_1AddV2(random_zoom_3/stateful_uniform_1/mul:z:0-random_zoom_3/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_zoom_3/stateful_uniform_1x
random_zoom_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_3/concat/axisá
random_zoom_3/concatConcatV2$random_zoom_3/stateful_uniform_1:z:0"random_zoom_3/stateful_uniform:z:0"random_zoom_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_zoom_3/concat
random_zoom_3/zoom_matrix/ShapeShaperandom_zoom_3/concat:output:0*
T0*
_output_shapes
:2!
random_zoom_3/zoom_matrix/Shape¨
-random_zoom_3/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_zoom_3/zoom_matrix/strided_slice/stack¬
/random_zoom_3/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_3/zoom_matrix/strided_slice/stack_1¬
/random_zoom_3/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_3/zoom_matrix/strided_slice/stack_2þ
'random_zoom_3/zoom_matrix/strided_sliceStridedSlice(random_zoom_3/zoom_matrix/Shape:output:06random_zoom_3/zoom_matrix/strided_slice/stack:output:08random_zoom_3/zoom_matrix/strided_slice/stack_1:output:08random_zoom_3/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'random_zoom_3/zoom_matrix/strided_slice
random_zoom_3/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
random_zoom_3/zoom_matrix/sub/yª
random_zoom_3/zoom_matrix/subSubrandom_zoom_3/Cast_1:y:0(random_zoom_3/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
random_zoom_3/zoom_matrix/sub
#random_zoom_3/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#random_zoom_3/zoom_matrix/truediv/yÃ
!random_zoom_3/zoom_matrix/truedivRealDiv!random_zoom_3/zoom_matrix/sub:z:0,random_zoom_3/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom_3/zoom_matrix/truediv·
/random_zoom_3/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_3/zoom_matrix/strided_slice_1/stack»
1random_zoom_3/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_1/stack_1»
1random_zoom_3/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_1/stack_2Å
)random_zoom_3/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_1/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_1/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_1
!random_zoom_3/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_3/zoom_matrix/sub_1/xÛ
random_zoom_3/zoom_matrix/sub_1Sub*random_zoom_3/zoom_matrix/sub_1/x:output:02random_zoom_3/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
random_zoom_3/zoom_matrix/sub_1Ã
random_zoom_3/zoom_matrix/mulMul%random_zoom_3/zoom_matrix/truediv:z:0#random_zoom_3/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_zoom_3/zoom_matrix/mul
!random_zoom_3/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_3/zoom_matrix/sub_2/y®
random_zoom_3/zoom_matrix/sub_2Subrandom_zoom_3/Cast:y:0*random_zoom_3/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2!
random_zoom_3/zoom_matrix/sub_2
%random_zoom_3/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%random_zoom_3/zoom_matrix/truediv_1/yË
#random_zoom_3/zoom_matrix/truediv_1RealDiv#random_zoom_3/zoom_matrix/sub_2:z:0.random_zoom_3/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_3/zoom_matrix/truediv_1·
/random_zoom_3/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_3/zoom_matrix/strided_slice_2/stack»
1random_zoom_3/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_2/stack_1»
1random_zoom_3/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_2/stack_2Å
)random_zoom_3/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_2/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_2/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_2
!random_zoom_3/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!random_zoom_3/zoom_matrix/sub_3/xÛ
random_zoom_3/zoom_matrix/sub_3Sub*random_zoom_3/zoom_matrix/sub_3/x:output:02random_zoom_3/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
random_zoom_3/zoom_matrix/sub_3É
random_zoom_3/zoom_matrix/mul_1Mul'random_zoom_3/zoom_matrix/truediv_1:z:0#random_zoom_3/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
random_zoom_3/zoom_matrix/mul_1·
/random_zoom_3/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_3/zoom_matrix/strided_slice_3/stack»
1random_zoom_3/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_3/stack_1»
1random_zoom_3/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_3/stack_2Å
)random_zoom_3/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_3/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_3/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_3
(random_zoom_3/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_3/zoom_matrix/zeros/packed/1ë
&random_zoom_3/zoom_matrix/zeros/packedPack0random_zoom_3/zoom_matrix/strided_slice:output:01random_zoom_3/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_3/zoom_matrix/zeros/packed
%random_zoom_3/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom_3/zoom_matrix/zeros/ConstÝ
random_zoom_3/zoom_matrix/zerosFill/random_zoom_3/zoom_matrix/zeros/packed:output:0.random_zoom_3/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
random_zoom_3/zoom_matrix/zeros
*random_zoom_3/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_3/zoom_matrix/zeros_1/packed/1ñ
(random_zoom_3/zoom_matrix/zeros_1/packedPack0random_zoom_3/zoom_matrix/strided_slice:output:03random_zoom_3/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_3/zoom_matrix/zeros_1/packed
'random_zoom_3/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_3/zoom_matrix/zeros_1/Constå
!random_zoom_3/zoom_matrix/zeros_1Fill1random_zoom_3/zoom_matrix/zeros_1/packed:output:00random_zoom_3/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!random_zoom_3/zoom_matrix/zeros_1·
/random_zoom_3/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_3/zoom_matrix/strided_slice_4/stack»
1random_zoom_3/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_4/stack_1»
1random_zoom_3/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_4/stack_2Å
)random_zoom_3/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_4/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_4/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_4
*random_zoom_3/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_3/zoom_matrix/zeros_2/packed/1ñ
(random_zoom_3/zoom_matrix/zeros_2/packedPack0random_zoom_3/zoom_matrix/strided_slice:output:03random_zoom_3/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_3/zoom_matrix/zeros_2/packed
'random_zoom_3/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_3/zoom_matrix/zeros_2/Constå
!random_zoom_3/zoom_matrix/zeros_2Fill1random_zoom_3/zoom_matrix/zeros_2/packed:output:00random_zoom_3/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!random_zoom_3/zoom_matrix/zeros_2
%random_zoom_3/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_3/zoom_matrix/concat/axisí
 random_zoom_3/zoom_matrix/concatConcatV22random_zoom_3/zoom_matrix/strided_slice_3:output:0(random_zoom_3/zoom_matrix/zeros:output:0!random_zoom_3/zoom_matrix/mul:z:0*random_zoom_3/zoom_matrix/zeros_1:output:02random_zoom_3/zoom_matrix/strided_slice_4:output:0#random_zoom_3/zoom_matrix/mul_1:z:0*random_zoom_3/zoom_matrix/zeros_2:output:0.random_zoom_3/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_zoom_3/zoom_matrix/concat¹
random_zoom_3/transform/ShapeShapeKrandom_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_3/transform/Shape¤
+random_zoom_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom_3/transform/strided_slice/stack¨
-random_zoom_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_3/transform/strided_slice/stack_1¨
-random_zoom_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_3/transform/strided_slice/stack_2Þ
%random_zoom_3/transform/strided_sliceStridedSlice&random_zoom_3/transform/Shape:output:04random_zoom_3/transform/strided_slice/stack:output:06random_zoom_3/transform/strided_slice/stack_1:output:06random_zoom_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%random_zoom_3/transform/strided_slice
"random_zoom_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"random_zoom_3/transform/fill_valueÏ
2random_zoom_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0)random_zoom_3/zoom_matrix/concat:output:0.random_zoom_3/transform/strided_slice:output:0+random_zoom_3/transform/fill_value:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR24
2random_zoom_3/transform/ImageProjectiveTransformV3«
IdentityIdentityGrandom_zoom_3/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityä
NoOpNoOp7^random_flip_3/stateful_uniform_full_int/RngReadAndSkip^^random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlge^random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2^random_rotation_3/stateful_uniform/RngReadAndSkip.^random_zoom_3/stateful_uniform/RngReadAndSkip0^random_zoom_3/stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ::: : : 2p
6random_flip_3/stateful_uniform_full_int/RngReadAndSkip6random_flip_3/stateful_uniform_full_int/RngReadAndSkip2¾
]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2Ì
drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterdrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2f
1random_rotation_3/stateful_uniform/RngReadAndSkip1random_rotation_3/stateful_uniform/RngReadAndSkip2^
-random_zoom_3/stateful_uniform/RngReadAndSkip-random_zoom_3/stateful_uniform/RngReadAndSkip2b
/random_zoom_3/stateful_uniform_1/RngReadAndSkip/random_zoom_3/stateful_uniform_1/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
ä
c
G__inference_resizing_3_layer_call_and_return_conditional_losses_1118630

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   2
resize/size³
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(2
resize/ResizeBilinear
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
o
C__inference_add_12_layer_call_and_return_conditional_losses_1122033
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/1
Ù
H
,__inference_dropout_19_layer_call_fn_1122190

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_11194142
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs


S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_1119308

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesÀ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs


S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1121894

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesÀ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
ñ
o
C__inference_add_13_layer_call_and_return_conditional_losses_1122224
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@:V R
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
"
_user_specified_name
inputs/1

ý
8__inference_multi_head_attention_7_layer_call_fn_1122279	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity

identity_1¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_11194882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
õ

/__inference_random_zoom_3_layer_call_fn_1122613

inputs
unknown:	
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11187922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
Ì
Ç
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1121863

projection7
$embedding_3_embedding_lookup_1121856:	¥@
identity¢embedding_3/embedding_lookup\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :¥2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:¥2
range©
embedding_3/embedding_lookupResourceGather$embedding_3_embedding_lookup_1121856range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*7
_class-
+)loc:@embedding_3/embedding_lookup/1121856*
_output_shapes
:	¥@*
dtype02
embedding_3/embedding_lookup
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*7
_class-
+)loc:@embedding_3/embedding_lookup/1121856*
_output_shapes
:	¥@2'
%embedding_3/embedding_lookup/Identity¸
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	¥@2)
'embedding_3/embedding_lookup/Identity_1
addAddV2
projection0embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
addg
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identitym
NoOpNoOp^embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: 2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:X T
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
$
_user_specified_name
projection
ï
K
/__inference_random_flip_3_layer_call_fn_1122398

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11186362
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
Ã9

S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1119644	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const¨
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÖ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yè
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/GreaterEqual¡
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Cast¤
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/dropout/Mul_1¹
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1121829

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs
þ

3__inference_random_rotation_3_layer_call_fn_1122479

inputs
unknown:	
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11189232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÄi: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
¡
Õ
*__inference_model_11_layer_call_fn_1119573
input_4
unknown
	unknown_0
	unknown_1:1@
	unknown_2:@
	unknown_3:	¥@
	unknown_4:@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:	@

unknown_17:	

unknown_18:	@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@ 

unknown_28:@@

unknown_29:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_11_layer_call_and_return_conditional_losses_11195082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ì
Ç
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1119191

projection7
$embedding_3_embedding_lookup_1119184:	¥@
identity¢embedding_3/embedding_lookup\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :¥2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:¥2
range©
embedding_3/embedding_lookupResourceGather$embedding_3_embedding_lookup_1119184range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*7
_class-
+)loc:@embedding_3/embedding_lookup/1119184*
_output_shapes
:	¥@*
dtype02
embedding_3/embedding_lookup
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*7
_class-
+)loc:@embedding_3/embedding_lookup/1119184*
_output_shapes
:	¥@2'
%embedding_3/embedding_lookup/Identity¸
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	¥@2)
'embedding_3/embedding_lookup/Identity_1
addAddV2
projection0embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
addg
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identitym
NoOpNoOp^embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: 2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:X T
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
$
_user_specified_name
projection
Êd
â
E__inference_model_11_layer_call_and_return_conditional_losses_1120077

inputs
data_augmentation_1119989
data_augmentation_1119991'
data_augmentation_1119993:	'
data_augmentation_1119995:	'
data_augmentation_1119997:	"
dense_21_1120001:1@
dense_21_1120003:@*
patch_encoder_7_1120008:	¥@,
layer_normalization_12_1120011:@,
layer_normalization_12_1120013:@4
multi_head_attention_6_1120016:@@0
multi_head_attention_6_1120018:@4
multi_head_attention_6_1120020:@@0
multi_head_attention_6_1120022:@4
multi_head_attention_6_1120024:@@0
multi_head_attention_6_1120026:@4
multi_head_attention_6_1120028:@@,
multi_head_attention_6_1120030:@,
layer_normalization_13_1120035:@,
layer_normalization_13_1120037:@#
dense_22_1120040:	@
dense_22_1120042:	#
dense_23_1120046:	@
dense_23_1120048:@,
layer_normalization_14_1120053:@,
layer_normalization_14_1120055:@4
multi_head_attention_7_1120058:@@0
multi_head_attention_7_1120060:@4
multi_head_attention_7_1120062:@@0
multi_head_attention_7_1120064:@4
multi_head_attention_7_1120066:@@0
multi_head_attention_7_1120068:@4
multi_head_attention_7_1120070:@@,
multi_head_attention_7_1120072:@
identity¢)data_augmentation/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢"dropout_18/StatefulPartitionedCall¢"dropout_19/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.layer_normalization_13/StatefulPartitionedCall¢.layer_normalization_14/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢.multi_head_attention_7/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCall 
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinputsdata_augmentation_1119989data_augmentation_1119991data_augmentation_1119993data_augmentation_1119995data_augmentation_1119997*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11190372+
)data_augmentation/StatefulPartitionedCall
patches_7/PartitionedCallPartitionedCall2data_augmentation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11191202
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1120001dense_21_1120003*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11191522"
 dense_21/StatefulPartitionedCallÿ
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11198982
lambda_3/PartitionedCall³
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11191752
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1120008*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11191912)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1120011layer_normalization_12_1120013*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111921720
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1120016multi_head_attention_6_1120018multi_head_attention_6_1120020multi_head_attention_6_1120022multi_head_attention_6_1120024multi_head_attention_6_1120026multi_head_attention_6_1120028multi_head_attention_6_1120030*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111983620
.multi_head_attention_6/StatefulPartitionedCall»
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:00patch_encoder_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_12_layer_call_and_return_conditional_losses_11192842
add_12/PartitionedCallþ
.layer_normalization_13/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_13_1120035layer_normalization_13_1120037*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_111930820
.layer_normalization_13/StatefulPartitionedCallÑ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:0dense_22_1120040dense_22_1120042*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_11193522"
 dense_22/StatefulPartitionedCall
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_11197352$
"dropout_18/StatefulPartitionedCallÄ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_23_1120046dense_23_1120048*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_11194032"
 dense_23/StatefulPartitionedCallÃ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_11197022$
"dropout_19/StatefulPartitionedCall
add_13/PartitionedCallPartitionedCall+dropout_19/StatefulPartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_13_layer_call_and_return_conditional_losses_11194222
add_13/PartitionedCallþ
.layer_normalization_14/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_14_1120053layer_normalization_14_1120055*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_111944620
.layer_normalization_14/StatefulPartitionedCallº
.multi_head_attention_7/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:07layer_normalization_14/StatefulPartitionedCall:output:0multi_head_attention_7_1120058multi_head_attention_7_1120060multi_head_attention_7_1120062multi_head_attention_7_1120064multi_head_attention_7_1120066multi_head_attention_7_1120068multi_head_attention_7_1120070multi_head_attention_7_1120072*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_111964420
.multi_head_attention_7/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_7/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

IdentityÌ
NoOpNoOp*^data_augmentation/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^layer_normalization_13/StatefulPartitionedCall/^layer_normalization_14/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall/^multi_head_attention_7/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.layer_normalization_13/StatefulPartitionedCall.layer_normalization_13/StatefulPartitionedCall2`
.layer_normalization_14/StatefulPartitionedCall.layer_normalization_14/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2`
.multi_head_attention_7/StatefulPartitionedCall.multi_head_attention_7/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
ß
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_1122138

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeº
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÄ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/Mul_1k
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs

e
G__inference_dropout_18_layer_call_and_return_conditional_losses_1122126

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs
Ó
F
*__inference_lambda_3_layer_call_fn_1121808

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11191662
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs

Ô
*__inference_model_11_layer_call_fn_1120533

inputs
unknown
	unknown_0
	unknown_1:1@
	unknown_2:@
	unknown_3:	¥@
	unknown_4:@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:	@

unknown_17:	

unknown_18:	@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@ 

unknown_28:@@

unknown_29:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_11_layer_call_and_return_conditional_losses_11195082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
^
ò
E__inference_model_11_layer_call_and_return_conditional_losses_1120306
input_4
data_augmentation_1120224
data_augmentation_1120226"
dense_21_1120230:1@
dense_21_1120232:@*
patch_encoder_7_1120237:	¥@,
layer_normalization_12_1120240:@,
layer_normalization_12_1120242:@4
multi_head_attention_6_1120245:@@0
multi_head_attention_6_1120247:@4
multi_head_attention_6_1120249:@@0
multi_head_attention_6_1120251:@4
multi_head_attention_6_1120253:@@0
multi_head_attention_6_1120255:@4
multi_head_attention_6_1120257:@@,
multi_head_attention_6_1120259:@,
layer_normalization_13_1120264:@,
layer_normalization_13_1120266:@#
dense_22_1120269:	@
dense_22_1120271:	#
dense_23_1120275:	@
dense_23_1120277:@,
layer_normalization_14_1120282:@,
layer_normalization_14_1120284:@4
multi_head_attention_7_1120287:@@0
multi_head_attention_7_1120289:@4
multi_head_attention_7_1120291:@@0
multi_head_attention_7_1120293:@4
multi_head_attention_7_1120295:@@0
multi_head_attention_7_1120297:@4
multi_head_attention_7_1120299:@@,
multi_head_attention_7_1120301:@
identity¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.layer_normalization_13/StatefulPartitionedCall¢.layer_normalization_14/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢.multi_head_attention_7/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCallµ
!data_augmentation/PartitionedCallPartitionedCallinput_4data_augmentation_1120224data_augmentation_1120226*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11186512#
!data_augmentation/PartitionedCall
patches_7/PartitionedCallPartitionedCall*data_augmentation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11191202
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1120230dense_21_1120232*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11191522"
 dense_21/StatefulPartitionedCallÿ
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11191662
lambda_3/PartitionedCall³
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11191752
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1120237*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11191912)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1120240layer_normalization_12_1120242*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111921720
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1120245multi_head_attention_6_1120247multi_head_attention_6_1120249multi_head_attention_6_1120251multi_head_attention_6_1120253multi_head_attention_6_1120255multi_head_attention_6_1120257multi_head_attention_6_1120259*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111925920
.multi_head_attention_6/StatefulPartitionedCall»
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:00patch_encoder_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_12_layer_call_and_return_conditional_losses_11192842
add_12/PartitionedCallþ
.layer_normalization_13/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_13_1120264layer_normalization_13_1120266*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_111930820
.layer_normalization_13/StatefulPartitionedCallÑ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:0dense_22_1120269dense_22_1120271*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_11193522"
 dense_22/StatefulPartitionedCall
dropout_18/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_11193632
dropout_18/PartitionedCall¼
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_23_1120275dense_23_1120277*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_11194032"
 dense_23/StatefulPartitionedCall
dropout_19/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_11194142
dropout_19/PartitionedCall
add_13/PartitionedCallPartitionedCall#dropout_19/PartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_13_layer_call_and_return_conditional_losses_11194222
add_13/PartitionedCallþ
.layer_normalization_14/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_14_1120282layer_normalization_14_1120284*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_111944620
.layer_normalization_14/StatefulPartitionedCallº
.multi_head_attention_7/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:07layer_normalization_14/StatefulPartitionedCall:output:0multi_head_attention_7_1120287multi_head_attention_7_1120289multi_head_attention_7_1120291multi_head_attention_7_1120293multi_head_attention_7_1120295multi_head_attention_7_1120297multi_head_attention_7_1120299multi_head_attention_7_1120301*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_111948820
.multi_head_attention_7/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_7/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

IdentityÖ
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^layer_normalization_13/StatefulPartitionedCall/^layer_normalization_14/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall/^multi_head_attention_7/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.layer_normalization_13/StatefulPartitionedCall.layer_normalization_13/StatefulPartitionedCall2`
.layer_normalization_14/StatefulPartitionedCall.layer_normalization_14/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2`
.multi_head_attention_7/StatefulPartitionedCall.multi_head_attention_7/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
¡
f
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1122617

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
Ø
z
3__inference_data_augmentation_layer_call_fn_1118658
normalization_3_input
unknown
	unknown_0
identityý
PartitionedCallPartitionedCallnormalization_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11186512
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:::h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:
ù
Ð
%__inference_signature_wrapper_1120466
input_4
unknown
	unknown_0
	unknown_1:1@
	unknown_2:@
	unknown_3:	¥@
	unknown_4:@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:	@

unknown_17:	

unknown_18:	@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@ 

unknown_28:@@

unknown_29:@
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_11186102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:


S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_1119446

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesÀ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
0

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1121978	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp¸
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpÈ
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
query/einsum/Einsum
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	query/add²
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpÂ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
key/einsum/Einsum
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2	
key/add¸
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpÈ
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde2
value/einsum/Einsum
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
Mul¢
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
softmax/Softmax
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2
dropout/Identity¹
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd2
einsum_1/EinsumÙ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpø
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum³
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpÂ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity_1à
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@

_user_specified_namevalue
Ê-
Ó
__inference_adapt_step_1113455
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOpØ
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
output_shapes
:ÿÿÿÿÿÿÿÿÿ*
output_types
22
IteratorGetNext}
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:2
moments/StopGradient°
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indicesº
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1V
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	2
Shapey
GatherV2/indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis«
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addW
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1Q
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_2V
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4£
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
ß
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_1119735

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeº
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÄ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
dropout/Mul_1k
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs
ì
[
/__inference_concatenate_3_layer_call_fn_1121835
inputs_0
inputs_1
identityÝ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11191752
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ¤@:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
"
_user_specified_name
inputs/1
ß
G
+__inference_patches_7_layer_call_fn_1121757

images
identityÌ
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11191202
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameimages

e
G__inference_dropout_18_layer_call_and_return_conditional_losses_1119363

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs
¡

N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119037

inputs
normalization_3_sub_y
normalization_3_sqrt_x#
random_flip_3_1119027:	'
random_rotation_3_1119030:	#
random_zoom_3_1119033:	
identity¢%random_flip_3/StatefulPartitionedCall¢)random_rotation_3/StatefulPartitionedCall¢%random_zoom_3/StatefulPartitionedCall
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/sub}
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*&
_output_shapes
:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_3/Maximum/y¬
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum¯
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/truedivü
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11186302
resizing_3/PartitionedCall½
%random_flip_3/StatefulPartitionedCallStatefulPartitionedCall#resizing_3/PartitionedCall:output:0random_flip_3_1119027*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11189942'
%random_flip_3/StatefulPartitionedCallØ
)random_rotation_3/StatefulPartitionedCallStatefulPartitionedCall.random_flip_3/StatefulPartitionedCall:output:0random_rotation_3_1119030*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11189232+
)random_rotation_3/StatefulPartitionedCallÌ
%random_zoom_3/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_3/StatefulPartitionedCall:output:0random_zoom_3_1119033*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11187922'
%random_zoom_3/StatefulPartitionedCall
IdentityIdentity.random_zoom_3/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

IdentityÊ
NoOpNoOp&^random_flip_3/StatefulPartitionedCall*^random_rotation_3/StatefulPartitionedCall&^random_zoom_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ::: : : 2N
%random_flip_3/StatefulPartitionedCall%random_flip_3/StatefulPartitionedCall2V
)random_rotation_3/StatefulPartitionedCall)random_rotation_3/StatefulPartitionedCall2N
%random_zoom_3/StatefulPartitionedCall%random_zoom_3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ö
f
G__inference_dropout_19_layer_call_and_return_conditional_losses_1119702

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
Íd
ã
E__inference_model_11_layer_call_and_return_conditional_losses_1120397
input_4
data_augmentation_1120309
data_augmentation_1120311'
data_augmentation_1120313:	'
data_augmentation_1120315:	'
data_augmentation_1120317:	"
dense_21_1120321:1@
dense_21_1120323:@*
patch_encoder_7_1120328:	¥@,
layer_normalization_12_1120331:@,
layer_normalization_12_1120333:@4
multi_head_attention_6_1120336:@@0
multi_head_attention_6_1120338:@4
multi_head_attention_6_1120340:@@0
multi_head_attention_6_1120342:@4
multi_head_attention_6_1120344:@@0
multi_head_attention_6_1120346:@4
multi_head_attention_6_1120348:@@,
multi_head_attention_6_1120350:@,
layer_normalization_13_1120355:@,
layer_normalization_13_1120357:@#
dense_22_1120360:	@
dense_22_1120362:	#
dense_23_1120366:	@
dense_23_1120368:@,
layer_normalization_14_1120373:@,
layer_normalization_14_1120375:@4
multi_head_attention_7_1120378:@@0
multi_head_attention_7_1120380:@4
multi_head_attention_7_1120382:@@0
multi_head_attention_7_1120384:@4
multi_head_attention_7_1120386:@@0
multi_head_attention_7_1120388:@4
multi_head_attention_7_1120390:@@,
multi_head_attention_7_1120392:@
identity¢)data_augmentation/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢"dropout_18/StatefulPartitionedCall¢"dropout_19/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.layer_normalization_13/StatefulPartitionedCall¢.layer_normalization_14/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢.multi_head_attention_7/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCall¡
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinput_4data_augmentation_1120309data_augmentation_1120311data_augmentation_1120313data_augmentation_1120315data_augmentation_1120317*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11190372+
)data_augmentation/StatefulPartitionedCall
patches_7/PartitionedCallPartitionedCall2data_augmentation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11191202
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1120321dense_21_1120323*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11191522"
 dense_21/StatefulPartitionedCallÿ
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11198982
lambda_3/PartitionedCall³
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11191752
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1120328*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11191912)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1120331layer_normalization_12_1120333*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111921720
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1120336multi_head_attention_6_1120338multi_head_attention_6_1120340multi_head_attention_6_1120342multi_head_attention_6_1120344multi_head_attention_6_1120346multi_head_attention_6_1120348multi_head_attention_6_1120350*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111983620
.multi_head_attention_6/StatefulPartitionedCall»
add_12/PartitionedCallPartitionedCall7multi_head_attention_6/StatefulPartitionedCall:output:00patch_encoder_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_12_layer_call_and_return_conditional_losses_11192842
add_12/PartitionedCallþ
.layer_normalization_13/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0layer_normalization_13_1120355layer_normalization_13_1120357*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_111930820
.layer_normalization_13/StatefulPartitionedCallÑ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_13/StatefulPartitionedCall:output:0dense_22_1120360dense_22_1120362*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_11193522"
 dense_22/StatefulPartitionedCall
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_11197352$
"dropout_18/StatefulPartitionedCallÄ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_23_1120366dense_23_1120368*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_11194032"
 dense_23/StatefulPartitionedCallÃ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_19_layer_call_and_return_conditional_losses_11197022$
"dropout_19/StatefulPartitionedCall
add_13/PartitionedCallPartitionedCall+dropout_19/StatefulPartitionedCall:output:0add_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_13_layer_call_and_return_conditional_losses_11194222
add_13/PartitionedCallþ
.layer_normalization_14/StatefulPartitionedCallStatefulPartitionedCalladd_13/PartitionedCall:output:0layer_normalization_14_1120373layer_normalization_14_1120375*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_111944620
.layer_normalization_14/StatefulPartitionedCallº
.multi_head_attention_7/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_14/StatefulPartitionedCall:output:07layer_normalization_14/StatefulPartitionedCall:output:0multi_head_attention_7_1120378multi_head_attention_7_1120380multi_head_attention_7_1120382multi_head_attention_7_1120384multi_head_attention_7_1120386multi_head_attention_7_1120388multi_head_attention_7_1120390multi_head_attention_7_1120392*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥¥**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_111964420
.multi_head_attention_7/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_7/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

IdentityÌ
NoOpNoOp*^data_augmentation/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^layer_normalization_13/StatefulPartitionedCall/^layer_normalization_14/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall/^multi_head_attention_7/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.layer_normalization_13/StatefulPartitionedCall.layer_normalization_13/StatefulPartitionedCall2`
.layer_normalization_14/StatefulPartitionedCall.layer_normalization_14/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2`
.multi_head_attention_7/StatefulPartitionedCall.multi_head_attention_7/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
£
¡
8__inference_layer_normalization_12_layer_call_fn_1121872

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_11192172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
Þ
°
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119080
normalization_3_input
normalization_3_sub_y
normalization_3_sqrt_x
identity
normalization_3/subSubnormalization_3_inputnormalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/sub}
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*&
_output_shapes
:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_3/Maximum/y¬
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum¯
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_3/truedivü
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11186302
resizing_3/PartitionedCall
random_flip_3/PartitionedCallPartitionedCall#resizing_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11186362
random_flip_3/PartitionedCall
!random_rotation_3/PartitionedCallPartitionedCall&random_flip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11186422#
!random_rotation_3/PartitionedCall
random_zoom_3/PartitionedCallPartitionedCall*random_rotation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11186482
random_zoom_3/PartitionedCall
IdentityIdentity&random_zoom_3/PartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:::h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:


S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_1122255

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesÀ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
òÇ
¼"
"__inference__wrapped_model_1118610
input_44
0model_11_data_augmentation_normalization_3_sub_y5
1model_11_data_augmentation_normalization_3_sqrt_xE
3model_11_dense_21_tensordot_readvariableop_resource:1@?
1model_11_dense_21_biasadd_readvariableop_resource:@P
=model_11_patch_encoder_7_embedding_3_embedding_lookup_1118417:	¥@S
Emodel_11_layer_normalization_12_batchnorm_mul_readvariableop_resource:@O
Amodel_11_layer_normalization_12_batchnorm_readvariableop_resource:@a
Kmodel_11_multi_head_attention_6_query_einsum_einsum_readvariableop_resource:@@S
Amodel_11_multi_head_attention_6_query_add_readvariableop_resource:@_
Imodel_11_multi_head_attention_6_key_einsum_einsum_readvariableop_resource:@@Q
?model_11_multi_head_attention_6_key_add_readvariableop_resource:@a
Kmodel_11_multi_head_attention_6_value_einsum_einsum_readvariableop_resource:@@S
Amodel_11_multi_head_attention_6_value_add_readvariableop_resource:@l
Vmodel_11_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:@@Z
Lmodel_11_multi_head_attention_6_attention_output_add_readvariableop_resource:@S
Emodel_11_layer_normalization_13_batchnorm_mul_readvariableop_resource:@O
Amodel_11_layer_normalization_13_batchnorm_readvariableop_resource:@F
3model_11_dense_22_tensordot_readvariableop_resource:	@@
1model_11_dense_22_biasadd_readvariableop_resource:	F
3model_11_dense_23_tensordot_readvariableop_resource:	@?
1model_11_dense_23_biasadd_readvariableop_resource:@S
Emodel_11_layer_normalization_14_batchnorm_mul_readvariableop_resource:@O
Amodel_11_layer_normalization_14_batchnorm_readvariableop_resource:@a
Kmodel_11_multi_head_attention_7_query_einsum_einsum_readvariableop_resource:@@S
Amodel_11_multi_head_attention_7_query_add_readvariableop_resource:@_
Imodel_11_multi_head_attention_7_key_einsum_einsum_readvariableop_resource:@@Q
?model_11_multi_head_attention_7_key_add_readvariableop_resource:@a
Kmodel_11_multi_head_attention_7_value_einsum_einsum_readvariableop_resource:@@S
Amodel_11_multi_head_attention_7_value_add_readvariableop_resource:@l
Vmodel_11_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource:@@Z
Lmodel_11_multi_head_attention_7_attention_output_add_readvariableop_resource:@
identity¢(model_11/dense_21/BiasAdd/ReadVariableOp¢*model_11/dense_21/Tensordot/ReadVariableOp¢(model_11/dense_22/BiasAdd/ReadVariableOp¢*model_11/dense_22/Tensordot/ReadVariableOp¢(model_11/dense_23/BiasAdd/ReadVariableOp¢*model_11/dense_23/Tensordot/ReadVariableOp¢8model_11/layer_normalization_12/batchnorm/ReadVariableOp¢<model_11/layer_normalization_12/batchnorm/mul/ReadVariableOp¢8model_11/layer_normalization_13/batchnorm/ReadVariableOp¢<model_11/layer_normalization_13/batchnorm/mul/ReadVariableOp¢8model_11/layer_normalization_14/batchnorm/ReadVariableOp¢<model_11/layer_normalization_14/batchnorm/mul/ReadVariableOp¢Cmodel_11/multi_head_attention_6/attention_output/add/ReadVariableOp¢Mmodel_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp¢6model_11/multi_head_attention_6/key/add/ReadVariableOp¢@model_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¢8model_11/multi_head_attention_6/query/add/ReadVariableOp¢Bmodel_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp¢8model_11/multi_head_attention_6/value/add/ReadVariableOp¢Bmodel_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp¢Cmodel_11/multi_head_attention_7/attention_output/add/ReadVariableOp¢Mmodel_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp¢6model_11/multi_head_attention_7/key/add/ReadVariableOp¢@model_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp¢8model_11/multi_head_attention_7/query/add/ReadVariableOp¢Bmodel_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp¢8model_11/multi_head_attention_7/value/add/ReadVariableOp¢Bmodel_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp¢5model_11/patch_encoder_7/embedding_3/embedding_lookupÞ
.model_11/data_augmentation/normalization_3/subSubinput_40model_11_data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_11/data_augmentation/normalization_3/subÎ
/model_11/data_augmentation/normalization_3/SqrtSqrt1model_11_data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:21
/model_11/data_augmentation/normalization_3/Sqrt±
4model_11/data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö326
4model_11/data_augmentation/normalization_3/Maximum/y
2model_11/data_augmentation/normalization_3/MaximumMaximum3model_11/data_augmentation/normalization_3/Sqrt:y:0=model_11/data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:24
2model_11/data_augmentation/normalization_3/Maximum
2model_11/data_augmentation/normalization_3/truedivRealDiv2model_11/data_augmentation/normalization_3/sub:z:06model_11/data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2model_11/data_augmentation/normalization_3/truediv·
1model_11/data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   23
1model_11/data_augmentation/resizing_3/resize/sizeÕ
;model_11/data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear6model_11/data_augmentation/normalization_3/truediv:z:0:model_11/data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(2=
;model_11/data_augmentation/resizing_3/resize/ResizeBilinear¿
&model_11/patches_7/ExtractImagePatchesExtractImagePatchesLmodel_11/data_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksizes
*
paddingVALID*
rates
*
strides
2(
&model_11/patches_7/ExtractImagePatches
 model_11/patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ¤  1   2"
 model_11/patches_7/Reshape/shape×
model_11/patches_7/ReshapeReshape0model_11/patches_7/ExtractImagePatches:patches:0)model_11/patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
model_11/patches_7/ReshapeÌ
*model_11/dense_21/Tensordot/ReadVariableOpReadVariableOp3model_11_dense_21_tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02,
*model_11/dense_21/Tensordot/ReadVariableOp
 model_11/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 model_11/dense_21/Tensordot/axes
 model_11/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model_11/dense_21/Tensordot/free
!model_11/dense_21/Tensordot/ShapeShape#model_11/patches_7/Reshape:output:0*
T0*
_output_shapes
:2#
!model_11/dense_21/Tensordot/Shape
)model_11/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_11/dense_21/Tensordot/GatherV2/axis«
$model_11/dense_21/Tensordot/GatherV2GatherV2*model_11/dense_21/Tensordot/Shape:output:0)model_11/dense_21/Tensordot/free:output:02model_11/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_11/dense_21/Tensordot/GatherV2
+model_11/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_11/dense_21/Tensordot/GatherV2_1/axis±
&model_11/dense_21/Tensordot/GatherV2_1GatherV2*model_11/dense_21/Tensordot/Shape:output:0)model_11/dense_21/Tensordot/axes:output:04model_11/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_11/dense_21/Tensordot/GatherV2_1
!model_11/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_11/dense_21/Tensordot/ConstÈ
 model_11/dense_21/Tensordot/ProdProd-model_11/dense_21/Tensordot/GatherV2:output:0*model_11/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 model_11/dense_21/Tensordot/Prod
#model_11/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#model_11/dense_21/Tensordot/Const_1Ð
"model_11/dense_21/Tensordot/Prod_1Prod/model_11/dense_21/Tensordot/GatherV2_1:output:0,model_11/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"model_11/dense_21/Tensordot/Prod_1
'model_11/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_11/dense_21/Tensordot/concat/axis
"model_11/dense_21/Tensordot/concatConcatV2)model_11/dense_21/Tensordot/free:output:0)model_11/dense_21/Tensordot/axes:output:00model_11/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_11/dense_21/Tensordot/concatÔ
!model_11/dense_21/Tensordot/stackPack)model_11/dense_21/Tensordot/Prod:output:0+model_11/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!model_11/dense_21/Tensordot/stackä
%model_11/dense_21/Tensordot/transpose	Transpose#model_11/patches_7/Reshape:output:0+model_11/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12'
%model_11/dense_21/Tensordot/transposeç
#model_11/dense_21/Tensordot/ReshapeReshape)model_11/dense_21/Tensordot/transpose:y:0*model_11/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#model_11/dense_21/Tensordot/Reshapeæ
"model_11/dense_21/Tensordot/MatMulMatMul,model_11/dense_21/Tensordot/Reshape:output:02model_11/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"model_11/dense_21/Tensordot/MatMul
#model_11/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2%
#model_11/dense_21/Tensordot/Const_2
)model_11/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_11/dense_21/Tensordot/concat_1/axis
$model_11/dense_21/Tensordot/concat_1ConcatV2-model_11/dense_21/Tensordot/GatherV2:output:0,model_11/dense_21/Tensordot/Const_2:output:02model_11/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_11/dense_21/Tensordot/concat_1Ù
model_11/dense_21/TensordotReshape,model_11/dense_21/Tensordot/MatMul:product:0-model_11/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
model_11/dense_21/TensordotÂ
(model_11/dense_21/BiasAdd/ReadVariableOpReadVariableOp1model_11_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_11/dense_21/BiasAdd/ReadVariableOpÐ
model_11/dense_21/BiasAddBiasAdd$model_11/dense_21/Tensordot:output:00model_11/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
model_11/dense_21/BiasAdd
(model_11/lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_11/lambda_3/Mean/reduction_indicesÁ
model_11/lambda_3/MeanMean"model_11/dense_21/BiasAdd:output:01model_11/lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_11/lambda_3/Mean
model_11/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2!
model_11/lambda_3/Reshape/shapeÂ
model_11/lambda_3/ReshapeReshapemodel_11/lambda_3/Mean:output:0(model_11/lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_11/lambda_3/Reshape
"model_11/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/concatenate_3/concat/axisÿ
model_11/concatenate_3/concatConcatV2"model_11/lambda_3/Reshape:output:0"model_11/dense_21/BiasAdd:output:0+model_11/concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/concatenate_3/concat
$model_11/patch_encoder_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model_11/patch_encoder_7/range/start
$model_11/patch_encoder_7/range/limitConst*
_output_shapes
: *
dtype0*
value
B :¥2&
$model_11/patch_encoder_7/range/limit
$model_11/patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/patch_encoder_7/range/deltaó
model_11/patch_encoder_7/rangeRange-model_11/patch_encoder_7/range/start:output:0-model_11/patch_encoder_7/range/limit:output:0-model_11/patch_encoder_7/range/delta:output:0*
_output_shapes	
:¥2 
model_11/patch_encoder_7/range¦
5model_11/patch_encoder_7/embedding_3/embedding_lookupResourceGather=model_11_patch_encoder_7_embedding_3_embedding_lookup_1118417'model_11/patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*P
_classF
DBloc:@model_11/patch_encoder_7/embedding_3/embedding_lookup/1118417*
_output_shapes
:	¥@*
dtype027
5model_11/patch_encoder_7/embedding_3/embedding_lookupö
>model_11/patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity>model_11/patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*P
_classF
DBloc:@model_11/patch_encoder_7/embedding_3/embedding_lookup/1118417*
_output_shapes
:	¥@2@
>model_11/patch_encoder_7/embedding_3/embedding_lookup/Identity
@model_11/patch_encoder_7/embedding_3/embedding_lookup/Identity_1IdentityGmodel_11/patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	¥@2B
@model_11/patch_encoder_7/embedding_3/embedding_lookup/Identity_1ï
model_11/patch_encoder_7/addAddV2&model_11/concatenate_3/concat:output:0Imodel_11/patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/patch_encoder_7/addÊ
>model_11/layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2@
>model_11/layer_normalization_12/moments/mean/reduction_indices
,model_11/layer_normalization_12/moments/meanMean model_11/patch_encoder_7/add:z:0Gmodel_11/layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2.
,model_11/layer_normalization_12/moments/meanê
4model_11/layer_normalization_12/moments/StopGradientStopGradient5model_11/layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥26
4model_11/layer_normalization_12/moments/StopGradient£
9model_11/layer_normalization_12/moments/SquaredDifferenceSquaredDifference model_11/patch_encoder_7/add:z:0=model_11/layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2;
9model_11/layer_normalization_12/moments/SquaredDifferenceÒ
Bmodel_11/layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_11/layer_normalization_12/moments/variance/reduction_indicesÀ
0model_11/layer_normalization_12/moments/varianceMean=model_11/layer_normalization_12/moments/SquaredDifference:z:0Kmodel_11/layer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(22
0model_11/layer_normalization_12/moments/variance§
/model_11/layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½7521
/model_11/layer_normalization_12/batchnorm/add/y
-model_11/layer_normalization_12/batchnorm/addAddV29model_11/layer_normalization_12/moments/variance:output:08model_11/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2/
-model_11/layer_normalization_12/batchnorm/addÕ
/model_11/layer_normalization_12/batchnorm/RsqrtRsqrt1model_11/layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥21
/model_11/layer_normalization_12/batchnorm/Rsqrtþ
<model_11/layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_11_layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model_11/layer_normalization_12/batchnorm/mul/ReadVariableOp
-model_11/layer_normalization_12/batchnorm/mulMul3model_11/layer_normalization_12/batchnorm/Rsqrt:y:0Dmodel_11/layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_11/layer_normalization_12/batchnorm/mulõ
/model_11/layer_normalization_12/batchnorm/mul_1Mul model_11/patch_encoder_7/add:z:01model_11/layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_12/batchnorm/mul_1
/model_11/layer_normalization_12/batchnorm/mul_2Mul5model_11/layer_normalization_12/moments/mean:output:01model_11/layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_12/batchnorm/mul_2ò
8model_11/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOpAmodel_11_layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_11/layer_normalization_12/batchnorm/ReadVariableOp
-model_11/layer_normalization_12/batchnorm/subSub@model_11/layer_normalization_12/batchnorm/ReadVariableOp:value:03model_11/layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_11/layer_normalization_12/batchnorm/sub
/model_11/layer_normalization_12/batchnorm/add_1AddV23model_11/layer_normalization_12/batchnorm/mul_1:z:01model_11/layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_12/batchnorm/add_1
Bmodel_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_11_multi_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpÖ
3model_11/multi_head_attention_6/query/einsum/EinsumEinsum3model_11/layer_normalization_12/batchnorm/add_1:z:0Jmodel_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde25
3model_11/multi_head_attention_6/query/einsum/Einsumö
8model_11/multi_head_attention_6/query/add/ReadVariableOpReadVariableOpAmodel_11_multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_11/multi_head_attention_6/query/add/ReadVariableOp
)model_11/multi_head_attention_6/query/addAddV2<model_11/multi_head_attention_6/query/einsum/Einsum:output:0@model_11/multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2+
)model_11/multi_head_attention_6/query/add
@model_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOpImodel_11_multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02B
@model_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpÐ
1model_11/multi_head_attention_6/key/einsum/EinsumEinsum3model_11/layer_normalization_12/batchnorm/add_1:z:0Hmodel_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde23
1model_11/multi_head_attention_6/key/einsum/Einsumð
6model_11/multi_head_attention_6/key/add/ReadVariableOpReadVariableOp?model_11_multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype028
6model_11/multi_head_attention_6/key/add/ReadVariableOp
'model_11/multi_head_attention_6/key/addAddV2:model_11/multi_head_attention_6/key/einsum/Einsum:output:0>model_11/multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2)
'model_11/multi_head_attention_6/key/add
Bmodel_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_11_multi_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpÖ
3model_11/multi_head_attention_6/value/einsum/EinsumEinsum3model_11/layer_normalization_12/batchnorm/add_1:z:0Jmodel_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde25
3model_11/multi_head_attention_6/value/einsum/Einsumö
8model_11/multi_head_attention_6/value/add/ReadVariableOpReadVariableOpAmodel_11_multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_11/multi_head_attention_6/value/add/ReadVariableOp
)model_11/multi_head_attention_6/value/addAddV2<model_11/multi_head_attention_6/value/einsum/Einsum:output:0@model_11/multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2+
)model_11/multi_head_attention_6/value/add
%model_11/multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2'
%model_11/multi_head_attention_6/Mul/yë
#model_11/multi_head_attention_6/MulMul-model_11/multi_head_attention_6/query/add:z:0.model_11/multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2%
#model_11/multi_head_attention_6/Mul¢
-model_11/multi_head_attention_6/einsum/EinsumEinsum+model_11/multi_head_attention_6/key/add:z:0'model_11/multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2/
-model_11/multi_head_attention_6/einsum/Einsumá
/model_11/multi_head_attention_6/softmax/SoftmaxSoftmax6model_11/multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥21
/model_11/multi_head_attention_6/softmax/Softmaxç
0model_11/multi_head_attention_6/dropout/IdentityIdentity9model_11/multi_head_attention_6/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥22
0model_11/multi_head_attention_6/dropout/Identity¹
/model_11/multi_head_attention_6/einsum_1/EinsumEinsum9model_11/multi_head_attention_6/dropout/Identity:output:0-model_11/multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd21
/model_11/multi_head_attention_6/einsum_1/Einsum¹
Mmodel_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_11_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
Mmodel_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpø
>model_11/multi_head_attention_6/attention_output/einsum/EinsumEinsum8model_11/multi_head_attention_6/einsum_1/Einsum:output:0Umodel_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2@
>model_11/multi_head_attention_6/attention_output/einsum/Einsum
Cmodel_11/multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpLmodel_11_multi_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cmodel_11/multi_head_attention_6/attention_output/add/ReadVariableOpÂ
4model_11/multi_head_attention_6/attention_output/addAddV2Gmodel_11/multi_head_attention_6/attention_output/einsum/Einsum:output:0Kmodel_11/multi_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@26
4model_11/multi_head_attention_6/attention_output/addÆ
model_11/add_12/addAddV28model_11/multi_head_attention_6/attention_output/add:z:0 model_11/patch_encoder_7/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/add_12/addÊ
>model_11/layer_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2@
>model_11/layer_normalization_13/moments/mean/reduction_indices
,model_11/layer_normalization_13/moments/meanMeanmodel_11/add_12/add:z:0Gmodel_11/layer_normalization_13/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2.
,model_11/layer_normalization_13/moments/meanê
4model_11/layer_normalization_13/moments/StopGradientStopGradient5model_11/layer_normalization_13/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥26
4model_11/layer_normalization_13/moments/StopGradient
9model_11/layer_normalization_13/moments/SquaredDifferenceSquaredDifferencemodel_11/add_12/add:z:0=model_11/layer_normalization_13/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2;
9model_11/layer_normalization_13/moments/SquaredDifferenceÒ
Bmodel_11/layer_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_11/layer_normalization_13/moments/variance/reduction_indicesÀ
0model_11/layer_normalization_13/moments/varianceMean=model_11/layer_normalization_13/moments/SquaredDifference:z:0Kmodel_11/layer_normalization_13/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(22
0model_11/layer_normalization_13/moments/variance§
/model_11/layer_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½7521
/model_11/layer_normalization_13/batchnorm/add/y
-model_11/layer_normalization_13/batchnorm/addAddV29model_11/layer_normalization_13/moments/variance:output:08model_11/layer_normalization_13/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2/
-model_11/layer_normalization_13/batchnorm/addÕ
/model_11/layer_normalization_13/batchnorm/RsqrtRsqrt1model_11/layer_normalization_13/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥21
/model_11/layer_normalization_13/batchnorm/Rsqrtþ
<model_11/layer_normalization_13/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_11_layer_normalization_13_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model_11/layer_normalization_13/batchnorm/mul/ReadVariableOp
-model_11/layer_normalization_13/batchnorm/mulMul3model_11/layer_normalization_13/batchnorm/Rsqrt:y:0Dmodel_11/layer_normalization_13/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_11/layer_normalization_13/batchnorm/mulì
/model_11/layer_normalization_13/batchnorm/mul_1Mulmodel_11/add_12/add:z:01model_11/layer_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_13/batchnorm/mul_1
/model_11/layer_normalization_13/batchnorm/mul_2Mul5model_11/layer_normalization_13/moments/mean:output:01model_11/layer_normalization_13/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_13/batchnorm/mul_2ò
8model_11/layer_normalization_13/batchnorm/ReadVariableOpReadVariableOpAmodel_11_layer_normalization_13_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_11/layer_normalization_13/batchnorm/ReadVariableOp
-model_11/layer_normalization_13/batchnorm/subSub@model_11/layer_normalization_13/batchnorm/ReadVariableOp:value:03model_11/layer_normalization_13/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_11/layer_normalization_13/batchnorm/sub
/model_11/layer_normalization_13/batchnorm/add_1AddV23model_11/layer_normalization_13/batchnorm/mul_1:z:01model_11/layer_normalization_13/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_13/batchnorm/add_1Í
*model_11/dense_22/Tensordot/ReadVariableOpReadVariableOp3model_11_dense_22_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*model_11/dense_22/Tensordot/ReadVariableOp
 model_11/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 model_11/dense_22/Tensordot/axes
 model_11/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model_11/dense_22/Tensordot/free©
!model_11/dense_22/Tensordot/ShapeShape3model_11/layer_normalization_13/batchnorm/add_1:z:0*
T0*
_output_shapes
:2#
!model_11/dense_22/Tensordot/Shape
)model_11/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_11/dense_22/Tensordot/GatherV2/axis«
$model_11/dense_22/Tensordot/GatherV2GatherV2*model_11/dense_22/Tensordot/Shape:output:0)model_11/dense_22/Tensordot/free:output:02model_11/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_11/dense_22/Tensordot/GatherV2
+model_11/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_11/dense_22/Tensordot/GatherV2_1/axis±
&model_11/dense_22/Tensordot/GatherV2_1GatherV2*model_11/dense_22/Tensordot/Shape:output:0)model_11/dense_22/Tensordot/axes:output:04model_11/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_11/dense_22/Tensordot/GatherV2_1
!model_11/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_11/dense_22/Tensordot/ConstÈ
 model_11/dense_22/Tensordot/ProdProd-model_11/dense_22/Tensordot/GatherV2:output:0*model_11/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 model_11/dense_22/Tensordot/Prod
#model_11/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#model_11/dense_22/Tensordot/Const_1Ð
"model_11/dense_22/Tensordot/Prod_1Prod/model_11/dense_22/Tensordot/GatherV2_1:output:0,model_11/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"model_11/dense_22/Tensordot/Prod_1
'model_11/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_11/dense_22/Tensordot/concat/axis
"model_11/dense_22/Tensordot/concatConcatV2)model_11/dense_22/Tensordot/free:output:0)model_11/dense_22/Tensordot/axes:output:00model_11/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_11/dense_22/Tensordot/concatÔ
!model_11/dense_22/Tensordot/stackPack)model_11/dense_22/Tensordot/Prod:output:0+model_11/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!model_11/dense_22/Tensordot/stackô
%model_11/dense_22/Tensordot/transpose	Transpose3model_11/layer_normalization_13/batchnorm/add_1:z:0+model_11/dense_22/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2'
%model_11/dense_22/Tensordot/transposeç
#model_11/dense_22/Tensordot/ReshapeReshape)model_11/dense_22/Tensordot/transpose:y:0*model_11/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#model_11/dense_22/Tensordot/Reshapeç
"model_11/dense_22/Tensordot/MatMulMatMul,model_11/dense_22/Tensordot/Reshape:output:02model_11/dense_22/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"model_11/dense_22/Tensordot/MatMul
#model_11/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_11/dense_22/Tensordot/Const_2
)model_11/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_11/dense_22/Tensordot/concat_1/axis
$model_11/dense_22/Tensordot/concat_1ConcatV2-model_11/dense_22/Tensordot/GatherV2:output:0,model_11/dense_22/Tensordot/Const_2:output:02model_11/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_11/dense_22/Tensordot/concat_1Ú
model_11/dense_22/TensordotReshape,model_11/dense_22/Tensordot/MatMul:product:0-model_11/dense_22/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dense_22/TensordotÃ
(model_11/dense_22/BiasAdd/ReadVariableOpReadVariableOp1model_11_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_11/dense_22/BiasAdd/ReadVariableOpÑ
model_11/dense_22/BiasAddBiasAdd$model_11/dense_22/Tensordot:output:00model_11/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dense_22/BiasAdd
model_11/dense_22/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_11/dense_22/Gelu/mul/xÂ
model_11/dense_22/Gelu/mulMul%model_11/dense_22/Gelu/mul/x:output:0"model_11/dense_22/BiasAdd:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dense_22/Gelu/mul
model_11/dense_22/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
model_11/dense_22/Gelu/Cast/xÏ
model_11/dense_22/Gelu/truedivRealDiv"model_11/dense_22/BiasAdd:output:0&model_11/dense_22/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2 
model_11/dense_22/Gelu/truediv
model_11/dense_22/Gelu/ErfErf"model_11/dense_22/Gelu/truediv:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dense_22/Gelu/Erf
model_11/dense_22/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_11/dense_22/Gelu/add/xÀ
model_11/dense_22/Gelu/addAddV2%model_11/dense_22/Gelu/add/x:output:0model_11/dense_22/Gelu/Erf:y:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dense_22/Gelu/add»
model_11/dense_22/Gelu/mul_1Mulmodel_11/dense_22/Gelu/mul:z:0model_11/dense_22/Gelu/add:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dense_22/Gelu/mul_1¢
model_11/dropout_18/IdentityIdentity model_11/dense_22/Gelu/mul_1:z:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
model_11/dropout_18/IdentityÍ
*model_11/dense_23/Tensordot/ReadVariableOpReadVariableOp3model_11_dense_23_tensordot_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*model_11/dense_23/Tensordot/ReadVariableOp
 model_11/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 model_11/dense_23/Tensordot/axes
 model_11/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model_11/dense_23/Tensordot/free
!model_11/dense_23/Tensordot/ShapeShape%model_11/dropout_18/Identity:output:0*
T0*
_output_shapes
:2#
!model_11/dense_23/Tensordot/Shape
)model_11/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_11/dense_23/Tensordot/GatherV2/axis«
$model_11/dense_23/Tensordot/GatherV2GatherV2*model_11/dense_23/Tensordot/Shape:output:0)model_11/dense_23/Tensordot/free:output:02model_11/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_11/dense_23/Tensordot/GatherV2
+model_11/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_11/dense_23/Tensordot/GatherV2_1/axis±
&model_11/dense_23/Tensordot/GatherV2_1GatherV2*model_11/dense_23/Tensordot/Shape:output:0)model_11/dense_23/Tensordot/axes:output:04model_11/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_11/dense_23/Tensordot/GatherV2_1
!model_11/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_11/dense_23/Tensordot/ConstÈ
 model_11/dense_23/Tensordot/ProdProd-model_11/dense_23/Tensordot/GatherV2:output:0*model_11/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 model_11/dense_23/Tensordot/Prod
#model_11/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#model_11/dense_23/Tensordot/Const_1Ð
"model_11/dense_23/Tensordot/Prod_1Prod/model_11/dense_23/Tensordot/GatherV2_1:output:0,model_11/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"model_11/dense_23/Tensordot/Prod_1
'model_11/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_11/dense_23/Tensordot/concat/axis
"model_11/dense_23/Tensordot/concatConcatV2)model_11/dense_23/Tensordot/free:output:0)model_11/dense_23/Tensordot/axes:output:00model_11/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_11/dense_23/Tensordot/concatÔ
!model_11/dense_23/Tensordot/stackPack)model_11/dense_23/Tensordot/Prod:output:0+model_11/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!model_11/dense_23/Tensordot/stackç
%model_11/dense_23/Tensordot/transpose	Transpose%model_11/dropout_18/Identity:output:0+model_11/dense_23/Tensordot/concat:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2'
%model_11/dense_23/Tensordot/transposeç
#model_11/dense_23/Tensordot/ReshapeReshape)model_11/dense_23/Tensordot/transpose:y:0*model_11/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#model_11/dense_23/Tensordot/Reshapeæ
"model_11/dense_23/Tensordot/MatMulMatMul,model_11/dense_23/Tensordot/Reshape:output:02model_11/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"model_11/dense_23/Tensordot/MatMul
#model_11/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2%
#model_11/dense_23/Tensordot/Const_2
)model_11/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_11/dense_23/Tensordot/concat_1/axis
$model_11/dense_23/Tensordot/concat_1ConcatV2-model_11/dense_23/Tensordot/GatherV2:output:0,model_11/dense_23/Tensordot/Const_2:output:02model_11/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_11/dense_23/Tensordot/concat_1Ù
model_11/dense_23/TensordotReshape,model_11/dense_23/Tensordot/MatMul:product:0-model_11/dense_23/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dense_23/TensordotÂ
(model_11/dense_23/BiasAdd/ReadVariableOpReadVariableOp1model_11_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_11/dense_23/BiasAdd/ReadVariableOpÐ
model_11/dense_23/BiasAddBiasAdd$model_11/dense_23/Tensordot:output:00model_11/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dense_23/BiasAdd
model_11/dense_23/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_11/dense_23/Gelu/mul/xÁ
model_11/dense_23/Gelu/mulMul%model_11/dense_23/Gelu/mul/x:output:0"model_11/dense_23/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dense_23/Gelu/mul
model_11/dense_23/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?2
model_11/dense_23/Gelu/Cast/xÎ
model_11/dense_23/Gelu/truedivRealDiv"model_11/dense_23/BiasAdd:output:0&model_11/dense_23/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2 
model_11/dense_23/Gelu/truediv
model_11/dense_23/Gelu/ErfErf"model_11/dense_23/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dense_23/Gelu/Erf
model_11/dense_23/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model_11/dense_23/Gelu/add/x¿
model_11/dense_23/Gelu/addAddV2%model_11/dense_23/Gelu/add/x:output:0model_11/dense_23/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dense_23/Gelu/addº
model_11/dense_23/Gelu/mul_1Mulmodel_11/dense_23/Gelu/mul:z:0model_11/dense_23/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dense_23/Gelu/mul_1¡
model_11/dropout_19/IdentityIdentity model_11/dense_23/Gelu/mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/dropout_19/Identityª
model_11/add_13/addAddV2%model_11/dropout_19/Identity:output:0model_11/add_12/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_11/add_13/addÊ
>model_11/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2@
>model_11/layer_normalization_14/moments/mean/reduction_indices
,model_11/layer_normalization_14/moments/meanMeanmodel_11/add_13/add:z:0Gmodel_11/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2.
,model_11/layer_normalization_14/moments/meanê
4model_11/layer_normalization_14/moments/StopGradientStopGradient5model_11/layer_normalization_14/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥26
4model_11/layer_normalization_14/moments/StopGradient
9model_11/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencemodel_11/add_13/add:z:0=model_11/layer_normalization_14/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2;
9model_11/layer_normalization_14/moments/SquaredDifferenceÒ
Bmodel_11/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_11/layer_normalization_14/moments/variance/reduction_indicesÀ
0model_11/layer_normalization_14/moments/varianceMean=model_11/layer_normalization_14/moments/SquaredDifference:z:0Kmodel_11/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(22
0model_11/layer_normalization_14/moments/variance§
/model_11/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½7521
/model_11/layer_normalization_14/batchnorm/add/y
-model_11/layer_normalization_14/batchnorm/addAddV29model_11/layer_normalization_14/moments/variance:output:08model_11/layer_normalization_14/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2/
-model_11/layer_normalization_14/batchnorm/addÕ
/model_11/layer_normalization_14/batchnorm/RsqrtRsqrt1model_11/layer_normalization_14/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥21
/model_11/layer_normalization_14/batchnorm/Rsqrtþ
<model_11/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_11_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model_11/layer_normalization_14/batchnorm/mul/ReadVariableOp
-model_11/layer_normalization_14/batchnorm/mulMul3model_11/layer_normalization_14/batchnorm/Rsqrt:y:0Dmodel_11/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_11/layer_normalization_14/batchnorm/mulì
/model_11/layer_normalization_14/batchnorm/mul_1Mulmodel_11/add_13/add:z:01model_11/layer_normalization_14/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_14/batchnorm/mul_1
/model_11/layer_normalization_14/batchnorm/mul_2Mul5model_11/layer_normalization_14/moments/mean:output:01model_11/layer_normalization_14/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_14/batchnorm/mul_2ò
8model_11/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpAmodel_11_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_11/layer_normalization_14/batchnorm/ReadVariableOp
-model_11/layer_normalization_14/batchnorm/subSub@model_11/layer_normalization_14/batchnorm/ReadVariableOp:value:03model_11/layer_normalization_14/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_11/layer_normalization_14/batchnorm/sub
/model_11/layer_normalization_14/batchnorm/add_1AddV23model_11/layer_normalization_14/batchnorm/mul_1:z:01model_11/layer_normalization_14/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_11/layer_normalization_14/batchnorm/add_1
Bmodel_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_11_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpÖ
3model_11/multi_head_attention_7/query/einsum/EinsumEinsum3model_11/layer_normalization_14/batchnorm/add_1:z:0Jmodel_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde25
3model_11/multi_head_attention_7/query/einsum/Einsumö
8model_11/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpAmodel_11_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_11/multi_head_attention_7/query/add/ReadVariableOp
)model_11/multi_head_attention_7/query/addAddV2<model_11/multi_head_attention_7/query/einsum/Einsum:output:0@model_11/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2+
)model_11/multi_head_attention_7/query/add
@model_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpImodel_11_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02B
@model_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpÐ
1model_11/multi_head_attention_7/key/einsum/EinsumEinsum3model_11/layer_normalization_14/batchnorm/add_1:z:0Hmodel_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde23
1model_11/multi_head_attention_7/key/einsum/Einsumð
6model_11/multi_head_attention_7/key/add/ReadVariableOpReadVariableOp?model_11_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

:@*
dtype028
6model_11/multi_head_attention_7/key/add/ReadVariableOp
'model_11/multi_head_attention_7/key/addAddV2:model_11/multi_head_attention_7/key/einsum/Einsum:output:0>model_11/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2)
'model_11/multi_head_attention_7/key/add
Bmodel_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_11_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpÖ
3model_11/multi_head_attention_7/value/einsum/EinsumEinsum3model_11/layer_normalization_14/batchnorm/add_1:z:0Jmodel_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde25
3model_11/multi_head_attention_7/value/einsum/Einsumö
8model_11/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpAmodel_11_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_11/multi_head_attention_7/value/add/ReadVariableOp
)model_11/multi_head_attention_7/value/addAddV2<model_11/multi_head_attention_7/value/einsum/Einsum:output:0@model_11/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2+
)model_11/multi_head_attention_7/value/add
%model_11/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2'
%model_11/multi_head_attention_7/Mul/yë
#model_11/multi_head_attention_7/MulMul-model_11/multi_head_attention_7/query/add:z:0.model_11/multi_head_attention_7/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2%
#model_11/multi_head_attention_7/Mul¢
-model_11/multi_head_attention_7/einsum/EinsumEinsum+model_11/multi_head_attention_7/key/add:z:0'model_11/multi_head_attention_7/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2/
-model_11/multi_head_attention_7/einsum/Einsumá
/model_11/multi_head_attention_7/softmax/SoftmaxSoftmax6model_11/multi_head_attention_7/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥21
/model_11/multi_head_attention_7/softmax/Softmaxç
0model_11/multi_head_attention_7/dropout/IdentityIdentity9model_11/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥22
0model_11/multi_head_attention_7/dropout/Identity¹
/model_11/multi_head_attention_7/einsum_1/EinsumEinsum9model_11/multi_head_attention_7/dropout/Identity:output:0-model_11/multi_head_attention_7/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd21
/model_11/multi_head_attention_7/einsum_1/Einsum¹
Mmodel_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_11_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
Mmodel_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpø
>model_11/multi_head_attention_7/attention_output/einsum/EinsumEinsum8model_11/multi_head_attention_7/einsum_1/Einsum:output:0Umodel_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2@
>model_11/multi_head_attention_7/attention_output/einsum/Einsum
Cmodel_11/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpLmodel_11_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cmodel_11/multi_head_attention_7/attention_output/add/ReadVariableOpÂ
4model_11/multi_head_attention_7/attention_output/addAddV2Gmodel_11/multi_head_attention_7/attention_output/einsum/Einsum:output:0Kmodel_11/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@26
4model_11/multi_head_attention_7/attention_output/add
IdentityIdentity9model_11/multi_head_attention_7/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity 
NoOpNoOp)^model_11/dense_21/BiasAdd/ReadVariableOp+^model_11/dense_21/Tensordot/ReadVariableOp)^model_11/dense_22/BiasAdd/ReadVariableOp+^model_11/dense_22/Tensordot/ReadVariableOp)^model_11/dense_23/BiasAdd/ReadVariableOp+^model_11/dense_23/Tensordot/ReadVariableOp9^model_11/layer_normalization_12/batchnorm/ReadVariableOp=^model_11/layer_normalization_12/batchnorm/mul/ReadVariableOp9^model_11/layer_normalization_13/batchnorm/ReadVariableOp=^model_11/layer_normalization_13/batchnorm/mul/ReadVariableOp9^model_11/layer_normalization_14/batchnorm/ReadVariableOp=^model_11/layer_normalization_14/batchnorm/mul/ReadVariableOpD^model_11/multi_head_attention_6/attention_output/add/ReadVariableOpN^model_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp7^model_11/multi_head_attention_6/key/add/ReadVariableOpA^model_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp9^model_11/multi_head_attention_6/query/add/ReadVariableOpC^model_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9^model_11/multi_head_attention_6/value/add/ReadVariableOpC^model_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpD^model_11/multi_head_attention_7/attention_output/add/ReadVariableOpN^model_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp7^model_11/multi_head_attention_7/key/add/ReadVariableOpA^model_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp9^model_11/multi_head_attention_7/query/add/ReadVariableOpC^model_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9^model_11/multi_head_attention_7/value/add/ReadVariableOpC^model_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp6^model_11/patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_11/dense_21/BiasAdd/ReadVariableOp(model_11/dense_21/BiasAdd/ReadVariableOp2X
*model_11/dense_21/Tensordot/ReadVariableOp*model_11/dense_21/Tensordot/ReadVariableOp2T
(model_11/dense_22/BiasAdd/ReadVariableOp(model_11/dense_22/BiasAdd/ReadVariableOp2X
*model_11/dense_22/Tensordot/ReadVariableOp*model_11/dense_22/Tensordot/ReadVariableOp2T
(model_11/dense_23/BiasAdd/ReadVariableOp(model_11/dense_23/BiasAdd/ReadVariableOp2X
*model_11/dense_23/Tensordot/ReadVariableOp*model_11/dense_23/Tensordot/ReadVariableOp2t
8model_11/layer_normalization_12/batchnorm/ReadVariableOp8model_11/layer_normalization_12/batchnorm/ReadVariableOp2|
<model_11/layer_normalization_12/batchnorm/mul/ReadVariableOp<model_11/layer_normalization_12/batchnorm/mul/ReadVariableOp2t
8model_11/layer_normalization_13/batchnorm/ReadVariableOp8model_11/layer_normalization_13/batchnorm/ReadVariableOp2|
<model_11/layer_normalization_13/batchnorm/mul/ReadVariableOp<model_11/layer_normalization_13/batchnorm/mul/ReadVariableOp2t
8model_11/layer_normalization_14/batchnorm/ReadVariableOp8model_11/layer_normalization_14/batchnorm/ReadVariableOp2|
<model_11/layer_normalization_14/batchnorm/mul/ReadVariableOp<model_11/layer_normalization_14/batchnorm/mul/ReadVariableOp2
Cmodel_11/multi_head_attention_6/attention_output/add/ReadVariableOpCmodel_11/multi_head_attention_6/attention_output/add/ReadVariableOp2
Mmodel_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpMmodel_11/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2p
6model_11/multi_head_attention_6/key/add/ReadVariableOp6model_11/multi_head_attention_6/key/add/ReadVariableOp2
@model_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp@model_11/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2t
8model_11/multi_head_attention_6/query/add/ReadVariableOp8model_11/multi_head_attention_6/query/add/ReadVariableOp2
Bmodel_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpBmodel_11/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2t
8model_11/multi_head_attention_6/value/add/ReadVariableOp8model_11/multi_head_attention_6/value/add/ReadVariableOp2
Bmodel_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpBmodel_11/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2
Cmodel_11/multi_head_attention_7/attention_output/add/ReadVariableOpCmodel_11/multi_head_attention_7/attention_output/add/ReadVariableOp2
Mmodel_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpMmodel_11/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2p
6model_11/multi_head_attention_7/key/add/ReadVariableOp6model_11/multi_head_attention_7/key/add/ReadVariableOp2
@model_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp@model_11/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2t
8model_11/multi_head_attention_7/query/add/ReadVariableOp8model_11/multi_head_attention_7/query/add/ReadVariableOp2
Bmodel_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpBmodel_11/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2t
8model_11/multi_head_attention_7/value/add/ReadVariableOp8model_11/multi_head_attention_7/value/add/ReadVariableOp2
Bmodel_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpBmodel_11/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2n
5model_11/patch_encoder_7/embedding_3/embedding_lookup5model_11/patch_encoder_7/embedding_3/embedding_lookup:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
í 
ü
E__inference_dense_21_layer_call_and_return_conditional_losses_1119152

inputs3
!tensordot_readvariableop_resource:1@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤1
 
_user_specified_nameinputs
¡
f
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118648

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
¥
j
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1118642

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs


S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1119217

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesÀ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752
batchnorm/add/y
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_1
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
é
m
C__inference_add_12_layer_call_and_return_conditional_losses_1119284

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿ¥@:ÿÿÿÿÿÿÿÿÿ¥@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs:TP
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@
 
_user_specified_nameinputs
ä
c
G__inference_resizing_3_layer_call_and_return_conditional_losses_1122393

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   2
resize/size³
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(2
resize/ResizeBilinear
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
e
,__inference_dropout_18_layer_call_fn_1122121

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_11197352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¥22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 
_user_specified_nameinputs
ì

#__inference__traced_restore_1123003
file_prefix2
 assignvariableop_dense_21_kernel:1@.
 assignvariableop_1_dense_21_bias:@=
/assignvariableop_2_layer_normalization_12_gamma:@<
.assignvariableop_3_layer_normalization_12_beta:@=
/assignvariableop_4_layer_normalization_13_gamma:@<
.assignvariableop_5_layer_normalization_13_beta:@5
"assignvariableop_6_dense_22_kernel:	@/
 assignvariableop_7_dense_22_bias:	5
"assignvariableop_8_dense_23_kernel:	@.
 assignvariableop_9_dense_23_bias:@>
0assignvariableop_10_layer_normalization_14_gamma:@=
/assignvariableop_11_layer_normalization_14_beta:@&
assignvariableop_12_mean:*
assignvariableop_13_variance:#
assignvariableop_14_count:	 M
:assignvariableop_15_patch_encoder_7_embedding_3_embeddings:	¥@M
7assignvariableop_16_multi_head_attention_6_query_kernel:@@G
5assignvariableop_17_multi_head_attention_6_query_bias:@K
5assignvariableop_18_multi_head_attention_6_key_kernel:@@E
3assignvariableop_19_multi_head_attention_6_key_bias:@M
7assignvariableop_20_multi_head_attention_6_value_kernel:@@G
5assignvariableop_21_multi_head_attention_6_value_bias:@X
Bassignvariableop_22_multi_head_attention_6_attention_output_kernel:@@N
@assignvariableop_23_multi_head_attention_6_attention_output_bias:@M
7assignvariableop_24_multi_head_attention_7_query_kernel:@@G
5assignvariableop_25_multi_head_attention_7_query_bias:@K
5assignvariableop_26_multi_head_attention_7_key_kernel:@@E
3assignvariableop_27_multi_head_attention_7_key_bias:@M
7assignvariableop_28_multi_head_attention_7_value_kernel:@@G
5assignvariableop_29_multi_head_attention_7_value_bias:@X
Bassignvariableop_30_multi_head_attention_7_attention_output_kernel:@@N
@assignvariableop_31_multi_head_attention_7_attention_output_bias:@*
assignvariableop_32_variable:	,
assignvariableop_33_variable_1:	,
assignvariableop_34_variable_2:	
identity_36¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Á
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Í
valueÃBÀ$B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$				2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2´
AssignVariableOp_2AssignVariableOp/assignvariableop_2_layer_normalization_12_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_layer_normalization_12_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4´
AssignVariableOp_4AssignVariableOp/assignvariableop_4_layer_normalization_13_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_layer_normalization_13_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_23_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_23_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¸
AssignVariableOp_10AssignVariableOp0assignvariableop_10_layer_normalization_14_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_layer_normalization_14_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12 
AssignVariableOp_12AssignVariableOpassignvariableop_12_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¤
AssignVariableOp_13AssignVariableOpassignvariableop_13_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Â
AssignVariableOp_15AssignVariableOp:assignvariableop_15_patch_encoder_7_embedding_3_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¿
AssignVariableOp_16AssignVariableOp7assignvariableop_16_multi_head_attention_6_query_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17½
AssignVariableOp_17AssignVariableOp5assignvariableop_17_multi_head_attention_6_query_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18½
AssignVariableOp_18AssignVariableOp5assignvariableop_18_multi_head_attention_6_key_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19»
AssignVariableOp_19AssignVariableOp3assignvariableop_19_multi_head_attention_6_key_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¿
AssignVariableOp_20AssignVariableOp7assignvariableop_20_multi_head_attention_6_value_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21½
AssignVariableOp_21AssignVariableOp5assignvariableop_21_multi_head_attention_6_value_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ê
AssignVariableOp_22AssignVariableOpBassignvariableop_22_multi_head_attention_6_attention_output_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23È
AssignVariableOp_23AssignVariableOp@assignvariableop_23_multi_head_attention_6_attention_output_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¿
AssignVariableOp_24AssignVariableOp7assignvariableop_24_multi_head_attention_7_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25½
AssignVariableOp_25AssignVariableOp5assignvariableop_25_multi_head_attention_7_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26½
AssignVariableOp_26AssignVariableOp5assignvariableop_26_multi_head_attention_7_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27»
AssignVariableOp_27AssignVariableOp3assignvariableop_27_multi_head_attention_7_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¿
AssignVariableOp_28AssignVariableOp7assignvariableop_28_multi_head_attention_7_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29½
AssignVariableOp_29AssignVariableOp5assignvariableop_29_multi_head_attention_7_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ê
AssignVariableOp_30AssignVariableOpBassignvariableop_30_multi_head_attention_7_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31È
AssignVariableOp_31AssignVariableOp@assignvariableop_31_multi_head_attention_7_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_32¤
AssignVariableOp_32AssignVariableOpassignvariableop_32_variableIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_33¦
AssignVariableOp_33AssignVariableOpassignvariableop_33_variable_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_34¦
AssignVariableOp_34AssignVariableOpassignvariableop_34_variable_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpà
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35f
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_36È
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¡
f
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1118636

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄi:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi
 
_user_specified_nameinputs
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1119166

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¤@:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
E
input_4:
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿT
multi_head_attention_7:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ¥¥tensorflow/serving/predict:Ôü
Í
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
	variables
regularization_losses
trainable_variables
	keras_api

signatures
_default_save_signature
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer

layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer-4
	variables
regularization_losses
trainable_variables
 	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_sequential
§
!	variables
"regularization_losses
#trainable_variables
$	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
½

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
§
+	variables
,regularization_losses
-trainable_variables
.	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
§
/	variables
0regularization_losses
1trainable_variables
2	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
¿
3position_embedding
4	variables
5regularization_losses
6trainable_variables
7	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
Æ
8axis
	9gamma
:beta
;	variables
<regularization_losses
=trainable_variables
>	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer

?_query_dense
@
_key_dense
A_value_dense
B_softmax
C_dropout_layer
D_output_dense
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
§
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer
Æ
Maxis
	Ngamma
Obeta
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
§
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"
_tf_keras_layer
½

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
º__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layer
§
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer
§
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"
_tf_keras_layer
Æ
laxis
	mgamma
nbeta
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"
_tf_keras_layer

s_query_dense
t
_key_dense
u_value_dense
v_softmax
w_dropout_layer
x_output_dense
y	variables
zregularization_losses
{trainable_variables
|	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layer
§
}0
~1
2
%3
&4
5
96
:7
8
9
10
11
12
13
14
15
N16
O17
T18
U19
^20
_21
m22
n23
24
25
26
27
28
29
30
31"
trackable_list_wrapper
 "
trackable_list_wrapper

%0
&1
2
93
:4
5
6
7
8
9
10
11
12
N13
O14
T15
U16
^17
_18
m19
n20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
Ó
 layer_regularization_losses
layer_metrics
layers
	variables
regularization_losses
trainable_variables
non_trainable_variables
metrics
 __call__
_default_save_signature
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
-
Äserving_default"
signature_map
Ù

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
}mean
}
adapt_mean
~variance
~adapt_variance
	count
	keras_api
Å_adapt_function"
_tf_keras_layer
«
	variables
regularization_losses
trainable_variables
	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer
¶
	_rng
 	variables
¡regularization_losses
¢trainable_variables
£	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
¶
	¤_rng
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"
_tf_keras_layer
¶
	©_rng
ª	variables
«regularization_losses
¬trainable_variables
­	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
5
}0
~1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ®layer_regularization_losses
¯layer_metrics
°layers
	variables
regularization_losses
trainable_variables
±non_trainable_variables
²metrics
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ³layer_regularization_losses
´layer_metrics
µlayers
!	variables
"regularization_losses
#trainable_variables
¶non_trainable_variables
·metrics
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
!:1@2dense_21/kernel
:@2dense_21/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
µ
 ¸layer_regularization_losses
¹layer_metrics
ºlayers
'	variables
(regularization_losses
)trainable_variables
»non_trainable_variables
¼metrics
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ½layer_regularization_losses
¾layer_metrics
¿layers
+	variables
,regularization_losses
-trainable_variables
Ànon_trainable_variables
Ámetrics
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Âlayer_regularization_losses
Ãlayer_metrics
Älayers
/	variables
0regularization_losses
1trainable_variables
Ånon_trainable_variables
Æmetrics
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
¼

embeddings
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"
_tf_keras_layer
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
µ
 Ëlayer_regularization_losses
Ìlayer_metrics
Ílayers
4	variables
5regularization_losses
6trainable_variables
Înon_trainable_variables
Ïmetrics
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2layer_normalization_12/gamma
):'@2layer_normalization_12/beta
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
µ
 Ðlayer_regularization_losses
Ñlayer_metrics
Òlayers
;	variables
<regularization_losses
=trainable_variables
Ónon_trainable_variables
Ômetrics
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
ö
Õpartial_output_shape
Öfull_output_shape
kernel
	bias
×	variables
Øregularization_losses
Ùtrainable_variables
Ú	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
Ûpartial_output_shape
Üfull_output_shape
kernel
	bias
Ý	variables
Þregularization_losses
ßtrainable_variables
à	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
ápartial_output_shape
âfull_output_shape
kernel
	bias
ã	variables
äregularization_losses
åtrainable_variables
æ	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ç	variables
èregularization_losses
étrainable_variables
ê	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ë	variables
ìregularization_losses
ítrainable_variables
î	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
ïpartial_output_shape
ðfull_output_shape
kernel
	bias
ñ	variables
òregularization_losses
ótrainable_variables
ô	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
µ
 õlayer_regularization_losses
ölayer_metrics
÷layers
E	variables
Fregularization_losses
Gtrainable_variables
ønon_trainable_variables
ùmetrics
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 úlayer_regularization_losses
ûlayer_metrics
ülayers
I	variables
Jregularization_losses
Ktrainable_variables
ýnon_trainable_variables
þmetrics
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2layer_normalization_13/gamma
):'@2layer_normalization_13/beta
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
µ
 ÿlayer_regularization_losses
layer_metrics
layers
P	variables
Qregularization_losses
Rtrainable_variables
non_trainable_variables
metrics
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
": 	@2dense_22/kernel
:2dense_22/bias
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
V	variables
Wregularization_losses
Xtrainable_variables
non_trainable_variables
metrics
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
Z	variables
[regularization_losses
\trainable_variables
non_trainable_variables
metrics
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
": 	@2dense_23/kernel
:@2dense_23/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
`	variables
aregularization_losses
btrainable_variables
non_trainable_variables
metrics
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
d	variables
eregularization_losses
ftrainable_variables
non_trainable_variables
metrics
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
h	variables
iregularization_losses
jtrainable_variables
non_trainable_variables
metrics
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2layer_normalization_14/gamma
):'@2layer_normalization_14/beta
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
o	variables
pregularization_losses
qtrainable_variables
 non_trainable_variables
¡metrics
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
ö
¢partial_output_shape
£full_output_shape
kernel
	bias
¤	variables
¥regularization_losses
¦trainable_variables
§	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
¨partial_output_shape
©full_output_shape
kernel
	bias
ª	variables
«regularization_losses
¬trainable_variables
­	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
®partial_output_shape
¯full_output_shape
kernel
	bias
°	variables
±regularization_losses
²trainable_variables
³	keras_api
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
«
´	variables
µregularization_losses
¶trainable_variables
·	keras_api
â__call__
+ã&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¸	variables
¹regularization_losses
ºtrainable_variables
»	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
¼partial_output_shape
½full_output_shape
kernel
	bias
¾	variables
¿regularization_losses
Àtrainable_variables
Á	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"
_tf_keras_layer
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
µ
 Âlayer_regularization_losses
Ãlayer_metrics
Älayers
y	variables
zregularization_losses
{trainable_variables
Ånon_trainable_variables
Æmetrics
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
:2mean
:2variance
:	 2count
9:7	¥@2&patch_encoder_7/embedding_3/embeddings
9:7@@2#multi_head_attention_6/query/kernel
3:1@2!multi_head_attention_6/query/bias
7:5@@2!multi_head_attention_6/key/kernel
1:/@2multi_head_attention_6/key/bias
9:7@@2#multi_head_attention_6/value/kernel
3:1@2!multi_head_attention_6/value/bias
D:B@@2.multi_head_attention_6/attention_output/kernel
::8@2,multi_head_attention_6/attention_output/bias
9:7@@2#multi_head_attention_7/query/kernel
3:1@2!multi_head_attention_7/query/bias
7:5@@2!multi_head_attention_7/key/kernel
1:/@2multi_head_attention_7/key/bias
9:7@@2#multi_head_attention_7/value/kernel
3:1@2!multi_head_attention_7/value/bias
D:B@@2.multi_head_attention_7/attention_output/kernel
::8@2,multi_head_attention_7/attention_output/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¦
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
5
}0
~1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 Çlayer_regularization_losses
Èlayer_metrics
Élayers
	variables
regularization_losses
trainable_variables
Ênon_trainable_variables
Ëmetrics
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
/
Ì
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 Ílayer_regularization_losses
Îlayer_metrics
Ïlayers
 	variables
¡regularization_losses
¢trainable_variables
Ðnon_trainable_variables
Ñmetrics
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
/
Ò
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 Ólayer_regularization_losses
Ôlayer_metrics
Õlayers
¥	variables
¦regularization_losses
§trainable_variables
Önon_trainable_variables
×metrics
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
/
Ø
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 Ùlayer_regularization_losses
Úlayer_metrics
Ûlayers
ª	variables
«regularization_losses
¬trainable_variables
Ünon_trainable_variables
Ýmetrics
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
}0
~1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¸
 Þlayer_regularization_losses
ßlayer_metrics
àlayers
Ç	variables
Èregularization_losses
Étrainable_variables
ánon_trainable_variables
âmetrics
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
30"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 ãlayer_regularization_losses
älayer_metrics
ålayers
×	variables
Øregularization_losses
Ùtrainable_variables
ænon_trainable_variables
çmetrics
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 èlayer_regularization_losses
élayer_metrics
êlayers
Ý	variables
Þregularization_losses
ßtrainable_variables
ënon_trainable_variables
ìmetrics
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 ílayer_regularization_losses
îlayer_metrics
ïlayers
ã	variables
äregularization_losses
åtrainable_variables
ðnon_trainable_variables
ñmetrics
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 òlayer_regularization_losses
ólayer_metrics
ôlayers
ç	variables
èregularization_losses
étrainable_variables
õnon_trainable_variables
ömetrics
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ÷layer_regularization_losses
ølayer_metrics
ùlayers
ë	variables
ìregularization_losses
ítrainable_variables
únon_trainable_variables
ûmetrics
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 ülayer_regularization_losses
ýlayer_metrics
þlayers
ñ	variables
òregularization_losses
ótrainable_variables
ÿnon_trainable_variables
metrics
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
?0
@1
A2
B3
C4
D5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 layer_regularization_losses
layer_metrics
layers
¤	variables
¥regularization_losses
¦trainable_variables
non_trainable_variables
metrics
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 layer_regularization_losses
layer_metrics
layers
ª	variables
«regularization_losses
¬trainable_variables
non_trainable_variables
metrics
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 layer_regularization_losses
layer_metrics
layers
°	variables
±regularization_losses
²trainable_variables
non_trainable_variables
metrics
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
layer_metrics
layers
´	variables
µregularization_losses
¶trainable_variables
non_trainable_variables
metrics
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
layer_metrics
layers
¸	variables
¹regularization_losses
ºtrainable_variables
non_trainable_variables
metrics
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
 layer_regularization_losses
layer_metrics
layers
¾	variables
¿regularization_losses
Àtrainable_variables
non_trainable_variables
metrics
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
s0
t1
u2
v3
w4
x5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ÍBÊ
"__inference__wrapped_model_1118610input_4"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
*__inference_model_11_layer_call_fn_1119573
*__inference_model_11_layer_call_fn_1120533
*__inference_model_11_layer_call_fn_1120606
*__inference_model_11_layer_call_fn_1120221À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_model_11_layer_call_and_return_conditional_losses_1120850
E__inference_model_11_layer_call_and_return_conditional_losses_1121412
E__inference_model_11_layer_call_and_return_conditional_losses_1120306
E__inference_model_11_layer_call_and_return_conditional_losses_1120397À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
3__inference_data_augmentation_layer_call_fn_1118658
3__inference_data_augmentation_layer_call_fn_1121421
3__inference_data_augmentation_layer_call_fn_1121436
3__inference_data_augmentation_layer_call_fn_1119065À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1121449
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1121752
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119080
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119101À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_patches_7_layer_call_fn_1121757¢
²
FullArgSpec
args
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_patches_7_layer_call_and_return_conditional_losses_1121764¢
²
FullArgSpec
args
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_21_layer_call_fn_1121773¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_21_layer_call_and_return_conditional_losses_1121803¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_lambda_3_layer_call_fn_1121808
*__inference_lambda_3_layer_call_fn_1121813À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
E__inference_lambda_3_layer_call_and_return_conditional_losses_1121821
E__inference_lambda_3_layer_call_and_return_conditional_losses_1121829À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ù2Ö
/__inference_concatenate_3_layer_call_fn_1121835¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1121842¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
1__inference_patch_encoder_7_layer_call_fn_1121849¦
²
FullArgSpec!
args
jself
j
projection
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1121863¦
²
FullArgSpec!
args
jself
j
projection
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
8__inference_layer_normalization_12_layer_call_fn_1121872¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ý2ú
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1121894¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
8__inference_multi_head_attention_6_layer_call_fn_1121918
8__inference_multi_head_attention_6_layer_call_fn_1121942ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1121978
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1122021ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_add_12_layer_call_fn_1122027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_add_12_layer_call_and_return_conditional_losses_1122033¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
8__inference_layer_normalization_13_layer_call_fn_1122042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ý2ú
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_1122064¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_dense_22_layer_call_fn_1122073¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_22_layer_call_and_return_conditional_losses_1122111¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_dropout_18_layer_call_fn_1122116
,__inference_dropout_18_layer_call_fn_1122121´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_18_layer_call_and_return_conditional_losses_1122126
G__inference_dropout_18_layer_call_and_return_conditional_losses_1122138´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_dense_23_layer_call_fn_1122147¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_23_layer_call_and_return_conditional_losses_1122185¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_dropout_19_layer_call_fn_1122190
,__inference_dropout_19_layer_call_fn_1122195´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_19_layer_call_and_return_conditional_losses_1122200
G__inference_dropout_19_layer_call_and_return_conditional_losses_1122212´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_add_13_layer_call_fn_1122218¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_add_13_layer_call_and_return_conditional_losses_1122224¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
8__inference_layer_normalization_14_layer_call_fn_1122233¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ý2ú
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_1122255¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
8__inference_multi_head_attention_7_layer_call_fn_1122279
8__inference_multi_head_attention_7_layer_call_fn_1122303ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1122339
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1122382ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
%__inference_signature_wrapper_1120466input_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_adapt_step_1113455
²
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_resizing_3_layer_call_fn_1122387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_resizing_3_layer_call_and_return_conditional_losses_1122393¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_random_flip_3_layer_call_fn_1122398
/__inference_random_flip_3_layer_call_fn_1122405´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1122409
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1122467´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤2¡
3__inference_random_rotation_3_layer_call_fn_1122472
3__inference_random_rotation_3_layer_call_fn_1122479´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1122483
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1122601´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_random_zoom_3_layer_call_fn_1122606
/__inference_random_zoom_3_layer_call_fn_1122613´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1122617
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1122743´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1ò
"__inference__wrapped_model_1118610Ë2èé%&9:NOTU^_mn:¢7
0¢-
+(
input_4ÿÿÿÿÿÿÿÿÿ
ª "YªV
T
multi_head_attention_7:7
multi_head_attention_7ÿÿÿÿÿÿÿÿÿ¥¥x
__inference_adapt_step_1113455V}~K¢H
A¢>
<9'¢$
"ÿÿÿÿÿÿÿÿÿIteratorSpec
ª "
 Ú
C__inference_add_12_layer_call_and_return_conditional_losses_1122033d¢a
Z¢W
UR
'$
inputs/0ÿÿÿÿÿÿÿÿÿ¥@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 ²
(__inference_add_12_layer_call_fn_1122027d¢a
Z¢W
UR
'$
inputs/0ÿÿÿÿÿÿÿÿÿ¥@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@Ú
C__inference_add_13_layer_call_and_return_conditional_losses_1122224d¢a
Z¢W
UR
'$
inputs/0ÿÿÿÿÿÿÿÿÿ¥@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 ²
(__inference_add_13_layer_call_fn_1122218d¢a
Z¢W
UR
'$
inputs/0ÿÿÿÿÿÿÿÿÿ¥@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@à
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1121842c¢`
Y¢V
TQ
&#
inputs/0ÿÿÿÿÿÿÿÿÿ@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¤@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 ¸
/__inference_concatenate_3_layer_call_fn_1121835c¢`
Y¢V
TQ
&#
inputs/0ÿÿÿÿÿÿÿÿÿ@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¤@
ª "ÿÿÿÿÿÿÿÿÿ¥@Û
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119080èéP¢M
F¢C
96
normalization_3_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 á
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1119101
èéÌÒØP¢M
F¢C
96
normalization_3_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 Ë
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1121449yèéA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 Ñ
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1121752
èéÌÒØA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 ²
3__inference_data_augmentation_layer_call_fn_1118658{èéP¢M
F¢C
96
normalization_3_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿÄi¹
3__inference_data_augmentation_layer_call_fn_1119065
èéÌÒØP¢M
F¢C
96
normalization_3_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "!ÿÿÿÿÿÿÿÿÿÄi£
3__inference_data_augmentation_layer_call_fn_1121421lèéA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿÄi©
3__inference_data_augmentation_layer_call_fn_1121436r
èéÌÒØA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "!ÿÿÿÿÿÿÿÿÿÄi¯
E__inference_dense_21_layer_call_and_return_conditional_losses_1121803f%&4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¤1
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¤@
 
*__inference_dense_21_layer_call_fn_1121773Y%&4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¤1
ª "ÿÿÿÿÿÿÿÿÿ¤@°
E__inference_dense_22_layer_call_and_return_conditional_losses_1122111gTU4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ¥
 
*__inference_dense_22_layer_call_fn_1122073ZTU4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥°
E__inference_dense_23_layer_call_and_return_conditional_losses_1122185g^_5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿ¥
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
*__inference_dense_23_layer_call_fn_1122147Z^_5¢2
+¢(
&#
inputsÿÿÿÿÿÿÿÿÿ¥
ª "ÿÿÿÿÿÿÿÿÿ¥@³
G__inference_dropout_18_layer_call_and_return_conditional_losses_1122126h9¢6
/¢,
&#
inputsÿÿÿÿÿÿÿÿÿ¥
p 
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ¥
 ³
G__inference_dropout_18_layer_call_and_return_conditional_losses_1122138h9¢6
/¢,
&#
inputsÿÿÿÿÿÿÿÿÿ¥
p
ª "+¢(
!
0ÿÿÿÿÿÿÿÿÿ¥
 
,__inference_dropout_18_layer_call_fn_1122116[9¢6
/¢,
&#
inputsÿÿÿÿÿÿÿÿÿ¥
p 
ª "ÿÿÿÿÿÿÿÿÿ¥
,__inference_dropout_18_layer_call_fn_1122121[9¢6
/¢,
&#
inputsÿÿÿÿÿÿÿÿÿ¥
p
ª "ÿÿÿÿÿÿÿÿÿ¥±
G__inference_dropout_19_layer_call_and_return_conditional_losses_1122200f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 ±
G__inference_dropout_19_layer_call_and_return_conditional_losses_1122212f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
,__inference_dropout_19_layer_call_fn_1122190Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
p 
ª "ÿÿÿÿÿÿÿÿÿ¥@
,__inference_dropout_19_layer_call_fn_1122195Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
p
ª "ÿÿÿÿÿÿÿÿÿ¥@²
E__inference_lambda_3_layer_call_and_return_conditional_losses_1121821i<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ¤@

 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 ²
E__inference_lambda_3_layer_call_and_return_conditional_losses_1121829i<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ¤@

 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_lambda_3_layer_call_fn_1121808\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ¤@

 
p 
ª "ÿÿÿÿÿÿÿÿÿ@
*__inference_lambda_3_layer_call_fn_1121813\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ¤@

 
p
ª "ÿÿÿÿÿÿÿÿÿ@½
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1121894f9:4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
8__inference_layer_normalization_12_layer_call_fn_1121872Y9:4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@½
S__inference_layer_normalization_13_layer_call_and_return_conditional_losses_1122064fNO4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
8__inference_layer_normalization_13_layer_call_fn_1122042YNO4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@½
S__inference_layer_normalization_14_layer_call_and_return_conditional_losses_1122255fmn4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
8__inference_layer_normalization_14_layer_call_fn_1122233Ymn4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@ó
E__inference_model_11_layer_call_and_return_conditional_losses_1120306©2èé%&9:NOTU^_mnB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 ù
E__inference_model_11_layer_call_and_return_conditional_losses_1120397¯8èéÌÒØ%&9:NOTU^_mnB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 ò
E__inference_model_11_layer_call_and_return_conditional_losses_1120850¨2èé%&9:NOTU^_mnA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 ø
E__inference_model_11_layer_call_and_return_conditional_losses_1121412®8èéÌÒØ%&9:NOTU^_mnA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 Ë
*__inference_model_11_layer_call_fn_11195732èé%&9:NOTU^_mnB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥Ñ
*__inference_model_11_layer_call_fn_1120221¢8èéÌÒØ%&9:NOTU^_mnB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥Ê
*__inference_model_11_layer_call_fn_11205332èé%&9:NOTU^_mnA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥Ð
*__inference_model_11_layer_call_fn_1120606¡8èéÌÒØ%&9:NOTU^_mnA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥±
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1121978Ùi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p 
ª "Z¢W
P¢M
"
0/0ÿÿÿÿÿÿÿÿÿ¥@
'$
0/1ÿÿÿÿÿÿÿÿÿ¥¥
 ±
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1122021Ùi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p
ª "Z¢W
P¢M
"
0/0ÿÿÿÿÿÿÿÿÿ¥@
'$
0/1ÿÿÿÿÿÿÿÿÿ¥¥
 
8__inference_multi_head_attention_6_layer_call_fn_1121918Ëi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p 
ª "L¢I
 
0ÿÿÿÿÿÿÿÿÿ¥@
%"
1ÿÿÿÿÿÿÿÿÿ¥¥
8__inference_multi_head_attention_6_layer_call_fn_1121942Ëi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p
ª "L¢I
 
0ÿÿÿÿÿÿÿÿÿ¥@
%"
1ÿÿÿÿÿÿÿÿÿ¥¥±
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1122339Ùi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p 
ª "Z¢W
P¢M
"
0/0ÿÿÿÿÿÿÿÿÿ¥@
'$
0/1ÿÿÿÿÿÿÿÿÿ¥¥
 ±
S__inference_multi_head_attention_7_layer_call_and_return_conditional_losses_1122382Ùi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p
ª "Z¢W
P¢M
"
0/0ÿÿÿÿÿÿÿÿÿ¥@
'$
0/1ÿÿÿÿÿÿÿÿÿ¥¥
 
8__inference_multi_head_attention_7_layer_call_fn_1122279Ëi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p 
ª "L¢I
 
0ÿÿÿÿÿÿÿÿÿ¥@
%"
1ÿÿÿÿÿÿÿÿÿ¥¥
8__inference_multi_head_attention_7_layer_call_fn_1122303Ëi¢f
_¢\
$!
queryÿÿÿÿÿÿÿÿÿ¥@
$!
valueÿÿÿÿÿÿÿÿÿ¥@

 

 
p
p
ª "L¢I
 
0ÿÿÿÿÿÿÿÿÿ¥@
%"
1ÿÿÿÿÿÿÿÿÿ¥¥º
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1121863j8¢5
.¢+
)&

projectionÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
1__inference_patch_encoder_7_layer_call_fn_1121849]8¢5
.¢+
)&

projectionÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@°
F__inference_patches_7_layer_call_and_return_conditional_losses_1121764f8¢5
.¢+
)&
imagesÿÿÿÿÿÿÿÿÿÄi
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¤1
 
+__inference_patches_7_layer_call_fn_1121757Y8¢5
.¢+
)&
imagesÿÿÿÿÿÿÿÿÿÄi
ª "ÿÿÿÿÿÿÿÿÿ¤1¼
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1122409n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 À
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1122467rÌ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
/__inference_random_flip_3_layer_call_fn_1122398a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª "!ÿÿÿÿÿÿÿÿÿÄi
/__inference_random_flip_3_layer_call_fn_1122405eÌ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª "!ÿÿÿÿÿÿÿÿÿÄiÀ
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1122483n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 Ä
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1122601rÒ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
3__inference_random_rotation_3_layer_call_fn_1122472a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª "!ÿÿÿÿÿÿÿÿÿÄi
3__inference_random_rotation_3_layer_call_fn_1122479eÒ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª "!ÿÿÿÿÿÿÿÿÿÄi¼
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1122617n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 À
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1122743rØ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
/__inference_random_zoom_3_layer_call_fn_1122606a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª "!ÿÿÿÿÿÿÿÿÿÄi
/__inference_random_zoom_3_layer_call_fn_1122613eØ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª "!ÿÿÿÿÿÿÿÿÿÄi¶
G__inference_resizing_3_layer_call_and_return_conditional_losses_1122393k9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
,__inference_resizing_3_layer_call_fn_1122387^9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÄi
%__inference_signature_wrapper_1120466Ö2èé%&9:NOTU^_mnE¢B
¢ 
;ª8
6
input_4+(
input_4ÿÿÿÿÿÿÿÿÿ"YªV
T
multi_head_attention_7:7
multi_head_attention_7ÿÿÿÿÿÿÿÿÿ¥¥