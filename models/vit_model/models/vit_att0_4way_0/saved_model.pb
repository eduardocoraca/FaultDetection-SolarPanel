®!
«
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
 "serve*2.6.02unknown8Ü
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
¶I
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*ïH
valueåHBâH BÛH
Ù
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
­
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer-4
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
R
&	variables
'regularization_losses
(trainable_variables
)	keras_api
j
*position_embedding
+	variables
,regularization_losses
-trainable_variables
.	keras_api
q
/axis
	0gamma
1beta
2	variables
3regularization_losses
4trainable_variables
5	keras_api
»
6_query_dense
7
_key_dense
8_value_dense
9_softmax
:_dropout_layer
;_output_dense
<	variables
=regularization_losses
>trainable_variables
?	keras_api
v
@0
A1
B2
3
4
C5
06
17
D8
E9
F10
G11
H12
I13
J14
K15
 
^
0
1
C2
03
14
D5
E6
F7
G8
H9
I10
J11
K12
­
Llayer_regularization_losses
Mlayer_metrics

Nlayers

	variables
regularization_losses
trainable_variables
Onon_trainable_variables
Pmetrics
 
¥
Q
_keep_axis
R_reduce_axis
S_reduce_axis_mask
T_broadcast_shape
@mean
@
adapt_mean
Avariance
Aadapt_variance
	Bcount
U	keras_api
R
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
\
Z_rng
[	variables
\regularization_losses
]trainable_variables
^	keras_api
\
__rng
`	variables
aregularization_losses
btrainable_variables
c	keras_api
\
d_rng
e	variables
fregularization_losses
gtrainable_variables
h	keras_api

@0
A1
B2
 
 
­
ilayer_regularization_losses
jlayer_metrics

klayers
	variables
regularization_losses
trainable_variables
lnon_trainable_variables
mmetrics
 
 
 
­
nlayer_regularization_losses
olayer_metrics

players
	variables
regularization_losses
trainable_variables
qnon_trainable_variables
rmetrics
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
slayer_regularization_losses
tlayer_metrics

ulayers
	variables
regularization_losses
 trainable_variables
vnon_trainable_variables
wmetrics
 
 
 
­
xlayer_regularization_losses
ylayer_metrics

zlayers
"	variables
#regularization_losses
$trainable_variables
{non_trainable_variables
|metrics
 
 
 
¯
}layer_regularization_losses
~layer_metrics

layers
&	variables
'regularization_losses
(trainable_variables
non_trainable_variables
metrics
f
C
embeddings
	variables
regularization_losses
trainable_variables
	keras_api

C0
 

C0
²
 layer_regularization_losses
layer_metrics
layers
+	variables
,regularization_losses
-trainable_variables
non_trainable_variables
metrics
 
ge
VARIABLE_VALUElayer_normalization_12/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_12/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
²
 layer_regularization_losses
layer_metrics
layers
2	variables
3regularization_losses
4trainable_variables
non_trainable_variables
metrics

partial_output_shape
full_output_shape

Dkernel
Ebias
	variables
regularization_losses
trainable_variables
	keras_api

partial_output_shape
full_output_shape

Fkernel
Gbias
	variables
regularization_losses
trainable_variables
	keras_api

partial_output_shape
full_output_shape

Hkernel
Ibias
	variables
regularization_losses
 trainable_variables
¡	keras_api
V
¢	variables
£regularization_losses
¤trainable_variables
¥	keras_api
V
¦	variables
§regularization_losses
¨trainable_variables
©	keras_api

ªpartial_output_shape
«full_output_shape

Jkernel
Kbias
¬	variables
­regularization_losses
®trainable_variables
¯	keras_api
8
D0
E1
F2
G3
H4
I5
J6
K7
 
8
D0
E1
F2
G3
H4
I5
J6
K7
²
 °layer_regularization_losses
±layer_metrics
²layers
<	variables
=regularization_losses
>trainable_variables
³non_trainable_variables
´metrics
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
 
 
?
0
1
2
3
4
5
6
7
	8

@0
A1
B2
 
 
 
 
 
 
 
 
 
²
 µlayer_regularization_losses
¶layer_metrics
·layers
V	variables
Wregularization_losses
Xtrainable_variables
¸non_trainable_variables
¹metrics

º
_state_var
 
 
 
²
 »layer_regularization_losses
¼layer_metrics
½layers
[	variables
\regularization_losses
]trainable_variables
¾non_trainable_variables
¿metrics

À
_state_var
 
 
 
²
 Álayer_regularization_losses
Âlayer_metrics
Ãlayers
`	variables
aregularization_losses
btrainable_variables
Änon_trainable_variables
Åmetrics

Æ
_state_var
 
 
 
²
 Çlayer_regularization_losses
Èlayer_metrics
Élayers
e	variables
fregularization_losses
gtrainable_variables
Ênon_trainable_variables
Ëmetrics
 
 
#
0
1
2
3
4

@0
A1
B2
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

C0
 

C0
µ
 Ìlayer_regularization_losses
Ílayer_metrics
Îlayers
	variables
regularization_losses
trainable_variables
Ïnon_trainable_variables
Ðmetrics
 
 

*0
 
 
 
 
 
 
 
 
 

D0
E1
 

D0
E1
µ
 Ñlayer_regularization_losses
Òlayer_metrics
Ólayers
	variables
regularization_losses
trainable_variables
Ônon_trainable_variables
Õmetrics
 
 

F0
G1
 

F0
G1
µ
 Ölayer_regularization_losses
×layer_metrics
Ølayers
	variables
regularization_losses
trainable_variables
Ùnon_trainable_variables
Úmetrics
 
 

H0
I1
 

H0
I1
µ
 Ûlayer_regularization_losses
Ülayer_metrics
Ýlayers
	variables
regularization_losses
 trainable_variables
Þnon_trainable_variables
ßmetrics
 
 
 
µ
 àlayer_regularization_losses
álayer_metrics
âlayers
¢	variables
£regularization_losses
¤trainable_variables
ãnon_trainable_variables
ämetrics
 
 
 
µ
 ålayer_regularization_losses
ælayer_metrics
çlayers
¦	variables
§regularization_losses
¨trainable_variables
ènon_trainable_variables
émetrics
 
 

J0
K1
 

J0
K1
µ
 êlayer_regularization_losses
ëlayer_metrics
ìlayers
¬	variables
­regularization_losses
®trainable_variables
ínon_trainable_variables
îmetrics
 
 
*
60
71
82
93
:4
;5
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

serving_default_input_4Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
¡
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4ConstConst_1dense_21/kerneldense_21/bias&patch_encoder_7/embedding_3/embeddingslayer_normalization_12/gammalayer_normalization_12/beta#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1116555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp0layer_normalization_12/gamma/Read/ReadVariableOp/layer_normalization_12/beta/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp:patch_encoder_7/embedding_3/embeddings/Read/ReadVariableOp7multi_head_attention_6/query/kernel/Read/ReadVariableOp5multi_head_attention_6/query/bias/Read/ReadVariableOp5multi_head_attention_6/key/kernel/Read/ReadVariableOp3multi_head_attention_6/key/bias/Read/ReadVariableOp7multi_head_attention_6/value/kernel/Read/ReadVariableOp5multi_head_attention_6/value/bias/Read/ReadVariableOpBmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_6/attention_output/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpConst_2* 
Tin
2				*
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
 __inference__traced_save_1118207

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biaslayer_normalization_12/gammalayer_normalization_12/betameanvariancecount&patch_encoder_7/embedding_3/embeddings#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/biasVariable
Variable_1
Variable_2*
Tin
2*
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
#__inference__traced_restore_1118274¤Ð
Ø
Ç
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117968

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
ë
H
,__inference_resizing_3_layer_call_fn_1117754

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
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
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
£
¡
8__inference_layer_normalization_12_layer_call_fn_1117600

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
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_11160132
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
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117549

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
É
þ
*__inference_model_10_layer_call_fn_1116108
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

unknown_13:@
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11160752
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
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 22
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
¬§
ç
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1115588

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
U
ù
#__inference__traced_restore_1118274
file_prefix2
 assignvariableop_dense_21_kernel:1@.
 assignvariableop_1_dense_21_bias:@=
/assignvariableop_2_layer_normalization_12_gamma:@<
.assignvariableop_3_layer_normalization_12_beta:@%
assignvariableop_4_mean:)
assignvariableop_5_variance:"
assignvariableop_6_count:	 L
9assignvariableop_7_patch_encoder_7_embedding_3_embeddings:	¥@L
6assignvariableop_8_multi_head_attention_6_query_kernel:@@F
4assignvariableop_9_multi_head_attention_6_query_bias:@K
5assignvariableop_10_multi_head_attention_6_key_kernel:@@E
3assignvariableop_11_multi_head_attention_6_key_bias:@M
7assignvariableop_12_multi_head_attention_6_value_kernel:@@G
5assignvariableop_13_multi_head_attention_6_value_bias:@X
Bassignvariableop_14_multi_head_attention_6_attention_output_kernel:@@N
@assignvariableop_15_multi_head_attention_6_attention_output_bias:@*
assignvariableop_16_variable:	,
assignvariableop_17_variable_1:	,
assignvariableop_18_variable_2:	
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ã
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ï
valueÅBÂB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¶
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2				2
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

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5 
AssignVariableOp_5AssignVariableOpassignvariableop_5_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_countIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¾
AssignVariableOp_7AssignVariableOp9assignvariableop_7_patch_encoder_7_embedding_3_embeddingsIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8»
AssignVariableOp_8AssignVariableOp6assignvariableop_8_multi_head_attention_6_query_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¹
AssignVariableOp_9AssignVariableOp4assignvariableop_9_multi_head_attention_6_query_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_multi_head_attention_6_key_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11»
AssignVariableOp_11AssignVariableOp3assignvariableop_11_multi_head_attention_6_key_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¿
AssignVariableOp_12AssignVariableOp7assignvariableop_12_multi_head_attention_6_value_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13½
AssignVariableOp_13AssignVariableOp5assignvariableop_13_multi_head_attention_6_value_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ê
AssignVariableOp_14AssignVariableOpBassignvariableop_14_multi_head_attention_6_attention_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15È
AssignVariableOp_15AssignVariableOp@assignvariableop_15_multi_head_attention_6_attention_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¤
AssignVariableOp_16AssignVariableOpassignvariableop_16_variableIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17¦
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18¦
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19f
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_20è
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
Æ
ý
*__inference_model_10_layer_call_fn_1116590

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

unknown_13:@
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11160752
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
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 22
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
Ä
Ì
*__inference_model_10_layer_call_fn_1116426
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

unknown_16:@
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11163462
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
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : 22
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
2
á
E__inference_model_10_layer_call_and_return_conditional_losses_1116518
input_4
data_augmentation_1116472
data_augmentation_1116474'
data_augmentation_1116476:	'
data_augmentation_1116478:	'
data_augmentation_1116480:	"
dense_21_1116484:1@
dense_21_1116486:@*
patch_encoder_7_1116491:	¥@,
layer_normalization_12_1116494:@,
layer_normalization_12_1116496:@4
multi_head_attention_6_1116499:@@0
multi_head_attention_6_1116501:@4
multi_head_attention_6_1116503:@@0
multi_head_attention_6_1116505:@4
multi_head_attention_6_1116507:@@0
multi_head_attention_6_1116509:@4
multi_head_attention_6_1116511:@@,
multi_head_attention_6_1116513:@
identity¢)data_augmentation/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCall¡
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinput_4data_augmentation_1116472data_augmentation_1116474data_augmentation_1116476data_augmentation_1116478data_augmentation_1116480*
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332+
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
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1116484dense_21_1116486*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
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
E__inference_lambda_3_layer_call_and_return_conditional_losses_11162412
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
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1116491*
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
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116494layer_normalization_12_1116496*
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
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116499multi_head_attention_6_1116501multi_head_attention_6_1116503multi_head_attention_6_1116505multi_head_attention_6_1116507multi_head_attention_6_1116509multi_head_attention_6_1116511multi_head_attention_6_1116513*
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111617920
.multi_head_attention_6/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity©
NoOpNoOp*^data_augmentation/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : 2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
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
 

N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117480

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
Þ
°
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115876
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
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11154322
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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11154382#
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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11154442
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
¡
f
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117776

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
¡
f
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1115444

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
±
¡
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115447

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
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11154322
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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11154382#
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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11154442
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
í 
ü
E__inference_dense_21_layer_call_and_return_conditional_losses_1115948

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
Üf

J__inference_random_flip_3_layer_call_and_return_conditional_losses_1115790

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
¡
ù
%__inference_signature_wrapper_1116555
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

unknown_13:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_11154062
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
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 22
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
Üf

J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117834

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

v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1117570
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
0

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117706	
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


S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1116013

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
ä
c
G__inference_resizing_3_layer_call_and_return_conditional_losses_1117760

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
Ã9

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1116179	
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
÷
O
3__inference_random_rotation_3_layer_call_fn_1117839

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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11154382
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
¥
j
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117850

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
Ì
Ç
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1115987

projection7
$embedding_3_embedding_lookup_1115980:	¥@
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
embedding_3/embedding_lookupResourceGather$embedding_3_embedding_lookup_1115980range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*7
_class-
+)loc:@embedding_3/embedding_lookup/1115980*
_output_shapes
:	¥@*
dtype02
embedding_3/embedding_lookup
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*7
_class-
+)loc:@embedding_3/embedding_lookup/1115980*
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
þ

3__inference_random_rotation_3_layer_call_fn_1117846

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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11157192
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
í 
ü
E__inference_dense_21_layer_call_and_return_conditional_losses_1117531

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
Ó
F
*__inference_lambda_3_layer_call_fn_1117541

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
E__inference_lambda_3_layer_call_and_return_conditional_losses_11162412
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
ö.
¹
E__inference_model_10_layer_call_and_return_conditional_losses_1116075

inputs
data_augmentation_1115904
data_augmentation_1115906"
dense_21_1115949:1@
dense_21_1115951:@*
patch_encoder_7_1115988:	¥@,
layer_normalization_12_1116014:@,
layer_normalization_12_1116016:@4
multi_head_attention_6_1116056:@@0
multi_head_attention_6_1116058:@4
multi_head_attention_6_1116060:@@0
multi_head_attention_6_1116062:@4
multi_head_attention_6_1116064:@@0
multi_head_attention_6_1116066:@4
multi_head_attention_6_1116068:@@,
multi_head_attention_6_1116070:@
identity¢ dense_21/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCall´
!data_augmentation/PartitionedCallPartitionedCallinputsdata_augmentation_1115904data_augmentation_1115906*
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472#
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
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1115949dense_21_1115951*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
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
E__inference_lambda_3_layer_call_and_return_conditional_losses_11159622
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
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1115988*
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
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116014layer_normalization_12_1116016*
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
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116056multi_head_attention_6_1116058multi_head_attention_6_1116060multi_head_attention_6_1116062multi_head_attention_6_1116064multi_head_attention_6_1116066multi_head_attention_6_1116068multi_head_attention_6_1116070*
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111605520
.multi_head_attention_6/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityý
NoOpNoOp!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
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
Ó
F
*__inference_lambda_3_layer_call_fn_1117536

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
E__inference_lambda_3_layer_call_and_return_conditional_losses_11159622
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
Á
Ë
*__inference_model_10_layer_call_fn_1116631

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

unknown_16:@
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11163462
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
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : 22
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
É
¹
"__inference__wrapped_model_1115406
input_44
0model_10_data_augmentation_normalization_3_sub_y5
1model_10_data_augmentation_normalization_3_sqrt_xE
3model_10_dense_21_tensordot_readvariableop_resource:1@?
1model_10_dense_21_biasadd_readvariableop_resource:@P
=model_10_patch_encoder_7_embedding_3_embedding_lookup_1115351:	¥@S
Emodel_10_layer_normalization_12_batchnorm_mul_readvariableop_resource:@O
Amodel_10_layer_normalization_12_batchnorm_readvariableop_resource:@a
Kmodel_10_multi_head_attention_6_query_einsum_einsum_readvariableop_resource:@@S
Amodel_10_multi_head_attention_6_query_add_readvariableop_resource:@_
Imodel_10_multi_head_attention_6_key_einsum_einsum_readvariableop_resource:@@Q
?model_10_multi_head_attention_6_key_add_readvariableop_resource:@a
Kmodel_10_multi_head_attention_6_value_einsum_einsum_readvariableop_resource:@@S
Amodel_10_multi_head_attention_6_value_add_readvariableop_resource:@l
Vmodel_10_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:@@Z
Lmodel_10_multi_head_attention_6_attention_output_add_readvariableop_resource:@
identity¢(model_10/dense_21/BiasAdd/ReadVariableOp¢*model_10/dense_21/Tensordot/ReadVariableOp¢8model_10/layer_normalization_12/batchnorm/ReadVariableOp¢<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOp¢Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOp¢Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp¢6model_10/multi_head_attention_6/key/add/ReadVariableOp¢@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¢8model_10/multi_head_attention_6/query/add/ReadVariableOp¢Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp¢8model_10/multi_head_attention_6/value/add/ReadVariableOp¢Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp¢5model_10/patch_encoder_7/embedding_3/embedding_lookupÞ
.model_10/data_augmentation/normalization_3/subSubinput_40model_10_data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_10/data_augmentation/normalization_3/subÎ
/model_10/data_augmentation/normalization_3/SqrtSqrt1model_10_data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:21
/model_10/data_augmentation/normalization_3/Sqrt±
4model_10/data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö326
4model_10/data_augmentation/normalization_3/Maximum/y
2model_10/data_augmentation/normalization_3/MaximumMaximum3model_10/data_augmentation/normalization_3/Sqrt:y:0=model_10/data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:24
2model_10/data_augmentation/normalization_3/Maximum
2model_10/data_augmentation/normalization_3/truedivRealDiv2model_10/data_augmentation/normalization_3/sub:z:06model_10/data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2model_10/data_augmentation/normalization_3/truediv·
1model_10/data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"Ä   i   23
1model_10/data_augmentation/resizing_3/resize/sizeÕ
;model_10/data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear6model_10/data_augmentation/normalization_3/truediv:z:0:model_10/data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄi*
half_pixel_centers(2=
;model_10/data_augmentation/resizing_3/resize/ResizeBilinear¿
&model_10/patches_7/ExtractImagePatchesExtractImagePatchesLmodel_10/data_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
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
&model_10/patches_7/ExtractImagePatches
 model_10/patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ¤  1   2"
 model_10/patches_7/Reshape/shape×
model_10/patches_7/ReshapeReshape0model_10/patches_7/ExtractImagePatches:patches:0)model_10/patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12
model_10/patches_7/ReshapeÌ
*model_10/dense_21/Tensordot/ReadVariableOpReadVariableOp3model_10_dense_21_tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02,
*model_10/dense_21/Tensordot/ReadVariableOp
 model_10/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 model_10/dense_21/Tensordot/axes
 model_10/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model_10/dense_21/Tensordot/free
!model_10/dense_21/Tensordot/ShapeShape#model_10/patches_7/Reshape:output:0*
T0*
_output_shapes
:2#
!model_10/dense_21/Tensordot/Shape
)model_10/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_10/dense_21/Tensordot/GatherV2/axis«
$model_10/dense_21/Tensordot/GatherV2GatherV2*model_10/dense_21/Tensordot/Shape:output:0)model_10/dense_21/Tensordot/free:output:02model_10/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_10/dense_21/Tensordot/GatherV2
+model_10/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_10/dense_21/Tensordot/GatherV2_1/axis±
&model_10/dense_21/Tensordot/GatherV2_1GatherV2*model_10/dense_21/Tensordot/Shape:output:0)model_10/dense_21/Tensordot/axes:output:04model_10/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_10/dense_21/Tensordot/GatherV2_1
!model_10/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_10/dense_21/Tensordot/ConstÈ
 model_10/dense_21/Tensordot/ProdProd-model_10/dense_21/Tensordot/GatherV2:output:0*model_10/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 model_10/dense_21/Tensordot/Prod
#model_10/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#model_10/dense_21/Tensordot/Const_1Ð
"model_10/dense_21/Tensordot/Prod_1Prod/model_10/dense_21/Tensordot/GatherV2_1:output:0,model_10/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"model_10/dense_21/Tensordot/Prod_1
'model_10/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_10/dense_21/Tensordot/concat/axis
"model_10/dense_21/Tensordot/concatConcatV2)model_10/dense_21/Tensordot/free:output:0)model_10/dense_21/Tensordot/axes:output:00model_10/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_10/dense_21/Tensordot/concatÔ
!model_10/dense_21/Tensordot/stackPack)model_10/dense_21/Tensordot/Prod:output:0+model_10/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!model_10/dense_21/Tensordot/stackä
%model_10/dense_21/Tensordot/transpose	Transpose#model_10/patches_7/Reshape:output:0+model_10/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤12'
%model_10/dense_21/Tensordot/transposeç
#model_10/dense_21/Tensordot/ReshapeReshape)model_10/dense_21/Tensordot/transpose:y:0*model_10/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#model_10/dense_21/Tensordot/Reshapeæ
"model_10/dense_21/Tensordot/MatMulMatMul,model_10/dense_21/Tensordot/Reshape:output:02model_10/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2$
"model_10/dense_21/Tensordot/MatMul
#model_10/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2%
#model_10/dense_21/Tensordot/Const_2
)model_10/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_10/dense_21/Tensordot/concat_1/axis
$model_10/dense_21/Tensordot/concat_1ConcatV2-model_10/dense_21/Tensordot/GatherV2:output:0,model_10/dense_21/Tensordot/Const_2:output:02model_10/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_10/dense_21/Tensordot/concat_1Ù
model_10/dense_21/TensordotReshape,model_10/dense_21/Tensordot/MatMul:product:0-model_10/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
model_10/dense_21/TensordotÂ
(model_10/dense_21/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_10/dense_21/BiasAdd/ReadVariableOpÐ
model_10/dense_21/BiasAddBiasAdd$model_10/dense_21/Tensordot:output:00model_10/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤@2
model_10/dense_21/BiasAdd
(model_10/lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_10/lambda_3/Mean/reduction_indicesÁ
model_10/lambda_3/MeanMean"model_10/dense_21/BiasAdd:output:01model_10/lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_10/lambda_3/Mean
model_10/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   2!
model_10/lambda_3/Reshape/shapeÂ
model_10/lambda_3/ReshapeReshapemodel_10/lambda_3/Mean:output:0(model_10/lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model_10/lambda_3/Reshape
"model_10/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_10/concatenate_3/concat/axisÿ
model_10/concatenate_3/concatConcatV2"model_10/lambda_3/Reshape:output:0"model_10/dense_21/BiasAdd:output:0+model_10/concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_10/concatenate_3/concat
$model_10/patch_encoder_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model_10/patch_encoder_7/range/start
$model_10/patch_encoder_7/range/limitConst*
_output_shapes
: *
dtype0*
value
B :¥2&
$model_10/patch_encoder_7/range/limit
$model_10/patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2&
$model_10/patch_encoder_7/range/deltaó
model_10/patch_encoder_7/rangeRange-model_10/patch_encoder_7/range/start:output:0-model_10/patch_encoder_7/range/limit:output:0-model_10/patch_encoder_7/range/delta:output:0*
_output_shapes	
:¥2 
model_10/patch_encoder_7/range¦
5model_10/patch_encoder_7/embedding_3/embedding_lookupResourceGather=model_10_patch_encoder_7_embedding_3_embedding_lookup_1115351'model_10/patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*P
_classF
DBloc:@model_10/patch_encoder_7/embedding_3/embedding_lookup/1115351*
_output_shapes
:	¥@*
dtype027
5model_10/patch_encoder_7/embedding_3/embedding_lookupö
>model_10/patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity>model_10/patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*P
_classF
DBloc:@model_10/patch_encoder_7/embedding_3/embedding_lookup/1115351*
_output_shapes
:	¥@2@
>model_10/patch_encoder_7/embedding_3/embedding_lookup/Identity
@model_10/patch_encoder_7/embedding_3/embedding_lookup/Identity_1IdentityGmodel_10/patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	¥@2B
@model_10/patch_encoder_7/embedding_3/embedding_lookup/Identity_1ï
model_10/patch_encoder_7/addAddV2&model_10/concatenate_3/concat:output:0Imodel_10/patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2
model_10/patch_encoder_7/addÊ
>model_10/layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2@
>model_10/layer_normalization_12/moments/mean/reduction_indices
,model_10/layer_normalization_12/moments/meanMean model_10/patch_encoder_7/add:z:0Gmodel_10/layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(2.
,model_10/layer_normalization_12/moments/meanê
4model_10/layer_normalization_12/moments/StopGradientStopGradient5model_10/layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥26
4model_10/layer_normalization_12/moments/StopGradient£
9model_10/layer_normalization_12/moments/SquaredDifferenceSquaredDifference model_10/patch_encoder_7/add:z:0=model_10/layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2;
9model_10/layer_normalization_12/moments/SquaredDifferenceÒ
Bmodel_10/layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_10/layer_normalization_12/moments/variance/reduction_indicesÀ
0model_10/layer_normalization_12/moments/varianceMean=model_10/layer_normalization_12/moments/SquaredDifference:z:0Kmodel_10/layer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥*
	keep_dims(22
0model_10/layer_normalization_12/moments/variance§
/model_10/layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½7521
/model_10/layer_normalization_12/batchnorm/add/y
-model_10/layer_normalization_12/batchnorm/addAddV29model_10/layer_normalization_12/moments/variance:output:08model_10/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥2/
-model_10/layer_normalization_12/batchnorm/addÕ
/model_10/layer_normalization_12/batchnorm/RsqrtRsqrt1model_10/layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥21
/model_10/layer_normalization_12/batchnorm/Rsqrtþ
<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_10_layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOp
-model_10/layer_normalization_12/batchnorm/mulMul3model_10/layer_normalization_12/batchnorm/Rsqrt:y:0Dmodel_10/layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_10/layer_normalization_12/batchnorm/mulõ
/model_10/layer_normalization_12/batchnorm/mul_1Mul model_10/patch_encoder_7/add:z:01model_10/layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_10/layer_normalization_12/batchnorm/mul_1
/model_10/layer_normalization_12/batchnorm/mul_2Mul5model_10/layer_normalization_12/moments/mean:output:01model_10/layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_10/layer_normalization_12/batchnorm/mul_2ò
8model_10/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOpAmodel_10_layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_10/layer_normalization_12/batchnorm/ReadVariableOp
-model_10/layer_normalization_12/batchnorm/subSub@model_10/layer_normalization_12/batchnorm/ReadVariableOp:value:03model_10/layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2/
-model_10/layer_normalization_12/batchnorm/sub
/model_10/layer_normalization_12/batchnorm/add_1AddV23model_10/layer_normalization_12/batchnorm/mul_1:z:01model_10/layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@21
/model_10/layer_normalization_12/batchnorm/add_1
Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_10_multi_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpÖ
3model_10/multi_head_attention_6/query/einsum/EinsumEinsum3model_10/layer_normalization_12/batchnorm/add_1:z:0Jmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde25
3model_10/multi_head_attention_6/query/einsum/Einsumö
8model_10/multi_head_attention_6/query/add/ReadVariableOpReadVariableOpAmodel_10_multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_10/multi_head_attention_6/query/add/ReadVariableOp
)model_10/multi_head_attention_6/query/addAddV2<model_10/multi_head_attention_6/query/einsum/Einsum:output:0@model_10/multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2+
)model_10/multi_head_attention_6/query/add
@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOpImodel_10_multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02B
@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpÐ
1model_10/multi_head_attention_6/key/einsum/EinsumEinsum3model_10/layer_normalization_12/batchnorm/add_1:z:0Hmodel_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde23
1model_10/multi_head_attention_6/key/einsum/Einsumð
6model_10/multi_head_attention_6/key/add/ReadVariableOpReadVariableOp?model_10_multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype028
6model_10/multi_head_attention_6/key/add/ReadVariableOp
'model_10/multi_head_attention_6/key/addAddV2:model_10/multi_head_attention_6/key/einsum/Einsum:output:0>model_10/multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2)
'model_10/multi_head_attention_6/key/add
Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_10_multi_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpÖ
3model_10/multi_head_attention_6/value/einsum/EinsumEinsum3model_10/layer_normalization_12/batchnorm/add_1:z:0Jmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabc,cde->abde25
3model_10/multi_head_attention_6/value/einsum/Einsumö
8model_10/multi_head_attention_6/value/add/ReadVariableOpReadVariableOpAmodel_10_multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_10/multi_head_attention_6/value/add/ReadVariableOp
)model_10/multi_head_attention_6/value/addAddV2<model_10/multi_head_attention_6/value/einsum/Einsum:output:0@model_10/multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2+
)model_10/multi_head_attention_6/value/add
%model_10/multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2'
%model_10/multi_head_attention_6/Mul/yë
#model_10/multi_head_attention_6/MulMul-model_10/multi_head_attention_6/query/add:z:0.model_10/multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@2%
#model_10/multi_head_attention_6/Mul¢
-model_10/multi_head_attention_6/einsum/EinsumEinsum+model_10/multi_head_attention_6/key/add:z:0'model_10/multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥*
equationaecd,abcd->acbe2/
-model_10/multi_head_attention_6/einsum/Einsumá
/model_10/multi_head_attention_6/softmax/SoftmaxSoftmax6model_10/multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥21
/model_10/multi_head_attention_6/softmax/Softmaxç
0model_10/multi_head_attention_6/dropout/IdentityIdentity9model_10/multi_head_attention_6/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥22
0model_10/multi_head_attention_6/dropout/Identity¹
/model_10/multi_head_attention_6/einsum_1/EinsumEinsum9model_10/multi_head_attention_6/dropout/Identity:output:0-model_10/multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationacbe,aecd->abcd21
/model_10/multi_head_attention_6/einsum_1/Einsum¹
Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_10_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpø
>model_10/multi_head_attention_6/attention_output/einsum/EinsumEinsum8model_10/multi_head_attention_6/einsum_1/Einsum:output:0Umodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@*
equationabcd,cde->abe2@
>model_10/multi_head_attention_6/attention_output/einsum/Einsum
Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpLmodel_10_multi_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOpÂ
4model_10/multi_head_attention_6/attention_output/addAddV2Gmodel_10/multi_head_attention_6/attention_output/einsum/Einsum:output:0Kmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥@26
4model_10/multi_head_attention_6/attention_output/add
IdentityIdentity9model_10/multi_head_attention_6/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityê
NoOpNoOp)^model_10/dense_21/BiasAdd/ReadVariableOp+^model_10/dense_21/Tensordot/ReadVariableOp9^model_10/layer_normalization_12/batchnorm/ReadVariableOp=^model_10/layer_normalization_12/batchnorm/mul/ReadVariableOpD^model_10/multi_head_attention_6/attention_output/add/ReadVariableOpN^model_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp7^model_10/multi_head_attention_6/key/add/ReadVariableOpA^model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp9^model_10/multi_head_attention_6/query/add/ReadVariableOpC^model_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9^model_10/multi_head_attention_6/value/add/ReadVariableOpC^model_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp6^model_10/patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 2T
(model_10/dense_21/BiasAdd/ReadVariableOp(model_10/dense_21/BiasAdd/ReadVariableOp2X
*model_10/dense_21/Tensordot/ReadVariableOp*model_10/dense_21/Tensordot/ReadVariableOp2t
8model_10/layer_normalization_12/batchnorm/ReadVariableOp8model_10/layer_normalization_12/batchnorm/ReadVariableOp2|
<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOp<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOp2
Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOpCmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOp2
Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpMmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2p
6model_10/multi_head_attention_6/key/add/ReadVariableOp6model_10/multi_head_attention_6/key/add/ReadVariableOp2
@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2t
8model_10/multi_head_attention_6/query/add/ReadVariableOp8model_10/multi_head_attention_6/query/add/ReadVariableOp2
Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpBmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2t
8model_10/multi_head_attention_6/value/add/ReadVariableOp8model_10/multi_head_attention_6/value/add/ReadVariableOp2
Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpBmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2n
5model_10/patch_encoder_7/embedding_3/embedding_lookup5model_10/patch_encoder_7/embedding_3/embedding_lookup:Z V
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
ä
c
G__inference_resizing_3_layer_call_and_return_conditional_losses_1115426

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
±

â
3__inference_data_augmentation_layer_call_fn_1115861
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332
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
Ø
Ç
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1115719

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
ï
K
/__inference_random_zoom_3_layer_call_fn_1117973

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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11154442
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
ù.
º
E__inference_model_10_layer_call_and_return_conditional_losses_1116469
input_4
data_augmentation_1116429
data_augmentation_1116431"
dense_21_1116435:1@
dense_21_1116437:@*
patch_encoder_7_1116442:	¥@,
layer_normalization_12_1116445:@,
layer_normalization_12_1116447:@4
multi_head_attention_6_1116450:@@0
multi_head_attention_6_1116452:@4
multi_head_attention_6_1116454:@@0
multi_head_attention_6_1116456:@4
multi_head_attention_6_1116458:@@0
multi_head_attention_6_1116460:@4
multi_head_attention_6_1116462:@@,
multi_head_attention_6_1116464:@
identity¢ dense_21/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCallµ
!data_augmentation/PartitionedCallPartitionedCallinput_4data_augmentation_1116429data_augmentation_1116431*
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472#
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
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1116435dense_21_1116437*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
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
E__inference_lambda_3_layer_call_and_return_conditional_losses_11159622
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
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1116442*
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
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116445layer_normalization_12_1116447*
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
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116450multi_head_attention_6_1116452multi_head_attention_6_1116454multi_head_attention_6_1116456multi_head_attention_6_1116458multi_head_attention_6_1116460multi_head_attention_6_1116462multi_head_attention_6_1116464*
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111605520
.multi_head_attention_6/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityý
NoOpNoOp!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
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
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1115962

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
ç
¡
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117177

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
«
k
3__inference_data_augmentation_layer_call_fn_1117149

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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472
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
2
à
E__inference_model_10_layer_call_and_return_conditional_losses_1116346

inputs
data_augmentation_1116300
data_augmentation_1116302'
data_augmentation_1116304:	'
data_augmentation_1116306:	'
data_augmentation_1116308:	"
dense_21_1116312:1@
dense_21_1116314:@*
patch_encoder_7_1116319:	¥@,
layer_normalization_12_1116322:@,
layer_normalization_12_1116324:@4
multi_head_attention_6_1116327:@@0
multi_head_attention_6_1116329:@4
multi_head_attention_6_1116331:@@0
multi_head_attention_6_1116333:@4
multi_head_attention_6_1116335:@@0
multi_head_attention_6_1116337:@4
multi_head_attention_6_1116339:@@,
multi_head_attention_6_1116341:@
identity¢)data_augmentation/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢.layer_normalization_12/StatefulPartitionedCall¢.multi_head_attention_6/StatefulPartitionedCall¢'patch_encoder_7/StatefulPartitionedCall 
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinputsdata_augmentation_1116300data_augmentation_1116302data_augmentation_1116304data_augmentation_1116306data_augmentation_1116308*
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332+
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
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall»
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1116312dense_21_1116314*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
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
E__inference_lambda_3_layer_call_and_return_conditional_losses_11162412
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
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallÇ
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1116319*
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
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCall
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116322layer_normalization_12_1116324*
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
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCallº
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116327multi_head_attention_6_1116329multi_head_attention_6_1116331multi_head_attention_6_1116333multi_head_attention_6_1116335multi_head_attention_6_1116337multi_head_attention_6_1116339multi_head_attention_6_1116341*
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111617920
.multi_head_attention_6/StatefulPartitionedCall
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity©
NoOpNoOp*^data_augmentation/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : 2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
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
ì
[
/__inference_concatenate_3_layer_call_fn_1117563
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
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
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
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117557

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


S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1117622

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
çÒ
ö
E__inference_model_10_layer_call_and_return_conditional_losses_1117140

inputs+
'data_augmentation_normalization_3_sub_y,
(data_augmentation_normalization_3_sqrt_x_
Qdata_augmentation_random_flip_3_stateful_uniform_full_int_rngreadandskip_resource:	Z
Ldata_augmentation_random_rotation_3_stateful_uniform_rngreadandskip_resource:	V
Hdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resource:	<
*dense_21_tensordot_readvariableop_resource:1@6
(dense_21_biasadd_readvariableop_resource:@G
4patch_encoder_7_embedding_3_embedding_lookup_1117078:	¥@J
<layer_normalization_12_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_12_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_6_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_query_add_readvariableop_resource:@V
@multi_head_attention_6_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_6_key_add_readvariableop_resource:@X
Bmulti_head_attention_6_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_value_add_readvariableop_resource:@c
Mmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_6_attention_output_add_readvariableop_resource:@
identity¢Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip¢odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip¢?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip¢Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip¢dense_21/BiasAdd/ReadVariableOp¢!dense_21/Tensordot/ReadVariableOp¢/layer_normalization_12/batchnorm/ReadVariableOp¢3layer_normalization_12/batchnorm/mul/ReadVariableOp¢:multi_head_attention_6/attention_output/add/ReadVariableOp¢Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_6/key/add/ReadVariableOp¢7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/query/add/ReadVariableOp¢9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/value/add/ReadVariableOp¢9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp¢,patch_encoder_7/embedding_3/embedding_lookupÂ
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
,patch_encoder_7/embedding_3/embedding_lookupResourceGather4patch_encoder_7_embedding_3_embedding_lookup_1117078patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1117078*
_output_shapes
:	¥@*
dtype02.
,patch_encoder_7/embedding_3/embedding_lookupÒ
5patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity5patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1117078*
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
+multi_head_attention_6/attention_output/add
IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identity÷	
NoOpNoOpI^data_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipp^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgw^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterD^data_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip@^data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipB^data_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip ^dense_21/BiasAdd/ReadVariableOp"^dense_21/Tensordot/ReadVariableOp0^layer_normalization_12/batchnorm/ReadVariableOp4^layer_normalization_12/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp-^patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : 2
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipHdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip2â
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgodata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2ð
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCountervdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipCdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip2
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip2
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipAdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2F
!dense_21/Tensordot/ReadVariableOp!dense_21/Tensordot/ReadVariableOp2b
/layer_normalization_12/batchnorm/ReadVariableOp/layer_normalization_12/batchnorm/ReadVariableOp2j
3layer_normalization_12/batchnorm/mul/ReadVariableOp3layer_normalization_12/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_6/key/add/ReadVariableOp-multi_head_attention_6/key/add/ReadVariableOp2r
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/query/add/ReadVariableOp/multi_head_attention_6/query/add/ReadVariableOp2v
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/value/add/ReadVariableOp/multi_head_attention_6/value/add/ReadVariableOp2v
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2\
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1116055	
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

ý
8__inference_multi_head_attention_6_layer_call_fn_1117646	
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_11160552
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

ý
8__inference_multi_head_attention_6_layer_call_fn_1117670	
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_11161792
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
/__inference_random_zoom_3_layer_call_fn_1117980

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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11155882
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
¡
f
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1117984

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
¡
f
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1115432

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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1115438

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
ß
G
+__inference_patches_7_layer_call_fn_1117485

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
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
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
Î
¢
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115897
normalization_3_input
normalization_3_sub_y
normalization_3_sqrt_x#
random_flip_3_1115887:	'
random_rotation_3_1115890:	#
random_zoom_3_1115893:	
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
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
resizing_3/PartitionedCall½
%random_flip_3/StatefulPartitionedCallStatefulPartitionedCall#resizing_3/PartitionedCall:output:0random_flip_3_1115887*
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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11157902'
%random_flip_3/StatefulPartitionedCallØ
)random_rotation_3/StatefulPartitionedCallStatefulPartitionedCall.random_flip_3/StatefulPartitionedCall:output:0random_rotation_3_1115890*
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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11157192+
)random_rotation_3/StatefulPartitionedCallÌ
%random_zoom_3/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_3/StatefulPartitionedCall:output:0random_zoom_3_1115893*
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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11155882'
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
2
ª	
 __inference__traced_save_1118207
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop;
7savev2_layer_normalization_12_gamma_read_readvariableop:
6savev2_layer_normalization_12_beta_read_readvariableop#
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
Gsavev2_multi_head_attention_6_attention_output_bias_read_readvariableop'
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
ShardedFilename½
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ï
valueÅBÂB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names°
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¾	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop7savev2_layer_normalization_12_gamma_read_readvariableop6savev2_layer_normalization_12_beta_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableopAsavev2_patch_encoder_7_embedding_3_embeddings_read_readvariableop>savev2_multi_head_attention_6_query_kernel_read_readvariableop<savev2_multi_head_attention_6_query_bias_read_readvariableop<savev2_multi_head_attention_6_key_kernel_read_readvariableop:savev2_multi_head_attention_6_key_bias_read_readvariableop>savev2_multi_head_attention_6_value_kernel_read_readvariableop<savev2_multi_head_attention_6_value_bias_read_readvariableopIsavev2_multi_head_attention_6_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_6_attention_output_bias_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *"
dtypes
2				2
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

identity_1Identity_1:output:0*¼
_input_shapesª
§: :1@:@:@:@::: :	¥@:@@:@:@@:@:@@:@:@@:@:::: 2(
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
:: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	¥@:(	$
"
_output_shapes
:@@:$
 

_output_shapes

:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:($
"
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
ü
b
F__inference_patches_7_layer_call_and_return_conditional_losses_1115916

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


Ó
3__inference_data_augmentation_layer_call_fn_1117164

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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332
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
ü
b
F__inference_patches_7_layer_call_and_return_conditional_losses_1117492

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
Ì
Ç
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1117591

projection7
$embedding_3_embedding_lookup_1117584:	¥@
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
embedding_3/embedding_lookupResourceGather$embedding_3_embedding_lookup_1117584range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*7
_class-
+)loc:@embedding_3/embedding_lookup/1117584*
_output_shapes
:	¥@*
dtype02
embedding_3/embedding_lookup
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*7
_class-
+)loc:@embedding_3/embedding_lookup/1117584*
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
¬§
ç
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118110

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
þ

1__inference_patch_encoder_7_layer_call_fn_1117577

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
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872
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
Ã9

S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117749	
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
¡

N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115833

inputs
normalization_3_sub_y
normalization_3_sqrt_x#
random_flip_3_1115823:	'
random_rotation_3_1115826:	#
random_zoom_3_1115829:	
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
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
resizing_3/PartitionedCall½
%random_flip_3/StatefulPartitionedCallStatefulPartitionedCall#resizing_3/PartitionedCall:output:0random_flip_3_1115823*
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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11157902'
%random_flip_3/StatefulPartitionedCallØ
)random_rotation_3/StatefulPartitionedCallStatefulPartitionedCall.random_flip_3/StatefulPartitionedCall:output:0random_rotation_3_1115826*
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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11157192+
)random_rotation_3/StatefulPartitionedCallÌ
%random_zoom_3/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_3/StatefulPartitionedCall:output:0random_zoom_3_1115829*
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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11155882'
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
ï
K
/__inference_random_flip_3_layer_call_fn_1117765

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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11154322
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
õ

/__inference_random_flip_3_layer_call_fn_1117772

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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11157902
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
«°
ß
E__inference_model_10_layer_call_and_return_conditional_losses_1116737

inputs+
'data_augmentation_normalization_3_sub_y,
(data_augmentation_normalization_3_sqrt_x<
*dense_21_tensordot_readvariableop_resource:1@6
(dense_21_biasadd_readvariableop_resource:@G
4patch_encoder_7_embedding_3_embedding_lookup_1116682:	¥@J
<layer_normalization_12_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_12_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_6_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_query_add_readvariableop_resource:@V
@multi_head_attention_6_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_6_key_add_readvariableop_resource:@X
Bmulti_head_attention_6_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_6_value_add_readvariableop_resource:@c
Mmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_6_attention_output_add_readvariableop_resource:@
identity¢dense_21/BiasAdd/ReadVariableOp¢!dense_21/Tensordot/ReadVariableOp¢/layer_normalization_12/batchnorm/ReadVariableOp¢3layer_normalization_12/batchnorm/mul/ReadVariableOp¢:multi_head_attention_6/attention_output/add/ReadVariableOp¢Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_6/key/add/ReadVariableOp¢7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/query/add/ReadVariableOp¢9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_6/value/add/ReadVariableOp¢9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp¢,patch_encoder_7/embedding_3/embedding_lookupÂ
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
,patch_encoder_7/embedding_3/embedding_lookupResourceGather4patch_encoder_7_embedding_3_embedding_lookup_1116682patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1116682*
_output_shapes
:	¥@*
dtype02.
,patch_encoder_7/embedding_3/embedding_lookupÒ
5patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity5patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1116682*
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
+multi_head_attention_6/attention_output/add
IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥¥2

Identityõ
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp"^dense_21/Tensordot/ReadVariableOp0^layer_normalization_12/batchnorm/ReadVariableOp4^layer_normalization_12/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp-^patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2F
!dense_21/Tensordot/ReadVariableOp!dense_21/Tensordot/ReadVariableOp2b
/layer_normalization_12/batchnorm/ReadVariableOp/layer_normalization_12/batchnorm/ReadVariableOp2j
3layer_normalization_12/batchnorm/mul/ReadVariableOp3layer_normalization_12/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_6/key/add/ReadVariableOp-multi_head_attention_6/key/add/ReadVariableOp2r
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp7multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/query/add/ReadVariableOp/multi_head_attention_6/query/add/ReadVariableOp2v
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_6/value/add/ReadVariableOp/multi_head_attention_6/value/add/ReadVariableOp2v
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2\
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
Ø
z
3__inference_data_augmentation_layer_call_fn_1115454
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472
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
ý
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1115971

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


*__inference_dense_21_layer_call_fn_1117501

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
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482
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
ï
a
E__inference_lambda_3_layer_call_and_return_conditional_losses_1116241

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
multi_head_attention_6:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ¥¥tensorflow/serving/predict:Õ±
Î
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
ï_default_save_signature
ð__call__
+ñ&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer

layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer-4
	variables
regularization_losses
trainable_variables
	keras_api
ò__call__
+ó&call_and_return_all_conditional_losses"
_tf_keras_sequential
§
	variables
regularization_losses
trainable_variables
	keras_api
ô__call__
+õ&call_and_return_all_conditional_losses"
_tf_keras_layer
½

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
§
"	variables
#regularization_losses
$trainable_variables
%	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses"
_tf_keras_layer
§
&	variables
'regularization_losses
(trainable_variables
)	keras_api
ú__call__
+û&call_and_return_all_conditional_losses"
_tf_keras_layer
¿
*position_embedding
+	variables
,regularization_losses
-trainable_variables
.	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer
Æ
/axis
	0gamma
1beta
2	variables
3regularization_losses
4trainable_variables
5	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses"
_tf_keras_layer

6_query_dense
7
_key_dense
8_value_dense
9_softmax
:_dropout_layer
;_output_dense
<	variables
=regularization_losses
>trainable_variables
?	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

@0
A1
B2
3
4
C5
06
17
D8
E9
F10
G11
H12
I13
J14
K15"
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
C2
03
14
D5
E6
F7
G8
H9
I10
J11
K12"
trackable_list_wrapper
Î
Llayer_regularization_losses
Mlayer_metrics

Nlayers

	variables
regularization_losses
trainable_variables
Onon_trainable_variables
Pmetrics
ð__call__
ï_default_save_signature
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
Ô
Q
_keep_axis
R_reduce_axis
S_reduce_axis_mask
T_broadcast_shape
@mean
@
adapt_mean
Avariance
Aadapt_variance
	Bcount
U	keras_api
_adapt_function"
_tf_keras_layer
§
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
±
Z_rng
[	variables
\regularization_losses
]trainable_variables
^	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
±
__rng
`	variables
aregularization_losses
btrainable_variables
c	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
±
d_rng
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ilayer_regularization_losses
jlayer_metrics

klayers
	variables
regularization_losses
trainable_variables
lnon_trainable_variables
mmetrics
ò__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
nlayer_regularization_losses
olayer_metrics

players
	variables
regularization_losses
trainable_variables
qnon_trainable_variables
rmetrics
ô__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
!:1@2dense_21/kernel
:@2dense_21/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
slayer_regularization_losses
tlayer_metrics

ulayers
	variables
regularization_losses
 trainable_variables
vnon_trainable_variables
wmetrics
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
xlayer_regularization_losses
ylayer_metrics

zlayers
"	variables
#regularization_losses
$trainable_variables
{non_trainable_variables
|metrics
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
}layer_regularization_losses
~layer_metrics

layers
&	variables
'regularization_losses
(trainable_variables
non_trainable_variables
metrics
ú__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
»
C
embeddings
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
+	variables
,regularization_losses
-trainable_variables
non_trainable_variables
metrics
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2layer_normalization_12/gamma
):'@2layer_normalization_12/beta
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
layers
2	variables
3regularization_losses
4trainable_variables
non_trainable_variables
metrics
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
ô
partial_output_shape
full_output_shape

Dkernel
Ebias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
partial_output_shape
full_output_shape

Fkernel
Gbias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
partial_output_shape
full_output_shape

Hkernel
Ibias
	variables
regularization_losses
 trainable_variables
¡	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¢	variables
£regularization_losses
¤trainable_variables
¥	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¦	variables
§regularization_losses
¨trainable_variables
©	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ô
ªpartial_output_shape
«full_output_shape

Jkernel
Kbias
¬	variables
­regularization_losses
®trainable_variables
¯	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
X
D0
E1
F2
G3
H4
I5
J6
K7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
D0
E1
F2
G3
H4
I5
J6
K7"
trackable_list_wrapper
µ
 °layer_regularization_losses
±layer_metrics
²layers
<	variables
=regularization_losses
>trainable_variables
³non_trainable_variables
´metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
5
@0
A1
B2"
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
µ
 µlayer_regularization_losses
¶layer_metrics
·layers
V	variables
Wregularization_losses
Xtrainable_variables
¸non_trainable_variables
¹metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
º
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 »layer_regularization_losses
¼layer_metrics
½layers
[	variables
\regularization_losses
]trainable_variables
¾non_trainable_variables
¿metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
À
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Álayer_regularization_losses
Âlayer_metrics
Ãlayers
`	variables
aregularization_losses
btrainable_variables
Änon_trainable_variables
Åmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
Æ
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Çlayer_regularization_losses
Èlayer_metrics
Élayers
e	variables
fregularization_losses
gtrainable_variables
Ênon_trainable_variables
Ëmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
@0
A1
B2"
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
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
¸
 Ìlayer_regularization_losses
Ílayer_metrics
Îlayers
	variables
regularization_losses
trainable_variables
Ïnon_trainable_variables
Ðmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
*0"
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
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
¸
 Ñlayer_regularization_losses
Òlayer_metrics
Ólayers
	variables
regularization_losses
trainable_variables
Ônon_trainable_variables
Õmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
¸
 Ölayer_regularization_losses
×layer_metrics
Ølayers
	variables
regularization_losses
trainable_variables
Ùnon_trainable_variables
Úmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
¸
 Ûlayer_regularization_losses
Ülayer_metrics
Ýlayers
	variables
regularization_losses
 trainable_variables
Þnon_trainable_variables
ßmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 àlayer_regularization_losses
álayer_metrics
âlayers
¢	variables
£regularization_losses
¤trainable_variables
ãnon_trainable_variables
ämetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ålayer_regularization_losses
ælayer_metrics
çlayers
¦	variables
§regularization_losses
¨trainable_variables
ènon_trainable_variables
émetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
¸
 êlayer_regularization_losses
ëlayer_metrics
ìlayers
¬	variables
­regularization_losses
®trainable_variables
ínon_trainable_variables
îmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
60
71
82
93
:4
;5"
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
ÍBÊ
"__inference__wrapped_model_1115406input_4"
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
*__inference_model_10_layer_call_fn_1116108
*__inference_model_10_layer_call_fn_1116590
*__inference_model_10_layer_call_fn_1116631
*__inference_model_10_layer_call_fn_1116426À
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
E__inference_model_10_layer_call_and_return_conditional_losses_1116737
E__inference_model_10_layer_call_and_return_conditional_losses_1117140
E__inference_model_10_layer_call_and_return_conditional_losses_1116469
E__inference_model_10_layer_call_and_return_conditional_losses_1116518À
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
3__inference_data_augmentation_layer_call_fn_1115454
3__inference_data_augmentation_layer_call_fn_1117149
3__inference_data_augmentation_layer_call_fn_1117164
3__inference_data_augmentation_layer_call_fn_1115861À
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117177
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117480
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115876
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115897À
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
+__inference_patches_7_layer_call_fn_1117485¢
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
F__inference_patches_7_layer_call_and_return_conditional_losses_1117492¢
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
*__inference_dense_21_layer_call_fn_1117501¢
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
E__inference_dense_21_layer_call_and_return_conditional_losses_1117531¢
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
*__inference_lambda_3_layer_call_fn_1117536
*__inference_lambda_3_layer_call_fn_1117541À
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
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117549
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117557À
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
/__inference_concatenate_3_layer_call_fn_1117563¢
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
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1117570¢
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
1__inference_patch_encoder_7_layer_call_fn_1117577¦
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
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1117591¦
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
8__inference_layer_normalization_12_layer_call_fn_1117600¢
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
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1117622¢
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
8__inference_multi_head_attention_6_layer_call_fn_1117646
8__inference_multi_head_attention_6_layer_call_fn_1117670ü
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
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117706
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117749ü
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
%__inference_signature_wrapper_1116555input_4"
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
,__inference_resizing_3_layer_call_fn_1117754¢
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
G__inference_resizing_3_layer_call_and_return_conditional_losses_1117760¢
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
/__inference_random_flip_3_layer_call_fn_1117765
/__inference_random_flip_3_layer_call_fn_1117772´
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
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117776
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117834´
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
3__inference_random_rotation_3_layer_call_fn_1117839
3__inference_random_rotation_3_layer_call_fn_1117846´
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
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117850
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117968´
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
/__inference_random_zoom_3_layer_call_fn_1117973
/__inference_random_zoom_3_layer_call_fn_1117980´
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
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1117984
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118110´
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
	J
Const
J	
Const_1Ñ
"__inference__wrapped_model_1115406ªC01DEFGHIJK:¢7
0¢-
+(
input_4ÿÿÿÿÿÿÿÿÿ
ª "YªV
T
multi_head_attention_6:7
multi_head_attention_6ÿÿÿÿÿÿÿÿÿ¥¥x
__inference_adapt_step_1113455VB@AK¢H
A¢>
<9'¢$
"ÿÿÿÿÿÿÿÿÿIteratorSpec
ª "
 à
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1117570c¢`
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
/__inference_concatenate_3_layer_call_fn_1117563c¢`
Y¢V
TQ
&#
inputs/0ÿÿÿÿÿÿÿÿÿ@
'$
inputs/1ÿÿÿÿÿÿÿÿÿ¤@
ª "ÿÿÿÿÿÿÿÿÿ¥@Û
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115876P¢M
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115897
ºÀÆP¢M
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117177yA¢>
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
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117480
ºÀÆA¢>
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
3__inference_data_augmentation_layer_call_fn_1115454{P¢M
F¢C
96
normalization_3_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿÄi¹
3__inference_data_augmentation_layer_call_fn_1115861
ºÀÆP¢M
F¢C
96
normalization_3_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "!ÿÿÿÿÿÿÿÿÿÄi£
3__inference_data_augmentation_layer_call_fn_1117149lA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿÄi©
3__inference_data_augmentation_layer_call_fn_1117164r
ºÀÆA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "!ÿÿÿÿÿÿÿÿÿÄi¯
E__inference_dense_21_layer_call_and_return_conditional_losses_1117531f4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¤1
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¤@
 
*__inference_dense_21_layer_call_fn_1117501Y4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¤1
ª "ÿÿÿÿÿÿÿÿÿ¤@²
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117549i<¢9
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
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117557i<¢9
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
*__inference_lambda_3_layer_call_fn_1117536\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ¤@

 
p 
ª "ÿÿÿÿÿÿÿÿÿ@
*__inference_lambda_3_layer_call_fn_1117541\<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ¤@

 
p
ª "ÿÿÿÿÿÿÿÿÿ@½
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1117622f014¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
8__inference_layer_normalization_12_layer_call_fn_1117600Y014¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@Ò
E__inference_model_10_layer_call_and_return_conditional_losses_1116469C01DEFGHIJKB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 Ø
E__inference_model_10_layer_call_and_return_conditional_losses_1116518ºÀÆC01DEFGHIJKB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 Ñ
E__inference_model_10_layer_call_and_return_conditional_losses_1116737C01DEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 ×
E__inference_model_10_layer_call_and_return_conditional_losses_1117140ºÀÆC01DEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¥¥
 ©
*__inference_model_10_layer_call_fn_1116108{C01DEFGHIJKB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥°
*__inference_model_10_layer_call_fn_1116426ºÀÆC01DEFGHIJKB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥¨
*__inference_model_10_layer_call_fn_1116590zC01DEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥¯
*__inference_model_10_layer_call_fn_1116631ºÀÆC01DEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ¥¥©
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117706ÑDEFGHIJKi¢f
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
 ©
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117749ÑDEFGHIJKi¢f
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
 
8__inference_multi_head_attention_6_layer_call_fn_1117646ÃDEFGHIJKi¢f
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
1ÿÿÿÿÿÿÿÿÿ¥¥
8__inference_multi_head_attention_6_layer_call_fn_1117670ÃDEFGHIJKi¢f
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
1ÿÿÿÿÿÿÿÿÿ¥¥¹
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1117591iC8¢5
.¢+
)&

projectionÿÿÿÿÿÿÿÿÿ¥@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¥@
 
1__inference_patch_encoder_7_layer_call_fn_1117577\C8¢5
.¢+
)&

projectionÿÿÿÿÿÿÿÿÿ¥@
ª "ÿÿÿÿÿÿÿÿÿ¥@°
F__inference_patches_7_layer_call_and_return_conditional_losses_1117492f8¢5
.¢+
)&
imagesÿÿÿÿÿÿÿÿÿÄi
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¤1
 
+__inference_patches_7_layer_call_fn_1117485Y8¢5
.¢+
)&
imagesÿÿÿÿÿÿÿÿÿÄi
ª "ÿÿÿÿÿÿÿÿÿ¤1¼
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117776n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 À
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117834rº<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
/__inference_random_flip_3_layer_call_fn_1117765a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª "!ÿÿÿÿÿÿÿÿÿÄi
/__inference_random_flip_3_layer_call_fn_1117772eº<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª "!ÿÿÿÿÿÿÿÿÿÄiÀ
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117850n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 Ä
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117968rÀ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
3__inference_random_rotation_3_layer_call_fn_1117839a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª "!ÿÿÿÿÿÿÿÿÿÄi
3__inference_random_rotation_3_layer_call_fn_1117846eÀ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª "!ÿÿÿÿÿÿÿÿÿÄi¼
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1117984n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 À
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118110rÆ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
/__inference_random_zoom_3_layer_call_fn_1117973a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p 
ª "!ÿÿÿÿÿÿÿÿÿÄi
/__inference_random_zoom_3_layer_call_fn_1117980eÆ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÄi
p
ª "!ÿÿÿÿÿÿÿÿÿÄi¶
G__inference_resizing_3_layer_call_and_return_conditional_losses_1117760k9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÄi
 
,__inference_resizing_3_layer_call_fn_1117754^9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÄiß
%__inference_signature_wrapper_1116555µC01DEFGHIJKE¢B
¢ 
;ª8
6
input_4+(
input_4ÿÿÿÿÿÿÿÿÿ"YªV
T
multi_head_attention_6:7
multi_head_attention_6ÿÿÿÿÿÿÿÿÿ¥¥