ј«!
ФЈ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
┐
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
Г
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
Ї
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
Ї
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
dtypetypeѕ
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
њ
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
Ц
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	ѕ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
list(type)(0ѕ
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

2	љ
Й
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
executor_typestring ѕ
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02unknown8ё▄
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
љ
layer_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namelayer_normalization_12/gamma
Ѕ
0layer_normalization_12/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_12/gamma*
_output_shapes
:@*
dtype0
ј
layer_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_12/beta
Є
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
Е
&patch_encoder_7/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ц@*7
shared_name(&patch_encoder_7/embedding_3/embeddings
б
:patch_encoder_7/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp&patch_encoder_7/embedding_3/embeddings*
_output_shapes
:	Ц@*
dtype0
д
#multi_head_attention_6/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_6/query/kernel
Ъ
7multi_head_attention_6/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/query/kernel*"
_output_shapes
:@@*
dtype0
ъ
!multi_head_attention_6/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_6/query/bias
Ќ
5multi_head_attention_6/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/query/bias*
_output_shapes

:@*
dtype0
б
!multi_head_attention_6/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention_6/key/kernel
Џ
5multi_head_attention_6/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/key/kernel*"
_output_shapes
:@@*
dtype0
џ
multi_head_attention_6/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_6/key/bias
Њ
3multi_head_attention_6/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_6/key/bias*
_output_shapes

:@*
dtype0
д
#multi_head_attention_6/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_6/value/kernel
Ъ
7multi_head_attention_6/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_6/value/kernel*"
_output_shapes
:@@*
dtype0
ъ
!multi_head_attention_6/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_6/value/bias
Ќ
5multi_head_attention_6/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_6/value/bias*
_output_shapes

:@*
dtype0
╝
.multi_head_attention_6/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.multi_head_attention_6/attention_output/kernel
х
Bmulti_head_attention_6/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_6/attention_output/kernel*"
_output_shapes
:@@*
dtype0
░
,multi_head_attention_6/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention_6/attention_output/bias
Е
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
valueB*sБOC
l
Const_1Const*&
_output_shapes
:*
dtype0*%
valueB*Ѓ╝sD

NoOpNoOp
ХI
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*№H
valueтHBРH B█H
┘
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
Г
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
╗
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
Г
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
Ц
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
Г
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
Г
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
Г
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
Г
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
»
}layer_regularization_losses
~layer_metrics

layers
&	variables
'regularization_losses
(trainable_variables
ђnon_trainable_variables
Ђmetrics
f
C
embeddings
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
Ё	keras_api

C0
 

C0
▓
 єlayer_regularization_losses
Єlayer_metrics
ѕlayers
+	variables
,regularization_losses
-trainable_variables
Ѕnon_trainable_variables
іmetrics
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
▓
 Іlayer_regularization_losses
їlayer_metrics
Їlayers
2	variables
3regularization_losses
4trainable_variables
јnon_trainable_variables
Јmetrics
Ъ
љpartial_output_shape
Љfull_output_shape

Dkernel
Ebias
њ	variables
Њregularization_losses
ћtrainable_variables
Ћ	keras_api
Ъ
ќpartial_output_shape
Ќfull_output_shape

Fkernel
Gbias
ў	variables
Ўregularization_losses
џtrainable_variables
Џ	keras_api
Ъ
юpartial_output_shape
Юfull_output_shape

Hkernel
Ibias
ъ	variables
Ъregularization_losses
аtrainable_variables
А	keras_api
V
б	variables
Бregularization_losses
цtrainable_variables
Ц	keras_api
V
д	variables
Дregularization_losses
еtrainable_variables
Е	keras_api
Ъ
фpartial_output_shape
Фfull_output_shape

Jkernel
Kbias
г	variables
Гregularization_losses
«trainable_variables
»	keras_api
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
▓
 ░layer_regularization_losses
▒layer_metrics
▓layers
<	variables
=regularization_losses
>trainable_variables
│non_trainable_variables
┤metrics
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
▓
 хlayer_regularization_losses
Хlayer_metrics
иlayers
V	variables
Wregularization_losses
Xtrainable_variables
Иnon_trainable_variables
╣metrics

║
_state_var
 
 
 
▓
 ╗layer_regularization_losses
╝layer_metrics
йlayers
[	variables
\regularization_losses
]trainable_variables
Йnon_trainable_variables
┐metrics

└
_state_var
 
 
 
▓
 ┴layer_regularization_losses
┬layer_metrics
├layers
`	variables
aregularization_losses
btrainable_variables
─non_trainable_variables
┼metrics

к
_state_var
 
 
 
▓
 Кlayer_regularization_losses
╚layer_metrics
╔layers
e	variables
fregularization_losses
gtrainable_variables
╩non_trainable_variables
╦metrics
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
х
 ╠layer_regularization_losses
═layer_metrics
╬layers
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
¤non_trainable_variables
лmetrics
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
х
 Лlayer_regularization_losses
мlayer_metrics
Мlayers
њ	variables
Њregularization_losses
ћtrainable_variables
нnon_trainable_variables
Нmetrics
 
 

F0
G1
 

F0
G1
х
 оlayer_regularization_losses
Оlayer_metrics
пlayers
ў	variables
Ўregularization_losses
џtrainable_variables
┘non_trainable_variables
┌metrics
 
 

H0
I1
 

H0
I1
х
 █layer_regularization_losses
▄layer_metrics
Пlayers
ъ	variables
Ъregularization_losses
аtrainable_variables
яnon_trainable_variables
▀metrics
 
 
 
х
 Яlayer_regularization_losses
рlayer_metrics
Рlayers
б	variables
Бregularization_losses
цtrainable_variables
сnon_trainable_variables
Сmetrics
 
 
 
х
 тlayer_regularization_losses
Тlayer_metrics
уlayers
д	variables
Дregularization_losses
еtrainable_variables
Уnon_trainable_variables
жmetrics
 
 

J0
K1
 

J0
K1
х
 Жlayer_regularization_losses
вlayer_metrics
Вlayers
г	variables
Гregularization_losses
«trainable_variables
ьnon_trainable_variables
Ьmetrics
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
ј
serving_default_input_4Placeholder*1
_output_shapes
:         ўќ*
dtype0*&
shape:         ўќ
А
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4ConstConst_1dense_21/kerneldense_21/bias&patch_encoder_7/embedding_3/embeddingslayer_normalization_12/gammalayer_normalization_12/beta#multi_head_attention_6/query/kernel!multi_head_attention_6/query/bias!multi_head_attention_6/key/kernelmulti_head_attention_6/key/bias#multi_head_attention_6/value/kernel!multi_head_attention_6/value/bias.multi_head_attention_6/attention_output/kernel,multi_head_attention_6/attention_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1116555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ	
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
GPU2*0J 8ѓ *)
f$R"
 __inference__traced_save_1118207
Џ
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
GPU2*0J 8ѓ *,
f'R%
#__inference__traced_restore_1118274цл
пЏ
К
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117968

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipD
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЂ
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2
strided_slice_1/stackЁ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
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
CastЂ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_2/stackЁ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
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
 *§Г Й2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *§Г >2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
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
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/algИ
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub»
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2
stateful_uniform/mulћ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
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
:         2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_1/yё
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1Њ
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_2/yѓ
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2Ќ
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_1Ќ
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_3Ќ
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yф
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_5/yѓ
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_6/yё
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6Ў
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_7/yѓ
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7Ў
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_3Ќ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/addЌ
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y░
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shapeћ
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stackў
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1ў
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2┬
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
:         2
rotation_matrix/Cos_2Ъ
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stackБ
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1Б
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2э
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_2Ъ
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stackБ
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1Б
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2э
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2Ї
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/NegЪ
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stackБ
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1Б
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2щ
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_3Ъ
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stackБ
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1Б
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2э
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_3Ъ
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stackБ
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1Б
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2э
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5Ъ
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stackБ
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1Б
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2ч
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6ѓ
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1├
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
rotation_matrix/zeros/Constх
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axisе
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
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
transform/fill_value╚
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3Ю
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
в
H
,__inference_resizing_3_layer_call_fn_1117754

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ўќ:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs
Б
А
8__inference_layer_normalization_12_layer_call_fn_1117600

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_11160132
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ц@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ц@
 
_user_specified_nameinputs
№
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
:         @2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         @2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ц@:T P
,
_output_shapes
:         ц@
 
_user_specified_nameinputs
╔
■
*__inference_model_10_layer_call_fn_1116108
input_4
unknown
	unknown_0
	unknown_1:1@
	unknown_2:@
	unknown_3:	Ц@
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
identityѕбStatefulPartitionedCall«
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
:         ЦЦ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11160752
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ўќ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
гД
у
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1115588

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipб!stateful_uniform_1/RngReadAndSkipD
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЂ
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2
strided_slice_1/stackЁ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
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
CastЂ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_2/stackЁ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
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
stateful_uniform/shape/1А
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
 *═╠L?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
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
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg╝
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub│
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         2
stateful_uniform/mulў
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2
stateful_uniformz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1Д
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
 *═╠L?2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?2
stateful_uniform_1/max~
stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform_1/ConstА
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
stateful_uniform_1/Cast/xљ
stateful_uniform_1/Cast_1Cast stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform_1/Cast_1Ѓ
!stateful_uniform_1/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource"stateful_uniform_1/Cast/x:output:0stateful_uniform_1/Cast_1:y:0 ^stateful_uniform/RngReadAndSkip*
_output_shapes
:2#
!stateful_uniform_1/RngReadAndSkipџ
&stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&stateful_uniform_1/strided_slice/stackъ
(stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_1ъ
(stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_2┌
 stateful_uniform_1/strided_sliceStridedSlice)stateful_uniform_1/RngReadAndSkip:value:0/stateful_uniform_1/strided_slice/stack:output:01stateful_uniform_1/strided_slice/stack_1:output:01stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2"
 stateful_uniform_1/strided_sliceЪ
stateful_uniform_1/BitcastBitcast)stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcastъ
(stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice_1/stackб
*stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_1б
*stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_2м
"stateful_uniform_1/strided_slice_1StridedSlice)stateful_uniform_1/RngReadAndSkip:value:01stateful_uniform_1/strided_slice_1/stack:output:03stateful_uniform_1/strided_slice_1/stack_1:output:03stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2$
"stateful_uniform_1/strided_slice_1Ц
stateful_uniform_1/Bitcast_1Bitcast+stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcast_1ц
/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :21
/stateful_uniform_1/StatelessRandomUniformV2/alg╚
+stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2!stateful_uniform_1/shape:output:0%stateful_uniform_1/Bitcast_1:output:0#stateful_uniform_1/Bitcast:output:08stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2-
+stateful_uniform_1/StatelessRandomUniformV2џ
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub╗
stateful_uniform_1/mulMul4stateful_uniform_1/StatelessRandomUniformV2:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:         2
stateful_uniform_1/mulа
stateful_uniform_1AddV2stateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:         2
stateful_uniform_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЏ
concatConcatV2stateful_uniform_1:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shapeї
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stackљ
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1љ
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ф
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
 *  ђ?2
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
zoom_matrix/truediv/yІ
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truedivЏ
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stackЪ
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1Ъ
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ы
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
 *  ђ?2
zoom_matrix/sub_1/xБ
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_1І
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
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
zoom_matrix/truediv_1/yЊ
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1Џ
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stackЪ
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1Ъ
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ы
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
 *  ђ?2
zoom_matrix/sub_3/xБ
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_3Љ
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mul_1Џ
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stackЪ
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1Ъ
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ы
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
zoom_matrix/zeros/packed/1│
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
zoom_matrix/zeros/ConstЦ
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1╣
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
zoom_matrix/zeros_1/ConstГ
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_1Џ
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stackЪ
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1Ъ
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ы
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
zoom_matrix/zeros_2/packed/1╣
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
zoom_matrix/zeros_2/ConstГ
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisр
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
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
transform/fill_value─
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3Ю
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityћ
NoOpNoOp ^stateful_uniform/RngReadAndSkip"^stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip2F
!stateful_uniform_1/RngReadAndSkip!stateful_uniform_1/RngReadAndSkip:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
ёU
щ
#__inference__traced_restore_1118274
file_prefix2
 assignvariableop_dense_21_kernel:1@.
 assignvariableop_1_dense_21_bias:@=
/assignvariableop_2_layer_normalization_12_gamma:@<
.assignvariableop_3_layer_normalization_12_beta:@%
assignvariableop_4_mean:)
assignvariableop_5_variance:"
assignvariableop_6_count:	 L
9assignvariableop_7_patch_encoder_7_embedding_3_embeddings:	Ц@L
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
identity_20ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9├
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
value┼B┬B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesХ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЈ
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

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2┤
AssignVariableOp_2AssignVariableOp/assignvariableop_2_layer_normalization_12_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3│
AssignVariableOp_3AssignVariableOp.assignvariableop_3_layer_normalization_12_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ю
AssignVariableOp_4AssignVariableOpassignvariableop_4_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5а
AssignVariableOp_5AssignVariableOpassignvariableop_5_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ю
AssignVariableOp_6AssignVariableOpassignvariableop_6_countIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Й
AssignVariableOp_7AssignVariableOp9assignvariableop_7_patch_encoder_7_embedding_3_embeddingsIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╗
AssignVariableOp_8AssignVariableOp6assignvariableop_8_multi_head_attention_6_query_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╣
AssignVariableOp_9AssignVariableOp4assignvariableop_9_multi_head_attention_6_query_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10й
AssignVariableOp_10AssignVariableOp5assignvariableop_10_multi_head_attention_6_key_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╗
AssignVariableOp_11AssignVariableOp3assignvariableop_11_multi_head_attention_6_key_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12┐
AssignVariableOp_12AssignVariableOp7assignvariableop_12_multi_head_attention_6_value_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13й
AssignVariableOp_13AssignVariableOp5assignvariableop_13_multi_head_attention_6_value_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╩
AssignVariableOp_14AssignVariableOpBassignvariableop_14_multi_head_attention_6_attention_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╚
AssignVariableOp_15AssignVariableOp@assignvariableop_15_multi_head_attention_6_attention_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16ц
AssignVariableOp_16AssignVariableOpassignvariableop_16_variableIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17д
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18д
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpђ
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19f
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_20У
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
к
§
*__inference_model_10_layer_call_fn_1116590

inputs
unknown
	unknown_0
	unknown_1:1@
	unknown_2:@
	unknown_3:	Ц@
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
identityѕбStatefulPartitionedCallГ
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
:         ЦЦ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11160752
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
─
╠
*__inference_model_10_layer_call_fn_1116426
input_4
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:1@
	unknown_5:@
	unknown_6:	Ц@
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
identityѕбStatefulPartitionedCallН
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
:         ЦЦ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11163462
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ўќ::: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ўќ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
ъ2
р
E__inference_model_10_layer_call_and_return_conditional_losses_1116518
input_4
data_augmentation_1116472
data_augmentation_1116474'
data_augmentation_1116476:	'
data_augmentation_1116478:	'
data_augmentation_1116480:	"
dense_21_1116484:1@
dense_21_1116486:@*
patch_encoder_7_1116491:	Ц@,
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
identityѕб)data_augmentation/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб.layer_normalization_12/StatefulPartitionedCallб.multi_head_attention_6/StatefulPartitionedCallб'patch_encoder_7/StatefulPartitionedCallА
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinput_4data_augmentation_1116472data_augmentation_1116474data_augmentation_1116476data_augmentation_1116478data_augmentation_1116480*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332+
)data_augmentation/StatefulPartitionedCallї
patches_7/PartitionedCallPartitionedCall2data_augmentation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall╗
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1116484dense_21_1116486*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
 dense_21/StatefulPartitionedCall 
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11162412
lambda_3/PartitionedCall│
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallК
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1116491*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCallЈ
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116494layer_normalization_12_1116496*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCall║
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116499multi_head_attention_6_1116501multi_head_attention_6_1116503multi_head_attention_6_1116505multi_head_attention_6_1116507multi_head_attention_6_1116509multi_head_attention_6_1116511multi_head_attention_6_1116513*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         Ц@:         ЦЦ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111617920
.multi_head_attention_6/StatefulPartitionedCallю
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:         ЦЦ2

IdentityЕ
NoOpNoOp*^data_augmentation/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ўќ::: : : : : : : : : : : : : : : : 2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Z V
1
_output_shapes
:         ўќ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ёа
Ў
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117480

inputs
normalization_3_sub_y
normalization_3_sqrt_xM
?random_flip_3_stateful_uniform_full_int_rngreadandskip_resource:	H
:random_rotation_3_stateful_uniform_rngreadandskip_resource:	D
6random_zoom_3_stateful_uniform_rngreadandskip_resource:	
identityѕб6random_flip_3/stateful_uniform_full_int/RngReadAndSkipб]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбdrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterб1random_rotation_3/stateful_uniform/RngReadAndSkipб-random_zoom_3/stateful_uniform/RngReadAndSkipб/random_zoom_3/stateful_uniform_1/RngReadAndSkipї
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2
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
 *Ћ┐о32
normalization_3/Maximum/yг
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum»
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2
normalization_3/truedivЂ
resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   2
resizing_3/resize/sizeж
 resizing_3/resize/ResizeBilinearResizeBilinearnormalization_3/truediv:z:0resizing_3/resize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(2"
 resizing_3/resize/ResizeBilinearе
-random_flip_3/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_flip_3/stateful_uniform_full_int/shapeе
-random_flip_3/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_flip_3/stateful_uniform_full_int/Constш
,random_flip_3/stateful_uniform_full_int/ProdProd6random_flip_3/stateful_uniform_full_int/shape:output:06random_flip_3/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2.
,random_flip_3/stateful_uniform_full_int/Prodб
.random_flip_3/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip_3/stateful_uniform_full_int/Cast/x¤
.random_flip_3/stateful_uniform_full_int/Cast_1Cast5random_flip_3/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.random_flip_3/stateful_uniform_full_int/Cast_1╠
6random_flip_3/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip?random_flip_3_stateful_uniform_full_int_rngreadandskip_resource7random_flip_3/stateful_uniform_full_int/Cast/x:output:02random_flip_3/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:28
6random_flip_3/stateful_uniform_full_int/RngReadAndSkip─
;random_flip_3/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;random_flip_3/stateful_uniform_full_int/strided_slice/stack╚
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_1╚
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateful_uniform_full_int/strided_slice/stack_2п
5random_flip_3/stateful_uniform_full_int/strided_sliceStridedSlice>random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip_3/stateful_uniform_full_int/strided_slice/stack:output:0Frandom_flip_3/stateful_uniform_full_int/strided_slice/stack_1:output:0Frandom_flip_3/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask27
5random_flip_3/stateful_uniform_full_int/strided_sliceя
/random_flip_3/stateful_uniform_full_int/BitcastBitcast>random_flip_3/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type021
/random_flip_3/stateful_uniform_full_int/Bitcast╚
=random_flip_3/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateful_uniform_full_int/strided_slice_1/stack╠
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1╠
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2л
7random_flip_3/stateful_uniform_full_int/strided_slice_1StridedSlice>random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_flip_3/stateful_uniform_full_int/strided_slice_1/stack:output:0Hrandom_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Hrandom_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:29
7random_flip_3/stateful_uniform_full_int/strided_slice_1С
1random_flip_3/stateful_uniform_full_int/Bitcast_1Bitcast@random_flip_3/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type023
1random_flip_3/stateful_uniform_full_int/Bitcast_1ю
+random_flip_3/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_flip_3/stateful_uniform_full_int/algѓ
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
random_flip_3/zeros_like╣
random_flip_3/stackPack0random_flip_3/stateful_uniform_full_int:output:0!random_flip_3/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip_3/stackЌ
!random_flip_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_flip_3/strided_slice/stackЏ
#random_flip_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_flip_3/strided_slice/stack_1Џ
#random_flip_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_flip_3/strided_slice/stack_2▄
random_flip_3/strided_sliceStridedSlicerandom_flip_3/stack:output:0*random_flip_3/strided_slice/stack:output:0,random_flip_3/strided_slice/stack_1:output:0,random_flip_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip_3/strided_sliceх
Arandom_flip_3/stateless_random_flip_left_right/control_dependencyIdentity1resizing_3/resize/ResizeBilinear:resized_images:0*
T0*3
_class)
'%loc:@resizing_3/resize/ResizeBilinear*0
_output_shapes
:         ─i2C
Arandom_flip_3/stateless_random_flip_left_right/control_dependencyТ
4random_flip_3/stateless_random_flip_left_right/ShapeShapeJrandom_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:26
4random_flip_3/stateless_random_flip_left_right/Shapeм
Brandom_flip_3/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Brandom_flip_3/stateless_random_flip_left_right/strided_slice/stackо
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_1о
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Drandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_2Ч
<random_flip_3/stateless_random_flip_left_right/strided_sliceStridedSlice=random_flip_3/stateless_random_flip_left_right/Shape:output:0Krandom_flip_3/stateless_random_flip_left_right/strided_slice/stack:output:0Mrandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_1:output:0Mrandom_flip_3/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<random_flip_3/stateless_random_flip_left_right/strided_sliceЏ
Mrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shapePackErandom_flip_3/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2O
Mrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shape▀
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min▀
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/max┤
drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip_3/strided_slice:output:0* 
_output_shapes
::2f
drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterо
]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlge^random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2_
]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgъ
`random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Vrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0jrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0crandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2b
`random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Ь
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subSubTrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Trandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subІ
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mulMulirandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Orandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2M
Krandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mul­
Grandom_flip_3/stateless_random_flip_left_right/stateless_random_uniformAddV2Orandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Trandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2I
Grandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform┬
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/1┬
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/2┬
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2@
>random_flip_3/stateless_random_flip_left_right/Reshape/shape/3н
<random_flip_3/stateless_random_flip_left_right/Reshape/shapePackErandom_flip_3/stateless_random_flip_left_right/strided_slice:output:0Grandom_flip_3/stateless_random_flip_left_right/Reshape/shape/1:output:0Grandom_flip_3/stateless_random_flip_left_right/Reshape/shape/2:output:0Grandom_flip_3/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2>
<random_flip_3/stateless_random_flip_left_right/Reshape/shape╔
6random_flip_3/stateless_random_flip_left_right/ReshapeReshapeKrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform:z:0Erandom_flip_3/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         28
6random_flip_3/stateless_random_flip_left_right/Reshape­
4random_flip_3/stateless_random_flip_left_right/RoundRound?random_flip_3/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         26
4random_flip_3/stateless_random_flip_left_right/Round╚
=random_flip_3/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip_3/stateless_random_flip_left_right/ReverseV2/axisл
8random_flip_3/stateless_random_flip_left_right/ReverseV2	ReverseV2Jrandom_flip_3/stateless_random_flip_left_right/control_dependency:output:0Frandom_flip_3/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:         ─i2:
8random_flip_3/stateless_random_flip_left_right/ReverseV2Д
2random_flip_3/stateless_random_flip_left_right/mulMul8random_flip_3/stateless_random_flip_left_right/Round:y:0Arandom_flip_3/stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:         ─i24
2random_flip_3/stateless_random_flip_left_right/mul▒
4random_flip_3/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?26
4random_flip_3/stateless_random_flip_left_right/sub/xб
2random_flip_3/stateless_random_flip_left_right/subSub=random_flip_3/stateless_random_flip_left_right/sub/x:output:08random_flip_3/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         24
2random_flip_3/stateless_random_flip_left_right/sub▓
4random_flip_3/stateless_random_flip_left_right/mul_1Mul6random_flip_3/stateless_random_flip_left_right/sub:z:0Jrandom_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:         ─i26
4random_flip_3/stateless_random_flip_left_right/mul_1ъ
2random_flip_3/stateless_random_flip_left_right/addAddV26random_flip_3/stateless_random_flip_left_right/mul:z:08random_flip_3/stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:         ─i24
2random_flip_3/stateless_random_flip_left_right/addў
random_rotation_3/ShapeShape6random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation_3/Shapeў
%random_rotation_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_3/strided_slice/stackю
'random_rotation_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_3/strided_slice/stack_1ю
'random_rotation_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_3/strided_slice/stack_2╬
random_rotation_3/strided_sliceStridedSlice random_rotation_3/Shape:output:0.random_rotation_3/strided_slice/stack:output:00random_rotation_3/strided_slice/stack_1:output:00random_rotation_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_3/strided_sliceЦ
'random_rotation_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2)
'random_rotation_3/strided_slice_1/stackЕ
)random_rotation_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2+
)random_rotation_3/strided_slice_1/stack_1а
)random_rotation_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_3/strided_slice_1/stack_2п
!random_rotation_3/strided_slice_1StridedSlice random_rotation_3/Shape:output:00random_rotation_3/strided_slice_1/stack:output:02random_rotation_3/strided_slice_1/stack_1:output:02random_rotation_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_3/strided_slice_1ћ
random_rotation_3/CastCast*random_rotation_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_3/CastЦ
'random_rotation_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2)
'random_rotation_3/strided_slice_2/stackЕ
)random_rotation_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2+
)random_rotation_3/strided_slice_2/stack_1а
)random_rotation_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_3/strided_slice_2/stack_2п
!random_rotation_3/strided_slice_2StridedSlice random_rotation_3/Shape:output:00random_rotation_3/strided_slice_2/stack:output:02random_rotation_3/strided_slice_2/stack_1:output:02random_rotation_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_3/strided_slice_2ў
random_rotation_3/Cast_1Cast*random_rotation_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_3/Cast_1┤
(random_rotation_3/stateful_uniform/shapePack(random_rotation_3/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_3/stateful_uniform/shapeЋ
&random_rotation_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *§Г Й2(
&random_rotation_3/stateful_uniform/minЋ
&random_rotation_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *§Г >2(
&random_rotation_3/stateful_uniform/maxъ
(random_rotation_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(random_rotation_3/stateful_uniform/Constр
'random_rotation_3/stateful_uniform/ProdProd1random_rotation_3/stateful_uniform/shape:output:01random_rotation_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/stateful_uniform/Prodў
)random_rotation_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_rotation_3/stateful_uniform/Cast/x└
)random_rotation_3/stateful_uniform/Cast_1Cast0random_rotation_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)random_rotation_3/stateful_uniform/Cast_1│
1random_rotation_3/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_3_stateful_uniform_rngreadandskip_resource2random_rotation_3/stateful_uniform/Cast/x:output:0-random_rotation_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:23
1random_rotation_3/stateful_uniform/RngReadAndSkip║
6random_rotation_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_3/stateful_uniform/strided_slice/stackЙ
8random_rotation_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_3/stateful_uniform/strided_slice/stack_1Й
8random_rotation_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_3/stateful_uniform/strided_slice/stack_2║
0random_rotation_3/stateful_uniform/strided_sliceStridedSlice9random_rotation_3/stateful_uniform/RngReadAndSkip:value:0?random_rotation_3/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_3/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask22
0random_rotation_3/stateful_uniform/strided_slice¤
*random_rotation_3/stateful_uniform/BitcastBitcast9random_rotation_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation_3/stateful_uniform/BitcastЙ
8random_rotation_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_3/stateful_uniform/strided_slice_1/stack┬
:random_rotation_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_3/stateful_uniform/strided_slice_1/stack_1┬
:random_rotation_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_3/stateful_uniform/strided_slice_1/stack_2▓
2random_rotation_3/stateful_uniform/strided_slice_1StridedSlice9random_rotation_3/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_3/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_3/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:24
2random_rotation_3/stateful_uniform/strided_slice_1Н
,random_rotation_3/stateful_uniform/Bitcast_1Bitcast;random_rotation_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02.
,random_rotation_3/stateful_uniform/Bitcast_1─
?random_rotation_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2A
?random_rotation_3/stateful_uniform/StatelessRandomUniformV2/algц
;random_rotation_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_3/stateful_uniform/shape:output:05random_rotation_3/stateful_uniform/Bitcast_1:output:03random_rotation_3/stateful_uniform/Bitcast:output:0Hrandom_rotation_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2=
;random_rotation_3/stateful_uniform/StatelessRandomUniformV2┌
&random_rotation_3/stateful_uniform/subSub/random_rotation_3/stateful_uniform/max:output:0/random_rotation_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_3/stateful_uniform/subэ
&random_rotation_3/stateful_uniform/mulMulDrandom_rotation_3/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_3/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2(
&random_rotation_3/stateful_uniform/mul▄
"random_rotation_3/stateful_uniformAddV2*random_rotation_3/stateful_uniform/mul:z:0/random_rotation_3/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         2$
"random_rotation_3/stateful_uniformЌ
'random_rotation_3/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2)
'random_rotation_3/rotation_matrix/sub/yк
%random_rotation_3/rotation_matrix/subSubrandom_rotation_3/Cast_1:y:00random_rotation_3/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_3/rotation_matrix/subФ
%random_rotation_3/rotation_matrix/CosCos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2'
%random_rotation_3/rotation_matrix/CosЏ
)random_rotation_3/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2+
)random_rotation_3/rotation_matrix/sub_1/y╠
'random_rotation_3/rotation_matrix/sub_1Subrandom_rotation_3/Cast_1:y:02random_rotation_3/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_1█
%random_rotation_3/rotation_matrix/mulMul)random_rotation_3/rotation_matrix/Cos:y:0+random_rotation_3/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         2'
%random_rotation_3/rotation_matrix/mulФ
%random_rotation_3/rotation_matrix/SinSin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2'
%random_rotation_3/rotation_matrix/SinЏ
)random_rotation_3/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2+
)random_rotation_3/rotation_matrix/sub_2/y╩
'random_rotation_3/rotation_matrix/sub_2Subrandom_rotation_3/Cast:y:02random_rotation_3/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_2▀
'random_rotation_3/rotation_matrix/mul_1Mul)random_rotation_3/rotation_matrix/Sin:y:0+random_rotation_3/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/mul_1▀
'random_rotation_3/rotation_matrix/sub_3Sub)random_rotation_3/rotation_matrix/mul:z:0+random_rotation_3/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/sub_3▀
'random_rotation_3/rotation_matrix/sub_4Sub)random_rotation_3/rotation_matrix/sub:z:0+random_rotation_3/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/sub_4Ъ
+random_rotation_3/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_3/rotation_matrix/truediv/yЫ
)random_rotation_3/rotation_matrix/truedivRealDiv+random_rotation_3/rotation_matrix/sub_4:z:04random_rotation_3/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2+
)random_rotation_3/rotation_matrix/truedivЏ
)random_rotation_3/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2+
)random_rotation_3/rotation_matrix/sub_5/y╩
'random_rotation_3/rotation_matrix/sub_5Subrandom_rotation_3/Cast:y:02random_rotation_3/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_5»
'random_rotation_3/rotation_matrix/Sin_1Sin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/Sin_1Џ
)random_rotation_3/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2+
)random_rotation_3/rotation_matrix/sub_6/y╠
'random_rotation_3/rotation_matrix/sub_6Subrandom_rotation_3/Cast_1:y:02random_rotation_3/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_6р
'random_rotation_3/rotation_matrix/mul_2Mul+random_rotation_3/rotation_matrix/Sin_1:y:0+random_rotation_3/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/mul_2»
'random_rotation_3/rotation_matrix/Cos_1Cos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/Cos_1Џ
)random_rotation_3/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2+
)random_rotation_3/rotation_matrix/sub_7/y╩
'random_rotation_3/rotation_matrix/sub_7Subrandom_rotation_3/Cast:y:02random_rotation_3/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_3/rotation_matrix/sub_7р
'random_rotation_3/rotation_matrix/mul_3Mul+random_rotation_3/rotation_matrix/Cos_1:y:0+random_rotation_3/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/mul_3▀
%random_rotation_3/rotation_matrix/addAddV2+random_rotation_3/rotation_matrix/mul_2:z:0+random_rotation_3/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         2'
%random_rotation_3/rotation_matrix/add▀
'random_rotation_3/rotation_matrix/sub_8Sub+random_rotation_3/rotation_matrix/sub_5:z:0)random_rotation_3/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/sub_8Б
-random_rotation_3/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_3/rotation_matrix/truediv_1/yЭ
+random_rotation_3/rotation_matrix/truediv_1RealDiv+random_rotation_3/rotation_matrix/sub_8:z:06random_rotation_3/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2-
+random_rotation_3/rotation_matrix/truediv_1е
'random_rotation_3/rotation_matrix/ShapeShape&random_rotation_3/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_3/rotation_matrix/ShapeИ
5random_rotation_3/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_3/rotation_matrix/strided_slice/stack╝
7random_rotation_3/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_3/rotation_matrix/strided_slice/stack_1╝
7random_rotation_3/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_3/rotation_matrix/strided_slice/stack_2«
/random_rotation_3/rotation_matrix/strided_sliceStridedSlice0random_rotation_3/rotation_matrix/Shape:output:0>random_rotation_3/rotation_matrix/strided_slice/stack:output:0@random_rotation_3/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_3/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_3/rotation_matrix/strided_slice»
'random_rotation_3/rotation_matrix/Cos_2Cos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/Cos_2├
7random_rotation_3/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_1/stackК
9random_rotation_3/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_1/stack_1К
9random_rotation_3/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_1/stack_2с
1random_rotation_3/rotation_matrix/strided_slice_1StridedSlice+random_rotation_3/rotation_matrix/Cos_2:y:0@random_rotation_3/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_1»
'random_rotation_3/rotation_matrix/Sin_2Sin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/Sin_2├
7random_rotation_3/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_2/stackК
9random_rotation_3/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_2/stack_1К
9random_rotation_3/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_2/stack_2с
1random_rotation_3/rotation_matrix/strided_slice_2StridedSlice+random_rotation_3/rotation_matrix/Sin_2:y:0@random_rotation_3/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_2├
%random_rotation_3/rotation_matrix/NegNeg:random_rotation_3/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2'
%random_rotation_3/rotation_matrix/Neg├
7random_rotation_3/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_3/stackК
9random_rotation_3/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_3/stack_1К
9random_rotation_3/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_3/stack_2т
1random_rotation_3/rotation_matrix/strided_slice_3StridedSlice-random_rotation_3/rotation_matrix/truediv:z:0@random_rotation_3/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_3»
'random_rotation_3/rotation_matrix/Sin_3Sin&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/Sin_3├
7random_rotation_3/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_4/stackК
9random_rotation_3/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_4/stack_1К
9random_rotation_3/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_4/stack_2с
1random_rotation_3/rotation_matrix/strided_slice_4StridedSlice+random_rotation_3/rotation_matrix/Sin_3:y:0@random_rotation_3/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_4»
'random_rotation_3/rotation_matrix/Cos_3Cos&random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/Cos_3├
7random_rotation_3/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_5/stackК
9random_rotation_3/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_5/stack_1К
9random_rotation_3/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_5/stack_2с
1random_rotation_3/rotation_matrix/strided_slice_5StridedSlice+random_rotation_3/rotation_matrix/Cos_3:y:0@random_rotation_3/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_5├
7random_rotation_3/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_3/rotation_matrix/strided_slice_6/stackК
9random_rotation_3/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_3/rotation_matrix/strided_slice_6/stack_1К
9random_rotation_3/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_3/rotation_matrix/strided_slice_6/stack_2у
1random_rotation_3/rotation_matrix/strided_slice_6StridedSlice/random_rotation_3/rotation_matrix/truediv_1:z:0@random_rotation_3/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_3/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_3/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_3/rotation_matrix/strided_slice_6д
0random_rotation_3/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_3/rotation_matrix/zeros/packed/1І
.random_rotation_3/rotation_matrix/zeros/packedPack8random_rotation_3/rotation_matrix/strided_slice:output:09random_rotation_3/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_3/rotation_matrix/zeros/packedБ
-random_rotation_3/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_3/rotation_matrix/zeros/Const§
'random_rotation_3/rotation_matrix/zerosFill7random_rotation_3/rotation_matrix/zeros/packed:output:06random_rotation_3/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2)
'random_rotation_3/rotation_matrix/zerosа
-random_rotation_3/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_3/rotation_matrix/concat/axis▄
(random_rotation_3/rotation_matrix/concatConcatV2:random_rotation_3/rotation_matrix/strided_slice_1:output:0)random_rotation_3/rotation_matrix/Neg:y:0:random_rotation_3/rotation_matrix/strided_slice_3:output:0:random_rotation_3/rotation_matrix/strided_slice_4:output:0:random_rotation_3/rotation_matrix/strided_slice_5:output:0:random_rotation_3/rotation_matrix/strided_slice_6:output:00random_rotation_3/rotation_matrix/zeros:output:06random_rotation_3/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2*
(random_rotation_3/rotation_matrix/concatг
!random_rotation_3/transform/ShapeShape6random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_3/transform/Shapeг
/random_rotation_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_3/transform/strided_slice/stack░
1random_rotation_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_3/transform/strided_slice/stack_1░
1random_rotation_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_3/transform/strided_slice/stack_2Ш
)random_rotation_3/transform/strided_sliceStridedSlice*random_rotation_3/transform/Shape:output:08random_rotation_3/transform/strided_slice/stack:output:0:random_rotation_3/transform/strided_slice/stack_1:output:0:random_rotation_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_3/transform/strided_sliceЋ
&random_rotation_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_3/transform/fill_valueм
6random_rotation_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV36random_flip_3/stateless_random_flip_left_right/add:z:01random_rotation_3/rotation_matrix/concat:output:02random_rotation_3/transform/strided_slice:output:0/random_rotation_3/transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_3/transform/ImageProjectiveTransformV3Ц
random_zoom_3/ShapeShapeKrandom_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_3/Shapeљ
!random_zoom_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!random_zoom_3/strided_slice/stackћ
#random_zoom_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_3/strided_slice/stack_1ћ
#random_zoom_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#random_zoom_3/strided_slice/stack_2Х
random_zoom_3/strided_sliceStridedSlicerandom_zoom_3/Shape:output:0*random_zoom_3/strided_slice/stack:output:0,random_zoom_3/strided_slice/stack_1:output:0,random_zoom_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_3/strided_sliceЮ
#random_zoom_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2%
#random_zoom_3/strided_slice_1/stackА
%random_zoom_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2'
%random_zoom_3/strided_slice_1/stack_1ў
%random_zoom_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_3/strided_slice_1/stack_2└
random_zoom_3/strided_slice_1StridedSlicerandom_zoom_3/Shape:output:0,random_zoom_3/strided_slice_1/stack:output:0.random_zoom_3/strided_slice_1/stack_1:output:0.random_zoom_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_3/strided_slice_1ѕ
random_zoom_3/CastCast&random_zoom_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_3/CastЮ
#random_zoom_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2%
#random_zoom_3/strided_slice_2/stackА
%random_zoom_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2'
%random_zoom_3/strided_slice_2/stack_1ў
%random_zoom_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_zoom_3/strided_slice_2/stack_2└
random_zoom_3/strided_slice_2StridedSlicerandom_zoom_3/Shape:output:0,random_zoom_3/strided_slice_2/stack:output:0.random_zoom_3/strided_slice_2/stack_1:output:0.random_zoom_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_3/strided_slice_2ї
random_zoom_3/Cast_1Cast&random_zoom_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_3/Cast_1њ
&random_zoom_3/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_3/stateful_uniform/shape/1┘
$random_zoom_3/stateful_uniform/shapePack$random_zoom_3/strided_slice:output:0/random_zoom_3/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$random_zoom_3/stateful_uniform/shapeЇ
"random_zoom_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L?2$
"random_zoom_3/stateful_uniform/minЇ
"random_zoom_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?2$
"random_zoom_3/stateful_uniform/maxќ
$random_zoom_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$random_zoom_3/stateful_uniform/ConstЛ
#random_zoom_3/stateful_uniform/ProdProd-random_zoom_3/stateful_uniform/shape:output:0-random_zoom_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2%
#random_zoom_3/stateful_uniform/Prodљ
%random_zoom_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_3/stateful_uniform/Cast/x┤
%random_zoom_3/stateful_uniform/Cast_1Cast,random_zoom_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%random_zoom_3/stateful_uniform/Cast_1Ъ
-random_zoom_3/stateful_uniform/RngReadAndSkipRngReadAndSkip6random_zoom_3_stateful_uniform_rngreadandskip_resource.random_zoom_3/stateful_uniform/Cast/x:output:0)random_zoom_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:2/
-random_zoom_3/stateful_uniform/RngReadAndSkip▓
2random_zoom_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2random_zoom_3/stateful_uniform/strided_slice/stackХ
4random_zoom_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4random_zoom_3/stateful_uniform/strided_slice/stack_1Х
4random_zoom_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4random_zoom_3/stateful_uniform/strided_slice/stack_2б
,random_zoom_3/stateful_uniform/strided_sliceStridedSlice5random_zoom_3/stateful_uniform/RngReadAndSkip:value:0;random_zoom_3/stateful_uniform/strided_slice/stack:output:0=random_zoom_3/stateful_uniform/strided_slice/stack_1:output:0=random_zoom_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2.
,random_zoom_3/stateful_uniform/strided_slice├
&random_zoom_3/stateful_uniform/BitcastBitcast5random_zoom_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02(
&random_zoom_3/stateful_uniform/BitcastХ
4random_zoom_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4random_zoom_3/stateful_uniform/strided_slice_1/stack║
6random_zoom_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform/strided_slice_1/stack_1║
6random_zoom_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform/strided_slice_1/stack_2џ
.random_zoom_3/stateful_uniform/strided_slice_1StridedSlice5random_zoom_3/stateful_uniform/RngReadAndSkip:value:0=random_zoom_3/stateful_uniform/strided_slice_1/stack:output:0?random_zoom_3/stateful_uniform/strided_slice_1/stack_1:output:0?random_zoom_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:20
.random_zoom_3/stateful_uniform/strided_slice_1╔
(random_zoom_3/stateful_uniform/Bitcast_1Bitcast7random_zoom_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02*
(random_zoom_3/stateful_uniform/Bitcast_1╝
;random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2=
;random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algљ
7random_zoom_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2-random_zoom_3/stateful_uniform/shape:output:01random_zoom_3/stateful_uniform/Bitcast_1:output:0/random_zoom_3/stateful_uniform/Bitcast:output:0Drandom_zoom_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         29
7random_zoom_3/stateful_uniform/StatelessRandomUniformV2╩
"random_zoom_3/stateful_uniform/subSub+random_zoom_3/stateful_uniform/max:output:0+random_zoom_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_zoom_3/stateful_uniform/subв
"random_zoom_3/stateful_uniform/mulMul@random_zoom_3/stateful_uniform/StatelessRandomUniformV2:output:0&random_zoom_3/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         2$
"random_zoom_3/stateful_uniform/mulл
random_zoom_3/stateful_uniformAddV2&random_zoom_3/stateful_uniform/mul:z:0+random_zoom_3/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2 
random_zoom_3/stateful_uniformќ
(random_zoom_3/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_3/stateful_uniform_1/shape/1▀
&random_zoom_3/stateful_uniform_1/shapePack$random_zoom_3/strided_slice:output:01random_zoom_3/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_3/stateful_uniform_1/shapeЉ
$random_zoom_3/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L?2&
$random_zoom_3/stateful_uniform_1/minЉ
$random_zoom_3/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?2&
$random_zoom_3/stateful_uniform_1/maxџ
&random_zoom_3/stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_zoom_3/stateful_uniform_1/Const┘
%random_zoom_3/stateful_uniform_1/ProdProd/random_zoom_3/stateful_uniform_1/shape:output:0/random_zoom_3/stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 2'
%random_zoom_3/stateful_uniform_1/Prodћ
'random_zoom_3/stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_3/stateful_uniform_1/Cast/x║
'random_zoom_3/stateful_uniform_1/Cast_1Cast.random_zoom_3/stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'random_zoom_3/stateful_uniform_1/Cast_1О
/random_zoom_3/stateful_uniform_1/RngReadAndSkipRngReadAndSkip6random_zoom_3_stateful_uniform_rngreadandskip_resource0random_zoom_3/stateful_uniform_1/Cast/x:output:0+random_zoom_3/stateful_uniform_1/Cast_1:y:0.^random_zoom_3/stateful_uniform/RngReadAndSkip*
_output_shapes
:21
/random_zoom_3/stateful_uniform_1/RngReadAndSkipХ
4random_zoom_3/stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4random_zoom_3/stateful_uniform_1/strided_slice/stack║
6random_zoom_3/stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform_1/strided_slice/stack_1║
6random_zoom_3/stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform_1/strided_slice/stack_2«
.random_zoom_3/stateful_uniform_1/strided_sliceStridedSlice7random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0=random_zoom_3/stateful_uniform_1/strided_slice/stack:output:0?random_zoom_3/stateful_uniform_1/strided_slice/stack_1:output:0?random_zoom_3/stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask20
.random_zoom_3/stateful_uniform_1/strided_slice╔
(random_zoom_3/stateful_uniform_1/BitcastBitcast7random_zoom_3/stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02*
(random_zoom_3/stateful_uniform_1/Bitcast║
6random_zoom_3/stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6random_zoom_3/stateful_uniform_1/strided_slice_1/stackЙ
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1Й
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2д
0random_zoom_3/stateful_uniform_1/strided_slice_1StridedSlice7random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0?random_zoom_3/stateful_uniform_1/strided_slice_1/stack:output:0Arandom_zoom_3/stateful_uniform_1/strided_slice_1/stack_1:output:0Arandom_zoom_3/stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:22
0random_zoom_3/stateful_uniform_1/strided_slice_1¤
*random_zoom_3/stateful_uniform_1/Bitcast_1Bitcast9random_zoom_3/stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02,
*random_zoom_3/stateful_uniform_1/Bitcast_1└
=random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/algю
9random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2/random_zoom_3/stateful_uniform_1/shape:output:03random_zoom_3/stateful_uniform_1/Bitcast_1:output:01random_zoom_3/stateful_uniform_1/Bitcast:output:0Frandom_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2;
9random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2м
$random_zoom_3/stateful_uniform_1/subSub-random_zoom_3/stateful_uniform_1/max:output:0-random_zoom_3/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2&
$random_zoom_3/stateful_uniform_1/subз
$random_zoom_3/stateful_uniform_1/mulMulBrandom_zoom_3/stateful_uniform_1/StatelessRandomUniformV2:output:0(random_zoom_3/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:         2&
$random_zoom_3/stateful_uniform_1/mulп
 random_zoom_3/stateful_uniform_1AddV2(random_zoom_3/stateful_uniform_1/mul:z:0-random_zoom_3/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:         2"
 random_zoom_3/stateful_uniform_1x
random_zoom_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_3/concat/axisр
random_zoom_3/concatConcatV2$random_zoom_3/stateful_uniform_1:z:0"random_zoom_3/stateful_uniform:z:0"random_zoom_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
random_zoom_3/concatЈ
random_zoom_3/zoom_matrix/ShapeShaperandom_zoom_3/concat:output:0*
T0*
_output_shapes
:2!
random_zoom_3/zoom_matrix/Shapeе
-random_zoom_3/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_zoom_3/zoom_matrix/strided_slice/stackг
/random_zoom_3/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_3/zoom_matrix/strided_slice/stack_1г
/random_zoom_3/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_zoom_3/zoom_matrix/strided_slice/stack_2■
'random_zoom_3/zoom_matrix/strided_sliceStridedSlice(random_zoom_3/zoom_matrix/Shape:output:06random_zoom_3/zoom_matrix/strided_slice/stack:output:08random_zoom_3/zoom_matrix/strided_slice/stack_1:output:08random_zoom_3/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'random_zoom_3/zoom_matrix/strided_sliceЄ
random_zoom_3/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2!
random_zoom_3/zoom_matrix/sub/yф
random_zoom_3/zoom_matrix/subSubrandom_zoom_3/Cast_1:y:0(random_zoom_3/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
random_zoom_3/zoom_matrix/subЈ
#random_zoom_3/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#random_zoom_3/zoom_matrix/truediv/y├
!random_zoom_3/zoom_matrix/truedivRealDiv!random_zoom_3/zoom_matrix/sub:z:0,random_zoom_3/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2#
!random_zoom_3/zoom_matrix/truedivи
/random_zoom_3/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_3/zoom_matrix/strided_slice_1/stack╗
1random_zoom_3/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_1/stack_1╗
1random_zoom_3/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_1/stack_2┼
)random_zoom_3/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_1/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_1/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_1І
!random_zoom_3/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2#
!random_zoom_3/zoom_matrix/sub_1/x█
random_zoom_3/zoom_matrix/sub_1Sub*random_zoom_3/zoom_matrix/sub_1/x:output:02random_zoom_3/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         2!
random_zoom_3/zoom_matrix/sub_1├
random_zoom_3/zoom_matrix/mulMul%random_zoom_3/zoom_matrix/truediv:z:0#random_zoom_3/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2
random_zoom_3/zoom_matrix/mulІ
!random_zoom_3/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2#
!random_zoom_3/zoom_matrix/sub_2/y«
random_zoom_3/zoom_matrix/sub_2Subrandom_zoom_3/Cast:y:0*random_zoom_3/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2!
random_zoom_3/zoom_matrix/sub_2Њ
%random_zoom_3/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%random_zoom_3/zoom_matrix/truediv_1/y╦
#random_zoom_3/zoom_matrix/truediv_1RealDiv#random_zoom_3/zoom_matrix/sub_2:z:0.random_zoom_3/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2%
#random_zoom_3/zoom_matrix/truediv_1и
/random_zoom_3/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_3/zoom_matrix/strided_slice_2/stack╗
1random_zoom_3/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_2/stack_1╗
1random_zoom_3/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_2/stack_2┼
)random_zoom_3/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_2/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_2/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_2І
!random_zoom_3/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2#
!random_zoom_3/zoom_matrix/sub_3/x█
random_zoom_3/zoom_matrix/sub_3Sub*random_zoom_3/zoom_matrix/sub_3/x:output:02random_zoom_3/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2!
random_zoom_3/zoom_matrix/sub_3╔
random_zoom_3/zoom_matrix/mul_1Mul'random_zoom_3/zoom_matrix/truediv_1:z:0#random_zoom_3/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         2!
random_zoom_3/zoom_matrix/mul_1и
/random_zoom_3/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            21
/random_zoom_3/zoom_matrix/strided_slice_3/stack╗
1random_zoom_3/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_3/stack_1╗
1random_zoom_3/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_3/stack_2┼
)random_zoom_3/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_3/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_3/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_3ќ
(random_zoom_3/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(random_zoom_3/zoom_matrix/zeros/packed/1в
&random_zoom_3/zoom_matrix/zeros/packedPack0random_zoom_3/zoom_matrix/strided_slice:output:01random_zoom_3/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&random_zoom_3/zoom_matrix/zeros/packedЊ
%random_zoom_3/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%random_zoom_3/zoom_matrix/zeros/ConstП
random_zoom_3/zoom_matrix/zerosFill/random_zoom_3/zoom_matrix/zeros/packed:output:0.random_zoom_3/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2!
random_zoom_3/zoom_matrix/zerosџ
*random_zoom_3/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_3/zoom_matrix/zeros_1/packed/1ы
(random_zoom_3/zoom_matrix/zeros_1/packedPack0random_zoom_3/zoom_matrix/strided_slice:output:03random_zoom_3/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_3/zoom_matrix/zeros_1/packedЌ
'random_zoom_3/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_3/zoom_matrix/zeros_1/Constт
!random_zoom_3/zoom_matrix/zeros_1Fill1random_zoom_3/zoom_matrix/zeros_1/packed:output:00random_zoom_3/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         2#
!random_zoom_3/zoom_matrix/zeros_1и
/random_zoom_3/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           21
/random_zoom_3/zoom_matrix/strided_slice_4/stack╗
1random_zoom_3/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           23
1random_zoom_3/zoom_matrix/strided_slice_4/stack_1╗
1random_zoom_3/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         23
1random_zoom_3/zoom_matrix/strided_slice_4/stack_2┼
)random_zoom_3/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_3/concat:output:08random_zoom_3/zoom_matrix/strided_slice_4/stack:output:0:random_zoom_3/zoom_matrix/strided_slice_4/stack_1:output:0:random_zoom_3/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2+
)random_zoom_3/zoom_matrix/strided_slice_4џ
*random_zoom_3/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2,
*random_zoom_3/zoom_matrix/zeros_2/packed/1ы
(random_zoom_3/zoom_matrix/zeros_2/packedPack0random_zoom_3/zoom_matrix/strided_slice:output:03random_zoom_3/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2*
(random_zoom_3/zoom_matrix/zeros_2/packedЌ
'random_zoom_3/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_zoom_3/zoom_matrix/zeros_2/Constт
!random_zoom_3/zoom_matrix/zeros_2Fill1random_zoom_3/zoom_matrix/zeros_2/packed:output:00random_zoom_3/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         2#
!random_zoom_3/zoom_matrix/zeros_2љ
%random_zoom_3/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%random_zoom_3/zoom_matrix/concat/axisь
 random_zoom_3/zoom_matrix/concatConcatV22random_zoom_3/zoom_matrix/strided_slice_3:output:0(random_zoom_3/zoom_matrix/zeros:output:0!random_zoom_3/zoom_matrix/mul:z:0*random_zoom_3/zoom_matrix/zeros_1:output:02random_zoom_3/zoom_matrix/strided_slice_4:output:0#random_zoom_3/zoom_matrix/mul_1:z:0*random_zoom_3/zoom_matrix/zeros_2:output:0.random_zoom_3/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2"
 random_zoom_3/zoom_matrix/concat╣
random_zoom_3/transform/ShapeShapeKrandom_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_3/transform/Shapeц
+random_zoom_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_zoom_3/transform/strided_slice/stackе
-random_zoom_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_3/transform/strided_slice/stack_1е
-random_zoom_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-random_zoom_3/transform/strided_slice/stack_2я
%random_zoom_3/transform/strided_sliceStridedSlice&random_zoom_3/transform/Shape:output:04random_zoom_3/transform/strided_slice/stack:output:06random_zoom_3/transform/strided_slice/stack_1:output:06random_zoom_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%random_zoom_3/transform/strided_sliceЇ
"random_zoom_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"random_zoom_3/transform/fill_value¤
2random_zoom_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0)random_zoom_3/zoom_matrix/concat:output:0.random_zoom_3/transform/strided_slice:output:0+random_zoom_3/transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR24
2random_zoom_3/transform/ImageProjectiveTransformV3Ф
IdentityIdentityGrandom_zoom_3/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:         ─i2

IdentityС
NoOpNoOp7^random_flip_3/stateful_uniform_full_int/RngReadAndSkip^^random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlge^random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2^random_rotation_3/stateful_uniform/RngReadAndSkip.^random_zoom_3/stateful_uniform/RngReadAndSkip0^random_zoom_3/stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         ўќ::: : : 2p
6random_flip_3/stateful_uniform_full_int/RngReadAndSkip6random_flip_3/stateful_uniform_full_int/RngReadAndSkip2Й
]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg]random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2╠
drandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterdrandom_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2f
1random_rotation_3/stateful_uniform/RngReadAndSkip1random_rotation_3/stateful_uniform/RngReadAndSkip2^
-random_zoom_3/stateful_uniform/RngReadAndSkip-random_zoom_3/stateful_uniform/RngReadAndSkip2b
/random_zoom_3/stateful_uniform_1/RngReadAndSkip/random_zoom_3/stateful_uniform_1/RngReadAndSkip:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
я
░
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115876
normalization_3_input
normalization_3_sub_y
normalization_3_sqrt_x
identityЏ
normalization_3/subSubnormalization_3_inputnormalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2
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
 *Ћ┐о32
normalization_3/Maximum/yг
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum»
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2
normalization_3/truedivЧ
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
resizing_3/PartitionedCallЇ
random_flip_3/PartitionedCallPartitionedCall#resizing_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11154322
random_flip_3/PartitionedCallю
!random_rotation_3/PartitionedCallPartitionedCall&random_flip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11154382#
!random_rotation_3/PartitionedCallћ
random_zoom_3/PartitionedCallPartitionedCall*random_rotation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11154442
random_zoom_3/PartitionedCallЃ
IdentityIdentity&random_zoom_3/PartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         ўќ:::h d
1
_output_shapes
:         ўќ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:
А
f
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117776

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
А
f
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1115444

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
▒
А
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115447

inputs
normalization_3_sub_y
normalization_3_sqrt_x
identityї
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2
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
 *Ћ┐о32
normalization_3/Maximum/yг
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum»
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2
normalization_3/truedivЧ
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
resizing_3/PartitionedCallЇ
random_flip_3/PartitionedCallPartitionedCall#resizing_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11154322
random_flip_3/PartitionedCallю
!random_rotation_3/PartitionedCallPartitionedCall&random_flip_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11154382#
!random_rotation_3/PartitionedCallћ
random_zoom_3/PartitionedCallPartitionedCall*random_rotation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11154442
random_zoom_3/PartitionedCallЃ
IdentityIdentity&random_zoom_3/PartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         ўќ:::Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
ь 
Ч
E__inference_dense_21_layer_call_and_return_conditional_losses_1115948

inputs3
!tensordot_readvariableop_resource:1@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpќ
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЉ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         ц12
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/Reshapeъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         ц@2
	Tensordotї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ц@2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ц@2

Identityѓ
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ц1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         ц1
 
_user_specified_nameinputs
▄f
ђ
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1115790

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityѕб(stateful_uniform_full_int/RngReadAndSkipбOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shapeї
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Constй
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prodє
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/xЦ
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1є
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkipе
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stackг
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1г
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2ё
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice┤
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcastг
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack░
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1░
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2Ч
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1║
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1ђ
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg«
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

zeros_likeЂ
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
strided_slice/stack_2ѕ
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceн
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*0
_output_shapes
:         ─i25
3stateless_random_flip_left_right/control_dependency╝
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/ShapeХ
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stack║
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1║
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2е
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceы
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shape├
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/min├
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxі
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterг
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg╩
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subМ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2?
=stateless_random_flip_left_right/stateless_random_uniform/mulИ
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2;
9stateless_random_flip_left_right/stateless_random_uniformд
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1д
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2д
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3ђ
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shapeЉ
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2*
(stateless_random_flip_left_right/Reshapeк
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2(
&stateless_random_flip_left_right/Roundг
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axisў
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:         ─i2,
*stateless_random_flip_left_right/ReverseV2№
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:         ─i2&
$stateless_random_flip_left_right/mulЋ
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2(
&stateless_random_flip_left_right/sub/xЖ
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2&
$stateless_random_flip_left_right/subЩ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:         ─i2(
&stateless_random_flip_left_right/mul_1Т
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:         ─i2&
$stateless_random_flip_left_right/addї
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityц
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2б
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
А
щ
%__inference_signature_wrapper_1116555
input_4
unknown
	unknown_0
	unknown_1:1@
	unknown_2:@
	unknown_3:	Ц@
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
identityѕбStatefulPartitionedCallІ
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
:         ЦЦ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__wrapped_model_11154062
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ўќ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
▄f
ђ
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117834

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityѕб(stateful_uniform_full_int/RngReadAndSkipбOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shapeї
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Constй
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prodє
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/xЦ
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1є
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkipе
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stackг
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1г
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2ё
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice┤
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcastг
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack░
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1░
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2Ч
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1║
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1ђ
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg«
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

zeros_likeЂ
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
strided_slice/stack_2ѕ
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceн
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*0
_output_shapes
:         ─i25
3stateless_random_flip_left_right/control_dependency╝
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/ShapeХ
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stack║
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1║
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2е
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceы
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shape├
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/min├
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxі
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterг
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg╩
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subМ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2?
=stateless_random_flip_left_right/stateless_random_uniform/mulИ
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2;
9stateless_random_flip_left_right/stateless_random_uniformд
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1д
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2д
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3ђ
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shapeЉ
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2*
(stateless_random_flip_left_right/Reshapeк
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2(
&stateless_random_flip_left_right/Roundг
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axisў
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:         ─i2,
*stateless_random_flip_left_right/ReverseV2№
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:         ─i2&
$stateless_random_flip_left_right/mulЋ
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2(
&stateless_random_flip_left_right/sub/xЖ
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2&
$stateless_random_flip_left_right/subЩ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:         ─i2(
&stateless_random_flip_left_right/mul_1Т
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:         ─i2&
$stateless_random_flip_left_right/addї
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityц
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2б
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
Ё
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
concat/axisє
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:         Ц@2
concath
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:         Ц@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         @:         ц@:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ц@
"
_user_specified_name
inputs/1
Ю0
ї
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

identity_1ѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp╚
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpџ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpњ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp╚
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpџ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
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
:         Ц@2
Mulб
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2
einsum/EinsumЂ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ2
softmax/SoftmaxЄ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/Identity╣
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpЭ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityѓ

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identity_1Я
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Ц@:         Ц@: : : : : : : : 2J
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
:         Ц@

_user_specified_namequery:SO
,
_output_shapes
:         Ц@

_user_specified_namevalue
і
њ
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1116013

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЮ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2
moments/meanі
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:         Ц2
moments/StopGradientЕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         Ц@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices└
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yЊ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:         Ц2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:         Ц2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЌ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/mul_1і
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpЊ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityѕ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ц@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ц@
 
_user_specified_nameinputs
С
c
G__inference_resizing_3_layer_call_and_return_conditional_losses_1117760

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   2
resize/size│
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(2
resize/ResizeBilinearЃ
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ўќ:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs
├9
ї
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

identity_1ѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp╚
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpџ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpњ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp╚
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpџ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
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
:         Ц@2
Mulб
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2
einsum/EinsumЂ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/dropout/Constе
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeо
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:         ЦЦ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2 
dropout/dropout/GreaterEqual/yУ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/dropout/GreaterEqualА
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ЦЦ2
dropout/dropout/Castц
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/dropout/Mul_1╣
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpЭ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityѓ

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identity_1Я
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Ц@:         Ц@: : : : : : : : 2J
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
:         Ц@

_user_specified_namequery:SO
,
_output_shapes
:         Ц@

_user_specified_namevalue
э
O
3__inference_random_rotation_3_layer_call_fn_1117839

inputs
identityп
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11154382
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
Ц
j
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117850

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
╠
К
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1115987

projection7
$embedding_3_embedding_lookup_1115980:	Ц@
identityѕбembedding_3/embedding_lookup\
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
B :Ц2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:Ц2
rangeЕ
embedding_3/embedding_lookupResourceGather$embedding_3_embedding_lookup_1115980range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*7
_class-
+)loc:@embedding_3/embedding_lookup/1115980*
_output_shapes
:	Ц@*
dtype02
embedding_3/embedding_lookupњ
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*7
_class-
+)loc:@embedding_3/embedding_lookup/1115980*
_output_shapes
:	Ц@2'
%embedding_3/embedding_lookup/IdentityИ
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Ц@2)
'embedding_3/embedding_lookup/Identity_1ѕ
addAddV2
projection0embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         Ц@2
addg
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identitym
NoOpNoOp^embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:         Ц@: 2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:X T
,
_output_shapes
:         Ц@
$
_user_specified_name
projection
■
Ѓ
3__inference_random_rotation_3_layer_call_fn_1117846

inputs
unknown:	
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11157192
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
ь 
Ч
E__inference_dense_21_layer_call_and_return_conditional_losses_1117531

inputs3
!tensordot_readvariableop_resource:1@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpќ
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЉ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         ц12
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/Reshapeъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         ц@2
	Tensordotї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ц@2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ц@2

Identityѓ
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ц1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         ц1
 
_user_specified_nameinputs
М
F
*__inference_lambda_3_layer_call_fn_1117541

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11162412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ц@:T P
,
_output_shapes
:         ц@
 
_user_specified_nameinputs
Ш.
╣
E__inference_model_10_layer_call_and_return_conditional_losses_1116075

inputs
data_augmentation_1115904
data_augmentation_1115906"
dense_21_1115949:1@
dense_21_1115951:@*
patch_encoder_7_1115988:	Ц@,
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
identityѕб dense_21/StatefulPartitionedCallб.layer_normalization_12/StatefulPartitionedCallб.multi_head_attention_6/StatefulPartitionedCallб'patch_encoder_7/StatefulPartitionedCall┤
!data_augmentation/PartitionedCallPartitionedCallinputsdata_augmentation_1115904data_augmentation_1115906*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472#
!data_augmentation/PartitionedCallё
patches_7/PartitionedCallPartitionedCall*data_augmentation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall╗
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1115949dense_21_1115951*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
 dense_21/StatefulPartitionedCall 
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11159622
lambda_3/PartitionedCall│
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallК
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1115988*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCallЈ
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116014layer_normalization_12_1116016*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCall║
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116056multi_head_attention_6_1116058multi_head_attention_6_1116060multi_head_attention_6_1116062multi_head_attention_6_1116064multi_head_attention_6_1116066multi_head_attention_6_1116068multi_head_attention_6_1116070*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         Ц@:         ЦЦ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111605520
.multi_head_attention_6/StatefulPartitionedCallю
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identity§
NoOpNoOp!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
М
F
*__inference_lambda_3_layer_call_fn_1117536

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11159622
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ц@:T P
,
_output_shapes
:         ц@
 
_user_specified_nameinputs
┴
╦
*__inference_model_10_layer_call_fn_1116631

inputs
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:1@
	unknown_5:@
	unknown_6:	Ц@
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
identityѕбStatefulPartitionedCallн
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
:         ЦЦ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_11163462
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ўќ::: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
ѕ╔
╣
"__inference__wrapped_model_1115406
input_44
0model_10_data_augmentation_normalization_3_sub_y5
1model_10_data_augmentation_normalization_3_sqrt_xE
3model_10_dense_21_tensordot_readvariableop_resource:1@?
1model_10_dense_21_biasadd_readvariableop_resource:@P
=model_10_patch_encoder_7_embedding_3_embedding_lookup_1115351:	Ц@S
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
identityѕб(model_10/dense_21/BiasAdd/ReadVariableOpб*model_10/dense_21/Tensordot/ReadVariableOpб8model_10/layer_normalization_12/batchnorm/ReadVariableOpб<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOpбCmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOpбMmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpб6model_10/multi_head_attention_6/key/add/ReadVariableOpб@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpб8model_10/multi_head_attention_6/query/add/ReadVariableOpбBmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpб8model_10/multi_head_attention_6/value/add/ReadVariableOpбBmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpб5model_10/patch_encoder_7/embedding_3/embedding_lookupя
.model_10/data_augmentation/normalization_3/subSubinput_40model_10_data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:         ўќ20
.model_10/data_augmentation/normalization_3/sub╬
/model_10/data_augmentation/normalization_3/SqrtSqrt1model_10_data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:21
/model_10/data_augmentation/normalization_3/Sqrt▒
4model_10/data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ћ┐о326
4model_10/data_augmentation/normalization_3/Maximum/yў
2model_10/data_augmentation/normalization_3/MaximumMaximum3model_10/data_augmentation/normalization_3/Sqrt:y:0=model_10/data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:24
2model_10/data_augmentation/normalization_3/MaximumЏ
2model_10/data_augmentation/normalization_3/truedivRealDiv2model_10/data_augmentation/normalization_3/sub:z:06model_10/data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ24
2model_10/data_augmentation/normalization_3/truedivи
1model_10/data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   23
1model_10/data_augmentation/resizing_3/resize/sizeН
;model_10/data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear6model_10/data_augmentation/normalization_3/truediv:z:0:model_10/data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(2=
;model_10/data_augmentation/resizing_3/resize/ResizeBilinear┐
&model_10/patches_7/ExtractImagePatchesExtractImagePatchesLmodel_10/data_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:         1*
ksizes
*
paddingVALID*
rates
*
strides
2(
&model_10/patches_7/ExtractImagePatchesЎ
 model_10/patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    ц  1   2"
 model_10/patches_7/Reshape/shapeО
model_10/patches_7/ReshapeReshape0model_10/patches_7/ExtractImagePatches:patches:0)model_10/patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:         ц12
model_10/patches_7/Reshape╠
*model_10/dense_21/Tensordot/ReadVariableOpReadVariableOp3model_10_dense_21_tensordot_readvariableop_resource*
_output_shapes

:1@*
dtype02,
*model_10/dense_21/Tensordot/ReadVariableOpј
 model_10/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 model_10/dense_21/Tensordot/axesЋ
 model_10/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model_10/dense_21/Tensordot/freeЎ
!model_10/dense_21/Tensordot/ShapeShape#model_10/patches_7/Reshape:output:0*
T0*
_output_shapes
:2#
!model_10/dense_21/Tensordot/Shapeў
)model_10/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_10/dense_21/Tensordot/GatherV2/axisФ
$model_10/dense_21/Tensordot/GatherV2GatherV2*model_10/dense_21/Tensordot/Shape:output:0)model_10/dense_21/Tensordot/free:output:02model_10/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_10/dense_21/Tensordot/GatherV2ю
+model_10/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_10/dense_21/Tensordot/GatherV2_1/axis▒
&model_10/dense_21/Tensordot/GatherV2_1GatherV2*model_10/dense_21/Tensordot/Shape:output:0)model_10/dense_21/Tensordot/axes:output:04model_10/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_10/dense_21/Tensordot/GatherV2_1љ
!model_10/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_10/dense_21/Tensordot/Const╚
 model_10/dense_21/Tensordot/ProdProd-model_10/dense_21/Tensordot/GatherV2:output:0*model_10/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 model_10/dense_21/Tensordot/Prodћ
#model_10/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#model_10/dense_21/Tensordot/Const_1л
"model_10/dense_21/Tensordot/Prod_1Prod/model_10/dense_21/Tensordot/GatherV2_1:output:0,model_10/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"model_10/dense_21/Tensordot/Prod_1ћ
'model_10/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_10/dense_21/Tensordot/concat/axisі
"model_10/dense_21/Tensordot/concatConcatV2)model_10/dense_21/Tensordot/free:output:0)model_10/dense_21/Tensordot/axes:output:00model_10/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_10/dense_21/Tensordot/concatн
!model_10/dense_21/Tensordot/stackPack)model_10/dense_21/Tensordot/Prod:output:0+model_10/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!model_10/dense_21/Tensordot/stackС
%model_10/dense_21/Tensordot/transpose	Transpose#model_10/patches_7/Reshape:output:0+model_10/dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:         ц12'
%model_10/dense_21/Tensordot/transposeу
#model_10/dense_21/Tensordot/ReshapeReshape)model_10/dense_21/Tensordot/transpose:y:0*model_10/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2%
#model_10/dense_21/Tensordot/ReshapeТ
"model_10/dense_21/Tensordot/MatMulMatMul,model_10/dense_21/Tensordot/Reshape:output:02model_10/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2$
"model_10/dense_21/Tensordot/MatMulћ
#model_10/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2%
#model_10/dense_21/Tensordot/Const_2ў
)model_10/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_10/dense_21/Tensordot/concat_1/axisЌ
$model_10/dense_21/Tensordot/concat_1ConcatV2-model_10/dense_21/Tensordot/GatherV2:output:0,model_10/dense_21/Tensordot/Const_2:output:02model_10/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_10/dense_21/Tensordot/concat_1┘
model_10/dense_21/TensordotReshape,model_10/dense_21/Tensordot/MatMul:product:0-model_10/dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         ц@2
model_10/dense_21/Tensordot┬
(model_10/dense_21/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_10/dense_21/BiasAdd/ReadVariableOpл
model_10/dense_21/BiasAddBiasAdd$model_10/dense_21/Tensordot:output:00model_10/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ц@2
model_10/dense_21/BiasAddќ
(model_10/lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_10/lambda_3/Mean/reduction_indices┴
model_10/lambda_3/MeanMean"model_10/dense_21/BiasAdd:output:01model_10/lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
model_10/lambda_3/MeanЌ
model_10/lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2!
model_10/lambda_3/Reshape/shape┬
model_10/lambda_3/ReshapeReshapemodel_10/lambda_3/Mean:output:0(model_10/lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:         @2
model_10/lambda_3/Reshapeі
"model_10/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_10/concatenate_3/concat/axis 
model_10/concatenate_3/concatConcatV2"model_10/lambda_3/Reshape:output:0"model_10/dense_21/BiasAdd:output:0+model_10/concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:         Ц@2
model_10/concatenate_3/concatј
$model_10/patch_encoder_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model_10/patch_encoder_7/range/startЈ
$model_10/patch_encoder_7/range/limitConst*
_output_shapes
: *
dtype0*
value
B :Ц2&
$model_10/patch_encoder_7/range/limitј
$model_10/patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2&
$model_10/patch_encoder_7/range/deltaз
model_10/patch_encoder_7/rangeRange-model_10/patch_encoder_7/range/start:output:0-model_10/patch_encoder_7/range/limit:output:0-model_10/patch_encoder_7/range/delta:output:0*
_output_shapes	
:Ц2 
model_10/patch_encoder_7/rangeд
5model_10/patch_encoder_7/embedding_3/embedding_lookupResourceGather=model_10_patch_encoder_7_embedding_3_embedding_lookup_1115351'model_10/patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*P
_classF
DBloc:@model_10/patch_encoder_7/embedding_3/embedding_lookup/1115351*
_output_shapes
:	Ц@*
dtype027
5model_10/patch_encoder_7/embedding_3/embedding_lookupШ
>model_10/patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity>model_10/patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*P
_classF
DBloc:@model_10/patch_encoder_7/embedding_3/embedding_lookup/1115351*
_output_shapes
:	Ц@2@
>model_10/patch_encoder_7/embedding_3/embedding_lookup/IdentityЃ
@model_10/patch_encoder_7/embedding_3/embedding_lookup/Identity_1IdentityGmodel_10/patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Ц@2B
@model_10/patch_encoder_7/embedding_3/embedding_lookup/Identity_1№
model_10/patch_encoder_7/addAddV2&model_10/concatenate_3/concat:output:0Imodel_10/patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         Ц@2
model_10/patch_encoder_7/add╩
>model_10/layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2@
>model_10/layer_normalization_12/moments/mean/reduction_indicesЌ
,model_10/layer_normalization_12/moments/meanMean model_10/patch_encoder_7/add:z:0Gmodel_10/layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2.
,model_10/layer_normalization_12/moments/meanЖ
4model_10/layer_normalization_12/moments/StopGradientStopGradient5model_10/layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:         Ц26
4model_10/layer_normalization_12/moments/StopGradientБ
9model_10/layer_normalization_12/moments/SquaredDifferenceSquaredDifference model_10/patch_encoder_7/add:z:0=model_10/layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:         Ц@2;
9model_10/layer_normalization_12/moments/SquaredDifferenceм
Bmodel_10/layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2D
Bmodel_10/layer_normalization_12/moments/variance/reduction_indices└
0model_10/layer_normalization_12/moments/varianceMean=model_10/layer_normalization_12/moments/SquaredDifference:z:0Kmodel_10/layer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(22
0model_10/layer_normalization_12/moments/varianceД
/model_10/layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є521
/model_10/layer_normalization_12/batchnorm/add/yЊ
-model_10/layer_normalization_12/batchnorm/addAddV29model_10/layer_normalization_12/moments/variance:output:08model_10/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:         Ц2/
-model_10/layer_normalization_12/batchnorm/addН
/model_10/layer_normalization_12/batchnorm/RsqrtRsqrt1model_10/layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:         Ц21
/model_10/layer_normalization_12/batchnorm/Rsqrt■
<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_10_layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOpЌ
-model_10/layer_normalization_12/batchnorm/mulMul3model_10/layer_normalization_12/batchnorm/Rsqrt:y:0Dmodel_10/layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2/
-model_10/layer_normalization_12/batchnorm/mulш
/model_10/layer_normalization_12/batchnorm/mul_1Mul model_10/patch_encoder_7/add:z:01model_10/layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@21
/model_10/layer_normalization_12/batchnorm/mul_1і
/model_10/layer_normalization_12/batchnorm/mul_2Mul5model_10/layer_normalization_12/moments/mean:output:01model_10/layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@21
/model_10/layer_normalization_12/batchnorm/mul_2Ы
8model_10/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOpAmodel_10_layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_10/layer_normalization_12/batchnorm/ReadVariableOpЊ
-model_10/layer_normalization_12/batchnorm/subSub@model_10/layer_normalization_12/batchnorm/ReadVariableOp:value:03model_10/layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:         Ц@2/
-model_10/layer_normalization_12/batchnorm/subі
/model_10/layer_normalization_12/batchnorm/add_1AddV23model_10/layer_normalization_12/batchnorm/mul_1:z:01model_10/layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ц@21
/model_10/layer_normalization_12/batchnorm/add_1ў
Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_10_multi_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpо
3model_10/multi_head_attention_6/query/einsum/EinsumEinsum3model_10/layer_normalization_12/batchnorm/add_1:z:0Jmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde25
3model_10/multi_head_attention_6/query/einsum/EinsumШ
8model_10/multi_head_attention_6/query/add/ReadVariableOpReadVariableOpAmodel_10_multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_10/multi_head_attention_6/query/add/ReadVariableOpџ
)model_10/multi_head_attention_6/query/addAddV2<model_10/multi_head_attention_6/query/einsum/Einsum:output:0@model_10/multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2+
)model_10/multi_head_attention_6/query/addњ
@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOpImodel_10_multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02B
@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOpл
1model_10/multi_head_attention_6/key/einsum/EinsumEinsum3model_10/layer_normalization_12/batchnorm/add_1:z:0Hmodel_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde23
1model_10/multi_head_attention_6/key/einsum/Einsum­
6model_10/multi_head_attention_6/key/add/ReadVariableOpReadVariableOp?model_10_multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype028
6model_10/multi_head_attention_6/key/add/ReadVariableOpњ
'model_10/multi_head_attention_6/key/addAddV2:model_10/multi_head_attention_6/key/einsum/Einsum:output:0>model_10/multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2)
'model_10/multi_head_attention_6/key/addў
Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpKmodel_10_multi_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpо
3model_10/multi_head_attention_6/value/einsum/EinsumEinsum3model_10/layer_normalization_12/batchnorm/add_1:z:0Jmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde25
3model_10/multi_head_attention_6/value/einsum/EinsumШ
8model_10/multi_head_attention_6/value/add/ReadVariableOpReadVariableOpAmodel_10_multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02:
8model_10/multi_head_attention_6/value/add/ReadVariableOpџ
)model_10/multi_head_attention_6/value/addAddV2<model_10/multi_head_attention_6/value/einsum/Einsum:output:0@model_10/multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2+
)model_10/multi_head_attention_6/value/addЊ
%model_10/multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2'
%model_10/multi_head_attention_6/Mul/yв
#model_10/multi_head_attention_6/MulMul-model_10/multi_head_attention_6/query/add:z:0.model_10/multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:         Ц@2%
#model_10/multi_head_attention_6/Mulб
-model_10/multi_head_attention_6/einsum/EinsumEinsum+model_10/multi_head_attention_6/key/add:z:0'model_10/multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2/
-model_10/multi_head_attention_6/einsum/Einsumр
/model_10/multi_head_attention_6/softmax/SoftmaxSoftmax6model_10/multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ21
/model_10/multi_head_attention_6/softmax/Softmaxу
0model_10/multi_head_attention_6/dropout/IdentityIdentity9model_10/multi_head_attention_6/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:         ЦЦ22
0model_10/multi_head_attention_6/dropout/Identity╣
/model_10/multi_head_attention_6/einsum_1/EinsumEinsum9model_10/multi_head_attention_6/dropout/Identity:output:0-model_10/multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd21
/model_10/multi_head_attention_6/einsum_1/Einsum╣
Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_10_multi_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpЭ
>model_10/multi_head_attention_6/attention_output/einsum/EinsumEinsum8model_10/multi_head_attention_6/einsum_1/Einsum:output:0Umodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe2@
>model_10/multi_head_attention_6/attention_output/einsum/EinsumЊ
Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpLmodel_10_multi_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02E
Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOp┬
4model_10/multi_head_attention_6/attention_output/addAddV2Gmodel_10/multi_head_attention_6/attention_output/einsum/Einsum:output:0Kmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@26
4model_10/multi_head_attention_6/attention_output/addъ
IdentityIdentity9model_10/multi_head_attention_6/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

IdentityЖ
NoOpNoOp)^model_10/dense_21/BiasAdd/ReadVariableOp+^model_10/dense_21/Tensordot/ReadVariableOp9^model_10/layer_normalization_12/batchnorm/ReadVariableOp=^model_10/layer_normalization_12/batchnorm/mul/ReadVariableOpD^model_10/multi_head_attention_6/attention_output/add/ReadVariableOpN^model_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp7^model_10/multi_head_attention_6/key/add/ReadVariableOpA^model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp9^model_10/multi_head_attention_6/query/add/ReadVariableOpC^model_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp9^model_10/multi_head_attention_6/value/add/ReadVariableOpC^model_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp6^model_10/patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 2T
(model_10/dense_21/BiasAdd/ReadVariableOp(model_10/dense_21/BiasAdd/ReadVariableOp2X
*model_10/dense_21/Tensordot/ReadVariableOp*model_10/dense_21/Tensordot/ReadVariableOp2t
8model_10/layer_normalization_12/batchnorm/ReadVariableOp8model_10/layer_normalization_12/batchnorm/ReadVariableOp2|
<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOp<model_10/layer_normalization_12/batchnorm/mul/ReadVariableOp2і
Cmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOpCmodel_10/multi_head_attention_6/attention_output/add/ReadVariableOp2ъ
Mmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpMmodel_10/multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp2p
6model_10/multi_head_attention_6/key/add/ReadVariableOp6model_10/multi_head_attention_6/key/add/ReadVariableOp2ё
@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp@model_10/multi_head_attention_6/key/einsum/Einsum/ReadVariableOp2t
8model_10/multi_head_attention_6/query/add/ReadVariableOp8model_10/multi_head_attention_6/query/add/ReadVariableOp2ѕ
Bmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOpBmodel_10/multi_head_attention_6/query/einsum/Einsum/ReadVariableOp2t
8model_10/multi_head_attention_6/value/add/ReadVariableOp8model_10/multi_head_attention_6/value/add/ReadVariableOp2ѕ
Bmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOpBmodel_10/multi_head_attention_6/value/einsum/Einsum/ReadVariableOp2n
5model_10/patch_encoder_7/embedding_3/embedding_lookup5model_10/patch_encoder_7/embedding_3/embedding_lookup:Z V
1
_output_shapes
:         ўќ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
С
c
G__inference_resizing_3_layer_call_and_return_conditional_losses_1115426

inputs
identityk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   2
resize/size│
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(2
resize/ResizeBilinearЃ
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ўќ:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs
▒

Р
3__inference_data_augmentation_layer_call_fn_1115861
normalization_3_input
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallnormalization_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         ўќ::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
1
_output_shapes
:         ўќ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:
пЏ
К
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1115719

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipD
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЂ
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2
strided_slice_1/stackЁ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
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
CastЂ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_2/stackЁ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
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
 *§Г Й2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *§Г >2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
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
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/algИ
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub»
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2
stateful_uniform/mulћ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
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
:         2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_1/yё
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1Њ
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_2/yѓ
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2Ќ
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_1Ќ
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_3Ќ
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yф
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_5/yѓ
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_6/yё
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6Ў
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_7/yѓ
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7Ў
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_3Ќ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/addЌ
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y░
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shapeћ
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stackў
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1ў
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2┬
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
:         2
rotation_matrix/Cos_2Ъ
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stackБ
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1Б
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2э
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_2Ъ
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stackБ
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1Б
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2э
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2Ї
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/NegЪ
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stackБ
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1Б
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2щ
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_3Ъ
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stackБ
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1Б
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2э
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_3Ъ
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stackБ
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1Б
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2э
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5Ъ
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stackБ
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1Б
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2ч
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6ѓ
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1├
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
rotation_matrix/zeros/Constх
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axisе
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
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
transform/fill_value╚
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3Ю
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
№
K
/__inference_random_zoom_3_layer_call_fn_1117973

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11154442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
щ.
║
E__inference_model_10_layer_call_and_return_conditional_losses_1116469
input_4
data_augmentation_1116429
data_augmentation_1116431"
dense_21_1116435:1@
dense_21_1116437:@*
patch_encoder_7_1116442:	Ц@,
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
identityѕб dense_21/StatefulPartitionedCallб.layer_normalization_12/StatefulPartitionedCallб.multi_head_attention_6/StatefulPartitionedCallб'patch_encoder_7/StatefulPartitionedCallх
!data_augmentation/PartitionedCallPartitionedCallinput_4data_augmentation_1116429data_augmentation_1116431*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472#
!data_augmentation/PartitionedCallё
patches_7/PartitionedCallPartitionedCall*data_augmentation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall╗
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1116435dense_21_1116437*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
 dense_21/StatefulPartitionedCall 
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11159622
lambda_3/PartitionedCall│
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallК
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1116442*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCallЈ
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116445layer_normalization_12_1116447*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCall║
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116450multi_head_attention_6_1116452multi_head_attention_6_1116454multi_head_attention_6_1116456multi_head_attention_6_1116458multi_head_attention_6_1116460multi_head_attention_6_1116462multi_head_attention_6_1116464*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         Ц@:         ЦЦ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111605520
.multi_head_attention_6/StatefulPartitionedCallю
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identity§
NoOpNoOp!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Z V
1
_output_shapes
:         ўќ
!
_user_specified_name	input_4:,(
&
_output_shapes
::,(
&
_output_shapes
:
№
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
:         @2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         @2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ц@:T P
,
_output_shapes
:         ц@
 
_user_specified_nameinputs
у
А
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117177

inputs
normalization_3_sub_y
normalization_3_sqrt_x
identityї
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2
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
 *Ћ┐о32
normalization_3/Maximum/yг
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum»
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2
normalization_3/truedivЂ
resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   2
resizing_3/resize/sizeж
 resizing_3/resize/ResizeBilinearResizeBilinearnormalization_3/truediv:z:0resizing_3/resize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(2"
 resizing_3/resize/ResizeBilinearј
IdentityIdentity1resizing_3/resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         ўќ:::Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ф
k
3__inference_data_augmentation_layer_call_fn_1117149

inputs
unknown
	unknown_0
identityЬ
PartitionedCallPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         ўќ:::Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Џ2
Я
E__inference_model_10_layer_call_and_return_conditional_losses_1116346

inputs
data_augmentation_1116300
data_augmentation_1116302'
data_augmentation_1116304:	'
data_augmentation_1116306:	'
data_augmentation_1116308:	"
dense_21_1116312:1@
dense_21_1116314:@*
patch_encoder_7_1116319:	Ц@,
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
identityѕб)data_augmentation/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб.layer_normalization_12/StatefulPartitionedCallб.multi_head_attention_6/StatefulPartitionedCallб'patch_encoder_7/StatefulPartitionedCallа
)data_augmentation/StatefulPartitionedCallStatefulPartitionedCallinputsdata_augmentation_1116300data_augmentation_1116302data_augmentation_1116304data_augmentation_1116306data_augmentation_1116308*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332+
)data_augmentation/StatefulPartitionedCallї
patches_7/PartitionedCallPartitionedCall2data_augmentation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
patches_7/PartitionedCall╗
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"patches_7/PartitionedCall:output:0dense_21_1116312dense_21_1116314*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482"
 dense_21/StatefulPartitionedCall 
lambda_3/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_lambda_3_layer_call_and_return_conditional_losses_11162412
lambda_3/PartitionedCall│
concatenate_3/PartitionedCallPartitionedCall!lambda_3/PartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
concatenate_3/PartitionedCallК
'patch_encoder_7/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0patch_encoder_7_1116319*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872)
'patch_encoder_7/StatefulPartitionedCallЈ
.layer_normalization_12/StatefulPartitionedCallStatefulPartitionedCall0patch_encoder_7/StatefulPartitionedCall:output:0layer_normalization_12_1116322layer_normalization_12_1116324*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_111601320
.layer_normalization_12/StatefulPartitionedCall║
.multi_head_attention_6/StatefulPartitionedCallStatefulPartitionedCall7layer_normalization_12/StatefulPartitionedCall:output:07layer_normalization_12/StatefulPartitionedCall:output:0multi_head_attention_6_1116327multi_head_attention_6_1116329multi_head_attention_6_1116331multi_head_attention_6_1116333multi_head_attention_6_1116335multi_head_attention_6_1116337multi_head_attention_6_1116339multi_head_attention_6_1116341*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         Ц@:         ЦЦ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_111617920
.multi_head_attention_6/StatefulPartitionedCallю
IdentityIdentity7multi_head_attention_6/StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:         ЦЦ2

IdentityЕ
NoOpNoOp*^data_augmentation/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall/^layer_normalization_12/StatefulPartitionedCall/^multi_head_attention_6/StatefulPartitionedCall(^patch_encoder_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ўќ::: : : : : : : : : : : : : : : : 2V
)data_augmentation/StatefulPartitionedCall)data_augmentation/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2`
.layer_normalization_12/StatefulPartitionedCall.layer_normalization_12/StatefulPartitionedCall2`
.multi_head_attention_6/StatefulPartitionedCall.multi_head_attention_6/StatefulPartitionedCall2R
'patch_encoder_7/StatefulPartitionedCall'patch_encoder_7/StatefulPartitionedCall:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
В
[
/__inference_concatenate_3_layer_call_fn_1117563
inputs_0
inputs_1
identityП
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_11159712
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ц@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         @:         ц@:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ц@
"
_user_specified_name
inputs/1
╩-
М
__inference_adapt_step_1113455
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бIteratorGetNextбReadVariableOpбReadVariableOp_1бReadVariableOp_2бadd/ReadVariableOpп
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*1
_output_shapes
:         ўќ*0
output_shapes
:         ўќ*
output_types
22
IteratorGetNext}
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*1
_output_shapes
:         ўќ2
CastЋ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indicesЎ
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(2
moments/meanё
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*1
_output_shapes
:         ўќ2
moments/SquaredDifferenceЮ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indices║
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(2
moments/varianceѓ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeezeі
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
GatherV2/axisФ
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
 *  ђ?2
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
add_4Б
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOpў
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1џ
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
№
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
:         @2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         @2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ц@:T P
,
_output_shapes
:         ц@
 
_user_specified_nameinputs
і
њ
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1117622

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЮ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2
moments/meanі
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:         Ц2
moments/StopGradientЕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         Ц@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices└
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yЊ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:         Ц2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:         Ц2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЌ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/mul_1і
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpЊ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/subі
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ц@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityѕ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ц@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ц@
 
_user_specified_nameinputs
ум
Ш
E__inference_model_10_layer_call_and_return_conditional_losses_1117140

inputs+
'data_augmentation_normalization_3_sub_y,
(data_augmentation_normalization_3_sqrt_x_
Qdata_augmentation_random_flip_3_stateful_uniform_full_int_rngreadandskip_resource:	Z
Ldata_augmentation_random_rotation_3_stateful_uniform_rngreadandskip_resource:	V
Hdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resource:	<
*dense_21_tensordot_readvariableop_resource:1@6
(dense_21_biasadd_readvariableop_resource:@G
4patch_encoder_7_embedding_3_embedding_lookup_1117078:	Ц@J
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
identityѕбHdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipбodata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбvdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterбCdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipб?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipбAdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipбdense_21/BiasAdd/ReadVariableOpб!dense_21/Tensordot/ReadVariableOpб/layer_normalization_12/batchnorm/ReadVariableOpб3layer_normalization_12/batchnorm/mul/ReadVariableOpб:multi_head_attention_6/attention_output/add/ReadVariableOpбDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpб-multi_head_attention_6/key/add/ReadVariableOpб7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpб/multi_head_attention_6/query/add/ReadVariableOpб9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpб/multi_head_attention_6/value/add/ReadVariableOpб9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpб,patch_encoder_7/embedding_3/embedding_lookup┬
%data_augmentation/normalization_3/subSubinputs'data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2'
%data_augmentation/normalization_3/sub│
&data_augmentation/normalization_3/SqrtSqrt(data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:2(
&data_augmentation/normalization_3/SqrtЪ
+data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ћ┐о32-
+data_augmentation/normalization_3/Maximum/yЗ
)data_augmentation/normalization_3/MaximumMaximum*data_augmentation/normalization_3/Sqrt:y:04data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2+
)data_augmentation/normalization_3/Maximumэ
)data_augmentation/normalization_3/truedivRealDiv)data_augmentation/normalization_3/sub:z:0-data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2+
)data_augmentation/normalization_3/truedivЦ
(data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   2*
(data_augmentation/resizing_3/resize/size▒
2data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear-data_augmentation/normalization_3/truediv:z:01data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(24
2data_augmentation/resizing_3/resize/ResizeBilinear╠
?data_augmentation/random_flip_3/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2A
?data_augmentation/random_flip_3/stateful_uniform_full_int/shape╠
?data_augmentation/random_flip_3/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?data_augmentation/random_flip_3/stateful_uniform_full_int/Constй
>data_augmentation/random_flip_3/stateful_uniform_full_int/ProdProdHdata_augmentation/random_flip_3/stateful_uniform_full_int/shape:output:0Hdata_augmentation/random_flip_3/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2@
>data_augmentation/random_flip_3/stateful_uniform_full_int/Prodк
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2B
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast/xЁ
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast_1CastGdata_augmentation/random_flip_3/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@data_augmentation/random_flip_3/stateful_uniform_full_int/Cast_1д
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipQdata_augmentation_random_flip_3_stateful_uniform_full_int_rngreadandskip_resourceIdata_augmentation/random_flip_3/stateful_uniform_full_int/Cast/x:output:0Ddata_augmentation/random_flip_3/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2J
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipУ
Mdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stackВ
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_1В
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_2─
Gdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_sliceStridedSlicePdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Vdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack:output:0Xdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_1:output:0Xdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2I
Gdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_sliceћ
Adata_augmentation/random_flip_3/stateful_uniform_full_int/BitcastBitcastPdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02C
Adata_augmentation/random_flip_3/stateful_uniform_full_int/BitcastВ
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack­
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1­
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2╝
Idata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1StridedSlicePdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip:value:0Xdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack:output:0Zdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Zdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2K
Idata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1џ
Cdata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast_1BitcastRdata_augmentation/random_flip_3/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02E
Cdata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast_1└
=data_augmentation/random_flip_3/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=data_augmentation/random_flip_3/stateful_uniform_full_int/algЬ
9data_augmentation/random_flip_3/stateful_uniform_full_intStatelessRandomUniformFullIntV2Hdata_augmentation/random_flip_3/stateful_uniform_full_int/shape:output:0Ldata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast_1:output:0Jdata_augmentation/random_flip_3/stateful_uniform_full_int/Bitcast:output:0Fdata_augmentation/random_flip_3/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2;
9data_augmentation/random_flip_3/stateful_uniform_full_intб
*data_augmentation/random_flip_3/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2,
*data_augmentation/random_flip_3/zeros_likeЂ
%data_augmentation/random_flip_3/stackPackBdata_augmentation/random_flip_3/stateful_uniform_full_int:output:03data_augmentation/random_flip_3/zeros_like:output:0*
N*
T0	*
_output_shapes

:2'
%data_augmentation/random_flip_3/stack╗
3data_augmentation/random_flip_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3data_augmentation/random_flip_3/strided_slice/stack┐
5data_augmentation/random_flip_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5data_augmentation/random_flip_3/strided_slice/stack_1┐
5data_augmentation/random_flip_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5data_augmentation/random_flip_3/strided_slice/stack_2╚
-data_augmentation/random_flip_3/strided_sliceStridedSlice.data_augmentation/random_flip_3/stack:output:0<data_augmentation/random_flip_3/strided_slice/stack:output:0>data_augmentation/random_flip_3/strided_slice/stack_1:output:0>data_augmentation/random_flip_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2/
-data_augmentation/random_flip_3/strided_slice§
Sdata_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependencyIdentityCdata_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
T0*E
_class;
97loc:@data_augmentation/resizing_3/resize/ResizeBilinear*0
_output_shapes
:         ─i2U
Sdata_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependencyю
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/ShapeShape\data_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/ShapeШ
Tdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stackЩ
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_1Щ
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
Vdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_2У
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_sliceStridedSliceOdata_augmentation/random_flip_3/stateless_random_flip_left_right/Shape:output:0]data_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack:output:0_data_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_1:output:0_data_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2P
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_sliceЛ
_data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shapePackWdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2a
_data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shapeЃ
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/minЃ
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/maxЖ
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter6data_augmentation/random_flip_3/strided_slice:output:0* 
_output_shapes
::2x
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgw^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2q
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgІ
rdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2hdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0|data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ђdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0udata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2t
rdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subSubfdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/max:output:0fdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/subМ
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mulMul{data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0adata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2_
]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mulИ
Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniformAddV2adata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0fdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2[
Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniformТ
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2R
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/1Т
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2R
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/2Т
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2R
Pdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/3└
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shapePackWdata_augmentation/random_flip_3/stateless_random_flip_left_right/strided_slice:output:0Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/1:output:0Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/2:output:0Ydata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2P
Ndata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shapeЉ
Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReshapeReshape]data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform:z:0Wdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2J
Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshapeд
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/RoundRoundQdata_augmentation/random_flip_3/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/RoundВ
Odata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2Q
Odata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2/axisў
Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2	ReverseV2\data_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency:output:0Xdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*0
_output_shapes
:         ─i2L
Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2№
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/mulMulJdata_augmentation/random_flip_3/stateless_random_flip_left_right/Round:y:0Sdata_augmentation/random_flip_3/stateless_random_flip_left_right/ReverseV2:output:0*
T0*0
_output_shapes
:         ─i2F
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/mulН
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub/xЖ
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/subSubOdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub/x:output:0Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2F
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/subЩ
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul_1MulHdata_augmentation/random_flip_3/stateless_random_flip_left_right/sub:z:0\data_augmentation/random_flip_3/stateless_random_flip_left_right/control_dependency:output:0*
T0*0
_output_shapes
:         ─i2H
Fdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul_1Т
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/addAddV2Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul:z:0Jdata_augmentation/random_flip_3/stateless_random_flip_left_right/mul_1:z:0*
T0*0
_output_shapes
:         ─i2F
Ddata_augmentation/random_flip_3/stateless_random_flip_left_right/add╬
)data_augmentation/random_rotation_3/ShapeShapeHdata_augmentation/random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2+
)data_augmentation/random_rotation_3/Shape╝
7data_augmentation/random_rotation_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7data_augmentation/random_rotation_3/strided_slice/stack└
9data_augmentation/random_rotation_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9data_augmentation/random_rotation_3/strided_slice/stack_1└
9data_augmentation/random_rotation_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9data_augmentation/random_rotation_3/strided_slice/stack_2║
1data_augmentation/random_rotation_3/strided_sliceStridedSlice2data_augmentation/random_rotation_3/Shape:output:0@data_augmentation/random_rotation_3/strided_slice/stack:output:0Bdata_augmentation/random_rotation_3/strided_slice/stack_1:output:0Bdata_augmentation/random_rotation_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1data_augmentation/random_rotation_3/strided_slice╔
9data_augmentation/random_rotation_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2;
9data_augmentation/random_rotation_3/strided_slice_1/stack═
;data_augmentation/random_rotation_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2=
;data_augmentation/random_rotation_3/strided_slice_1/stack_1─
;data_augmentation/random_rotation_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;data_augmentation/random_rotation_3/strided_slice_1/stack_2─
3data_augmentation/random_rotation_3/strided_slice_1StridedSlice2data_augmentation/random_rotation_3/Shape:output:0Bdata_augmentation/random_rotation_3/strided_slice_1/stack:output:0Ddata_augmentation/random_rotation_3/strided_slice_1/stack_1:output:0Ddata_augmentation/random_rotation_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3data_augmentation/random_rotation_3/strided_slice_1╩
(data_augmentation/random_rotation_3/CastCast<data_augmentation/random_rotation_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(data_augmentation/random_rotation_3/Cast╔
9data_augmentation/random_rotation_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2;
9data_augmentation/random_rotation_3/strided_slice_2/stack═
;data_augmentation/random_rotation_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2=
;data_augmentation/random_rotation_3/strided_slice_2/stack_1─
;data_augmentation/random_rotation_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;data_augmentation/random_rotation_3/strided_slice_2/stack_2─
3data_augmentation/random_rotation_3/strided_slice_2StridedSlice2data_augmentation/random_rotation_3/Shape:output:0Bdata_augmentation/random_rotation_3/strided_slice_2/stack:output:0Ddata_augmentation/random_rotation_3/strided_slice_2/stack_1:output:0Ddata_augmentation/random_rotation_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3data_augmentation/random_rotation_3/strided_slice_2╬
*data_augmentation/random_rotation_3/Cast_1Cast<data_augmentation/random_rotation_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*data_augmentation/random_rotation_3/Cast_1Ж
:data_augmentation/random_rotation_3/stateful_uniform/shapePack:data_augmentation/random_rotation_3/strided_slice:output:0*
N*
T0*
_output_shapes
:2<
:data_augmentation/random_rotation_3/stateful_uniform/shape╣
8data_augmentation/random_rotation_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *§Г Й2:
8data_augmentation/random_rotation_3/stateful_uniform/min╣
8data_augmentation/random_rotation_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *§Г >2:
8data_augmentation/random_rotation_3/stateful_uniform/max┬
:data_augmentation/random_rotation_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:data_augmentation/random_rotation_3/stateful_uniform/ConstЕ
9data_augmentation/random_rotation_3/stateful_uniform/ProdProdCdata_augmentation/random_rotation_3/stateful_uniform/shape:output:0Cdata_augmentation/random_rotation_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/stateful_uniform/Prod╝
;data_augmentation/random_rotation_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2=
;data_augmentation/random_rotation_3/stateful_uniform/Cast/xШ
;data_augmentation/random_rotation_3/stateful_uniform/Cast_1CastBdata_augmentation/random_rotation_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;data_augmentation/random_rotation_3/stateful_uniform/Cast_1Ї
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipRngReadAndSkipLdata_augmentation_random_rotation_3_stateful_uniform_rngreadandskip_resourceDdata_augmentation/random_rotation_3/stateful_uniform/Cast/x:output:0?data_augmentation/random_rotation_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:2E
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipя
Hdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stackР
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_1Р
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_2д
Bdata_augmentation/random_rotation_3/stateful_uniform/strided_sliceStridedSliceKdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip:value:0Qdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack:output:0Sdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_1:output:0Sdata_augmentation/random_rotation_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2D
Bdata_augmentation/random_rotation_3/stateful_uniform/strided_sliceЁ
<data_augmentation/random_rotation_3/stateful_uniform/BitcastBitcastKdata_augmentation/random_rotation_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02>
<data_augmentation/random_rotation_3/stateful_uniform/BitcastР
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stackТ
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_1Т
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Ldata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_2ъ
Ddata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1StridedSliceKdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip:value:0Sdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack:output:0Udata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_1:output:0Udata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2F
Ddata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1І
>data_augmentation/random_rotation_3/stateful_uniform/Bitcast_1BitcastMdata_augmentation/random_rotation_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02@
>data_augmentation/random_rotation_3/stateful_uniform/Bitcast_1У
Qdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2S
Qdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2/algљ
Mdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Cdata_augmentation/random_rotation_3/stateful_uniform/shape:output:0Gdata_augmentation/random_rotation_3/stateful_uniform/Bitcast_1:output:0Edata_augmentation/random_rotation_3/stateful_uniform/Bitcast:output:0Zdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2O
Mdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2б
8data_augmentation/random_rotation_3/stateful_uniform/subSubAdata_augmentation/random_rotation_3/stateful_uniform/max:output:0Adata_augmentation/random_rotation_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2:
8data_augmentation/random_rotation_3/stateful_uniform/sub┐
8data_augmentation/random_rotation_3/stateful_uniform/mulMulVdata_augmentation/random_rotation_3/stateful_uniform/StatelessRandomUniformV2:output:0<data_augmentation/random_rotation_3/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2:
8data_augmentation/random_rotation_3/stateful_uniform/mulц
4data_augmentation/random_rotation_3/stateful_uniformAddV2<data_augmentation/random_rotation_3/stateful_uniform/mul:z:0Adata_augmentation/random_rotation_3/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         26
4data_augmentation/random_rotation_3/stateful_uniform╗
9data_augmentation/random_rotation_3/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2;
9data_augmentation/random_rotation_3/rotation_matrix/sub/yј
7data_augmentation/random_rotation_3/rotation_matrix/subSub.data_augmentation/random_rotation_3/Cast_1:y:0Bdata_augmentation/random_rotation_3/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 29
7data_augmentation/random_rotation_3/rotation_matrix/subр
7data_augmentation/random_rotation_3/rotation_matrix/CosCos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         29
7data_augmentation/random_rotation_3/rotation_matrix/Cos┐
;data_augmentation/random_rotation_3/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_1/yћ
9data_augmentation/random_rotation_3/rotation_matrix/sub_1Sub.data_augmentation/random_rotation_3/Cast_1:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_1Б
7data_augmentation/random_rotation_3/rotation_matrix/mulMul;data_augmentation/random_rotation_3/rotation_matrix/Cos:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         29
7data_augmentation/random_rotation_3/rotation_matrix/mulр
7data_augmentation/random_rotation_3/rotation_matrix/SinSin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         29
7data_augmentation/random_rotation_3/rotation_matrix/Sin┐
;data_augmentation/random_rotation_3/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_2/yњ
9data_augmentation/random_rotation_3/rotation_matrix/sub_2Sub,data_augmentation/random_rotation_3/Cast:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_2Д
9data_augmentation/random_rotation_3/rotation_matrix/mul_1Mul;data_augmentation/random_rotation_3/rotation_matrix/Sin:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/mul_1Д
9data_augmentation/random_rotation_3/rotation_matrix/sub_3Sub;data_augmentation/random_rotation_3/rotation_matrix/mul:z:0=data_augmentation/random_rotation_3/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_3Д
9data_augmentation/random_rotation_3/rotation_matrix/sub_4Sub;data_augmentation/random_rotation_3/rotation_matrix/sub:z:0=data_augmentation/random_rotation_3/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_4├
=data_augmentation/random_rotation_3/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2?
=data_augmentation/random_rotation_3/rotation_matrix/truediv/y║
;data_augmentation/random_rotation_3/rotation_matrix/truedivRealDiv=data_augmentation/random_rotation_3/rotation_matrix/sub_4:z:0Fdata_augmentation/random_rotation_3/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2=
;data_augmentation/random_rotation_3/rotation_matrix/truediv┐
;data_augmentation/random_rotation_3/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_5/yњ
9data_augmentation/random_rotation_3/rotation_matrix/sub_5Sub,data_augmentation/random_rotation_3/Cast:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_5т
9data_augmentation/random_rotation_3/rotation_matrix/Sin_1Sin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/Sin_1┐
;data_augmentation/random_rotation_3/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_6/yћ
9data_augmentation/random_rotation_3/rotation_matrix/sub_6Sub.data_augmentation/random_rotation_3/Cast_1:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_6Е
9data_augmentation/random_rotation_3/rotation_matrix/mul_2Mul=data_augmentation/random_rotation_3/rotation_matrix/Sin_1:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/mul_2т
9data_augmentation/random_rotation_3/rotation_matrix/Cos_1Cos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/Cos_1┐
;data_augmentation/random_rotation_3/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2=
;data_augmentation/random_rotation_3/rotation_matrix/sub_7/yњ
9data_augmentation/random_rotation_3/rotation_matrix/sub_7Sub,data_augmentation/random_rotation_3/Cast:y:0Ddata_augmentation/random_rotation_3/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_7Е
9data_augmentation/random_rotation_3/rotation_matrix/mul_3Mul=data_augmentation/random_rotation_3/rotation_matrix/Cos_1:y:0=data_augmentation/random_rotation_3/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/mul_3Д
7data_augmentation/random_rotation_3/rotation_matrix/addAddV2=data_augmentation/random_rotation_3/rotation_matrix/mul_2:z:0=data_augmentation/random_rotation_3/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         29
7data_augmentation/random_rotation_3/rotation_matrix/addД
9data_augmentation/random_rotation_3/rotation_matrix/sub_8Sub=data_augmentation/random_rotation_3/rotation_matrix/sub_5:z:0;data_augmentation/random_rotation_3/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/sub_8К
?data_augmentation/random_rotation_3/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2A
?data_augmentation/random_rotation_3/rotation_matrix/truediv_1/y└
=data_augmentation/random_rotation_3/rotation_matrix/truediv_1RealDiv=data_augmentation/random_rotation_3/rotation_matrix/sub_8:z:0Hdata_augmentation/random_rotation_3/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2?
=data_augmentation/random_rotation_3/rotation_matrix/truediv_1я
9data_augmentation/random_rotation_3/rotation_matrix/ShapeShape8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*
_output_shapes
:2;
9data_augmentation/random_rotation_3/rotation_matrix/Shape▄
Gdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stackЯ
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_1Я
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_2џ
Adata_augmentation/random_rotation_3/rotation_matrix/strided_sliceStridedSliceBdata_augmentation/random_rotation_3/rotation_matrix/Shape:output:0Pdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack:output:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_1:output:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Adata_augmentation/random_rotation_3/rotation_matrix/strided_sliceт
9data_augmentation/random_rotation_3/rotation_matrix/Cos_2Cos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/Cos_2у
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stackв
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_1в
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_2¤
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Cos_2:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1т
9data_augmentation/random_rotation_3/rotation_matrix/Sin_2Sin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/Sin_2у
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stackв
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_1в
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_2¤
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Sin_2:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2щ
7data_augmentation/random_rotation_3/rotation_matrix/NegNegLdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         29
7data_augmentation/random_rotation_3/rotation_matrix/Negу
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stackв
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_1в
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_2Л
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3StridedSlice?data_augmentation/random_rotation_3/rotation_matrix/truediv:z:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3т
9data_augmentation/random_rotation_3/rotation_matrix/Sin_3Sin8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/Sin_3у
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stackв
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_1в
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_2¤
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Sin_3:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4т
9data_augmentation/random_rotation_3/rotation_matrix/Cos_3Cos8data_augmentation/random_rotation_3/stateful_uniform:z:0*
T0*#
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/Cos_3у
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stackв
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_1в
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_2¤
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5StridedSlice=data_augmentation/random_rotation_3/rotation_matrix/Cos_3:y:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5у
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2K
Idata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stackв
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_1в
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2M
Kdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_2М
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6StridedSliceAdata_augmentation/random_rotation_3/rotation_matrix/truediv_1:z:0Rdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_1:output:0Tdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2E
Cdata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6╩
Bdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2D
Bdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed/1М
@data_augmentation/random_rotation_3/rotation_matrix/zeros/packedPackJdata_augmentation/random_rotation_3/rotation_matrix/strided_slice:output:0Kdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2B
@data_augmentation/random_rotation_3/rotation_matrix/zeros/packedК
?data_augmentation/random_rotation_3/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?data_augmentation/random_rotation_3/rotation_matrix/zeros/Const┼
9data_augmentation/random_rotation_3/rotation_matrix/zerosFillIdata_augmentation/random_rotation_3/rotation_matrix/zeros/packed:output:0Hdata_augmentation/random_rotation_3/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2;
9data_augmentation/random_rotation_3/rotation_matrix/zeros─
?data_augmentation/random_rotation_3/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2A
?data_augmentation/random_rotation_3/rotation_matrix/concat/axisљ
:data_augmentation/random_rotation_3/rotation_matrix/concatConcatV2Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_1:output:0;data_augmentation/random_rotation_3/rotation_matrix/Neg:y:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_3:output:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_4:output:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_5:output:0Ldata_augmentation/random_rotation_3/rotation_matrix/strided_slice_6:output:0Bdata_augmentation/random_rotation_3/rotation_matrix/zeros:output:0Hdata_augmentation/random_rotation_3/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2<
:data_augmentation/random_rotation_3/rotation_matrix/concatР
3data_augmentation/random_rotation_3/transform/ShapeShapeHdata_augmentation/random_flip_3/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:25
3data_augmentation/random_rotation_3/transform/Shapeл
Adata_augmentation/random_rotation_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Adata_augmentation/random_rotation_3/transform/strided_slice/stackн
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_1н
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cdata_augmentation/random_rotation_3/transform/strided_slice/stack_2Р
;data_augmentation/random_rotation_3/transform/strided_sliceStridedSlice<data_augmentation/random_rotation_3/transform/Shape:output:0Jdata_augmentation/random_rotation_3/transform/strided_slice/stack:output:0Ldata_augmentation/random_rotation_3/transform/strided_slice/stack_1:output:0Ldata_augmentation/random_rotation_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2=
;data_augmentation/random_rotation_3/transform/strided_slice╣
8data_augmentation/random_rotation_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8data_augmentation/random_rotation_3/transform/fill_valueЙ
Hdata_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Hdata_augmentation/random_flip_3/stateless_random_flip_left_right/add:z:0Cdata_augmentation/random_rotation_3/rotation_matrix/concat:output:0Ddata_augmentation/random_rotation_3/transform/strided_slice:output:0Adata_augmentation/random_rotation_3/transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2J
Hdata_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3█
%data_augmentation/random_zoom_3/ShapeShape]data_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2'
%data_augmentation/random_zoom_3/Shape┤
3data_augmentation/random_zoom_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3data_augmentation/random_zoom_3/strided_slice/stackИ
5data_augmentation/random_zoom_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5data_augmentation/random_zoom_3/strided_slice/stack_1И
5data_augmentation/random_zoom_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5data_augmentation/random_zoom_3/strided_slice/stack_2б
-data_augmentation/random_zoom_3/strided_sliceStridedSlice.data_augmentation/random_zoom_3/Shape:output:0<data_augmentation/random_zoom_3/strided_slice/stack:output:0>data_augmentation/random_zoom_3/strided_slice/stack_1:output:0>data_augmentation/random_zoom_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-data_augmentation/random_zoom_3/strided_slice┴
5data_augmentation/random_zoom_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        27
5data_augmentation/random_zoom_3/strided_slice_1/stack┼
7data_augmentation/random_zoom_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        29
7data_augmentation/random_zoom_3/strided_slice_1/stack_1╝
7data_augmentation/random_zoom_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7data_augmentation/random_zoom_3/strided_slice_1/stack_2г
/data_augmentation/random_zoom_3/strided_slice_1StridedSlice.data_augmentation/random_zoom_3/Shape:output:0>data_augmentation/random_zoom_3/strided_slice_1/stack:output:0@data_augmentation/random_zoom_3/strided_slice_1/stack_1:output:0@data_augmentation/random_zoom_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/data_augmentation/random_zoom_3/strided_slice_1Й
$data_augmentation/random_zoom_3/CastCast8data_augmentation/random_zoom_3/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$data_augmentation/random_zoom_3/Cast┴
5data_augmentation/random_zoom_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        27
5data_augmentation/random_zoom_3/strided_slice_2/stack┼
7data_augmentation/random_zoom_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         29
7data_augmentation/random_zoom_3/strided_slice_2/stack_1╝
7data_augmentation/random_zoom_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7data_augmentation/random_zoom_3/strided_slice_2/stack_2г
/data_augmentation/random_zoom_3/strided_slice_2StridedSlice.data_augmentation/random_zoom_3/Shape:output:0>data_augmentation/random_zoom_3/strided_slice_2/stack:output:0@data_augmentation/random_zoom_3/strided_slice_2/stack_1:output:0@data_augmentation/random_zoom_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/data_augmentation/random_zoom_3/strided_slice_2┬
&data_augmentation/random_zoom_3/Cast_1Cast8data_augmentation/random_zoom_3/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&data_augmentation/random_zoom_3/Cast_1Х
8data_augmentation/random_zoom_3/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8data_augmentation/random_zoom_3/stateful_uniform/shape/1А
6data_augmentation/random_zoom_3/stateful_uniform/shapePack6data_augmentation/random_zoom_3/strided_slice:output:0Adata_augmentation/random_zoom_3/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:28
6data_augmentation/random_zoom_3/stateful_uniform/shape▒
4data_augmentation/random_zoom_3/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L?26
4data_augmentation/random_zoom_3/stateful_uniform/min▒
4data_augmentation/random_zoom_3/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?26
4data_augmentation/random_zoom_3/stateful_uniform/max║
6data_augmentation/random_zoom_3/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6data_augmentation/random_zoom_3/stateful_uniform/ConstЎ
5data_augmentation/random_zoom_3/stateful_uniform/ProdProd?data_augmentation/random_zoom_3/stateful_uniform/shape:output:0?data_augmentation/random_zoom_3/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 27
5data_augmentation/random_zoom_3/stateful_uniform/Prod┤
7data_augmentation/random_zoom_3/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :29
7data_augmentation/random_zoom_3/stateful_uniform/Cast/xЖ
7data_augmentation/random_zoom_3/stateful_uniform/Cast_1Cast>data_augmentation/random_zoom_3/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 29
7data_augmentation/random_zoom_3/stateful_uniform/Cast_1щ
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipRngReadAndSkipHdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resource@data_augmentation/random_zoom_3/stateful_uniform/Cast/x:output:0;data_augmentation/random_zoom_3/stateful_uniform/Cast_1:y:0*
_output_shapes
:2A
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipо
Ddata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Ddata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack┌
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_1┌
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_2ј
>data_augmentation/random_zoom_3/stateful_uniform/strided_sliceStridedSliceGdata_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip:value:0Mdata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack:output:0Odata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_1:output:0Odata_augmentation/random_zoom_3/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2@
>data_augmentation/random_zoom_3/stateful_uniform/strided_sliceщ
8data_augmentation/random_zoom_3/stateful_uniform/BitcastBitcastGdata_augmentation/random_zoom_3/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02:
8data_augmentation/random_zoom_3/stateful_uniform/Bitcast┌
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2H
Fdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stackя
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_1я
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_2є
@data_augmentation/random_zoom_3/stateful_uniform/strided_slice_1StridedSliceGdata_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip:value:0Odata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack:output:0Qdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_1:output:0Qdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2B
@data_augmentation/random_zoom_3/stateful_uniform/strided_slice_1 
:data_augmentation/random_zoom_3/stateful_uniform/Bitcast_1BitcastIdata_augmentation/random_zoom_3/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02<
:data_augmentation/random_zoom_3/stateful_uniform/Bitcast_1Я
Mdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2O
Mdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2/algЧ
Idata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2?data_augmentation/random_zoom_3/stateful_uniform/shape:output:0Cdata_augmentation/random_zoom_3/stateful_uniform/Bitcast_1:output:0Adata_augmentation/random_zoom_3/stateful_uniform/Bitcast:output:0Vdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2K
Idata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2њ
4data_augmentation/random_zoom_3/stateful_uniform/subSub=data_augmentation/random_zoom_3/stateful_uniform/max:output:0=data_augmentation/random_zoom_3/stateful_uniform/min:output:0*
T0*
_output_shapes
: 26
4data_augmentation/random_zoom_3/stateful_uniform/sub│
4data_augmentation/random_zoom_3/stateful_uniform/mulMulRdata_augmentation/random_zoom_3/stateful_uniform/StatelessRandomUniformV2:output:08data_augmentation/random_zoom_3/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         26
4data_augmentation/random_zoom_3/stateful_uniform/mulў
0data_augmentation/random_zoom_3/stateful_uniformAddV28data_augmentation/random_zoom_3/stateful_uniform/mul:z:0=data_augmentation/random_zoom_3/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         22
0data_augmentation/random_zoom_3/stateful_uniform║
:data_augmentation/random_zoom_3/stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:data_augmentation/random_zoom_3/stateful_uniform_1/shape/1Д
8data_augmentation/random_zoom_3/stateful_uniform_1/shapePack6data_augmentation/random_zoom_3/strided_slice:output:0Cdata_augmentation/random_zoom_3/stateful_uniform_1/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8data_augmentation/random_zoom_3/stateful_uniform_1/shapeх
6data_augmentation/random_zoom_3/stateful_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L?28
6data_augmentation/random_zoom_3/stateful_uniform_1/minх
6data_augmentation/random_zoom_3/stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?28
6data_augmentation/random_zoom_3/stateful_uniform_1/maxЙ
8data_augmentation/random_zoom_3/stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8data_augmentation/random_zoom_3/stateful_uniform_1/ConstА
7data_augmentation/random_zoom_3/stateful_uniform_1/ProdProdAdata_augmentation/random_zoom_3/stateful_uniform_1/shape:output:0Adata_augmentation/random_zoom_3/stateful_uniform_1/Const:output:0*
T0*
_output_shapes
: 29
7data_augmentation/random_zoom_3/stateful_uniform_1/ProdИ
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2;
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast/x­
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast_1Cast@data_augmentation/random_zoom_3/stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9data_augmentation/random_zoom_3/stateful_uniform_1/Cast_1├
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipRngReadAndSkipHdata_augmentation_random_zoom_3_stateful_uniform_rngreadandskip_resourceBdata_augmentation/random_zoom_3/stateful_uniform_1/Cast/x:output:0=data_augmentation/random_zoom_3/stateful_uniform_1/Cast_1:y:0@^data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip*
_output_shapes
:2C
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip┌
Fdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stackя
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_1я
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_2џ
@data_augmentation/random_zoom_3/stateful_uniform_1/strided_sliceStridedSliceIdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0Odata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack:output:0Qdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_1:output:0Qdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2B
@data_augmentation/random_zoom_3/stateful_uniform_1/strided_slice 
:data_augmentation/random_zoom_3/stateful_uniform_1/BitcastBitcastIdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02<
:data_augmentation/random_zoom_3/stateful_uniform_1/Bitcastя
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stackР
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1Р
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2њ
Bdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1StridedSliceIdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip:value:0Qdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack:output:0Sdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_1:output:0Sdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2D
Bdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1Ё
<data_augmentation/random_zoom_3/stateful_uniform_1/Bitcast_1BitcastKdata_augmentation/random_zoom_3/stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02>
<data_augmentation/random_zoom_3/stateful_uniform_1/Bitcast_1С
Odata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2Q
Odata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/algѕ
Kdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2Adata_augmentation/random_zoom_3/stateful_uniform_1/shape:output:0Edata_augmentation/random_zoom_3/stateful_uniform_1/Bitcast_1:output:0Cdata_augmentation/random_zoom_3/stateful_uniform_1/Bitcast:output:0Xdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2M
Kdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2џ
6data_augmentation/random_zoom_3/stateful_uniform_1/subSub?data_augmentation/random_zoom_3/stateful_uniform_1/max:output:0?data_augmentation/random_zoom_3/stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 28
6data_augmentation/random_zoom_3/stateful_uniform_1/sub╗
6data_augmentation/random_zoom_3/stateful_uniform_1/mulMulTdata_augmentation/random_zoom_3/stateful_uniform_1/StatelessRandomUniformV2:output:0:data_augmentation/random_zoom_3/stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:         28
6data_augmentation/random_zoom_3/stateful_uniform_1/mulа
2data_augmentation/random_zoom_3/stateful_uniform_1AddV2:data_augmentation/random_zoom_3/stateful_uniform_1/mul:z:0?data_augmentation/random_zoom_3/stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:         24
2data_augmentation/random_zoom_3/stateful_uniform_1ю
+data_augmentation/random_zoom_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+data_augmentation/random_zoom_3/concat/axis╗
&data_augmentation/random_zoom_3/concatConcatV26data_augmentation/random_zoom_3/stateful_uniform_1:z:04data_augmentation/random_zoom_3/stateful_uniform:z:04data_augmentation/random_zoom_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2(
&data_augmentation/random_zoom_3/concat┼
1data_augmentation/random_zoom_3/zoom_matrix/ShapeShape/data_augmentation/random_zoom_3/concat:output:0*
T0*
_output_shapes
:23
1data_augmentation/random_zoom_3/zoom_matrix/Shape╠
?data_augmentation/random_zoom_3/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?data_augmentation/random_zoom_3/zoom_matrix/strided_slice/stackл
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_1л
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_2Ж
9data_augmentation/random_zoom_3/zoom_matrix/strided_sliceStridedSlice:data_augmentation/random_zoom_3/zoom_matrix/Shape:output:0Hdata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_1:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9data_augmentation/random_zoom_3/zoom_matrix/strided_sliceФ
1data_augmentation/random_zoom_3/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?23
1data_augmentation/random_zoom_3/zoom_matrix/sub/yЫ
/data_augmentation/random_zoom_3/zoom_matrix/subSub*data_augmentation/random_zoom_3/Cast_1:y:0:data_augmentation/random_zoom_3/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 21
/data_augmentation/random_zoom_3/zoom_matrix/sub│
5data_augmentation/random_zoom_3/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @27
5data_augmentation/random_zoom_3/zoom_matrix/truediv/yІ
3data_augmentation/random_zoom_3/zoom_matrix/truedivRealDiv3data_augmentation/random_zoom_3/zoom_matrix/sub:z:0>data_augmentation/random_zoom_3/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 25
3data_augmentation/random_zoom_3/zoom_matrix/truediv█
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_1▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_2▒
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_1StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_1»
3data_augmentation/random_zoom_3/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?25
3data_augmentation/random_zoom_3/zoom_matrix/sub_1/xБ
1data_augmentation/random_zoom_3/zoom_matrix/sub_1Sub<data_augmentation/random_zoom_3/zoom_matrix/sub_1/x:output:0Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         23
1data_augmentation/random_zoom_3/zoom_matrix/sub_1І
/data_augmentation/random_zoom_3/zoom_matrix/mulMul7data_augmentation/random_zoom_3/zoom_matrix/truediv:z:05data_augmentation/random_zoom_3/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         21
/data_augmentation/random_zoom_3/zoom_matrix/mul»
3data_augmentation/random_zoom_3/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?25
3data_augmentation/random_zoom_3/zoom_matrix/sub_2/yШ
1data_augmentation/random_zoom_3/zoom_matrix/sub_2Sub(data_augmentation/random_zoom_3/Cast:y:0<data_augmentation/random_zoom_3/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 23
1data_augmentation/random_zoom_3/zoom_matrix/sub_2и
7data_augmentation/random_zoom_3/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @29
7data_augmentation/random_zoom_3/zoom_matrix/truediv_1/yЊ
5data_augmentation/random_zoom_3/zoom_matrix/truediv_1RealDiv5data_augmentation/random_zoom_3/zoom_matrix/sub_2:z:0@data_augmentation/random_zoom_3/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 27
5data_augmentation/random_zoom_3/zoom_matrix/truediv_1█
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_1▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_2▒
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_2StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_2»
3data_augmentation/random_zoom_3/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?25
3data_augmentation/random_zoom_3/zoom_matrix/sub_3/xБ
1data_augmentation/random_zoom_3/zoom_matrix/sub_3Sub<data_augmentation/random_zoom_3/zoom_matrix/sub_3/x:output:0Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         23
1data_augmentation/random_zoom_3/zoom_matrix/sub_3Љ
1data_augmentation/random_zoom_3/zoom_matrix/mul_1Mul9data_augmentation/random_zoom_3/zoom_matrix/truediv_1:z:05data_augmentation/random_zoom_3/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         23
1data_augmentation/random_zoom_3/zoom_matrix/mul_1█
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_1▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_2▒
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_3StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_3║
:data_augmentation/random_zoom_3/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:data_augmentation/random_zoom_3/zoom_matrix/zeros/packed/1│
8data_augmentation/random_zoom_3/zoom_matrix/zeros/packedPackBdata_augmentation/random_zoom_3/zoom_matrix/strided_slice:output:0Cdata_augmentation/random_zoom_3/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2:
8data_augmentation/random_zoom_3/zoom_matrix/zeros/packedи
7data_augmentation/random_zoom_3/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7data_augmentation/random_zoom_3/zoom_matrix/zeros/ConstЦ
1data_augmentation/random_zoom_3/zoom_matrix/zerosFillAdata_augmentation/random_zoom_3/zoom_matrix/zeros/packed:output:0@data_augmentation/random_zoom_3/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         23
1data_augmentation/random_zoom_3/zoom_matrix/zerosЙ
<data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed/1╣
:data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packedPackBdata_augmentation/random_zoom_3/zoom_matrix/strided_slice:output:0Edata_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2<
:data_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed╗
9data_augmentation/random_zoom_3/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9data_augmentation/random_zoom_3/zoom_matrix/zeros_1/ConstГ
3data_augmentation/random_zoom_3/zoom_matrix/zeros_1FillCdata_augmentation/random_zoom_3/zoom_matrix/zeros_1/packed:output:0Bdata_augmentation/random_zoom_3/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         25
3data_augmentation/random_zoom_3/zoom_matrix/zeros_1█
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2C
Adata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_1▀
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2E
Cdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_2▒
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_4StridedSlice/data_augmentation/random_zoom_3/concat:output:0Jdata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_1:output:0Ldata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2=
;data_augmentation/random_zoom_3/zoom_matrix/strided_slice_4Й
<data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed/1╣
:data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packedPackBdata_augmentation/random_zoom_3/zoom_matrix/strided_slice:output:0Edata_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2<
:data_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed╗
9data_augmentation/random_zoom_3/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9data_augmentation/random_zoom_3/zoom_matrix/zeros_2/ConstГ
3data_augmentation/random_zoom_3/zoom_matrix/zeros_2FillCdata_augmentation/random_zoom_3/zoom_matrix/zeros_2/packed:output:0Bdata_augmentation/random_zoom_3/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         25
3data_augmentation/random_zoom_3/zoom_matrix/zeros_2┤
7data_augmentation/random_zoom_3/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :29
7data_augmentation/random_zoom_3/zoom_matrix/concat/axisА
2data_augmentation/random_zoom_3/zoom_matrix/concatConcatV2Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_3:output:0:data_augmentation/random_zoom_3/zoom_matrix/zeros:output:03data_augmentation/random_zoom_3/zoom_matrix/mul:z:0<data_augmentation/random_zoom_3/zoom_matrix/zeros_1:output:0Ddata_augmentation/random_zoom_3/zoom_matrix/strided_slice_4:output:05data_augmentation/random_zoom_3/zoom_matrix/mul_1:z:0<data_augmentation/random_zoom_3/zoom_matrix/zeros_2:output:0@data_augmentation/random_zoom_3/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         24
2data_augmentation/random_zoom_3/zoom_matrix/concat№
/data_augmentation/random_zoom_3/transform/ShapeShape]data_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:21
/data_augmentation/random_zoom_3/transform/Shape╚
=data_augmentation/random_zoom_3/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=data_augmentation/random_zoom_3/transform/strided_slice/stack╠
?data_augmentation/random_zoom_3/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?data_augmentation/random_zoom_3/transform/strided_slice/stack_1╠
?data_augmentation/random_zoom_3/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?data_augmentation/random_zoom_3/transform/strided_slice/stack_2╩
7data_augmentation/random_zoom_3/transform/strided_sliceStridedSlice8data_augmentation/random_zoom_3/transform/Shape:output:0Fdata_augmentation/random_zoom_3/transform/strided_slice/stack:output:0Hdata_augmentation/random_zoom_3/transform/strided_slice/stack_1:output:0Hdata_augmentation/random_zoom_3/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:29
7data_augmentation/random_zoom_3/transform/strided_slice▒
4data_augmentation/random_zoom_3/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4data_augmentation/random_zoom_3/transform/fill_value╗
Ddata_augmentation/random_zoom_3/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3]data_augmentation/random_rotation_3/transform/ImageProjectiveTransformV3:transformed_images:0;data_augmentation/random_zoom_3/zoom_matrix/concat:output:0@data_augmentation/random_zoom_3/transform/strided_slice:output:0=data_augmentation/random_zoom_3/transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2F
Ddata_augmentation/random_zoom_3/transform/ImageProjectiveTransformV3║
patches_7/ExtractImagePatchesExtractImagePatchesYdata_augmentation/random_zoom_3/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*/
_output_shapes
:         1*
ksizes
*
paddingVALID*
rates
*
strides
2
patches_7/ExtractImagePatchesЄ
patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    ц  1   2
patches_7/Reshape/shape│
patches_7/ReshapeReshape'patches_7/ExtractImagePatches:patches:0 patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:         ц12
patches_7/Reshape▒
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
dense_21/Tensordot/axesЃ
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
dense_21/Tensordot/Shapeє
 dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/GatherV2/axis■
dense_21/Tensordot/GatherV2GatherV2!dense_21/Tensordot/Shape:output:0 dense_21/Tensordot/free:output:0)dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_21/Tensordot/GatherV2і
"dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_21/Tensordot/GatherV2_1/axisё
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
dense_21/Tensordot/Constц
dense_21/Tensordot/ProdProd$dense_21/Tensordot/GatherV2:output:0!dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prodѓ
dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_21/Tensordot/Const_1г
dense_21/Tensordot/Prod_1Prod&dense_21/Tensordot/GatherV2_1:output:0#dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prod_1ѓ
dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_21/Tensordot/concat/axisП
dense_21/Tensordot/concatConcatV2 dense_21/Tensordot/free:output:0 dense_21/Tensordot/axes:output:0'dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat░
dense_21/Tensordot/stackPack dense_21/Tensordot/Prod:output:0"dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/stack└
dense_21/Tensordot/transpose	Transposepatches_7/Reshape:output:0"dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:         ц12
dense_21/Tensordot/transpose├
dense_21/Tensordot/ReshapeReshape dense_21/Tensordot/transpose:y:0!dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_21/Tensordot/Reshape┬
dense_21/Tensordot/MatMulMatMul#dense_21/Tensordot/Reshape:output:0)dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_21/Tensordot/MatMulѓ
dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_21/Tensordot/Const_2є
 dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/concat_1/axisЖ
dense_21/Tensordot/concat_1ConcatV2$dense_21/Tensordot/GatherV2:output:0#dense_21/Tensordot/Const_2:output:0)dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat_1х
dense_21/TensordotReshape#dense_21/Tensordot/MatMul:product:0$dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         ц@2
dense_21/TensordotД
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_21/BiasAdd/ReadVariableOpг
dense_21/BiasAddBiasAdddense_21/Tensordot:output:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ц@2
dense_21/BiasAddё
lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2!
lambda_3/Mean/reduction_indicesЮ
lambda_3/MeanMeandense_21/BiasAdd:output:0(lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
lambda_3/MeanЁ
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
lambda_3/Reshape/shapeъ
lambda_3/ReshapeReshapelambda_3/Mean:output:0lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:         @2
lambda_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisм
concatenate_3/concatConcatV2lambda_3/Reshape:output:0dense_21/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:         Ц@2
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
B :Ц2
patch_encoder_7/range/limit|
patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder_7/range/deltaк
patch_encoder_7/rangeRange$patch_encoder_7/range/start:output:0$patch_encoder_7/range/limit:output:0$patch_encoder_7/range/delta:output:0*
_output_shapes	
:Ц2
patch_encoder_7/rangeщ
,patch_encoder_7/embedding_3/embedding_lookupResourceGather4patch_encoder_7_embedding_3_embedding_lookup_1117078patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1117078*
_output_shapes
:	Ц@*
dtype02.
,patch_encoder_7/embedding_3/embedding_lookupм
5patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity5patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1117078*
_output_shapes
:	Ц@27
5patch_encoder_7/embedding_3/embedding_lookup/IdentityУ
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1Identity>patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Ц@29
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1╦
patch_encoder_7/addAddV2concatenate_3/concat:output:0@patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         Ц@2
patch_encoder_7/addИ
5layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_12/moments/mean/reduction_indicesз
#layer_normalization_12/moments/meanMeanpatch_encoder_7/add:z:0>layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2%
#layer_normalization_12/moments/mean¤
+layer_normalization_12/moments/StopGradientStopGradient,layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:         Ц2-
+layer_normalization_12/moments/StopGradient 
0layer_normalization_12/moments/SquaredDifferenceSquaredDifferencepatch_encoder_7/add:z:04layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:         Ц@22
0layer_normalization_12/moments/SquaredDifference└
9layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_12/moments/variance/reduction_indicesю
'layer_normalization_12/moments/varianceMean4layer_normalization_12/moments/SquaredDifference:z:0Blayer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2)
'layer_normalization_12/moments/varianceЋ
&layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52(
&layer_normalization_12/batchnorm/add/y№
$layer_normalization_12/batchnorm/addAddV20layer_normalization_12/moments/variance:output:0/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:         Ц2&
$layer_normalization_12/batchnorm/add║
&layer_normalization_12/batchnorm/RsqrtRsqrt(layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:         Ц2(
&layer_normalization_12/batchnorm/Rsqrtс
3layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_12/batchnorm/mul/ReadVariableOpз
$layer_normalization_12/batchnorm/mulMul*layer_normalization_12/batchnorm/Rsqrt:y:0;layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2&
$layer_normalization_12/batchnorm/mulЛ
&layer_normalization_12/batchnorm/mul_1Mulpatch_encoder_7/add:z:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2(
&layer_normalization_12/batchnorm/mul_1Т
&layer_normalization_12/batchnorm/mul_2Mul,layer_normalization_12/moments/mean:output:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2(
&layer_normalization_12/batchnorm/mul_2О
/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_12/batchnorm/ReadVariableOp№
$layer_normalization_12/batchnorm/subSub7layer_normalization_12/batchnorm/ReadVariableOp:value:0*layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:         Ц@2&
$layer_normalization_12/batchnorm/subТ
&layer_normalization_12/batchnorm/add_1AddV2*layer_normalization_12/batchnorm/mul_1:z:0(layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ц@2(
&layer_normalization_12/batchnorm/add_1§
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp▓
*multi_head_attention_6/query/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2,
*multi_head_attention_6/query/einsum/Einsum█
/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp8multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/query/add/ReadVariableOpШ
 multi_head_attention_6/query/addAddV23multi_head_attention_6/query/einsum/Einsum:output:07multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2"
 multi_head_attention_6/query/addэ
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpг
(multi_head_attention_6/key/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0?multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2*
(multi_head_attention_6/key/einsum/EinsumН
-multi_head_attention_6/key/add/ReadVariableOpReadVariableOp6multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_6/key/add/ReadVariableOpЬ
multi_head_attention_6/key/addAddV21multi_head_attention_6/key/einsum/Einsum:output:05multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2 
multi_head_attention_6/key/add§
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp▓
*multi_head_attention_6/value/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2,
*multi_head_attention_6/value/einsum/Einsum█
/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp8multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/value/add/ReadVariableOpШ
 multi_head_attention_6/value/addAddV23multi_head_attention_6/value/einsum/Einsum:output:07multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2"
 multi_head_attention_6/value/addЂ
multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_6/Mul/yК
multi_head_attention_6/MulMul$multi_head_attention_6/query/add:z:0%multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:         Ц@2
multi_head_attention_6/Mul■
$multi_head_attention_6/einsum/EinsumEinsum"multi_head_attention_6/key/add:z:0multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2&
$multi_head_attention_6/einsum/Einsumк
&multi_head_attention_6/softmax/SoftmaxSoftmax-multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ2(
&multi_head_attention_6/softmax/SoftmaxА
,multi_head_attention_6/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2.
,multi_head_attention_6/dropout/dropout/Constё
*multi_head_attention_6/dropout/dropout/MulMul0multi_head_attention_6/softmax/Softmax:softmax:05multi_head_attention_6/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:         ЦЦ2,
*multi_head_attention_6/dropout/dropout/Mul╝
,multi_head_attention_6/dropout/dropout/ShapeShape0multi_head_attention_6/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_6/dropout/dropout/ShapeЏ
Cmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_6/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:         ЦЦ*
dtype02E
Cmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniform│
5multi_head_attention_6/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=27
5multi_head_attention_6/dropout/dropout/GreaterEqual/y─
3multi_head_attention_6/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_6/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_6/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ЦЦ25
3multi_head_attention_6/dropout/dropout/GreaterEqualТ
+multi_head_attention_6/dropout/dropout/CastCast7multi_head_attention_6/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ЦЦ2-
+multi_head_attention_6/dropout/dropout/Castђ
,multi_head_attention_6/dropout/dropout/Mul_1Mul.multi_head_attention_6/dropout/dropout/Mul:z:0/multi_head_attention_6/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:         ЦЦ2.
,multi_head_attention_6/dropout/dropout/Mul_1Ћ
&multi_head_attention_6/einsum_1/EinsumEinsum0multi_head_attention_6/dropout/dropout/Mul_1:z:0$multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd2(
&multi_head_attention_6/einsum_1/Einsumъ
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpн
5multi_head_attention_6/attention_output/einsum/EinsumEinsum/multi_head_attention_6/einsum_1/Einsum:output:0Lmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe27
5multi_head_attention_6/attention_output/einsum/EinsumЭ
:multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_6/attention_output/add/ReadVariableOpъ
+multi_head_attention_6/attention_output/addAddV2>multi_head_attention_6/attention_output/einsum/Einsum:output:0Bmulti_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2-
+multi_head_attention_6/attention_output/addЋ
IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityэ	
NoOpNoOpI^data_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipp^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgw^data_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterD^data_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip@^data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkipB^data_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip ^dense_21/BiasAdd/ReadVariableOp"^dense_21/Tensordot/ReadVariableOp0^layer_normalization_12/batchnorm/ReadVariableOp4^layer_normalization_12/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp-^patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:         ўќ::: : : : : : : : : : : : : : : : 2ћ
Hdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkipHdata_augmentation/random_flip_3/stateful_uniform_full_int/RngReadAndSkip2Р
odata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgodata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2­
vdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCountervdata_augmentation/random_flip_3/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2і
Cdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkipCdata_augmentation/random_rotation_3/stateful_uniform/RngReadAndSkip2ѓ
?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip?data_augmentation/random_zoom_3/stateful_uniform/RngReadAndSkip2є
Adata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkipAdata_augmentation/random_zoom_3/stateful_uniform_1/RngReadAndSkip2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2F
!dense_21/Tensordot/ReadVariableOp!dense_21/Tensordot/ReadVariableOp2b
/layer_normalization_12/batchnorm/ReadVariableOp/layer_normalization_12/batchnorm/ReadVariableOp2j
3layer_normalization_12/batchnorm/mul/ReadVariableOp3layer_normalization_12/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2ї
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
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ю0
ї
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

identity_1ѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp╚
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpџ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpњ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp╚
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpџ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
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
:         Ц@2
Mulб
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2
einsum/EinsumЂ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ2
softmax/SoftmaxЄ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/Identity╣
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpЭ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityѓ

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identity_1Я
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Ц@:         Ц@: : : : : : : : 2J
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
:         Ц@

_user_specified_namequery:SO
,
_output_shapes
:         Ц@

_user_specified_namevalue
Ў
§
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

identity_1ѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         Ц@:         ЦЦ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_11160552
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц@2

IdentityЅ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:         ЦЦ2

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
@:         Ц@:         Ц@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:         Ц@

_user_specified_namequery:SO
,
_output_shapes
:         Ц@

_user_specified_namevalue
Ў
§
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

identity_1ѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         Ц@:         ЦЦ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_11161792
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц@2

IdentityЅ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:         ЦЦ2

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
@:         Ц@:         Ц@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:         Ц@

_user_specified_namequery:SO
,
_output_shapes
:         Ц@

_user_specified_namevalue
ш

/__inference_random_zoom_3_layer_call_fn_1117980

inputs
unknown:	
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11155882
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
А
f
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1117984

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
А
f
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1115432

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
Ц
j
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1115438

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
▀
G
+__inference_patches_7_layer_call_fn_1117485

images
identity╠
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_patches_7_layer_call_and_return_conditional_losses_11159162
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ц12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameimages
╬
б
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115897
normalization_3_input
normalization_3_sub_y
normalization_3_sqrt_x#
random_flip_3_1115887:	'
random_rotation_3_1115890:	#
random_zoom_3_1115893:	
identityѕб%random_flip_3/StatefulPartitionedCallб)random_rotation_3/StatefulPartitionedCallб%random_zoom_3/StatefulPartitionedCallЏ
normalization_3/subSubnormalization_3_inputnormalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2
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
 *Ћ┐о32
normalization_3/Maximum/yг
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum»
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2
normalization_3/truedivЧ
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
resizing_3/PartitionedCallй
%random_flip_3/StatefulPartitionedCallStatefulPartitionedCall#resizing_3/PartitionedCall:output:0random_flip_3_1115887*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11157902'
%random_flip_3/StatefulPartitionedCallп
)random_rotation_3/StatefulPartitionedCallStatefulPartitionedCall.random_flip_3/StatefulPartitionedCall:output:0random_rotation_3_1115890*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11157192+
)random_rotation_3/StatefulPartitionedCall╠
%random_zoom_3/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_3/StatefulPartitionedCall:output:0random_zoom_3_1115893*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11155882'
%random_zoom_3/StatefulPartitionedCallњ
IdentityIdentity.random_zoom_3/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identity╩
NoOpNoOp&^random_flip_3/StatefulPartitionedCall*^random_rotation_3/StatefulPartitionedCall&^random_zoom_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         ўќ::: : : 2N
%random_flip_3/StatefulPartitionedCall%random_flip_3/StatefulPartitionedCall2V
)random_rotation_3/StatefulPartitionedCall)random_rotation_3/StatefulPartitionedCall2N
%random_zoom_3/StatefulPartitionedCall%random_zoom_3/StatefulPartitionedCall:h d
1
_output_shapes
:         ўќ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:
Љ2
ф	
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

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameй
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
value┼B┬B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-3/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer-4/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names░
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЙ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop7savev2_layer_normalization_12_gamma_read_readvariableop6savev2_layer_normalization_12_beta_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableopAsavev2_patch_encoder_7_embedding_3_embeddings_read_readvariableop>savev2_multi_head_attention_6_query_kernel_read_readvariableop<savev2_multi_head_attention_6_query_bias_read_readvariableop<savev2_multi_head_attention_6_key_kernel_read_readvariableop:savev2_multi_head_attention_6_key_bias_read_readvariableop>savev2_multi_head_attention_6_value_kernel_read_readvariableop<savev2_multi_head_attention_6_value_bias_read_readvariableopIsavev2_multi_head_attention_6_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_6_attention_output_bias_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *"
dtypes
2				2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*╝
_input_shapesф
Д: :1@:@:@:@::: :	Ц@:@@:@:@@:@:@@:@:@@:@:::: 2(
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
:	Ц@:(	$
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
Ч
b
F__inference_patches_7_layer_call_and_return_conditional_losses_1115916

images
identityМ
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:         1*
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
valueB"    ц  1   2
Reshape/shapeІ
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:         ц12	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:         ц12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameimages
ё

М
3__inference_data_augmentation_layer_call_fn_1117164

inputs
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11158332
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         ўќ::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
Ч
b
F__inference_patches_7_layer_call_and_return_conditional_losses_1117492

images
identityМ
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:         1*
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
valueB"    ц  1   2
Reshape/shapeІ
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:         ц12	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:         ц12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameimages
╠
К
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1117591

projection7
$embedding_3_embedding_lookup_1117584:	Ц@
identityѕбembedding_3/embedding_lookup\
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
B :Ц2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:Ц2
rangeЕ
embedding_3/embedding_lookupResourceGather$embedding_3_embedding_lookup_1117584range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*7
_class-
+)loc:@embedding_3/embedding_lookup/1117584*
_output_shapes
:	Ц@*
dtype02
embedding_3/embedding_lookupњ
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*7
_class-
+)loc:@embedding_3/embedding_lookup/1117584*
_output_shapes
:	Ц@2'
%embedding_3/embedding_lookup/IdentityИ
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Ц@2)
'embedding_3/embedding_lookup/Identity_1ѕ
addAddV2
projection0embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         Ц@2
addg
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identitym
NoOpNoOp^embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:         Ц@: 2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:X T
,
_output_shapes
:         Ц@
$
_user_specified_name
projection
гД
у
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118110

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipб!stateful_uniform_1/RngReadAndSkipD
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЂ
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        2
strided_slice_1/stackЁ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
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
CastЂ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        2
strided_slice_2/stackЁ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
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
stateful_uniform/shape/1А
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
 *═╠L?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
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
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg╝
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub│
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         2
stateful_uniform/mulў
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2
stateful_uniformz
stateful_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_1/shape/1Д
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
 *═╠L?2
stateful_uniform_1/minu
stateful_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ?2
stateful_uniform_1/max~
stateful_uniform_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform_1/ConstА
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
stateful_uniform_1/Cast/xљ
stateful_uniform_1/Cast_1Cast stateful_uniform_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform_1/Cast_1Ѓ
!stateful_uniform_1/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource"stateful_uniform_1/Cast/x:output:0stateful_uniform_1/Cast_1:y:0 ^stateful_uniform/RngReadAndSkip*
_output_shapes
:2#
!stateful_uniform_1/RngReadAndSkipџ
&stateful_uniform_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&stateful_uniform_1/strided_slice/stackъ
(stateful_uniform_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_1ъ
(stateful_uniform_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice/stack_2┌
 stateful_uniform_1/strided_sliceStridedSlice)stateful_uniform_1/RngReadAndSkip:value:0/stateful_uniform_1/strided_slice/stack:output:01stateful_uniform_1/strided_slice/stack_1:output:01stateful_uniform_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2"
 stateful_uniform_1/strided_sliceЪ
stateful_uniform_1/BitcastBitcast)stateful_uniform_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcastъ
(stateful_uniform_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform_1/strided_slice_1/stackб
*stateful_uniform_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_1б
*stateful_uniform_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*stateful_uniform_1/strided_slice_1/stack_2м
"stateful_uniform_1/strided_slice_1StridedSlice)stateful_uniform_1/RngReadAndSkip:value:01stateful_uniform_1/strided_slice_1/stack:output:03stateful_uniform_1/strided_slice_1/stack_1:output:03stateful_uniform_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2$
"stateful_uniform_1/strided_slice_1Ц
stateful_uniform_1/Bitcast_1Bitcast+stateful_uniform_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform_1/Bitcast_1ц
/stateful_uniform_1/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :21
/stateful_uniform_1/StatelessRandomUniformV2/alg╚
+stateful_uniform_1/StatelessRandomUniformV2StatelessRandomUniformV2!stateful_uniform_1/shape:output:0%stateful_uniform_1/Bitcast_1:output:0#stateful_uniform_1/Bitcast:output:08stateful_uniform_1/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2-
+stateful_uniform_1/StatelessRandomUniformV2џ
stateful_uniform_1/subSubstateful_uniform_1/max:output:0stateful_uniform_1/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform_1/sub╗
stateful_uniform_1/mulMul4stateful_uniform_1/StatelessRandomUniformV2:output:0stateful_uniform_1/sub:z:0*
T0*'
_output_shapes
:         2
stateful_uniform_1/mulа
stateful_uniform_1AddV2stateful_uniform_1/mul:z:0stateful_uniform_1/min:output:0*
T0*'
_output_shapes
:         2
stateful_uniform_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЏ
concatConcatV2stateful_uniform_1:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shapeї
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stackљ
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1љ
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ф
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
 *  ђ?2
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
zoom_matrix/truediv/yІ
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truedivЏ
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stackЪ
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1Ъ
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ы
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
 *  ђ?2
zoom_matrix/sub_1/xБ
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_1І
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
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
zoom_matrix/truediv_1/yЊ
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1Џ
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stackЪ
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1Ъ
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ы
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
 *  ђ?2
zoom_matrix/sub_3/xБ
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_3Љ
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mul_1Џ
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stackЪ
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1Ъ
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ы
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
zoom_matrix/zeros/packed/1│
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
zoom_matrix/zeros/ConstЦ
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1╣
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
zoom_matrix/zeros_1/ConstГ
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_1Џ
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stackЪ
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1Ъ
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ы
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

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
zoom_matrix/zeros_2/packed/1╣
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
zoom_matrix/zeros_2/ConstГ
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisр
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
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
transform/fill_value─
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*0
_output_shapes
:         ─i*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3Ю
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityћ
NoOpNoOp ^stateful_uniform/RngReadAndSkip"^stateful_uniform_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip2F
!stateful_uniform_1/RngReadAndSkip!stateful_uniform_1/RngReadAndSkip:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
■
і
1__inference_patch_encoder_7_layer_call_fn_1117577

projection
unknown:	Ц@
identityѕбStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCall
projectionunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_11159872
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:         Ц@: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
,
_output_shapes
:         Ц@
$
_user_specified_name
projection
├9
ї
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

identity_1ѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp╚
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpџ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpњ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp╚
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpџ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2
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
:         Ц@2
Mulб
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2
einsum/EinsumЂ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/dropout/Constе
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeо
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:         ЦЦ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2 
dropout/dropout/GreaterEqual/yУ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/dropout/GreaterEqualА
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ЦЦ2
dropout/dropout/Castц
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:         ЦЦ2
dropout/dropout/Mul_1╣
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpЭ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2
attention_output/addx
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:         Ц@2

Identityѓ

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identity_1Я
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Ц@:         Ц@: : : : : : : : 2J
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
:         Ц@

_user_specified_namequery:SO
,
_output_shapes
:         Ц@

_user_specified_namevalue
А
Њ
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115833

inputs
normalization_3_sub_y
normalization_3_sqrt_x#
random_flip_3_1115823:	'
random_rotation_3_1115826:	#
random_zoom_3_1115829:	
identityѕб%random_flip_3/StatefulPartitionedCallб)random_rotation_3/StatefulPartitionedCallб%random_zoom_3/StatefulPartitionedCallї
normalization_3/subSubinputsnormalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2
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
 *Ћ┐о32
normalization_3/Maximum/yг
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization_3/Maximum»
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2
normalization_3/truedivЧ
resizing_3/PartitionedCallPartitionedCallnormalization_3/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_resizing_3_layer_call_and_return_conditional_losses_11154262
resizing_3/PartitionedCallй
%random_flip_3/StatefulPartitionedCallStatefulPartitionedCall#resizing_3/PartitionedCall:output:0random_flip_3_1115823*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11157902'
%random_flip_3/StatefulPartitionedCallп
)random_rotation_3/StatefulPartitionedCallStatefulPartitionedCall.random_flip_3/StatefulPartitionedCall:output:0random_rotation_3_1115826*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_11157192+
)random_rotation_3/StatefulPartitionedCall╠
%random_zoom_3/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_3/StatefulPartitionedCall:output:0random_zoom_3_1115829*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_11155882'
%random_zoom_3/StatefulPartitionedCallњ
IdentityIdentity.random_zoom_3/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identity╩
NoOpNoOp&^random_flip_3/StatefulPartitionedCall*^random_rotation_3/StatefulPartitionedCall&^random_zoom_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         ўќ::: : : 2N
%random_flip_3/StatefulPartitionedCall%random_flip_3/StatefulPartitionedCall2V
)random_rotation_3/StatefulPartitionedCall)random_rotation_3/StatefulPartitionedCall2N
%random_zoom_3/StatefulPartitionedCall%random_zoom_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
№
K
/__inference_random_flip_3_layer_call_fn_1117765

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11154322
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ─i:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
ш

/__inference_random_flip_3_layer_call_fn_1117772

inputs
unknown:	
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_random_flip_3_layer_call_and_return_conditional_losses_11157902
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ─i2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ─i: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ─i
 
_user_specified_nameinputs
Ф░
▀
E__inference_model_10_layer_call_and_return_conditional_losses_1116737

inputs+
'data_augmentation_normalization_3_sub_y,
(data_augmentation_normalization_3_sqrt_x<
*dense_21_tensordot_readvariableop_resource:1@6
(dense_21_biasadd_readvariableop_resource:@G
4patch_encoder_7_embedding_3_embedding_lookup_1116682:	Ц@J
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
identityѕбdense_21/BiasAdd/ReadVariableOpб!dense_21/Tensordot/ReadVariableOpб/layer_normalization_12/batchnorm/ReadVariableOpб3layer_normalization_12/batchnorm/mul/ReadVariableOpб:multi_head_attention_6/attention_output/add/ReadVariableOpбDmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpб-multi_head_attention_6/key/add/ReadVariableOpб7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpб/multi_head_attention_6/query/add/ReadVariableOpб9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpб/multi_head_attention_6/value/add/ReadVariableOpб9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpб,patch_encoder_7/embedding_3/embedding_lookup┬
%data_augmentation/normalization_3/subSubinputs'data_augmentation_normalization_3_sub_y*
T0*1
_output_shapes
:         ўќ2'
%data_augmentation/normalization_3/sub│
&data_augmentation/normalization_3/SqrtSqrt(data_augmentation_normalization_3_sqrt_x*
T0*&
_output_shapes
:2(
&data_augmentation/normalization_3/SqrtЪ
+data_augmentation/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ћ┐о32-
+data_augmentation/normalization_3/Maximum/yЗ
)data_augmentation/normalization_3/MaximumMaximum*data_augmentation/normalization_3/Sqrt:y:04data_augmentation/normalization_3/Maximum/y:output:0*
T0*&
_output_shapes
:2+
)data_augmentation/normalization_3/Maximumэ
)data_augmentation/normalization_3/truedivRealDiv)data_augmentation/normalization_3/sub:z:0-data_augmentation/normalization_3/Maximum:z:0*
T0*1
_output_shapes
:         ўќ2+
)data_augmentation/normalization_3/truedivЦ
(data_augmentation/resizing_3/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"─   i   2*
(data_augmentation/resizing_3/resize/size▒
2data_augmentation/resizing_3/resize/ResizeBilinearResizeBilinear-data_augmentation/normalization_3/truediv:z:01data_augmentation/resizing_3/resize/size:output:0*
T0*0
_output_shapes
:         ─i*
half_pixel_centers(24
2data_augmentation/resizing_3/resize/ResizeBilinearц
patches_7/ExtractImagePatchesExtractImagePatchesCdata_augmentation/resizing_3/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:         1*
ksizes
*
paddingVALID*
rates
*
strides
2
patches_7/ExtractImagePatchesЄ
patches_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    ц  1   2
patches_7/Reshape/shape│
patches_7/ReshapeReshape'patches_7/ExtractImagePatches:patches:0 patches_7/Reshape/shape:output:0*
T0*,
_output_shapes
:         ц12
patches_7/Reshape▒
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
dense_21/Tensordot/axesЃ
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
dense_21/Tensordot/Shapeє
 dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/GatherV2/axis■
dense_21/Tensordot/GatherV2GatherV2!dense_21/Tensordot/Shape:output:0 dense_21/Tensordot/free:output:0)dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_21/Tensordot/GatherV2і
"dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_21/Tensordot/GatherV2_1/axisё
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
dense_21/Tensordot/Constц
dense_21/Tensordot/ProdProd$dense_21/Tensordot/GatherV2:output:0!dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prodѓ
dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_21/Tensordot/Const_1г
dense_21/Tensordot/Prod_1Prod&dense_21/Tensordot/GatherV2_1:output:0#dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_21/Tensordot/Prod_1ѓ
dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_21/Tensordot/concat/axisП
dense_21/Tensordot/concatConcatV2 dense_21/Tensordot/free:output:0 dense_21/Tensordot/axes:output:0'dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat░
dense_21/Tensordot/stackPack dense_21/Tensordot/Prod:output:0"dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/stack└
dense_21/Tensordot/transpose	Transposepatches_7/Reshape:output:0"dense_21/Tensordot/concat:output:0*
T0*,
_output_shapes
:         ц12
dense_21/Tensordot/transpose├
dense_21/Tensordot/ReshapeReshape dense_21/Tensordot/transpose:y:0!dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_21/Tensordot/Reshape┬
dense_21/Tensordot/MatMulMatMul#dense_21/Tensordot/Reshape:output:0)dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_21/Tensordot/MatMulѓ
dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_21/Tensordot/Const_2є
 dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_21/Tensordot/concat_1/axisЖ
dense_21/Tensordot/concat_1ConcatV2$dense_21/Tensordot/GatherV2:output:0#dense_21/Tensordot/Const_2:output:0)dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_21/Tensordot/concat_1х
dense_21/TensordotReshape#dense_21/Tensordot/MatMul:product:0$dense_21/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         ц@2
dense_21/TensordotД
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_21/BiasAdd/ReadVariableOpг
dense_21/BiasAddBiasAdddense_21/Tensordot:output:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ц@2
dense_21/BiasAddё
lambda_3/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2!
lambda_3/Mean/reduction_indicesЮ
lambda_3/MeanMeandense_21/BiasAdd:output:0(lambda_3/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
lambda_3/MeanЁ
lambda_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
lambda_3/Reshape/shapeъ
lambda_3/ReshapeReshapelambda_3/Mean:output:0lambda_3/Reshape/shape:output:0*
T0*+
_output_shapes
:         @2
lambda_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisм
concatenate_3/concatConcatV2lambda_3/Reshape:output:0dense_21/BiasAdd:output:0"concatenate_3/concat/axis:output:0*
N*
T0*,
_output_shapes
:         Ц@2
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
B :Ц2
patch_encoder_7/range/limit|
patch_encoder_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder_7/range/deltaк
patch_encoder_7/rangeRange$patch_encoder_7/range/start:output:0$patch_encoder_7/range/limit:output:0$patch_encoder_7/range/delta:output:0*
_output_shapes	
:Ц2
patch_encoder_7/rangeщ
,patch_encoder_7/embedding_3/embedding_lookupResourceGather4patch_encoder_7_embedding_3_embedding_lookup_1116682patch_encoder_7/range:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1116682*
_output_shapes
:	Ц@*
dtype02.
,patch_encoder_7/embedding_3/embedding_lookupм
5patch_encoder_7/embedding_3/embedding_lookup/IdentityIdentity5patch_encoder_7/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*G
_class=
;9loc:@patch_encoder_7/embedding_3/embedding_lookup/1116682*
_output_shapes
:	Ц@27
5patch_encoder_7/embedding_3/embedding_lookup/IdentityУ
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1Identity>patch_encoder_7/embedding_3/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Ц@29
7patch_encoder_7/embedding_3/embedding_lookup/Identity_1╦
patch_encoder_7/addAddV2concatenate_3/concat:output:0@patch_encoder_7/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         Ц@2
patch_encoder_7/addИ
5layer_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_12/moments/mean/reduction_indicesз
#layer_normalization_12/moments/meanMeanpatch_encoder_7/add:z:0>layer_normalization_12/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2%
#layer_normalization_12/moments/mean¤
+layer_normalization_12/moments/StopGradientStopGradient,layer_normalization_12/moments/mean:output:0*
T0*,
_output_shapes
:         Ц2-
+layer_normalization_12/moments/StopGradient 
0layer_normalization_12/moments/SquaredDifferenceSquaredDifferencepatch_encoder_7/add:z:04layer_normalization_12/moments/StopGradient:output:0*
T0*,
_output_shapes
:         Ц@22
0layer_normalization_12/moments/SquaredDifference└
9layer_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_12/moments/variance/reduction_indicesю
'layer_normalization_12/moments/varianceMean4layer_normalization_12/moments/SquaredDifference:z:0Blayer_normalization_12/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:         Ц*
	keep_dims(2)
'layer_normalization_12/moments/varianceЋ
&layer_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52(
&layer_normalization_12/batchnorm/add/y№
$layer_normalization_12/batchnorm/addAddV20layer_normalization_12/moments/variance:output:0/layer_normalization_12/batchnorm/add/y:output:0*
T0*,
_output_shapes
:         Ц2&
$layer_normalization_12/batchnorm/add║
&layer_normalization_12/batchnorm/RsqrtRsqrt(layer_normalization_12/batchnorm/add:z:0*
T0*,
_output_shapes
:         Ц2(
&layer_normalization_12/batchnorm/Rsqrtс
3layer_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_12/batchnorm/mul/ReadVariableOpз
$layer_normalization_12/batchnorm/mulMul*layer_normalization_12/batchnorm/Rsqrt:y:0;layer_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2&
$layer_normalization_12/batchnorm/mulЛ
&layer_normalization_12/batchnorm/mul_1Mulpatch_encoder_7/add:z:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2(
&layer_normalization_12/batchnorm/mul_1Т
&layer_normalization_12/batchnorm/mul_2Mul,layer_normalization_12/moments/mean:output:0(layer_normalization_12/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ц@2(
&layer_normalization_12/batchnorm/mul_2О
/layer_normalization_12/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_12_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_12/batchnorm/ReadVariableOp№
$layer_normalization_12/batchnorm/subSub7layer_normalization_12/batchnorm/ReadVariableOp:value:0*layer_normalization_12/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:         Ц@2&
$layer_normalization_12/batchnorm/subТ
&layer_normalization_12/batchnorm/add_1AddV2*layer_normalization_12/batchnorm/mul_1:z:0(layer_normalization_12/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ц@2(
&layer_normalization_12/batchnorm/add_1§
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/query/einsum/Einsum/ReadVariableOp▓
*multi_head_attention_6/query/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2,
*multi_head_attention_6/query/einsum/Einsum█
/multi_head_attention_6/query/add/ReadVariableOpReadVariableOp8multi_head_attention_6_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/query/add/ReadVariableOpШ
 multi_head_attention_6/query/addAddV23multi_head_attention_6/query/einsum/Einsum:output:07multi_head_attention_6/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2"
 multi_head_attention_6/query/addэ
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_6_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_6/key/einsum/Einsum/ReadVariableOpг
(multi_head_attention_6/key/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0?multi_head_attention_6/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2*
(multi_head_attention_6/key/einsum/EinsumН
-multi_head_attention_6/key/add/ReadVariableOpReadVariableOp6multi_head_attention_6_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_6/key/add/ReadVariableOpЬ
multi_head_attention_6/key/addAddV21multi_head_attention_6/key/einsum/Einsum:output:05multi_head_attention_6/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2 
multi_head_attention_6/key/add§
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_6_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_6/value/einsum/Einsum/ReadVariableOp▓
*multi_head_attention_6/value/einsum/EinsumEinsum*layer_normalization_12/batchnorm/add_1:z:0Amulti_head_attention_6/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:         Ц@*
equationabc,cde->abde2,
*multi_head_attention_6/value/einsum/Einsum█
/multi_head_attention_6/value/add/ReadVariableOpReadVariableOp8multi_head_attention_6_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_6/value/add/ReadVariableOpШ
 multi_head_attention_6/value/addAddV23multi_head_attention_6/value/einsum/Einsum:output:07multi_head_attention_6/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ц@2"
 multi_head_attention_6/value/addЂ
multi_head_attention_6/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_6/Mul/yК
multi_head_attention_6/MulMul$multi_head_attention_6/query/add:z:0%multi_head_attention_6/Mul/y:output:0*
T0*0
_output_shapes
:         Ц@2
multi_head_attention_6/Mul■
$multi_head_attention_6/einsum/EinsumEinsum"multi_head_attention_6/key/add:z:0multi_head_attention_6/Mul:z:0*
N*
T0*1
_output_shapes
:         ЦЦ*
equationaecd,abcd->acbe2&
$multi_head_attention_6/einsum/Einsumк
&multi_head_attention_6/softmax/SoftmaxSoftmax-multi_head_attention_6/einsum/Einsum:output:0*
T0*1
_output_shapes
:         ЦЦ2(
&multi_head_attention_6/softmax/Softmax╠
'multi_head_attention_6/dropout/IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:         ЦЦ2)
'multi_head_attention_6/dropout/IdentityЋ
&multi_head_attention_6/einsum_1/EinsumEinsum0multi_head_attention_6/dropout/Identity:output:0$multi_head_attention_6/value/add:z:0*
N*
T0*0
_output_shapes
:         Ц@*
equationacbe,aecd->abcd2(
&multi_head_attention_6/einsum_1/Einsumъ
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_6_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOpн
5multi_head_attention_6/attention_output/einsum/EinsumEinsum/multi_head_attention_6/einsum_1/Einsum:output:0Lmulti_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:         Ц@*
equationabcd,cde->abe27
5multi_head_attention_6/attention_output/einsum/EinsumЭ
:multi_head_attention_6/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_6_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_6/attention_output/add/ReadVariableOpъ
+multi_head_attention_6/attention_output/addAddV2>multi_head_attention_6/attention_output/einsum/Einsum:output:0Bmulti_head_attention_6/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ц@2-
+multi_head_attention_6/attention_output/addЋ
IdentityIdentity0multi_head_attention_6/softmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:         ЦЦ2

Identityш
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp"^dense_21/Tensordot/ReadVariableOp0^layer_normalization_12/batchnorm/ReadVariableOp4^layer_normalization_12/batchnorm/mul/ReadVariableOp;^multi_head_attention_6/attention_output/add/ReadVariableOpE^multi_head_attention_6/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_6/key/add/ReadVariableOp8^multi_head_attention_6/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/query/add/ReadVariableOp:^multi_head_attention_6/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_6/value/add/ReadVariableOp:^multi_head_attention_6/value/einsum/Einsum/ReadVariableOp-^patch_encoder_7/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:         ўќ::: : : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2F
!dense_21/Tensordot/ReadVariableOp!dense_21/Tensordot/ReadVariableOp2b
/layer_normalization_12/batchnorm/ReadVariableOp/layer_normalization_12/batchnorm/ReadVariableOp2j
3layer_normalization_12/batchnorm/mul/ReadVariableOp3layer_normalization_12/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_6/attention_output/add/ReadVariableOp:multi_head_attention_6/attention_output/add/ReadVariableOp2ї
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
:         ўќ
 
_user_specified_nameinputs:,(
&
_output_shapes
::,(
&
_output_shapes
:
п
z
3__inference_data_augmentation_layer_call_fn_1115454
normalization_3_input
unknown
	unknown_0
identity§
PartitionedCallPartitionedCallnormalization_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ─i* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_data_augmentation_layer_call_and_return_conditional_losses_11154472
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ─i2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         ўќ:::h d
1
_output_shapes
:         ўќ
/
_user_specified_namenormalization_3_input:,(
&
_output_shapes
::,(
&
_output_shapes
:
§
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
concat/axisё
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*,
_output_shapes
:         Ц@2
concath
IdentityIdentityconcat:output:0*
T0*,
_output_shapes
:         Ц@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         @:         ц@:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ц@
 
_user_specified_nameinputs
І
Ќ
*__inference_dense_21_layer_call_fn_1117501

inputs
unknown:1@
	unknown_0:@
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ц@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_11159482
StatefulPartitionedCallђ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ц@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ц1: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ц1
 
_user_specified_nameinputs
№
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
:         @2
Means
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       @   2
Reshape/shapez
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         @2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ц@:T P
,
_output_shapes
:         ц@
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*═
serving_default╣
E
input_4:
serving_default_input_4:0         ўќT
multi_head_attention_6:
StatefulPartitionedCall:0         ЦЦtensorflow/serving/predict:Н▒
╬
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
№_default_save_signature
­__call__
+ы&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
Є
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
Ы__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_sequential
Д
	variables
regularization_losses
trainable_variables
	keras_api
З__call__
+ш&call_and_return_all_conditional_losses"
_tf_keras_layer
й

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
Ш__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
"	variables
#regularization_losses
$trainable_variables
%	keras_api
Э__call__
+щ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
&	variables
'regularization_losses
(trainable_variables
)	keras_api
Щ__call__
+ч&call_and_return_all_conditional_losses"
_tf_keras_layer
┐
*position_embedding
+	variables
,regularization_losses
-trainable_variables
.	keras_api
Ч__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
к
/axis
	0gamma
1beta
2	variables
3regularization_losses
4trainable_variables
5	keras_api
■__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
љ
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
ђ__call__
+Ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
ќ
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
╬
Llayer_regularization_losses
Mlayer_metrics

Nlayers

	variables
regularization_losses
trainable_variables
Onon_trainable_variables
Pmetrics
­__call__
№_default_save_signature
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
-
ѓserving_default"
signature_map
н
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
Ѓ_adapt_function"
_tf_keras_layer
Д
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
▒
Z_rng
[	variables
\regularization_losses
]trainable_variables
^	keras_api
є__call__
+Є&call_and_return_all_conditional_losses"
_tf_keras_layer
▒
__rng
`	variables
aregularization_losses
btrainable_variables
c	keras_api
ѕ__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
▒
d_rng
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
і__call__
+І&call_and_return_all_conditional_losses"
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
░
ilayer_regularization_losses
jlayer_metrics

klayers
	variables
regularization_losses
trainable_variables
lnon_trainable_variables
mmetrics
Ы__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
nlayer_regularization_losses
olayer_metrics

players
	variables
regularization_losses
trainable_variables
qnon_trainable_variables
rmetrics
З__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
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
░
slayer_regularization_losses
tlayer_metrics

ulayers
	variables
regularization_losses
 trainable_variables
vnon_trainable_variables
wmetrics
Ш__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
xlayer_regularization_losses
ylayer_metrics

zlayers
"	variables
#regularization_losses
$trainable_variables
{non_trainable_variables
|metrics
Э__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
}layer_regularization_losses
~layer_metrics

layers
&	variables
'regularization_losses
(trainable_variables
ђnon_trainable_variables
Ђmetrics
Щ__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
╗
C
embeddings
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
Ё	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
х
 єlayer_regularization_losses
Єlayer_metrics
ѕlayers
+	variables
,regularization_losses
-trainable_variables
Ѕnon_trainable_variables
іmetrics
Ч__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
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
х
 Іlayer_regularization_losses
їlayer_metrics
Їlayers
2	variables
3regularization_losses
4trainable_variables
јnon_trainable_variables
Јmetrics
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
З
љpartial_output_shape
Љfull_output_shape

Dkernel
Ebias
њ	variables
Њregularization_losses
ћtrainable_variables
Ћ	keras_api
ј__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_layer
З
ќpartial_output_shape
Ќfull_output_shape

Fkernel
Gbias
ў	variables
Ўregularization_losses
џtrainable_variables
Џ	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
З
юpartial_output_shape
Юfull_output_shape

Hkernel
Ibias
ъ	variables
Ъregularization_losses
аtrainable_variables
А	keras_api
њ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
б	variables
Бregularization_losses
цtrainable_variables
Ц	keras_api
ћ__call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
д	variables
Дregularization_losses
еtrainable_variables
Е	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
З
фpartial_output_shape
Фfull_output_shape

Jkernel
Kbias
г	variables
Гregularization_losses
«trainable_variables
»	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses"
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
х
 ░layer_regularization_losses
▒layer_metrics
▓layers
<	variables
=regularization_losses
>trainable_variables
│non_trainable_variables
┤metrics
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
:2mean
:2variance
:	 2count
9:7	Ц@2&patch_encoder_7/embedding_3/embeddings
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
х
 хlayer_regularization_losses
Хlayer_metrics
иlayers
V	variables
Wregularization_losses
Xtrainable_variables
Иnon_trainable_variables
╣metrics
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
/
║
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 ╗layer_regularization_losses
╝layer_metrics
йlayers
[	variables
\regularization_losses
]trainable_variables
Йnon_trainable_variables
┐metrics
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
/
└
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 ┴layer_regularization_losses
┬layer_metrics
├layers
`	variables
aregularization_losses
btrainable_variables
─non_trainable_variables
┼metrics
ѕ__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
/
к
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Кlayer_regularization_losses
╚layer_metrics
╔layers
e	variables
fregularization_losses
gtrainable_variables
╩non_trainable_variables
╦metrics
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
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
И
 ╠layer_regularization_losses
═layer_metrics
╬layers
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
¤non_trainable_variables
лmetrics
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
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
И
 Лlayer_regularization_losses
мlayer_metrics
Мlayers
њ	variables
Њregularization_losses
ћtrainable_variables
нnon_trainable_variables
Нmetrics
ј__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
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
И
 оlayer_regularization_losses
Оlayer_metrics
пlayers
ў	variables
Ўregularization_losses
џtrainable_variables
┘non_trainable_variables
┌metrics
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
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
И
 █layer_regularization_losses
▄layer_metrics
Пlayers
ъ	variables
Ъregularization_losses
аtrainable_variables
яnon_trainable_variables
▀metrics
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Яlayer_regularization_losses
рlayer_metrics
Рlayers
б	variables
Бregularization_losses
цtrainable_variables
сnon_trainable_variables
Сmetrics
ћ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 тlayer_regularization_losses
Тlayer_metrics
уlayers
д	variables
Дregularization_losses
еtrainable_variables
Уnon_trainable_variables
жmetrics
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
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
И
 Жlayer_regularization_losses
вlayer_metrics
Вlayers
г	variables
Гregularization_losses
«trainable_variables
ьnon_trainable_variables
Ьmetrics
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
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
═B╩
"__inference__wrapped_model_1115406input_4"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
*__inference_model_10_layer_call_fn_1116108
*__inference_model_10_layer_call_fn_1116590
*__inference_model_10_layer_call_fn_1116631
*__inference_model_10_layer_call_fn_1116426└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Р2▀
E__inference_model_10_layer_call_and_return_conditional_losses_1116737
E__inference_model_10_layer_call_and_return_conditional_losses_1117140
E__inference_model_10_layer_call_and_return_conditional_losses_1116469
E__inference_model_10_layer_call_and_return_conditional_losses_1116518└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
џ2Ќ
3__inference_data_augmentation_layer_call_fn_1115454
3__inference_data_augmentation_layer_call_fn_1117149
3__inference_data_augmentation_layer_call_fn_1117164
3__inference_data_augmentation_layer_call_fn_1115861└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117177
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117480
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115876
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115897└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Н2м
+__inference_patches_7_layer_call_fn_1117485б
Ў▓Ћ
FullArgSpec
argsџ
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_patches_7_layer_call_and_return_conditional_losses_1117492б
Ў▓Ћ
FullArgSpec
argsџ
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_dense_21_layer_call_fn_1117501б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_dense_21_layer_call_and_return_conditional_losses_1117531б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъ2Џ
*__inference_lambda_3_layer_call_fn_1117536
*__inference_lambda_3_layer_call_fn_1117541└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
н2Л
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117549
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117557└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┘2о
/__inference_concatenate_3_layer_call_fn_1117563б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1117570б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▀2▄
1__inference_patch_encoder_7_layer_call_fn_1117577д
Ю▓Ў
FullArgSpec!
argsџ
jself
j
projection
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Щ2э
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1117591д
Ю▓Ў
FullArgSpec!
argsџ
jself
j
projection
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Р2▀
8__inference_layer_normalization_12_layer_call_fn_1117600б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§2Щ
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1117622б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
8__inference_multi_head_attention_6_layer_call_fn_1117646
8__inference_multi_head_attention_6_layer_call_fn_1117670Ч
з▓№
FullArgSpece
args]џZ
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
defaultsџ

 

 
p 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
г2Е
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117706
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117749Ч
з▓№
FullArgSpece
args]џZ
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
defaultsџ

 

 
p 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠B╔
%__inference_signature_wrapper_1116555input_4"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
└2й
__inference_adapt_step_1113455џ
Њ▓Ј
FullArgSpec
argsџ

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
о2М
,__inference_resizing_3_layer_call_fn_1117754б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_resizing_3_layer_call_and_return_conditional_losses_1117760б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю2Ў
/__inference_random_flip_3_layer_call_fn_1117765
/__inference_random_flip_3_layer_call_fn_1117772┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117776
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117834┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ц2А
3__inference_random_rotation_3_layer_call_fn_1117839
3__inference_random_rotation_3_layer_call_fn_1117846┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117850
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117968┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ю2Ў
/__inference_random_zoom_3_layer_call_fn_1117973
/__inference_random_zoom_3_layer_call_fn_1117980┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1117984
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118110┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
	J
Const
J	
Const_1Л
"__inference__wrapped_model_1115406фџЏC01DEFGHIJK:б7
0б-
+і(
input_4         ўќ
ф "YфV
T
multi_head_attention_6:і7
multi_head_attention_6         ЦЦx
__inference_adapt_step_1113455VB@AKбH
Aб>
<њ9'б$
"і         ўќIteratorSpec
ф "
 Я
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1117570Љcб`
YбV
TџQ
&і#
inputs/0         @
'і$
inputs/1         ц@
ф "*б'
 і
0         Ц@
џ И
/__inference_concatenate_3_layer_call_fn_1117563ёcб`
YбV
TџQ
&і#
inputs/0         @
'і$
inputs/1         ц@
ф "і         Ц@█
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115876ѕџЏPбM
FбC
9і6
normalization_3_input         ўќ
p 

 
ф ".б+
$і!
0         ─i
џ р
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1115897ј
џЏ║└кPбM
FбC
9і6
normalization_3_input         ўќ
p

 
ф ".б+
$і!
0         ─i
џ ╦
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117177yџЏAб>
7б4
*і'
inputs         ўќ
p 

 
ф ".б+
$і!
0         ─i
џ Л
N__inference_data_augmentation_layer_call_and_return_conditional_losses_1117480
џЏ║└кAб>
7б4
*і'
inputs         ўќ
p

 
ф ".б+
$і!
0         ─i
џ ▓
3__inference_data_augmentation_layer_call_fn_1115454{џЏPбM
FбC
9і6
normalization_3_input         ўќ
p 

 
ф "!і         ─i╣
3__inference_data_augmentation_layer_call_fn_1115861Ђ
џЏ║└кPбM
FбC
9і6
normalization_3_input         ўќ
p

 
ф "!і         ─iБ
3__inference_data_augmentation_layer_call_fn_1117149lџЏAб>
7б4
*і'
inputs         ўќ
p 

 
ф "!і         ─iЕ
3__inference_data_augmentation_layer_call_fn_1117164r
џЏ║└кAб>
7б4
*і'
inputs         ўќ
p

 
ф "!і         ─i»
E__inference_dense_21_layer_call_and_return_conditional_losses_1117531f4б1
*б'
%і"
inputs         ц1
ф "*б'
 і
0         ц@
џ Є
*__inference_dense_21_layer_call_fn_1117501Y4б1
*б'
%і"
inputs         ц1
ф "і         ц@▓
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117549i<б9
2б/
%і"
inputs         ц@

 
p 
ф ")б&
і
0         @
џ ▓
E__inference_lambda_3_layer_call_and_return_conditional_losses_1117557i<б9
2б/
%і"
inputs         ц@

 
p
ф ")б&
і
0         @
џ і
*__inference_lambda_3_layer_call_fn_1117536\<б9
2б/
%і"
inputs         ц@

 
p 
ф "і         @і
*__inference_lambda_3_layer_call_fn_1117541\<б9
2б/
%і"
inputs         ц@

 
p
ф "і         @й
S__inference_layer_normalization_12_layer_call_and_return_conditional_losses_1117622f014б1
*б'
%і"
inputs         Ц@
ф "*б'
 і
0         Ц@
џ Ћ
8__inference_layer_normalization_12_layer_call_fn_1117600Y014б1
*б'
%і"
inputs         Ц@
ф "і         Ц@м
E__inference_model_10_layer_call_and_return_conditional_losses_1116469ѕџЏC01DEFGHIJKBб?
8б5
+і(
input_4         ўќ
p 

 
ф "/б,
%і"
0         ЦЦ
џ п
E__inference_model_10_layer_call_and_return_conditional_losses_1116518јџЏ║└кC01DEFGHIJKBб?
8б5
+і(
input_4         ўќ
p

 
ф "/б,
%і"
0         ЦЦ
џ Л
E__inference_model_10_layer_call_and_return_conditional_losses_1116737ЄџЏC01DEFGHIJKAб>
7б4
*і'
inputs         ўќ
p 

 
ф "/б,
%і"
0         ЦЦ
џ О
E__inference_model_10_layer_call_and_return_conditional_losses_1117140ЇџЏ║└кC01DEFGHIJKAб>
7б4
*і'
inputs         ўќ
p

 
ф "/б,
%і"
0         ЦЦ
џ Е
*__inference_model_10_layer_call_fn_1116108{џЏC01DEFGHIJKBб?
8б5
+і(
input_4         ўќ
p 

 
ф ""і         ЦЦ░
*__inference_model_10_layer_call_fn_1116426ЂџЏ║└кC01DEFGHIJKBб?
8б5
+і(
input_4         ўќ
p

 
ф ""і         ЦЦе
*__inference_model_10_layer_call_fn_1116590zџЏC01DEFGHIJKAб>
7б4
*і'
inputs         ўќ
p 

 
ф ""і         ЦЦ»
*__inference_model_10_layer_call_fn_1116631ђџЏ║└кC01DEFGHIJKAб>
7б4
*і'
inputs         ўќ
p

 
ф ""і         ЦЦЕ
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117706ЛDEFGHIJKiбf
_б\
$і!
query         Ц@
$і!
value         Ц@

 

 
p
p 
ф "ZбW
PбM
"і
0/0         Ц@
'і$
0/1         ЦЦ
џ Е
S__inference_multi_head_attention_6_layer_call_and_return_conditional_losses_1117749ЛDEFGHIJKiбf
_б\
$і!
query         Ц@
$і!
value         Ц@

 

 
p
p
ф "ZбW
PбM
"і
0/0         Ц@
'і$
0/1         ЦЦ
џ ђ
8__inference_multi_head_attention_6_layer_call_fn_1117646├DEFGHIJKiбf
_б\
$і!
query         Ц@
$і!
value         Ц@

 

 
p
p 
ф "LбI
 і
0         Ц@
%і"
1         ЦЦђ
8__inference_multi_head_attention_6_layer_call_fn_1117670├DEFGHIJKiбf
_б\
$і!
query         Ц@
$і!
value         Ц@

 

 
p
p
ф "LбI
 і
0         Ц@
%і"
1         ЦЦ╣
L__inference_patch_encoder_7_layer_call_and_return_conditional_losses_1117591iC8б5
.б+
)і&

projection         Ц@
ф "*б'
 і
0         Ц@
џ Љ
1__inference_patch_encoder_7_layer_call_fn_1117577\C8б5
.б+
)і&

projection         Ц@
ф "і         Ц@░
F__inference_patches_7_layer_call_and_return_conditional_losses_1117492f8б5
.б+
)і&
images         ─i
ф "*б'
 і
0         ц1
џ ѕ
+__inference_patches_7_layer_call_fn_1117485Y8б5
.б+
)і&
images         ─i
ф "і         ц1╝
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117776n<б9
2б/
)і&
inputs         ─i
p 
ф ".б+
$і!
0         ─i
џ └
J__inference_random_flip_3_layer_call_and_return_conditional_losses_1117834r║<б9
2б/
)і&
inputs         ─i
p
ф ".б+
$і!
0         ─i
џ ћ
/__inference_random_flip_3_layer_call_fn_1117765a<б9
2б/
)і&
inputs         ─i
p 
ф "!і         ─iў
/__inference_random_flip_3_layer_call_fn_1117772e║<б9
2б/
)і&
inputs         ─i
p
ф "!і         ─i└
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117850n<б9
2б/
)і&
inputs         ─i
p 
ф ".б+
$і!
0         ─i
џ ─
N__inference_random_rotation_3_layer_call_and_return_conditional_losses_1117968r└<б9
2б/
)і&
inputs         ─i
p
ф ".б+
$і!
0         ─i
џ ў
3__inference_random_rotation_3_layer_call_fn_1117839a<б9
2б/
)і&
inputs         ─i
p 
ф "!і         ─iю
3__inference_random_rotation_3_layer_call_fn_1117846e└<б9
2б/
)і&
inputs         ─i
p
ф "!і         ─i╝
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1117984n<б9
2б/
)і&
inputs         ─i
p 
ф ".б+
$і!
0         ─i
џ └
J__inference_random_zoom_3_layer_call_and_return_conditional_losses_1118110rк<б9
2б/
)і&
inputs         ─i
p
ф ".б+
$і!
0         ─i
џ ћ
/__inference_random_zoom_3_layer_call_fn_1117973a<б9
2б/
)і&
inputs         ─i
p 
ф "!і         ─iў
/__inference_random_zoom_3_layer_call_fn_1117980eк<б9
2б/
)і&
inputs         ─i
p
ф "!і         ─iХ
G__inference_resizing_3_layer_call_and_return_conditional_losses_1117760k9б6
/б,
*і'
inputs         ўќ
ф ".б+
$і!
0         ─i
џ ј
,__inference_resizing_3_layer_call_fn_1117754^9б6
/б,
*і'
inputs         ўќ
ф "!і         ─i▀
%__inference_signature_wrapper_1116555хџЏC01DEFGHIJKEбB
б 
;ф8
6
input_4+і(
input_4         ўќ"YфV
T
multi_head_attention_6:і7
multi_head_attention_6         ЦЦ