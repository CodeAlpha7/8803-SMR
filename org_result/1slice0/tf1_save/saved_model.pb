Ű
%ŕ$
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
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
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.52v1.15.4-39-g3db52be7be8íś
n
PlaceholderPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Placeholder_1Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
h
Placeholder_3Placeholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
Placeholder_4Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
ą
7main/pi/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*'
_class
loc:@main/pi/dense/kernel
¤
6main/pi/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*'
_class
loc:@main/pi/dense/kernel*
valueB
 *    
Ś
8main/pi/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *(?*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
: 

Amain/pi/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7main/pi/dense/kernel/Initializer/truncated_normal/shape*
T0*
dtype0*'
_class
loc:@main/pi/dense/kernel*
seedŃÂń*
_output_shapes
:	*
seed2

5main/pi/dense/kernel/Initializer/truncated_normal/mulMulAmain/pi/dense/kernel/Initializer/truncated_normal/TruncatedNormal8main/pi/dense/kernel/Initializer/truncated_normal/stddev*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	
ú
1main/pi/dense/kernel/Initializer/truncated_normalAdd5main/pi/dense/kernel/Initializer/truncated_normal/mul6main/pi/dense/kernel/Initializer/truncated_normal/mean*
_output_shapes
:	*'
_class
loc:@main/pi/dense/kernel*
T0
ł
main/pi/dense/kernel
VariableV2*
shape:	*
	container *
dtype0*'
_class
loc:@main/pi/dense/kernel*
shared_name *
_output_shapes
:	
ę
main/pi/dense/kernel/AssignAssignmain/pi/dense/kernel1main/pi/dense/kernel/Initializer/truncated_normal*
validate_shape(*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	*
use_locking(

main/pi/dense/kernel/readIdentitymain/pi/dense/kernel*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	

$main/pi/dense/bias/Initializer/zerosConst*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:*
valueB*    *
dtype0
§
main/pi/dense/bias
VariableV2*
shape:*
shared_name *%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:*
	container *
dtype0
Ó
main/pi/dense/bias/AssignAssignmain/pi/dense/bias$main/pi/dense/bias/Initializer/zeros*
use_locking(*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:*
validate_shape(*
T0

main/pi/dense/bias/readIdentitymain/pi/dense/bias*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:

main/pi/dense/MatMulMatMulPlaceholdermain/pi/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_b( 

main/pi/dense/BiasAddBiasAddmain/pi/dense/MatMulmain/pi/dense/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
d
main/pi/dense/ReluRelumain/pi/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
9main/pi/dense_1/kernel/Initializer/truncated_normal/shapeConst*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0
¨
8main/pi/dense_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *)
_class
loc:@main/pi/dense_1/kernel*
dtype0
Ş
:main/pi/dense_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *Eń>*
_output_shapes
: *)
_class
loc:@main/pi/dense_1/kernel*
dtype0

Cmain/pi/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9main/pi/dense_1/kernel/Initializer/truncated_normal/shape*)
_class
loc:@main/pi/dense_1/kernel*
seedŃÂń*
T0*
dtype0* 
_output_shapes
:
*
seed2

7main/pi/dense_1/kernel/Initializer/truncated_normal/mulMulCmain/pi/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal:main/pi/dense_1/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*)
_class
loc:@main/pi/dense_1/kernel

3main/pi/dense_1/kernel/Initializer/truncated_normalAdd7main/pi/dense_1/kernel/Initializer/truncated_normal/mul8main/pi/dense_1/kernel/Initializer/truncated_normal/mean*
T0*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:

š
main/pi/dense_1/kernel
VariableV2*
dtype0*
shared_name *)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
*
	container *
shape:

ó
main/pi/dense_1/kernel/AssignAssignmain/pi/dense_1/kernel3main/pi/dense_1/kernel/Initializer/truncated_normal*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0

main/pi/dense_1/kernel/readIdentitymain/pi/dense_1/kernel*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:


&main/pi/dense_1/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*'
_class
loc:@main/pi/dense_1/bias
Ť
main/pi/dense_1/bias
VariableV2*'
_class
loc:@main/pi/dense_1/bias*
dtype0*
_output_shapes	
:*
shared_name *
	container *
shape:
Ű
main/pi/dense_1/bias/AssignAssignmain/pi/dense_1/bias&main/pi/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*'
_class
loc:@main/pi/dense_1/bias

main/pi/dense_1/bias/readIdentitymain/pi/dense_1/bias*
_output_shapes	
:*
T0*'
_class
loc:@main/pi/dense_1/bias
Ş
main/pi/dense_1/MatMulMatMulmain/pi/dense/Relumain/pi/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

main/pi/dense_1/BiasAddBiasAddmain/pi/dense_1/MatMulmain/pi/dense_1/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
h
main/pi/dense_1/ReluRelumain/pi/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ł
7main/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5main/pi/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
: *
valueB
 *Č~Yž
Ľ
5main/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *Č~Y>*)
_class
loc:@main/pi/dense_2/kernel

?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_2/kernel/Initializer/random_uniform/shape*
T0*
_output_shapes
:	*
seed2(*
seedŃÂń*
dtype0*)
_class
loc:@main/pi/dense_2/kernel
ö
5main/pi/dense_2/kernel/Initializer/random_uniform/subSub5main/pi/dense_2/kernel/Initializer/random_uniform/max5main/pi/dense_2/kernel/Initializer/random_uniform/min*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
: *
T0

5main/pi/dense_2/kernel/Initializer/random_uniform/mulMul?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*)
_class
loc:@main/pi/dense_2/kernel
ű
1main/pi/dense_2/kernel/Initializer/random_uniformAdd5main/pi/dense_2/kernel/Initializer/random_uniform/mul5main/pi/dense_2/kernel/Initializer/random_uniform/min*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	*
T0
ˇ
main/pi/dense_2/kernel
VariableV2*
	container *
shared_name *
dtype0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	*
shape:	
đ
main/pi/dense_2/kernel/AssignAssignmain/pi/dense_2/kernel1main/pi/dense_2/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel

main/pi/dense_2/kernel/readIdentitymain/pi/dense_2/kernel*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	

&main/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias*
dtype0
Š
main/pi/dense_2/bias
VariableV2*
	container *
shape:*
_output_shapes
:*
dtype0*
shared_name *'
_class
loc:@main/pi/dense_2/bias
Ú
main/pi/dense_2/bias/AssignAssignmain/pi/dense_2/bias&main/pi/dense_2/bias/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:

main/pi/dense_2/bias/readIdentitymain/pi/dense_2/bias*'
_class
loc:@main/pi/dense_2/bias*
T0*
_output_shapes
:
Ť
main/pi/dense_2/MatMulMatMulmain/pi/dense_1/Relumain/pi/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0*
transpose_a( 

main/pi/dense_2/BiasAddBiasAddmain/pi/dense_2/MatMulmain/pi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
m
main/pi/dense_2/SigmoidSigmoidmain/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
main/pi/mul/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
l
main/pi/mulMulmain/pi/mul/xmain/pi/dense_2/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
main/q1/concat/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0

main/q1/concatConcatV2PlaceholderPlaceholder_1main/q1/concat/axis*
T0*
N*

Tidx0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
7main/q1/dense/kernel/Initializer/truncated_normal/shapeConst*'
_class
loc:@main/q1/dense/kernel*
dtype0*
valueB"      *
_output_shapes
:
¤
6main/q1/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0*'
_class
loc:@main/q1/dense/kernel
Ś
8main/q1/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *'
_class
loc:@main/q1/dense/kernel*
valueB
 *ëř>

Amain/q1/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7main/q1/dense/kernel/Initializer/truncated_normal/shape*
seedŃÂń*'
_class
loc:@main/q1/dense/kernel*
dtype0*
T0*
_output_shapes
:	*
seed2=

5main/q1/dense/kernel/Initializer/truncated_normal/mulMulAmain/q1/dense/kernel/Initializer/truncated_normal/TruncatedNormal8main/q1/dense/kernel/Initializer/truncated_normal/stddev*'
_class
loc:@main/q1/dense/kernel*
T0*
_output_shapes
:	
ú
1main/q1/dense/kernel/Initializer/truncated_normalAdd5main/q1/dense/kernel/Initializer/truncated_normal/mul6main/q1/dense/kernel/Initializer/truncated_normal/mean*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	
ł
main/q1/dense/kernel
VariableV2*'
_class
loc:@main/q1/dense/kernel*
	container *
dtype0*
shared_name *
shape:	*
_output_shapes
:	
ę
main/q1/dense/kernel/AssignAssignmain/q1/dense/kernel1main/q1/dense/kernel/Initializer/truncated_normal*
_output_shapes
:	*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
T0*
use_locking(

main/q1/dense/kernel/readIdentitymain/q1/dense/kernel*
_output_shapes
:	*
T0*'
_class
loc:@main/q1/dense/kernel

$main/q1/dense/bias/Initializer/zerosConst*
dtype0*%
_class
loc:@main/q1/dense/bias*
_output_shapes	
:*
valueB*    
§
main/q1/dense/bias
VariableV2*
shape:*
shared_name *%
_class
loc:@main/q1/dense/bias*
_output_shapes	
:*
	container *
dtype0
Ó
main/q1/dense/bias/AssignAssignmain/q1/dense/bias$main/q1/dense/bias/Initializer/zeros*
_output_shapes	
:*
T0*
use_locking(*%
_class
loc:@main/q1/dense/bias*
validate_shape(

main/q1/dense/bias/readIdentitymain/q1/dense/bias*
_output_shapes	
:*
T0*%
_class
loc:@main/q1/dense/bias
˘
main/q1/dense/MatMulMatMulmain/q1/concatmain/q1/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_b( 

main/q1/dense/BiasAddBiasAddmain/q1/dense/MatMulmain/q1/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
d
main/q1/dense/ReluRelumain/q1/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
9main/q1/dense_1/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"      *
_output_shapes
:*)
_class
loc:@main/q1/dense_1/kernel
¨
8main/q1/dense_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
valueB
 *    
Ş
:main/q1/dense_1/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *)
_class
loc:@main/q1/dense_1/kernel*
valueB
 *Eń>

Cmain/q1/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9main/q1/dense_1/kernel/Initializer/truncated_normal/shape*
seedŃÂń*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*
dtype0*
seed2M

7main/q1/dense_1/kernel/Initializer/truncated_normal/mulMulCmain/q1/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal:main/q1/dense_1/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*)
_class
loc:@main/q1/dense_1/kernel*
T0

3main/q1/dense_1/kernel/Initializer/truncated_normalAdd7main/q1/dense_1/kernel/Initializer/truncated_normal/mul8main/q1/dense_1/kernel/Initializer/truncated_normal/mean*)
_class
loc:@main/q1/dense_1/kernel*
T0* 
_output_shapes
:

š
main/q1/dense_1/kernel
VariableV2*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*
shape:
*
dtype0*
	container *
shared_name 
ó
main/q1/dense_1/kernel/AssignAssignmain/q1/dense_1/kernel3main/q1/dense_1/kernel/Initializer/truncated_normal*
validate_shape(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
use_locking(* 
_output_shapes
:


main/q1/dense_1/kernel/readIdentitymain/q1/dense_1/kernel*)
_class
loc:@main/q1/dense_1/kernel*
T0* 
_output_shapes
:


&main/q1/dense_1/bias/Initializer/zerosConst*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes	
:*
valueB*    *
dtype0
Ť
main/q1/dense_1/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *'
_class
loc:@main/q1/dense_1/bias*
_output_shapes	
:
Ű
main/q1/dense_1/bias/AssignAssignmain/q1/dense_1/bias&main/q1/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
validate_shape(*'
_class
loc:@main/q1/dense_1/bias*
T0*
use_locking(

main/q1/dense_1/bias/readIdentitymain/q1/dense_1/bias*
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias*
T0
Ş
main/q1/dense_1/MatMulMatMulmain/q1/dense/Relumain/q1/dense_1/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

main/q1/dense_1/BiasAddBiasAddmain/q1/dense_1/MatMulmain/q1/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
h
main/q1/dense_1/ReluRelumain/q1/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ł
7main/q1/dense_2/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@main/q1/dense_2/kernel*
valueB"      *
_output_shapes
:*
dtype0
Ľ
5main/q1/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
: *
valueB
 *n×\ž
Ľ
5main/q1/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *n×\>*
_output_shapes
: *)
_class
loc:@main/q1/dense_2/kernel*
dtype0

?main/q1/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q1/dense_2/kernel/Initializer/random_uniform/shape*
seedŃÂń*
seed2]*
_output_shapes
:	*)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
T0
ö
5main/q1/dense_2/kernel/Initializer/random_uniform/subSub5main/q1/dense_2/kernel/Initializer/random_uniform/max5main/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
: 

5main/q1/dense_2/kernel/Initializer/random_uniform/mulMul?main/q1/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/q1/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*)
_class
loc:@main/q1/dense_2/kernel*
T0
ű
1main/q1/dense_2/kernel/Initializer/random_uniformAdd5main/q1/dense_2/kernel/Initializer/random_uniform/mul5main/q1/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*)
_class
loc:@main/q1/dense_2/kernel
ˇ
main/q1/dense_2/kernel
VariableV2*
shape:	*
dtype0*)
_class
loc:@main/q1/dense_2/kernel*
	container *
_output_shapes
:	*
shared_name 
đ
main/q1/dense_2/kernel/AssignAssignmain/q1/dense_2/kernel1main/q1/dense_2/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*)
_class
loc:@main/q1/dense_2/kernel

main/q1/dense_2/kernel/readIdentitymain/q1/dense_2/kernel*)
_class
loc:@main/q1/dense_2/kernel*
T0*
_output_shapes
:	

&main/q1/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *'
_class
loc:@main/q1/dense_2/bias
Š
main/q1/dense_2/bias
VariableV2*
shape:*
shared_name *'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:*
dtype0*
	container 
Ú
main/q1/dense_2/bias/AssignAssignmain/q1/dense_2/bias&main/q1/dense_2/bias/Initializer/zeros*
T0*
_output_shapes
:*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
use_locking(

main/q1/dense_2/bias/readIdentitymain/q1/dense_2/bias*
_output_shapes
:*
T0*'
_class
loc:@main/q1/dense_2/bias
Ť
main/q1/dense_2/MatMulMatMulmain/q1/dense_1/Relumain/q1/dense_2/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙

main/q1/dense_2/BiasAddBiasAddmain/q1/dense_2/MatMulmain/q1/dense_2/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
x
main/q1/SqueezeSqueezemain/q1/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
^
main/q2/concat/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0

main/q2/concatConcatV2PlaceholderPlaceholder_1main/q2/concat/axis*
N*

Tidx0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
7main/q2/dense/kernel/Initializer/truncated_normal/shapeConst*'
_class
loc:@main/q2/dense/kernel*
valueB"      *
_output_shapes
:*
dtype0
¤
6main/q2/dense/kernel/Initializer/truncated_normal/meanConst*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
Ś
8main/q2/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*'
_class
loc:@main/q2/dense/kernel*
valueB
 *ëř>*
_output_shapes
: 

Amain/q2/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal7main/q2/dense/kernel/Initializer/truncated_normal/shape*
seedŃÂń*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
:	*
dtype0*
seed2p*
T0

5main/q2/dense/kernel/Initializer/truncated_normal/mulMulAmain/q2/dense/kernel/Initializer/truncated_normal/TruncatedNormal8main/q2/dense/kernel/Initializer/truncated_normal/stddev*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
:	*
T0
ú
1main/q2/dense/kernel/Initializer/truncated_normalAdd5main/q2/dense/kernel/Initializer/truncated_normal/mul6main/q2/dense/kernel/Initializer/truncated_normal/mean*'
_class
loc:@main/q2/dense/kernel*
T0*
_output_shapes
:	
ł
main/q2/dense/kernel
VariableV2*'
_class
loc:@main/q2/dense/kernel*
dtype0*
_output_shapes
:	*
	container *
shape:	*
shared_name 
ę
main/q2/dense/kernel/AssignAssignmain/q2/dense/kernel1main/q2/dense/kernel/Initializer/truncated_normal*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0*'
_class
loc:@main/q2/dense/kernel

main/q2/dense/kernel/readIdentitymain/q2/dense/kernel*
_output_shapes
:	*
T0*'
_class
loc:@main/q2/dense/kernel

$main/q2/dense/bias/Initializer/zerosConst*
_output_shapes	
:*%
_class
loc:@main/q2/dense/bias*
dtype0*
valueB*    
§
main/q2/dense/bias
VariableV2*
_output_shapes	
:*
dtype0*
	container *%
_class
loc:@main/q2/dense/bias*
shared_name *
shape:
Ó
main/q2/dense/bias/AssignAssignmain/q2/dense/bias$main/q2/dense/bias/Initializer/zeros*%
_class
loc:@main/q2/dense/bias*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0

main/q2/dense/bias/readIdentitymain/q2/dense/bias*
_output_shapes	
:*
T0*%
_class
loc:@main/q2/dense/bias
˘
main/q2/dense/MatMulMatMulmain/q2/concatmain/q2/dense/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 

main/q2/dense/BiasAddBiasAddmain/q2/dense/MatMulmain/q2/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
d
main/q2/dense/ReluRelumain/q2/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
9main/q2/dense_1/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *)
_class
loc:@main/q2/dense_1/kernel*
dtype0
¨
8main/q2/dense_1/kernel/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes
: 
Ş
:main/q2/dense_1/kernel/Initializer/truncated_normal/stddevConst*)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
valueB
 *Eń>*
_output_shapes
: 

Cmain/q2/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9main/q2/dense_1/kernel/Initializer/truncated_normal/shape*
dtype0*
seedŃÂń*)
_class
loc:@main/q2/dense_1/kernel*
T0* 
_output_shapes
:
*
seed2

7main/q2/dense_1/kernel/Initializer/truncated_normal/mulMulCmain/q2/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal:main/q2/dense_1/kernel/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:


3main/q2/dense_1/kernel/Initializer/truncated_normalAdd7main/q2/dense_1/kernel/Initializer/truncated_normal/mul8main/q2/dense_1/kernel/Initializer/truncated_normal/mean*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:

š
main/q2/dense_1/kernel
VariableV2*
dtype0*
shared_name * 
_output_shapes
:
*
shape:
*
	container *)
_class
loc:@main/q2/dense_1/kernel
ó
main/q2/dense_1/kernel/AssignAssignmain/q2/dense_1/kernel3main/q2/dense_1/kernel/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*)
_class
loc:@main/q2/dense_1/kernel

main/q2/dense_1/kernel/readIdentitymain/q2/dense_1/kernel*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:


&main/q2/dense_1/bias/Initializer/zerosConst*
dtype0*'
_class
loc:@main/q2/dense_1/bias*
valueB*    *
_output_shapes	
:
Ť
main/q2/dense_1/bias
VariableV2*
	container *
shape:*'
_class
loc:@main/q2/dense_1/bias*
dtype0*
_output_shapes	
:*
shared_name 
Ű
main/q2/dense_1/bias/AssignAssignmain/q2/dense_1/bias&main/q2/dense_1/bias/Initializer/zeros*
T0*
_output_shapes	
:*'
_class
loc:@main/q2/dense_1/bias*
use_locking(*
validate_shape(

main/q2/dense_1/bias/readIdentitymain/q2/dense_1/bias*
T0*
_output_shapes	
:*'
_class
loc:@main/q2/dense_1/bias
Ş
main/q2/dense_1/MatMulMatMulmain/q2/dense/Relumain/q2/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

main/q2/dense_1/BiasAddBiasAddmain/q2/dense_1/MatMulmain/q2/dense_1/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
h
main/q2/dense_1/ReluRelumain/q2/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7main/q2/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *)
_class
loc:@main/q2/dense_2/kernel*
dtype0
Ľ
5main/q2/dense_2/kernel/Initializer/random_uniform/minConst*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *n×\ž
Ľ
5main/q2/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *n×\>*)
_class
loc:@main/q2/dense_2/kernel

?main/q2/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q2/dense_2/kernel/Initializer/random_uniform/shape*)
_class
loc:@main/q2/dense_2/kernel*
T0*
seedŃÂń*
dtype0*
_output_shapes
:	*
seed2
ö
5main/q2/dense_2/kernel/Initializer/random_uniform/subSub5main/q2/dense_2/kernel/Initializer/random_uniform/max5main/q2/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *)
_class
loc:@main/q2/dense_2/kernel*
T0

5main/q2/dense_2/kernel/Initializer/random_uniform/mulMul?main/q2/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/q2/dense_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	
ű
1main/q2/dense_2/kernel/Initializer/random_uniformAdd5main/q2/dense_2/kernel/Initializer/random_uniform/mul5main/q2/dense_2/kernel/Initializer/random_uniform/min*)
_class
loc:@main/q2/dense_2/kernel*
T0*
_output_shapes
:	
ˇ
main/q2/dense_2/kernel
VariableV2*
shared_name *)
_class
loc:@main/q2/dense_2/kernel*
shape:	*
dtype0*
	container *
_output_shapes
:	
đ
main/q2/dense_2/kernel/AssignAssignmain/q2/dense_2/kernel1main/q2/dense_2/kernel/Initializer/random_uniform*
use_locking(*)
_class
loc:@main/q2/dense_2/kernel*
T0*
_output_shapes
:	*
validate_shape(

main/q2/dense_2/kernel/readIdentitymain/q2/dense_2/kernel*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	

&main/q2/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:*
dtype0
Š
main/q2/dense_2/bias
VariableV2*
	container *
dtype0*
_output_shapes
:*
shape:*
shared_name *'
_class
loc:@main/q2/dense_2/bias
Ú
main/q2/dense_2/bias/AssignAssignmain/q2/dense_2/bias&main/q2/dense_2/bias/Initializer/zeros*
T0*
_output_shapes
:*
validate_shape(*'
_class
loc:@main/q2/dense_2/bias*
use_locking(

main/q2/dense_2/bias/readIdentitymain/q2/dense_2/bias*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:*
T0
Ť
main/q2/dense_2/MatMulMatMulmain/q2/dense_1/Relumain/q2/dense_2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( 

main/q2/dense_2/BiasAddBiasAddmain/q2/dense_2/MatMulmain/q2/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
main/q2/SqueezeSqueezemain/q2/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
`
main/q1_1/concat/axisConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 

main/q1_1/concatConcatV2Placeholdermain/pi/mulmain/q1_1/concat/axis*

Tidx0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
N
Ś
main/q1_1/dense/MatMulMatMulmain/q1_1/concatmain/q1/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 

main/q1_1/dense/BiasAddBiasAddmain/q1_1/dense/MatMulmain/q1/dense/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
main/q1_1/dense/ReluRelumain/q1_1/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
main/q1_1/dense_1/MatMulMatMulmain/q1_1/dense/Relumain/q1/dense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
Ł
main/q1_1/dense_1/BiasAddBiasAddmain/q1_1/dense_1/MatMulmain/q1/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
l
main/q1_1/dense_1/ReluRelumain/q1_1/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
main/q1_1/dense_2/MatMulMatMulmain/q1_1/dense_1/Relumain/q1/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
˘
main/q1_1/dense_2/BiasAddBiasAddmain/q1_1/dense_2/MatMulmain/q1/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
main/q1_1/SqueezeSqueezemain/q1_1/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
9target/pi/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *
_output_shapes
:*)
_class
loc:@target/pi/dense/kernel*
dtype0
¨
8target/pi/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *)
_class
loc:@target/pi/dense/kernel
Ş
:target/pi/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *(?*
_output_shapes
: *)
_class
loc:@target/pi/dense/kernel

Ctarget/pi/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9target/pi/dense/kernel/Initializer/truncated_normal/shape*
dtype0*
T0*
seed2Ź*)
_class
loc:@target/pi/dense/kernel*
_output_shapes
:	*
seedŃÂń

7target/pi/dense/kernel/Initializer/truncated_normal/mulMulCtarget/pi/dense/kernel/Initializer/truncated_normal/TruncatedNormal:target/pi/dense/kernel/Initializer/truncated_normal/stddev*
_output_shapes
:	*)
_class
loc:@target/pi/dense/kernel*
T0

3target/pi/dense/kernel/Initializer/truncated_normalAdd7target/pi/dense/kernel/Initializer/truncated_normal/mul8target/pi/dense/kernel/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	*)
_class
loc:@target/pi/dense/kernel
ˇ
target/pi/dense/kernel
VariableV2*
_output_shapes
:	*
	container *
shared_name *)
_class
loc:@target/pi/dense/kernel*
dtype0*
shape:	
ň
target/pi/dense/kernel/AssignAssigntarget/pi/dense/kernel3target/pi/dense/kernel/Initializer/truncated_normal*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*)
_class
loc:@target/pi/dense/kernel

target/pi/dense/kernel/readIdentitytarget/pi/dense/kernel*
_output_shapes
:	*
T0*)
_class
loc:@target/pi/dense/kernel

&target/pi/dense/bias/Initializer/zerosConst*'
_class
loc:@target/pi/dense/bias*
dtype0*
_output_shapes	
:*
valueB*    
Ť
target/pi/dense/bias
VariableV2*
_output_shapes	
:*
shared_name *
	container *
shape:*'
_class
loc:@target/pi/dense/bias*
dtype0
Ű
target/pi/dense/bias/AssignAssigntarget/pi/dense/bias&target/pi/dense/bias/Initializer/zeros*
_output_shapes	
:*
T0*'
_class
loc:@target/pi/dense/bias*
use_locking(*
validate_shape(

target/pi/dense/bias/readIdentitytarget/pi/dense/bias*
T0*'
_class
loc:@target/pi/dense/bias*
_output_shapes	
:
Ľ
target/pi/dense/MatMulMatMulPlaceholder_2target/pi/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_b( 

target/pi/dense/BiasAddBiasAddtarget/pi/dense/MatMultarget/pi/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
h
target/pi/dense/ReluRelutarget/pi/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
;target/pi/dense_1/kernel/Initializer/truncated_normal/shapeConst*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
:*
valueB"      
Ź
:target/pi/dense_1/kernel/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes
: 
Ž
<target/pi/dense_1/kernel/Initializer/truncated_normal/stddevConst*+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes
: *
valueB
 *Eń>*
dtype0

Etarget/pi/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;target/pi/dense_1/kernel/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
seed2ź*
seedŃÂń* 
_output_shapes
:


9target/pi/dense_1/kernel/Initializer/truncated_normal/mulMulEtarget/pi/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal<target/pi/dense_1/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*+
_class!
loc:@target/pi/dense_1/kernel

5target/pi/dense_1/kernel/Initializer/truncated_normalAdd9target/pi/dense_1/kernel/Initializer/truncated_normal/mul:target/pi/dense_1/kernel/Initializer/truncated_normal/mean*
T0* 
_output_shapes
:
*+
_class!
loc:@target/pi/dense_1/kernel
˝
target/pi/dense_1/kernel
VariableV2*
dtype0*
shared_name *
shape:
*
	container *+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:

ű
target/pi/dense_1/kernel/AssignAssigntarget/pi/dense_1/kernel5target/pi/dense_1/kernel/Initializer/truncated_normal* 
_output_shapes
:
*+
_class!
loc:@target/pi/dense_1/kernel*
T0*
use_locking(*
validate_shape(

target/pi/dense_1/kernel/readIdentitytarget/pi/dense_1/kernel* 
_output_shapes
:
*+
_class!
loc:@target/pi/dense_1/kernel*
T0
˘
(target/pi/dense_1/bias/Initializer/zerosConst*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:*
valueB*    *
dtype0
Ż
target/pi/dense_1/bias
VariableV2*
shared_name *
	container *)
_class
loc:@target/pi/dense_1/bias*
shape:*
_output_shapes	
:*
dtype0
ă
target/pi/dense_1/bias/AssignAssigntarget/pi/dense_1/bias(target/pi/dense_1/bias/Initializer/zeros*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0

target/pi/dense_1/bias/readIdentitytarget/pi/dense_1/bias*
T0*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:
°
target/pi/dense_1/MatMulMatMultarget/pi/dense/Relutarget/pi/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 
Ľ
target/pi/dense_1/BiasAddBiasAddtarget/pi/dense_1/MatMultarget/pi/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
target/pi/dense_1/ReluRelutarget/pi/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
9target/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0*+
_class!
loc:@target/pi/dense_2/kernel
Š
7target/pi/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Č~Yž*+
_class!
loc:@target/pi/dense_2/kernel
Š
7target/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*+
_class!
loc:@target/pi/dense_2/kernel*
valueB
 *Č~Y>*
_output_shapes
: 

Atarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_2/kernel/Initializer/random_uniform/shape*
seed2Ě*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
seedŃÂń*
_output_shapes
:	*
T0
ţ
7target/pi/dense_2/kernel/Initializer/random_uniform/subSub7target/pi/dense_2/kernel/Initializer/random_uniform/max7target/pi/dense_2/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
_output_shapes
: 

7target/pi/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*+
_class!
loc:@target/pi/dense_2/kernel

3target/pi/dense_2/kernel/Initializer/random_uniformAdd7target/pi/dense_2/kernel/Initializer/random_uniform/mul7target/pi/dense_2/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	*
T0
ť
target/pi/dense_2/kernel
VariableV2*
shared_name *
dtype0*
shape:	*
_output_shapes
:	*+
_class!
loc:@target/pi/dense_2/kernel*
	container 
ř
target/pi/dense_2/kernel/AssignAssigntarget/pi/dense_2/kernel3target/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*
_output_shapes
:	*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(

target/pi/dense_2/kernel/readIdentitytarget/pi/dense_2/kernel*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	
 
(target/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*)
_class
loc:@target/pi/dense_2/bias*
dtype0
­
target/pi/dense_2/bias
VariableV2*
shared_name *)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
shape:*
	container *
dtype0
â
target/pi/dense_2/bias/AssignAssigntarget/pi/dense_2/bias(target/pi/dense_2/bias/Initializer/zeros*
validate_shape(*
T0*
_output_shapes
:*)
_class
loc:@target/pi/dense_2/bias*
use_locking(

target/pi/dense_2/bias/readIdentitytarget/pi/dense_2/bias*
T0*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:
ą
target/pi/dense_2/MatMulMatMultarget/pi/dense_1/Relutarget/pi/dense_2/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
target/pi/dense_2/BiasAddBiasAddtarget/pi/dense_2/MatMultarget/pi/dense_2/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
q
target/pi/dense_2/SigmoidSigmoidtarget/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
target/pi/mul/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
r
target/pi/mulMultarget/pi/mul/xtarget/pi/dense_2/Sigmoid*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
target/q1/concat/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

target/q1/concatConcatV2Placeholder_2Placeholder_1target/q1/concat/axis*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tidx0
ľ
9target/q1/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *
_output_shapes
:*)
_class
loc:@target/q1/dense/kernel*
dtype0
¨
8target/q1/dense/kernel/Initializer/truncated_normal/meanConst*)
_class
loc:@target/q1/dense/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
Ş
:target/q1/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *ëř>*
dtype0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes
: 

Ctarget/q1/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9target/q1/dense/kernel/Initializer/truncated_normal/shape*
seedŃÂń*
_output_shapes
:	*)
_class
loc:@target/q1/dense/kernel*
dtype0*
seed2á*
T0

7target/q1/dense/kernel/Initializer/truncated_normal/mulMulCtarget/q1/dense/kernel/Initializer/truncated_normal/TruncatedNormal:target/q1/dense/kernel/Initializer/truncated_normal/stddev*
_output_shapes
:	*)
_class
loc:@target/q1/dense/kernel*
T0

3target/q1/dense/kernel/Initializer/truncated_normalAdd7target/q1/dense/kernel/Initializer/truncated_normal/mul8target/q1/dense/kernel/Initializer/truncated_normal/mean*
T0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes
:	
ˇ
target/q1/dense/kernel
VariableV2*
	container *)
_class
loc:@target/q1/dense/kernel*
shared_name *
_output_shapes
:	*
dtype0*
shape:	
ň
target/q1/dense/kernel/AssignAssigntarget/q1/dense/kernel3target/q1/dense/kernel/Initializer/truncated_normal*
T0*
validate_shape(*)
_class
loc:@target/q1/dense/kernel*
use_locking(*
_output_shapes
:	

target/q1/dense/kernel/readIdentitytarget/q1/dense/kernel*
T0*
_output_shapes
:	*)
_class
loc:@target/q1/dense/kernel

&target/q1/dense/bias/Initializer/zerosConst*
_output_shapes	
:*'
_class
loc:@target/q1/dense/bias*
valueB*    *
dtype0
Ť
target/q1/dense/bias
VariableV2*
_output_shapes	
:*
dtype0*
shared_name *
shape:*'
_class
loc:@target/q1/dense/bias*
	container 
Ű
target/q1/dense/bias/AssignAssigntarget/q1/dense/bias&target/q1/dense/bias/Initializer/zeros*
use_locking(*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes	
:*
T0

target/q1/dense/bias/readIdentitytarget/q1/dense/bias*
_output_shapes	
:*'
_class
loc:@target/q1/dense/bias*
T0
¨
target/q1/dense/MatMulMatMultarget/q1/concattarget/q1/dense/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

target/q1/dense/BiasAddBiasAddtarget/q1/dense/MatMultarget/q1/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
h
target/q1/dense/ReluRelutarget/q1/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
;target/q1/dense_1/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*+
_class!
loc:@target/q1/dense_1/kernel*
dtype0*
valueB"      
Ź
:target/q1/dense_1/kernel/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *+
_class!
loc:@target/q1/dense_1/kernel*
_output_shapes
: 
Ž
<target/q1/dense_1/kernel/Initializer/truncated_normal/stddevConst*+
_class!
loc:@target/q1/dense_1/kernel*
valueB
 *Eń>*
_output_shapes
: *
dtype0

Etarget/q1/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;target/q1/dense_1/kernel/Initializer/truncated_normal/shape*+
_class!
loc:@target/q1/dense_1/kernel*
T0*
dtype0* 
_output_shapes
:
*
seedŃÂń*
seed2ń

9target/q1/dense_1/kernel/Initializer/truncated_normal/mulMulEtarget/q1/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal<target/q1/dense_1/kernel/Initializer/truncated_normal/stddev*+
_class!
loc:@target/q1/dense_1/kernel*
T0* 
_output_shapes
:


5target/q1/dense_1/kernel/Initializer/truncated_normalAdd9target/q1/dense_1/kernel/Initializer/truncated_normal/mul:target/q1/dense_1/kernel/Initializer/truncated_normal/mean*+
_class!
loc:@target/q1/dense_1/kernel* 
_output_shapes
:
*
T0
˝
target/q1/dense_1/kernel
VariableV2* 
_output_shapes
:
*
dtype0*
	container *
shared_name *+
_class!
loc:@target/q1/dense_1/kernel*
shape:

ű
target/q1/dense_1/kernel/AssignAssigntarget/q1/dense_1/kernel5target/q1/dense_1/kernel/Initializer/truncated_normal*+
_class!
loc:@target/q1/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(

target/q1/dense_1/kernel/readIdentitytarget/q1/dense_1/kernel*+
_class!
loc:@target/q1/dense_1/kernel* 
_output_shapes
:
*
T0
˘
(target/q1/dense_1/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*)
_class
loc:@target/q1/dense_1/bias
Ż
target/q1/dense_1/bias
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shape:*)
_class
loc:@target/q1/dense_1/bias*
shared_name 
ă
target/q1/dense_1/bias/AssignAssigntarget/q1/dense_1/bias(target/q1/dense_1/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*)
_class
loc:@target/q1/dense_1/bias

target/q1/dense_1/bias/readIdentitytarget/q1/dense_1/bias*
T0*)
_class
loc:@target/q1/dense_1/bias*
_output_shapes	
:
°
target/q1/dense_1/MatMulMatMultarget/q1/dense/Relutarget/q1/dense_1/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0
Ľ
target/q1/dense_1/BiasAddBiasAddtarget/q1/dense_1/MatMultarget/q1/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
l
target/q1/dense_1/ReluRelutarget/q1/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
9target/q1/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*+
_class!
loc:@target/q1/dense_2/kernel*
valueB"      
Š
7target/q1/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *n×\ž*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
: *
dtype0
Š
7target/q1/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *n×\>*
dtype0*
_output_shapes
: *+
_class!
loc:@target/q1/dense_2/kernel

Atarget/q1/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q1/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*+
_class!
loc:@target/q1/dense_2/kernel*
T0*
seed2*
seedŃÂń
ţ
7target/q1/dense_2/kernel/Initializer/random_uniform/subSub7target/q1/dense_2/kernel/Initializer/random_uniform/max7target/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *+
_class!
loc:@target/q1/dense_2/kernel

7target/q1/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/q1/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/q1/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*+
_class!
loc:@target/q1/dense_2/kernel

3target/q1/dense_2/kernel/Initializer/random_uniformAdd7target/q1/dense_2/kernel/Initializer/random_uniform/mul7target/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	*+
_class!
loc:@target/q1/dense_2/kernel
ť
target/q1/dense_2/kernel
VariableV2*
_output_shapes
:	*+
_class!
loc:@target/q1/dense_2/kernel*
shared_name *
dtype0*
shape:	*
	container 
ř
target/q1/dense_2/kernel/AssignAssigntarget/q1/dense_2/kernel3target/q1/dense_2/kernel/Initializer/random_uniform*+
_class!
loc:@target/q1/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	

target/q1/dense_2/kernel/readIdentitytarget/q1/dense_2/kernel*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
:	*
T0
 
(target/q1/dense_2/bias/Initializer/zerosConst*
dtype0*)
_class
loc:@target/q1/dense_2/bias*
valueB*    *
_output_shapes
:
­
target/q1/dense_2/bias
VariableV2*
shared_name *
shape:*
dtype0*)
_class
loc:@target/q1/dense_2/bias*
_output_shapes
:*
	container 
â
target/q1/dense_2/bias/AssignAssigntarget/q1/dense_2/bias(target/q1/dense_2/bias/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*)
_class
loc:@target/q1/dense_2/bias

target/q1/dense_2/bias/readIdentitytarget/q1/dense_2/bias*)
_class
loc:@target/q1/dense_2/bias*
_output_shapes
:*
T0
ą
target/q1/dense_2/MatMulMatMultarget/q1/dense_1/Relutarget/q1/dense_2/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
target/q1/dense_2/BiasAddBiasAddtarget/q1/dense_2/MatMultarget/q1/dense_2/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
|
target/q1/SqueezeSqueezetarget/q1/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
`
target/q2/concat/axisConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 

target/q2/concatConcatV2Placeholder_2Placeholder_1target/q2/concat/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*
T0*

Tidx0
ľ
9target/q2/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*)
_class
loc:@target/q2/dense/kernel
¨
8target/q2/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *)
_class
loc:@target/q2/dense/kernel
Ş
:target/q2/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*)
_class
loc:@target/q2/dense/kernel*
valueB
 *ëř>*
_output_shapes
: 

Ctarget/q2/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9target/q2/dense/kernel/Initializer/truncated_normal/shape*
_output_shapes
:	*
seedŃÂń*
dtype0*
seed2*
T0*)
_class
loc:@target/q2/dense/kernel

7target/q2/dense/kernel/Initializer/truncated_normal/mulMulCtarget/q2/dense/kernel/Initializer/truncated_normal/TruncatedNormal:target/q2/dense/kernel/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	*)
_class
loc:@target/q2/dense/kernel

3target/q2/dense/kernel/Initializer/truncated_normalAdd7target/q2/dense/kernel/Initializer/truncated_normal/mul8target/q2/dense/kernel/Initializer/truncated_normal/mean*)
_class
loc:@target/q2/dense/kernel*
T0*
_output_shapes
:	
ˇ
target/q2/dense/kernel
VariableV2*
_output_shapes
:	*
shape:	*
dtype0*
shared_name *)
_class
loc:@target/q2/dense/kernel*
	container 
ň
target/q2/dense/kernel/AssignAssigntarget/q2/dense/kernel3target/q2/dense/kernel/Initializer/truncated_normal*
_output_shapes
:	*
validate_shape(*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel

target/q2/dense/kernel/readIdentitytarget/q2/dense/kernel*)
_class
loc:@target/q2/dense/kernel*
T0*
_output_shapes
:	

&target/q2/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*'
_class
loc:@target/q2/dense/bias*
valueB*    
Ť
target/q2/dense/bias
VariableV2*
	container *
shape:*
_output_shapes	
:*'
_class
loc:@target/q2/dense/bias*
dtype0*
shared_name 
Ű
target/q2/dense/bias/AssignAssigntarget/q2/dense/bias&target/q2/dense/bias/Initializer/zeros*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

target/q2/dense/bias/readIdentitytarget/q2/dense/bias*'
_class
loc:@target/q2/dense/bias*
T0*
_output_shapes	
:
¨
target/q2/dense/MatMulMatMultarget/q2/concattarget/q2/dense/kernel/read*
T0*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

target/q2/dense/BiasAddBiasAddtarget/q2/dense/MatMultarget/q2/dense/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
h
target/q2/dense/ReluRelutarget/q2/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
;target/q2/dense_1/kernel/Initializer/truncated_normal/shapeConst*+
_class!
loc:@target/q2/dense_1/kernel*
dtype0*
valueB"      *
_output_shapes
:
Ź
:target/q2/dense_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0*+
_class!
loc:@target/q2/dense_1/kernel
Ž
<target/q2/dense_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *Eń>*
_output_shapes
: *
dtype0*+
_class!
loc:@target/q2/dense_1/kernel

Etarget/q2/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;target/q2/dense_1/kernel/Initializer/truncated_normal/shape*
dtype0*
seedŃÂń*
T0*+
_class!
loc:@target/q2/dense_1/kernel* 
_output_shapes
:
*
seed2¤

9target/q2/dense_1/kernel/Initializer/truncated_normal/mulMulEtarget/q2/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal<target/q2/dense_1/kernel/Initializer/truncated_normal/stddev*+
_class!
loc:@target/q2/dense_1/kernel* 
_output_shapes
:
*
T0

5target/q2/dense_1/kernel/Initializer/truncated_normalAdd9target/q2/dense_1/kernel/Initializer/truncated_normal/mul:target/q2/dense_1/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*+
_class!
loc:@target/q2/dense_1/kernel
˝
target/q2/dense_1/kernel
VariableV2*+
_class!
loc:@target/q2/dense_1/kernel*
shared_name *
shape:
*
	container *
dtype0* 
_output_shapes
:

ű
target/q2/dense_1/kernel/AssignAssigntarget/q2/dense_1/kernel5target/q2/dense_1/kernel/Initializer/truncated_normal*
use_locking(* 
_output_shapes
:
*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
T0

target/q2/dense_1/kernel/readIdentitytarget/q2/dense_1/kernel*
T0*+
_class!
loc:@target/q2/dense_1/kernel* 
_output_shapes
:

˘
(target/q2/dense_1/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/q2/dense_1/bias*
dtype0*
_output_shapes	
:
Ż
target/q2/dense_1/bias
VariableV2*
	container *
shape:*
shared_name *
dtype0*)
_class
loc:@target/q2/dense_1/bias*
_output_shapes	
:
ă
target/q2/dense_1/bias/AssignAssigntarget/q2/dense_1/bias(target/q2/dense_1/bias/Initializer/zeros*
validate_shape(*
T0*)
_class
loc:@target/q2/dense_1/bias*
_output_shapes	
:*
use_locking(

target/q2/dense_1/bias/readIdentitytarget/q2/dense_1/bias*
T0*
_output_shapes	
:*)
_class
loc:@target/q2/dense_1/bias
°
target/q2/dense_1/MatMulMatMultarget/q2/dense/Relutarget/q2/dense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0*
transpose_b( 
Ľ
target/q2/dense_1/BiasAddBiasAddtarget/q2/dense_1/MatMultarget/q2/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
l
target/q2/dense_1/ReluRelutarget/q2/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
9target/q2/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:*+
_class!
loc:@target/q2/dense_2/kernel
Š
7target/q2/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*+
_class!
loc:@target/q2/dense_2/kernel*
valueB
 *n×\ž
Š
7target/q2/dense_2/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
: *
valueB
 *n×\>

Atarget/q2/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q2/dense_2/kernel/Initializer/random_uniform/shape*
T0*
seed2´*
dtype0*
_output_shapes
:	*+
_class!
loc:@target/q2/dense_2/kernel*
seedŃÂń
ţ
7target/q2/dense_2/kernel/Initializer/random_uniform/subSub7target/q2/dense_2/kernel/Initializer/random_uniform/max7target/q2/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@target/q2/dense_2/kernel

7target/q2/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/q2/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/q2/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*+
_class!
loc:@target/q2/dense_2/kernel*
T0

3target/q2/dense_2/kernel/Initializer/random_uniformAdd7target/q2/dense_2/kernel/Initializer/random_uniform/mul7target/q2/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
:	*+
_class!
loc:@target/q2/dense_2/kernel*
T0
ť
target/q2/dense_2/kernel
VariableV2*
dtype0*
shared_name *
_output_shapes
:	*+
_class!
loc:@target/q2/dense_2/kernel*
shape:	*
	container 
ř
target/q2/dense_2/kernel/AssignAssigntarget/q2/dense_2/kernel3target/q2/dense_2/kernel/Initializer/random_uniform*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0

target/q2/dense_2/kernel/readIdentitytarget/q2/dense_2/kernel*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes
:	*
T0
 
(target/q2/dense_2/bias/Initializer/zerosConst*)
_class
loc:@target/q2/dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0
­
target/q2/dense_2/bias
VariableV2*
shape:*
shared_name *)
_class
loc:@target/q2/dense_2/bias*
dtype0*
_output_shapes
:*
	container 
â
target/q2/dense_2/bias/AssignAssigntarget/q2/dense_2/bias(target/q2/dense_2/bias/Initializer/zeros*
T0*
use_locking(*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:

target/q2/dense_2/bias/readIdentitytarget/q2/dense_2/bias*
_output_shapes
:*)
_class
loc:@target/q2/dense_2/bias*
T0
ą
target/q2/dense_2/MatMulMatMultarget/q2/dense_1/Relutarget/q2/dense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 
¤
target/q2/dense_2/BiasAddBiasAddtarget/q2/dense_2/MatMultarget/q2/dense_2/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
|
target/q2/SqueezeSqueezetarget/q2/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
*
T0
b
target/q1_1/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

target/q1_1/concatConcatV2Placeholder_2target/pi/multarget/q1_1/concat/axis*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*

Tidx0
Ź
target/q1_1/dense/MatMulMatMultarget/q1_1/concattarget/q1/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 
Ł
target/q1_1/dense/BiasAddBiasAddtarget/q1_1/dense/MatMultarget/q1/dense/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
target/q1_1/dense/ReluRelutarget/q1_1/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
target/q1_1/dense_1/MatMulMatMultarget/q1_1/dense/Relutarget/q1/dense_1/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
target/q1_1/dense_1/BiasAddBiasAddtarget/q1_1/dense_1/MatMultarget/q1/dense_1/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
p
target/q1_1/dense_1/ReluRelutarget/q1_1/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
target/q1_1/dense_2/MatMulMatMultarget/q1_1/dense_1/Relutarget/q1/dense_2/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
target/q1_1/dense_2/BiasAddBiasAddtarget/q1_1/dense_2/MatMultarget/q1/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

target/q1_1/SqueezeSqueezetarget/q1_1/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
squeeze_dims

[
target_1/ShapeShapetarget/pi/mul*
out_type0*
_output_shapes
:*
T0
`
target_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
b
target_1/random_normal/stddevConst*
valueB
 *ÍĚL>*
_output_shapes
: *
dtype0
Ż
+target_1/random_normal/RandomStandardNormalRandomStandardNormaltarget_1/Shape*
dtype0*
seed2Đ*
seedŃÂń*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

target_1/random_normal/mulMul+target_1/random_normal/RandomStandardNormaltarget_1/random_normal/stddev*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

target_1/random_normalAddtarget_1/random_normal/multarget_1/random_normal/mean*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 target_1/clip_by_value/Minimum/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 

target_1/clip_by_value/MinimumMinimumtarget_1/random_normal target_1/clip_by_value/Minimum/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
target_1/clip_by_value/yConst*
valueB
 *   ż*
_output_shapes
: *
dtype0

target_1/clip_by_valueMaximumtarget_1/clip_by_value/Minimumtarget_1/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
target_1/addAddV2target/pi/multarget_1/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
"target_1/clip_by_value_1/Minimum/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

 target_1/clip_by_value_1/MinimumMinimumtarget_1/add"target_1/clip_by_value_1/Minimum/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
target_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ż

target_1/clip_by_value_1Maximum target_1/clip_by_value_1/Minimumtarget_1/clip_by_value_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
target_1/pi/dense/MatMulMatMulPlaceholder_2target/pi/dense/kernel/read*
T0*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ł
target_1/pi/dense/BiasAddBiasAddtarget_1/pi/dense/MatMultarget/pi/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
target_1/pi/dense/ReluRelutarget_1/pi/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
target_1/pi/dense_1/MatMulMatMultarget_1/pi/dense/Relutarget/pi/dense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
Š
target_1/pi/dense_1/BiasAddBiasAddtarget_1/pi/dense_1/MatMultarget/pi/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
p
target_1/pi/dense_1/ReluRelutarget_1/pi/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
target_1/pi/dense_2/MatMulMatMultarget_1/pi/dense_1/Relutarget/pi/dense_2/kernel/read*
transpose_b( *
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
target_1/pi/dense_2/BiasAddBiasAddtarget_1/pi/dense_2/MatMultarget/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
target_1/pi/dense_2/SigmoidSigmoidtarget_1/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
target_1/pi/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
x
target_1/pi/mulMultarget_1/pi/mul/xtarget_1/pi/dense_2/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
target_1/q1/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
§
target_1/q1/concatConcatV2Placeholder_2target_1/clip_by_value_1target_1/q1/concat/axis*
T0*

Tidx0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
target_1/q1/dense/MatMulMatMultarget_1/q1/concattarget/q1/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( *
transpose_b( 
Ł
target_1/q1/dense/BiasAddBiasAddtarget_1/q1/dense/MatMultarget/q1/dense/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
l
target_1/q1/dense/ReluRelutarget_1/q1/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
target_1/q1/dense_1/MatMulMatMultarget_1/q1/dense/Relutarget/q1/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Š
target_1/q1/dense_1/BiasAddBiasAddtarget_1/q1/dense_1/MatMultarget/q1/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
p
target_1/q1/dense_1/ReluRelutarget_1/q1/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
target_1/q1/dense_2/MatMulMatMultarget_1/q1/dense_1/Relutarget/q1/dense_2/kernel/read*
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 
¨
target_1/q1/dense_2/BiasAddBiasAddtarget_1/q1/dense_2/MatMultarget/q1/dense_2/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC

target_1/q1/SqueezeSqueezetarget_1/q1/dense_2/BiasAdd*
T0*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
target_1/q2/concat/axisConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
§
target_1/q2/concatConcatV2Placeholder_2target_1/clip_by_value_1target_1/q2/concat/axis*
N*

Tidx0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
target_1/q2/dense/MatMulMatMultarget_1/q2/concattarget/q2/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0
Ł
target_1/q2/dense/BiasAddBiasAddtarget_1/q2/dense/MatMultarget/q2/dense/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
l
target_1/q2/dense/ReluRelutarget_1/q2/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
target_1/q2/dense_1/MatMulMatMultarget_1/q2/dense/Relutarget/q2/dense_1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 
Š
target_1/q2/dense_1/BiasAddBiasAddtarget_1/q2/dense_1/MatMultarget/q2/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
p
target_1/q2/dense_1/ReluRelutarget_1/q2/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
target_1/q2/dense_2/MatMulMatMultarget_1/q2/dense_1/Relutarget/q2/dense_2/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( *
T0
¨
target_1/q2/dense_2/BiasAddBiasAddtarget_1/q2/dense_2/MatMultarget/q2/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

target_1/q2/SqueezeSqueezetarget_1/q2/dense_2/BiasAdd*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
squeeze_dims

d
target_1/q1_1/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
˘
target_1/q1_1/concatConcatV2Placeholder_2target_1/pi/multarget_1/q1_1/concat/axis*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
N*
T0
°
target_1/q1_1/dense/MatMulMatMultarget_1/q1_1/concattarget/q1/dense/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
§
target_1/q1_1/dense/BiasAddBiasAddtarget_1/q1_1/dense/MatMultarget/q1/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
p
target_1/q1_1/dense/ReluRelutarget_1/q1_1/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
target_1/q1_1/dense_1/MatMulMatMultarget_1/q1_1/dense/Relutarget/q1/dense_1/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
­
target_1/q1_1/dense_1/BiasAddBiasAddtarget_1/q1_1/dense_1/MatMultarget/q1/dense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
t
target_1/q1_1/dense_1/ReluRelutarget_1/q1_1/dense_1/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
š
target_1/q1_1/dense_2/MatMulMatMultarget_1/q1_1/dense_1/Relutarget/q1/dense_2/kernel/read*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 
Ź
target_1/q1_1/dense_2/BiasAddBiasAddtarget_1/q1_1/dense_2/MatMultarget/q1/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

target_1/q1_1/SqueezeSqueezetarget_1/q1_1/dense_2/BiasAdd*
squeeze_dims
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
MinimumMinimumtarget_1/q1/Squeezetarget_1/q2/Squeeze*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
N
subSubsub/xPlaceholder_4*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *¤p}?
D
mulMulmul/xsub*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
H
mul_1MulmulMinimum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
P
addAddV2Placeholder_3mul_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
O
StopGradientStopGradientadd*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
d
MeanMeanmain/q1_1/SqueezeConst*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
1
NegNegMean*
_output_shapes
: *
T0
Y
sub_1Submain/q1/SqueezeStopGradient*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
F
powPowsub_1pow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Z
Mean_1MeanpowConst_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
Y
sub_2Submain/q2/SqueezeStopGradient*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
L
pow_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
J
pow_1Powsub_2pow_1/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Q
Const_2Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_2Meanpow_1Const_2*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
?
add_1AddV2Mean_1Mean_2*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *

index_type0*
T0
N
gradients/Neg_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
j
gradients/Mean_grad/ShapeShapemain/q1_1/Squeeze*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
gradients/Mean_grad/Shape_1Shapemain/q1_1/Squeeze*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

&gradients/main/q1_1/Squeeze_grad/ShapeShapemain/q1_1/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
¸
(gradients/main/q1_1/Squeeze_grad/ReshapeReshapegradients/Mean_grad/truediv&gradients/main/q1_1/Squeeze_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Š
4gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients/main/q1_1/Squeeze_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC
Ł
9gradients/main/q1_1/dense_2/BiasAdd_grad/tuple/group_depsNoOp)^gradients/main/q1_1/Squeeze_grad/Reshape5^gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGrad
˘
Agradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients/main/q1_1/Squeeze_grad/Reshape:^gradients/main/q1_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/main/q1_1/Squeeze_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Cgradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGrad:^gradients/main/q1_1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/main/q1_1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ń
.gradients/main/q1_1/dense_2/MatMul_grad/MatMulMatMulAgradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q1/dense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ĺ
0gradients/main/q1_1/dense_2/MatMul_grad/MatMul_1MatMulmain/q1_1/dense_1/ReluAgradients/main/q1_1/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	
¤
8gradients/main/q1_1/dense_2/MatMul_grad/tuple/group_depsNoOp/^gradients/main/q1_1/dense_2/MatMul_grad/MatMul1^gradients/main/q1_1/dense_2/MatMul_grad/MatMul_1
­
@gradients/main/q1_1/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients/main/q1_1/dense_2/MatMul_grad/MatMul9^gradients/main/q1_1/dense_2/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*A
_class7
53loc:@gradients/main/q1_1/dense_2/MatMul_grad/MatMul
Ş
Bgradients/main/q1_1/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients/main/q1_1/dense_2/MatMul_grad/MatMul_19^gradients/main/q1_1/dense_2/MatMul_grad/tuple/group_deps*C
_class9
75loc:@gradients/main/q1_1/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Ç
.gradients/main/q1_1/dense_1/Relu_grad/ReluGradReluGrad@gradients/main/q1_1/dense_2/MatMul_grad/tuple/control_dependencymain/q1_1/dense_1/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
°
4gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/main/q1_1/dense_1/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Š
9gradients/main/q1_1/dense_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGrad/^gradients/main/q1_1/dense_1/Relu_grad/ReluGrad
Ż
Agradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/main/q1_1/dense_1/Relu_grad/ReluGrad:^gradients/main/q1_1/dense_1/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/main/q1_1/dense_1/Relu_grad/ReluGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Cgradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients/main/q1_1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*G
_class=
;9loc:@gradients/main/q1_1/dense_1/BiasAdd_grad/BiasAddGrad
ń
.gradients/main/q1_1/dense_1/MatMul_grad/MatMulMatMulAgradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q1/dense_1/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
0gradients/main/q1_1/dense_1/MatMul_grad/MatMul_1MatMulmain/q1_1/dense/ReluAgradients/main/q1_1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( * 
_output_shapes
:
*
transpose_a(*
T0
¤
8gradients/main/q1_1/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients/main/q1_1/dense_1/MatMul_grad/MatMul1^gradients/main/q1_1/dense_1/MatMul_grad/MatMul_1
­
@gradients/main/q1_1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients/main/q1_1/dense_1/MatMul_grad/MatMul9^gradients/main/q1_1/dense_1/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*A
_class7
53loc:@gradients/main/q1_1/dense_1/MatMul_grad/MatMul
Ť
Bgradients/main/q1_1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients/main/q1_1/dense_1/MatMul_grad/MatMul_19^gradients/main/q1_1/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*C
_class9
75loc:@gradients/main/q1_1/dense_1/MatMul_grad/MatMul_1*
T0
Ă
,gradients/main/q1_1/dense/Relu_grad/ReluGradReluGrad@gradients/main/q1_1/dense_1/MatMul_grad/tuple/control_dependencymain/q1_1/dense/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/q1_1/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
data_formatNHWC*
T0
Ł
7gradients/main/q1_1/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGrad-^gradients/main/q1_1/dense/Relu_grad/ReluGrad
§
?gradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/q1_1/dense/Relu_grad/ReluGrad8^gradients/main/q1_1/dense/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*?
_class5
31loc:@gradients/main/q1_1/dense/Relu_grad/ReluGrad
¨
Agradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGrad8^gradients/main/q1_1/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/main/q1_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ę
,gradients/main/q1_1/dense/MatMul_grad/MatMulMatMul?gradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependencymain/q1/dense/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
.gradients/main/q1_1/dense/MatMul_grad/MatMul_1MatMulmain/q1_1/concat?gradients/main/q1_1/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_b( *
transpose_a(

6gradients/main/q1_1/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/main/q1_1/dense/MatMul_grad/MatMul/^gradients/main/q1_1/dense/MatMul_grad/MatMul_1
¤
>gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/q1_1/dense/MatMul_grad/MatMul7^gradients/main/q1_1/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/main/q1_1/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
@gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/q1_1/dense/MatMul_grad/MatMul_17^gradients/main/q1_1/dense/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients/main/q1_1/dense/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
f
$gradients/main/q1_1/concat_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0

#gradients/main/q1_1/concat_grad/modFloorModmain/q1_1/concat/axis$gradients/main/q1_1/concat_grad/Rank*
T0*
_output_shapes
: 
p
%gradients/main/q1_1/concat_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:

&gradients/main/q1_1/concat_grad/ShapeNShapeNPlaceholdermain/pi/mul*
T0*
N* 
_output_shapes
::*
out_type0
Ţ
,gradients/main/q1_1/concat_grad/ConcatOffsetConcatOffset#gradients/main/q1_1/concat_grad/mod&gradients/main/q1_1/concat_grad/ShapeN(gradients/main/q1_1/concat_grad/ShapeN:1*
N* 
_output_shapes
::

%gradients/main/q1_1/concat_grad/SliceSlice>gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependency,gradients/main/q1_1/concat_grad/ConcatOffset&gradients/main/q1_1/concat_grad/ShapeN*
Index0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

'gradients/main/q1_1/concat_grad/Slice_1Slice>gradients/main/q1_1/dense/MatMul_grad/tuple/control_dependency.gradients/main/q1_1/concat_grad/ConcatOffset:1(gradients/main/q1_1/concat_grad/ShapeN:1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Index0

0gradients/main/q1_1/concat_grad/tuple/group_depsNoOp&^gradients/main/q1_1/concat_grad/Slice(^gradients/main/q1_1/concat_grad/Slice_1

8gradients/main/q1_1/concat_grad/tuple/control_dependencyIdentity%gradients/main/q1_1/concat_grad/Slice1^gradients/main/q1_1/concat_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/main/q1_1/concat_grad/Slice*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients/main/q1_1/concat_grad/tuple/control_dependency_1Identity'gradients/main/q1_1/concat_grad/Slice_11^gradients/main/q1_1/concat_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@gradients/main/q1_1/concat_grad/Slice_1*
T0
k
 gradients/main/pi/mul_grad/ShapeShapemain/pi/mul/x*
out_type0*
_output_shapes
: *
T0
y
"gradients/main/pi/mul_grad/Shape_1Shapemain/pi/dense_2/Sigmoid*
T0*
_output_shapes
:*
out_type0
Ě
0gradients/main/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/main/pi/mul_grad/Shape"gradients/main/pi/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ź
gradients/main/pi/mul_grad/MulMul:gradients/main/q1_1/concat_grad/tuple/control_dependency_1main/pi/dense_2/Sigmoid*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
gradients/main/pi/mul_grad/SumSumgradients/main/pi/mul_grad/Mul0gradients/main/pi/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

"gradients/main/pi/mul_grad/ReshapeReshapegradients/main/pi/mul_grad/Sum gradients/main/pi/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
¤
 gradients/main/pi/mul_grad/Mul_1Mulmain/pi/mul/x:gradients/main/q1_1/concat_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
 gradients/main/pi/mul_grad/Sum_1Sum gradients/main/pi/mul_grad/Mul_12gradients/main/pi/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
ľ
$gradients/main/pi/mul_grad/Reshape_1Reshape gradients/main/pi/mul_grad/Sum_1"gradients/main/pi/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/main/pi/mul_grad/tuple/group_depsNoOp#^gradients/main/pi/mul_grad/Reshape%^gradients/main/pi/mul_grad/Reshape_1
é
3gradients/main/pi/mul_grad/tuple/control_dependencyIdentity"gradients/main/pi/mul_grad/Reshape,^gradients/main/pi/mul_grad/tuple/group_deps*
_output_shapes
: *5
_class+
)'loc:@gradients/main/pi/mul_grad/Reshape*
T0

5gradients/main/pi/mul_grad/tuple/control_dependency_1Identity$gradients/main/pi/mul_grad/Reshape_1,^gradients/main/pi/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*7
_class-
+)loc:@gradients/main/pi/mul_grad/Reshape_1*
T0
Ă
2gradients/main/pi/dense_2/Sigmoid_grad/SigmoidGradSigmoidGradmain/pi/dense_2/Sigmoid5gradients/main/pi/mul_grad/tuple/control_dependency_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/main/pi/dense_2/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
Š
7gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/main/pi/dense_2/Sigmoid_grad/SigmoidGrad
˛
?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/main/pi/dense_2/Sigmoid_grad/SigmoidGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/main/pi/dense_2/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Agradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/main/pi/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
í
,gradients/main/pi/dense_2/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencymain/pi/dense_2/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_b(
ß
.gradients/main/pi/dense_2/MatMul_grad/MatMul_1MatMulmain/pi/dense_1/Relu?gradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	*
T0*
transpose_b( 

6gradients/main/pi/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_2/MatMul_grad/MatMul/^gradients/main/pi/dense_2/MatMul_grad/MatMul_1
Ľ
>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_2/MatMul_grad/MatMul7^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul
˘
@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_2/MatMul_grad/MatMul_17^gradients/main/pi/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	*A
_class7
53loc:@gradients/main/pi/dense_2/MatMul_grad/MatMul_1*
T0
Á
,gradients/main/pi/dense_1/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependencymain/pi/dense_1/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/main/pi/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad-^gradients/main/pi/dense_1/Relu_grad/ReluGrad
§
?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/Relu_grad/ReluGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*?
_class5
31loc:@gradients/main/pi/dense_1/Relu_grad/ReluGrad
¨
Agradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*E
_class;
97loc:@gradients/main/pi/dense_1/BiasAdd_grad/BiasAddGrad*
T0
í
,gradients/main/pi/dense_1/MatMul_grad/MatMulMatMul?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencymain/pi/dense_1/kernel/read*
T0*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ţ
.gradients/main/pi/dense_1/MatMul_grad/MatMul_1MatMulmain/pi/dense/Relu?gradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_b( *
transpose_a(

6gradients/main/pi/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/main/pi/dense_1/MatMul_grad/MatMul/^gradients/main/pi/dense_1/MatMul_grad/MatMul_1
Ľ
>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/main/pi/dense_1/MatMul_grad/MatMul7^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/main/pi/dense_1/MatMul_grad/MatMul_17^gradients/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/main/pi/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

˝
*gradients/main/pi/dense/Relu_grad/ReluGradReluGrad>gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependencymain/pi/dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
0gradients/main/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/main/pi/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC

5gradients/main/pi/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad+^gradients/main/pi/dense/Relu_grad/ReluGrad

=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/Relu_grad/ReluGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*=
_class3
1/loc:@gradients/main/pi/dense/Relu_grad/ReluGrad
 
?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad6^gradients/main/pi/dense/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/main/pi/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
ć
*gradients/main/pi/dense/MatMul_grad/MatMulMatMul=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependencymain/pi/dense/kernel/read*
transpose_b(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
Ň
,gradients/main/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder=gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

4gradients/main/pi/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/main/pi/dense/MatMul_grad/MatMul-^gradients/main/pi/dense/MatMul_grad/MatMul_1

<gradients/main/pi/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/main/pi/dense/MatMul_grad/MatMul5^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*=
_class3
1/loc:@gradients/main/pi/dense/MatMul_grad/MatMul

>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/main/pi/dense/MatMul_grad/MatMul_15^gradients/main/pi/dense/MatMul_grad/tuple/group_deps*?
_class5
31loc:@gradients/main/pi/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	

beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 

beta1_power
VariableV2*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
	container *
shared_name *
dtype0*
shape: 
ľ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
validate_shape(
q
beta1_power/readIdentitybeta1_power*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 

beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wž?*%
_class
loc:@main/pi/dense/bias*
dtype0

beta2_power
VariableV2*
_output_shapes
: *
dtype0*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape: 
ľ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes
: 
q
beta2_power/readIdentitybeta2_power*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
T0
Ť
+main/pi/dense/kernel/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense/kernel*
valueB	*    *
_output_shapes
:	*
dtype0
¸
main/pi/dense/kernel/Adam
VariableV2*
shape:	*
shared_name *'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	*
	container *
dtype0
î
 main/pi/dense/kernel/Adam/AssignAssignmain/pi/dense/kernel/Adam+main/pi/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	*
use_locking(

main/pi/dense/kernel/Adam/readIdentitymain/pi/dense/kernel/Adam*'
_class
loc:@main/pi/dense/kernel*
T0*
_output_shapes
:	
­
-main/pi/dense/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
valueB	*    *'
_class
loc:@main/pi/dense/kernel
ş
main/pi/dense/kernel/Adam_1
VariableV2*
shape:	*
shared_name *
dtype0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	*
	container 
ô
"main/pi/dense/kernel/Adam_1/AssignAssignmain/pi/dense/kernel/Adam_1-main/pi/dense/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	*
validate_shape(*'
_class
loc:@main/pi/dense/kernel*
use_locking(

 main/pi/dense/kernel/Adam_1/readIdentitymain/pi/dense/kernel/Adam_1*
_output_shapes
:	*
T0*'
_class
loc:@main/pi/dense/kernel

)main/pi/dense/bias/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:*
dtype0
Ź
main/pi/dense/bias/Adam
VariableV2*
	container *
dtype0*
shape:*
shared_name *%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:
â
main/pi/dense/bias/Adam/AssignAssignmain/pi/dense/bias/Adam)main/pi/dense/bias/Adam/Initializer/zeros*
use_locking(*%
_class
loc:@main/pi/dense/bias*
T0*
_output_shapes	
:*
validate_shape(

main/pi/dense/bias/Adam/readIdentitymain/pi/dense/bias/Adam*
_output_shapes	
:*
T0*%
_class
loc:@main/pi/dense/bias
Ą
+main/pi/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:*
valueB*    
Ž
main/pi/dense/bias/Adam_1
VariableV2*
dtype0*
	container *
shared_name *
shape:*
_output_shapes	
:*%
_class
loc:@main/pi/dense/bias
č
 main/pi/dense/bias/Adam_1/AssignAssignmain/pi/dense/bias/Adam_1+main/pi/dense/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:*%
_class
loc:@main/pi/dense/bias*
validate_shape(

main/pi/dense/bias/Adam_1/readIdentitymain/pi/dense/bias/Adam_1*
_output_shapes	
:*%
_class
loc:@main/pi/dense/bias*
T0
š
=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"      *)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
:
Ł
3main/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
: *
valueB
 *    

-main/pi/dense_1/kernel/Adam/Initializer/zerosFill=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*)
_class
loc:@main/pi/dense_1/kernel*
T0*

index_type0
ž
main/pi/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
*
shared_name *
dtype0*)
_class
loc:@main/pi/dense_1/kernel*
shape:
*
	container 
÷
"main/pi/dense_1/kernel/Adam/AssignAssignmain/pi/dense_1/kernel/Adam-main/pi/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
validate_shape(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(

 main/pi/dense_1/kernel/Adam/readIdentitymain/pi/dense_1/kernel/Adam*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:

ť
?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
:*
dtype0
Ľ
5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *)
_class
loc:@main/pi/dense_1/kernel

/main/pi/dense_1/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*)
_class
loc:@main/pi/dense_1/kernel*
T0*

index_type0
Ŕ
main/pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *
	container *
shape:
* 
_output_shapes
:
*)
_class
loc:@main/pi/dense_1/kernel*
dtype0
ý
$main/pi/dense_1/kernel/Adam_1/AssignAssignmain/pi/dense_1/kernel/Adam_1/main/pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0*)
_class
loc:@main/pi/dense_1/kernel
Ł
"main/pi/dense_1/kernel/Adam_1/readIdentitymain/pi/dense_1/kernel/Adam_1* 
_output_shapes
:
*
T0*)
_class
loc:@main/pi/dense_1/kernel
Ł
+main/pi/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
dtype0*
valueB*    *
_output_shapes	
:
°
main/pi/dense_1/bias/Adam
VariableV2*
_output_shapes	
:*'
_class
loc:@main/pi/dense_1/bias*
shared_name *
dtype0*
shape:*
	container 
ę
 main/pi/dense_1/bias/Adam/AssignAssignmain/pi/dense_1/bias/Adam+main/pi/dense_1/bias/Adam/Initializer/zeros*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias

main/pi/dense_1/bias/Adam/readIdentitymain/pi/dense_1/bias/Adam*
_output_shapes	
:*'
_class
loc:@main/pi/dense_1/bias*
T0
Ľ
-main/pi/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
main/pi/dense_1/bias/Adam_1
VariableV2*
shared_name *
_output_shapes	
:*'
_class
loc:@main/pi/dense_1/bias*
shape:*
dtype0*
	container 
đ
"main/pi/dense_1/bias/Adam_1/AssignAssignmain/pi/dense_1/bias/Adam_1-main/pi/dense_1/bias/Adam_1/Initializer/zeros*'
_class
loc:@main/pi/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(

 main/pi/dense_1/bias/Adam_1/readIdentitymain/pi/dense_1/bias/Adam_1*'
_class
loc:@main/pi/dense_1/bias*
T0*
_output_shapes	
:
Ż
-main/pi/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*)
_class
loc:@main/pi/dense_2/kernel*
valueB	*    
ź
main/pi/dense_2/kernel/Adam
VariableV2*)
_class
loc:@main/pi/dense_2/kernel*
shape:	*
shared_name *
	container *
_output_shapes
:	*
dtype0
ö
"main/pi/dense_2/kernel/Adam/AssignAssignmain/pi/dense_2/kernel/Adam-main/pi/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(

 main/pi/dense_2/kernel/Adam/readIdentitymain/pi/dense_2/kernel/Adam*
_output_shapes
:	*
T0*)
_class
loc:@main/pi/dense_2/kernel
ą
/main/pi/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	*
valueB	*    *
dtype0*)
_class
loc:@main/pi/dense_2/kernel
ž
main/pi/dense_2/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
	container *)
_class
loc:@main/pi/dense_2/kernel*
shape:	*
_output_shapes
:	
ü
$main/pi/dense_2/kernel/Adam_1/AssignAssignmain/pi/dense_2/kernel/Adam_1/main/pi/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel
˘
"main/pi/dense_2/kernel/Adam_1/readIdentitymain/pi/dense_2/kernel/Adam_1*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes
:	
Ą
+main/pi/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*'
_class
loc:@main/pi/dense_2/bias
Ž
main/pi/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
	container *'
_class
loc:@main/pi/dense_2/bias*
shape:*
dtype0*
shared_name 
é
 main/pi/dense_2/bias/Adam/AssignAssignmain/pi/dense_2/bias/Adam+main/pi/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
validate_shape(*
T0*'
_class
loc:@main/pi/dense_2/bias*
use_locking(

main/pi/dense_2/bias/Adam/readIdentitymain/pi/dense_2/bias/Adam*'
_class
loc:@main/pi/dense_2/bias*
T0*
_output_shapes
:
Ł
-main/pi/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*'
_class
loc:@main/pi/dense_2/bias
°
main/pi/dense_2/bias/Adam_1
VariableV2*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
shared_name *
	container *
dtype0*
shape:
ď
"main/pi/dense_2/bias/Adam_1/AssignAssignmain/pi/dense_2/bias/Adam_1-main/pi/dense_2/bias/Adam_1/Initializer/zeros*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(

 main/pi/dense_2/bias/Adam_1/readIdentitymain/pi/dense_2/bias/Adam_1*'
_class
loc:@main/pi/dense_2/bias*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o:
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
Q
Adam/epsilonConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 

*Adam/update_main/pi/dense/kernel/ApplyAdam	ApplyAdammain/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/pi/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *'
_class
loc:@main/pi/dense/kernel*
use_nesterov( *
_output_shapes
:	*
T0

(Adam/update_main/pi/dense/bias/ApplyAdam	ApplyAdammain/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1*%
_class
loc:@main/pi/dense/bias*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:
Ş
,Adam/update_main/pi/dense_1/kernel/ApplyAdam	ApplyAdammain/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *)
_class
loc:@main/pi/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
*
T0

*Adam/update_main/pi/dense_1/bias/ApplyAdam	ApplyAdammain/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_output_shapes	
:*'
_class
loc:@main/pi/dense_1/bias
Š
,Adam/update_main/pi/dense_2/kernel/ApplyAdam	ApplyAdammain/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
use_nesterov( *
T0*
use_locking( *)
_class
loc:@main/pi/dense_2/kernel

*Adam/update_main/pi/dense_2/bias/ApplyAdam	ApplyAdammain/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *'
_class
loc:@main/pi/dense_2/bias*
T0*
use_nesterov( *
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*%
_class
loc:@main/pi/dense/bias*
use_locking( *
validate_shape(*
_output_shapes
: *
T0


Adam/mul_1Mulbeta2_power/read
Adam/beta2)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam*
_output_shapes
: *%
_class
loc:@main/pi/dense/bias*
T0
Ą
Adam/Assign_1Assignbeta2_power
Adam/mul_1*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
use_locking( *
_output_shapes
: *
T0
ş
AdamNoOp^Adam/Assign^Adam/Assign_1)^Adam/update_main/pi/dense/bias/ApplyAdam+^Adam/update_main/pi/dense/kernel/ApplyAdam+^Adam/update_main/pi/dense_1/bias/ApplyAdam-^Adam/update_main/pi/dense_1/kernel/ApplyAdam+^Adam/update_main/pi/dense_2/bias/ApplyAdam-^Adam/update_main/pi/dense_2/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
B
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/Fill
˝
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/Fill(^gradients_1/add_1_grad/tuple/group_deps*
T0*
_output_shapes
: *#
_class
loc:@gradients_1/Fill
ż
1gradients_1/add_1_grad/tuple/control_dependency_1Identitygradients_1/Fill(^gradients_1/add_1_grad/tuple/group_deps*#
_class
loc:@gradients_1/Fill*
T0*
_output_shapes
: 
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
ľ
gradients_1/Mean_1_grad/ReshapeReshape/gradients_1/add_1_grad/tuple/control_dependency%gradients_1/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
`
gradients_1/Mean_1_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
T0*
_output_shapes
:*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
˘
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ś
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
c
!gradients_1/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0*
Truncate( 

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
ˇ
gradients_1/Mean_2_grad/ReshapeReshape1gradients_1/add_1_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients_1/Mean_2_grad/ShapeShapepow_1*
out_type0*
_output_shapes
:*
T0
¤
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
d
gradients_1/Mean_2_grad/Shape_1Shapepow_1*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_2_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_1/Mean_2_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
˘
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Ś
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
gradients_1/pow_grad/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
_
gradients_1/pow_grad/Shape_1Shapepow/y*
out_type0*
_output_shapes
: *
T0
ş
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
gradients_1/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0

gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
out_type0*
T0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
˛
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*

index_type0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ť
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
Ţ
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape
×
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1
a
gradients_1/pow_1_grad/ShapeShapesub_2*
_output_shapes
:*
out_type0*
T0
c
gradients_1/pow_1_grad/Shape_1Shapepow_1/y*
T0*
_output_shapes
: *
out_type0
Ŕ
,gradients_1/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_1_grad/Shapegradients_1/pow_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
y
gradients_1/pow_1_grad/mulMulgradients_1/Mean_2_grad/truedivpow_1/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients_1/pow_1_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
i
gradients_1/pow_1_grad/subSubpow_1/ygradients_1/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_1/pow_1_grad/PowPowsub_2gradients_1/pow_1_grad/sub*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pow_1_grad/mul_1Mulgradients_1/pow_1_grad/mulgradients_1/pow_1_grad/Pow*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
gradients_1/pow_1_grad/SumSumgradients_1/pow_1_grad/mul_1,gradients_1/pow_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:

gradients_1/pow_1_grad/ReshapeReshapegradients_1/pow_1_grad/Sumgradients_1/pow_1_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
e
 gradients_1/pow_1_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0

gradients_1/pow_1_grad/GreaterGreatersub_2 gradients_1/pow_1_grad/Greater/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&gradients_1/pow_1_grad/ones_like/ShapeShapesub_2*
_output_shapes
:*
out_type0*
T0
k
&gradients_1/pow_1_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
¸
 gradients_1/pow_1_grad/ones_likeFill&gradients_1/pow_1_grad/ones_like/Shape&gradients_1/pow_1_grad/ones_like/Const*

index_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pow_1_grad/SelectSelectgradients_1/pow_1_grad/Greatersub_2 gradients_1/pow_1_grad/ones_like*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
n
gradients_1/pow_1_grad/LogLoggradients_1/pow_1_grad/Select*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
!gradients_1/pow_1_grad/zeros_like	ZerosLikesub_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ś
gradients_1/pow_1_grad/Select_1Selectgradients_1/pow_1_grad/Greatergradients_1/pow_1_grad/Log!gradients_1/pow_1_grad/zeros_like*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
gradients_1/pow_1_grad/mul_2Mulgradients_1/Mean_2_grad/truedivpow_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/pow_1_grad/mul_3Mulgradients_1/pow_1_grad/mul_2gradients_1/pow_1_grad/Select_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
gradients_1/pow_1_grad/Sum_1Sumgradients_1/pow_1_grad/mul_3.gradients_1/pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 

 gradients_1/pow_1_grad/Reshape_1Reshapegradients_1/pow_1_grad/Sum_1gradients_1/pow_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_1/pow_1_grad/tuple/group_depsNoOp^gradients_1/pow_1_grad/Reshape!^gradients_1/pow_1_grad/Reshape_1
ć
/gradients_1/pow_1_grad/tuple/control_dependencyIdentitygradients_1/pow_1_grad/Reshape(^gradients_1/pow_1_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_1/pow_1_grad/Reshape*
T0
ß
1gradients_1/pow_1_grad/tuple/control_dependency_1Identity gradients_1/pow_1_grad/Reshape_1(^gradients_1/pow_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/pow_1_grad/Reshape_1*
_output_shapes
: *
T0
k
gradients_1/sub_1_grad/ShapeShapemain/q1/Squeeze*
out_type0*
T0*
_output_shapes
:
j
gradients_1/sub_1_grad/Shape_1ShapeStopGradient*
out_type0*
_output_shapes
:*
T0
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ž
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
~
gradients_1/sub_1_grad/NegNeg-gradients_1/pow_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
gradients_1/sub_1_grad/Sum_1Sumgradients_1/sub_1_grad/Neg.gradients_1/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ľ
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Sum_1gradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ć
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
ě
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
gradients_1/sub_2_grad/ShapeShapemain/q2/Squeeze*
_output_shapes
:*
out_type0*
T0
j
gradients_1/sub_2_grad/Shape_1ShapeStopGradient*
_output_shapes
:*
T0*
out_type0
Ŕ
,gradients_1/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_2_grad/Shapegradients_1/sub_2_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ŕ
gradients_1/sub_2_grad/SumSum/gradients_1/pow_1_grad/tuple/control_dependency,gradients_1/sub_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 

gradients_1/sub_2_grad/ReshapeReshapegradients_1/sub_2_grad/Sumgradients_1/sub_2_grad/Shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/sub_2_grad/NegNeg/gradients_1/pow_1_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
gradients_1/sub_2_grad/Sum_1Sumgradients_1/sub_2_grad/Neg.gradients_1/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ľ
 gradients_1/sub_2_grad/Reshape_1Reshapegradients_1/sub_2_grad/Sum_1gradients_1/sub_2_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_2_grad/tuple/group_depsNoOp^gradients_1/sub_2_grad/Reshape!^gradients_1/sub_2_grad/Reshape_1
ć
/gradients_1/sub_2_grad/tuple/control_dependencyIdentitygradients_1/sub_2_grad/Reshape(^gradients_1/sub_2_grad/tuple/group_deps*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*1
_class'
%#loc:@gradients_1/sub_2_grad/Reshape*
T0
ě
1gradients_1/sub_2_grad/tuple/control_dependency_1Identity gradients_1/sub_2_grad/Reshape_1(^gradients_1/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_2_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
&gradients_1/main/q1/Squeeze_grad/ShapeShapemain/q1/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:
Ě
(gradients_1/main/q1/Squeeze_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency&gradients_1/main/q1/Squeeze_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
}
&gradients_1/main/q2/Squeeze_grad/ShapeShapemain/q2/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
Ě
(gradients_1/main/q2/Squeeze_grad/ReshapeReshape/gradients_1/sub_2_grad/tuple/control_dependency&gradients_1/main/q2/Squeeze_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Š
4gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/main/q1/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
Ł
9gradients_1/main/q1/dense_2/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/main/q1/Squeeze_grad/Reshape5^gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGrad
˘
Agradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/main/q1/Squeeze_grad/Reshape:^gradients_1/main/q1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/main/q1/Squeeze_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Cgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*G
_class=
;9loc:@gradients_1/main/q1/dense_2/BiasAdd_grad/BiasAddGrad
Š
4gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/main/q2/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Ł
9gradients_1/main/q2/dense_2/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/main/q2/Squeeze_grad/Reshape5^gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGrad
˘
Agradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/main/q2/Squeeze_grad/Reshape:^gradients_1/main/q2/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*;
_class1
/-loc:@gradients_1/main/q2/Squeeze_grad/Reshape*
T0
Ż
Cgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q2/dense_2/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*G
_class=
;9loc:@gradients_1/main/q2/dense_2/BiasAdd_grad/BiasAddGrad
ń
.gradients_1/main/q1/dense_2/MatMul_grad/MatMulMatMulAgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q1/dense_2/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 
ă
0gradients_1/main/q1/dense_2/MatMul_grad/MatMul_1MatMulmain/q1/dense_1/ReluAgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
¤
8gradients_1/main/q1/dense_2/MatMul_grad/tuple/group_depsNoOp/^gradients_1/main/q1/dense_2/MatMul_grad/MatMul1^gradients_1/main/q1/dense_2/MatMul_grad/MatMul_1
­
@gradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q1/dense_2/MatMul_grad/MatMul9^gradients_1/main/q1/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q1/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Bgradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q1/dense_2/MatMul_grad/MatMul_19^gradients_1/main/q1/dense_2/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*C
_class9
75loc:@gradients_1/main/q1/dense_2/MatMul_grad/MatMul_1
ń
.gradients_1/main/q2/dense_2/MatMul_grad/MatMulMatMulAgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependencymain/q2/dense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ă
0gradients_1/main/q2/dense_2/MatMul_grad/MatMul_1MatMulmain/q2/dense_1/ReluAgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	*
T0*
transpose_a(
¤
8gradients_1/main/q2/dense_2/MatMul_grad/tuple/group_depsNoOp/^gradients_1/main/q2/dense_2/MatMul_grad/MatMul1^gradients_1/main/q2/dense_2/MatMul_grad/MatMul_1
­
@gradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q2/dense_2/MatMul_grad/MatMul9^gradients_1/main/q2/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*A
_class7
53loc:@gradients_1/main/q2/dense_2/MatMul_grad/MatMul
Ş
Bgradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q2/dense_2/MatMul_grad/MatMul_19^gradients_1/main/q2/dense_2/MatMul_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/main/q2/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Ĺ
.gradients_1/main/q1/dense_1/Relu_grad/ReluGradReluGrad@gradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependencymain/q1/dense_1/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
.gradients_1/main/q2/dense_1/Relu_grad/ReluGradReluGrad@gradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependencymain/q2/dense_1/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
4gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_1/main/q1/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Š
9gradients_1/main/q1/dense_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_1/main/q1/dense_1/Relu_grad/ReluGrad
Ż
Agradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_1/main/q1/dense_1/Relu_grad/ReluGrad:^gradients_1/main/q1/dense_1/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*A
_class7
53loc:@gradients_1/main/q1/dense_1/Relu_grad/ReluGrad*
T0
°
Cgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q1/dense_1/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/main/q1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
°
4gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_1/main/q2/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes	
:
Š
9gradients_1/main/q2/dense_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_1/main/q2/dense_1/Relu_grad/ReluGrad
Ż
Agradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_1/main/q2/dense_1/Relu_grad/ReluGrad:^gradients_1/main/q2/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q2/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Cgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_1/main/q2/dense_1/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/main/q2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ń
.gradients_1/main/q1/dense_1/MatMul_grad/MatMulMatMulAgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q1/dense_1/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0
â
0gradients_1/main/q1/dense_1/MatMul_grad/MatMul_1MatMulmain/q1/dense/ReluAgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:

¤
8gradients_1/main/q1/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients_1/main/q1/dense_1/MatMul_grad/MatMul1^gradients_1/main/q1/dense_1/MatMul_grad/MatMul_1
­
@gradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q1/dense_1/MatMul_grad/MatMul9^gradients_1/main/q1/dense_1/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/main/q1/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Bgradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q1/dense_1/MatMul_grad/MatMul_19^gradients_1/main/q1/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/q1/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

ń
.gradients_1/main/q2/dense_1/MatMul_grad/MatMulMatMulAgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependencymain/q2/dense_1/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
â
0gradients_1/main/q2/dense_1/MatMul_grad/MatMul_1MatMulmain/q2/dense/ReluAgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( * 
_output_shapes
:
*
T0*
transpose_a(
¤
8gradients_1/main/q2/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients_1/main/q2/dense_1/MatMul_grad/MatMul1^gradients_1/main/q2/dense_1/MatMul_grad/MatMul_1
­
@gradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/q2/dense_1/MatMul_grad/MatMul9^gradients_1/main/q2/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*A
_class7
53loc:@gradients_1/main/q2/dense_1/MatMul_grad/MatMul
Ť
Bgradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/q2/dense_1/MatMul_grad/MatMul_19^gradients_1/main/q2/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/q2/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients_1/main/q1/dense/Relu_grad/ReluGradReluGrad@gradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependencymain/q1/dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
,gradients_1/main/q2/dense/Relu_grad/ReluGradReluGrad@gradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependencymain/q2/dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
2gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/main/q1/dense/Relu_grad/ReluGrad*
T0*
_output_shapes	
:*
data_formatNHWC
Ł
7gradients_1/main/q1/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/main/q1/dense/Relu_grad/ReluGrad
§
?gradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/main/q1/dense/Relu_grad/ReluGrad8^gradients_1/main/q1/dense/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/main/q1/dense/Relu_grad/ReluGrad*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/main/q1/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*E
_class;
97loc:@gradients_1/main/q1/dense/BiasAdd_grad/BiasAddGrad*
T0
Ź
2gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/main/q2/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients_1/main/q2/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/main/q2/dense/Relu_grad/ReluGrad
§
?gradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/main/q2/dense/Relu_grad/ReluGrad8^gradients_1/main/q2/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients_1/main/q2/dense/Relu_grad/ReluGrad
¨
Agradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/main/q2/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/main/q2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ę
,gradients_1/main/q1/dense/MatMul_grad/MatMulMatMul?gradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependencymain/q1/dense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
transpose_a( 
Ů
.gradients_1/main/q1/dense/MatMul_grad/MatMul_1MatMulmain/q1/concat?gradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	*
T0*
transpose_a(

6gradients_1/main/q1/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/main/q1/dense/MatMul_grad/MatMul/^gradients_1/main/q1/dense/MatMul_grad/MatMul_1
¤
>gradients_1/main/q1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/main/q1/dense/MatMul_grad/MatMul7^gradients_1/main/q1/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/main/q1/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
@gradients_1/main/q1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/main/q1/dense/MatMul_grad/MatMul_17^gradients_1/main/q1/dense/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*A
_class7
53loc:@gradients_1/main/q1/dense/MatMul_grad/MatMul_1
ę
,gradients_1/main/q2/dense/MatMul_grad/MatMulMatMul?gradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependencymain/q2/dense/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
transpose_a( *
T0
Ů
.gradients_1/main/q2/dense/MatMul_grad/MatMul_1MatMulmain/q2/concat?gradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	*
T0*
transpose_a(

6gradients_1/main/q2/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/main/q2/dense/MatMul_grad/MatMul/^gradients_1/main/q2/dense/MatMul_grad/MatMul_1
¤
>gradients_1/main/q2/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/main/q2/dense/MatMul_grad/MatMul7^gradients_1/main/q2/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*?
_class5
31loc:@gradients_1/main/q2/dense/MatMul_grad/MatMul*
T0
˘
@gradients_1/main/q2/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/main/q2/dense/MatMul_grad/MatMul_17^gradients_1/main/q2/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/q2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	

beta1_power_1/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: *%
_class
loc:@main/q1/dense/bias

beta1_power_1
VariableV2*
shared_name *
	container *
dtype0*
shape: *%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
ť
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
T0
u
beta1_power_1/readIdentitybeta1_power_1*%
_class
loc:@main/q1/dense/bias*
T0*
_output_shapes
: 

beta2_power_1/initial_valueConst*%
_class
loc:@main/q1/dense/bias*
valueB
 *wž?*
_output_shapes
: *
dtype0

beta2_power_1
VariableV2*
shape: *
	container *
dtype0*
shared_name *%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
ť
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *%
_class
loc:@main/q1/dense/bias*
use_locking(*
T0
u
beta2_power_1/readIdentitybeta2_power_1*
T0*
_output_shapes
: *%
_class
loc:@main/q1/dense/bias
ľ
;main/q1/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_output_shapes
:*'
_class
loc:@main/q1/dense/kernel*
dtype0

1main/q1/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
: *
valueB
 *    

+main/q1/dense/kernel/Adam/Initializer/zerosFill;main/q1/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/q1/dense/kernel/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	*

index_type0*'
_class
loc:@main/q1/dense/kernel
¸
main/q1/dense/kernel/Adam
VariableV2*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	*
shape:	*
	container *
shared_name *
dtype0
î
 main/q1/dense/kernel/Adam/AssignAssignmain/q1/dense/kernel/Adam+main/q1/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	*
T0

main/q1/dense/kernel/Adam/readIdentitymain/q1/dense/kernel/Adam*
_output_shapes
:	*
T0*'
_class
loc:@main/q1/dense/kernel
ˇ
=main/q1/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:*
valueB"      
Ą
3main/q1/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *'
_class
loc:@main/q1/dense/kernel*
valueB
 *    *
dtype0

-main/q1/dense/kernel/Adam_1/Initializer/zerosFill=main/q1/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/q1/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*

index_type0*
T0*'
_class
loc:@main/q1/dense/kernel
ş
main/q1/dense/kernel/Adam_1
VariableV2*
_output_shapes
:	*
shared_name *
shape:	*
dtype0*'
_class
loc:@main/q1/dense/kernel*
	container 
ô
"main/q1/dense/kernel/Adam_1/AssignAssignmain/q1/dense/kernel/Adam_1-main/q1/dense/kernel/Adam_1/Initializer/zeros*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(

 main/q1/dense/kernel/Adam_1/readIdentitymain/q1/dense/kernel/Adam_1*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	

)main/q1/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@main/q1/dense/bias*
dtype0*
valueB*    *
_output_shapes	
:
Ź
main/q1/dense/bias/Adam
VariableV2*
shared_name *
_output_shapes	
:*%
_class
loc:@main/q1/dense/bias*
	container *
shape:*
dtype0
â
main/q1/dense/bias/Adam/AssignAssignmain/q1/dense/bias/Adam)main/q1/dense/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*%
_class
loc:@main/q1/dense/bias

main/q1/dense/bias/Adam/readIdentitymain/q1/dense/bias/Adam*%
_class
loc:@main/q1/dense/bias*
T0*
_output_shapes	
:
Ą
+main/q1/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*%
_class
loc:@main/q1/dense/bias*
valueB*    
Ž
main/q1/dense/bias/Adam_1
VariableV2*
shared_name *
	container *
shape:*
_output_shapes	
:*%
_class
loc:@main/q1/dense/bias*
dtype0
č
 main/q1/dense/bias/Adam_1/AssignAssignmain/q1/dense/bias/Adam_1+main/q1/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(

main/q1/dense/bias/Adam_1/readIdentitymain/q1/dense/bias/Adam_1*
_output_shapes	
:*
T0*%
_class
loc:@main/q1/dense/bias
š
=main/q1/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
valueB"      *
_output_shapes
:
Ł
3main/q1/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
valueB
 *    

-main/q1/dense_1/kernel/Adam/Initializer/zerosFill=main/q1/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/q1/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*

index_type0
ž
main/q1/dense_1/kernel/Adam
VariableV2*)
_class
loc:@main/q1/dense_1/kernel*
	container * 
_output_shapes
:
*
dtype0*
shape:
*
shared_name 
÷
"main/q1/dense_1/kernel/Adam/AssignAssignmain/q1/dense_1/kernel/Adam-main/q1/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*)
_class
loc:@main/q1/dense_1/kernel

 main/q1/dense_1/kernel/Adam/readIdentitymain/q1/dense_1/kernel/Adam*
T0* 
_output_shapes
:
*)
_class
loc:@main/q1/dense_1/kernel
ť
?main/q1/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *)
_class
loc:@main/q1/dense_1/kernel*
dtype0
Ľ
5main/q1/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *)
_class
loc:@main/q1/dense_1/kernel

/main/q1/dense_1/kernel/Adam_1/Initializer/zerosFill?main/q1/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/q1/dense_1/kernel/Adam_1/Initializer/zeros/Const*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*

index_type0*
T0
Ŕ
main/q1/dense_1/kernel/Adam_1
VariableV2*
	container *
shared_name *
shape:
*)
_class
loc:@main/q1/dense_1/kernel*
dtype0* 
_output_shapes
:

ý
$main/q1/dense_1/kernel/Adam_1/AssignAssignmain/q1/dense_1/kernel/Adam_1/main/q1/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*
use_locking(*
T0
Ł
"main/q1/dense_1/kernel/Adam_1/readIdentitymain/q1/dense_1/kernel/Adam_1*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*
T0
Ł
+main/q1/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes	
:*
valueB*    
°
main/q1/dense_1/bias/Adam
VariableV2*'
_class
loc:@main/q1/dense_1/bias*
shared_name *
_output_shapes	
:*
dtype0*
	container *
shape:
ę
 main/q1/dense_1/bias/Adam/AssignAssignmain/q1/dense_1/bias/Adam+main/q1/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias

main/q1/dense_1/bias/Adam/readIdentitymain/q1/dense_1/bias/Adam*
T0*
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias
Ľ
-main/q1/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*
dtype0*'
_class
loc:@main/q1/dense_1/bias*
valueB*    
˛
main/q1/dense_1/bias/Adam_1
VariableV2*
dtype0*
shape:*
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias*
	container *
shared_name 
đ
"main/q1/dense_1/bias/Adam_1/AssignAssignmain/q1/dense_1/bias/Adam_1-main/q1/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
T0*'
_class
loc:@main/q1/dense_1/bias*
use_locking(*
validate_shape(

 main/q1/dense_1/bias/Adam_1/readIdentitymain/q1/dense_1/bias/Adam_1*
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias*
T0
Ż
-main/q1/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	*
dtype0*)
_class
loc:@main/q1/dense_2/kernel*
valueB	*    
ź
main/q1/dense_2/kernel/Adam
VariableV2*
shape:	*
shared_name *
_output_shapes
:	*
dtype0*
	container *)
_class
loc:@main/q1/dense_2/kernel
ö
"main/q1/dense_2/kernel/Adam/AssignAssignmain/q1/dense_2/kernel/Adam-main/q1/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
T0*
use_locking(*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(

 main/q1/dense_2/kernel/Adam/readIdentitymain/q1/dense_2/kernel/Adam*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	*
T0
ą
/main/q1/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
valueB	*    *)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	
ž
main/q1/dense_2/kernel/Adam_1
VariableV2*
shared_name *
dtype0*
shape:	*
	container *
_output_shapes
:	*)
_class
loc:@main/q1/dense_2/kernel
ü
$main/q1/dense_2/kernel/Adam_1/AssignAssignmain/q1/dense_2/kernel/Adam_1/main/q1/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	*)
_class
loc:@main/q1/dense_2/kernel*
use_locking(*
T0
˘
"main/q1/dense_2/kernel/Adam_1/readIdentitymain/q1/dense_2/kernel/Adam_1*
_output_shapes
:	*
T0*)
_class
loc:@main/q1/dense_2/kernel
Ą
+main/q1/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:
Ž
main/q1/dense_2/bias/Adam
VariableV2*
_output_shapes
:*'
_class
loc:@main/q1/dense_2/bias*
dtype0*
shape:*
	container *
shared_name 
é
 main/q1/dense_2/bias/Adam/AssignAssignmain/q1/dense_2/bias/Adam+main/q1/dense_2/bias/Adam/Initializer/zeros*'
_class
loc:@main/q1/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(

main/q1/dense_2/bias/Adam/readIdentitymain/q1/dense_2/bias/Adam*'
_class
loc:@main/q1/dense_2/bias*
T0*
_output_shapes
:
Ł
-main/q1/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:*
dtype0*
valueB*    
°
main/q1/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*'
_class
loc:@main/q1/dense_2/bias*
	container *
shape:*
dtype0*
shared_name 
ď
"main/q1/dense_2/bias/Adam_1/AssignAssignmain/q1/dense_2/bias/Adam_1-main/q1/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*'
_class
loc:@main/q1/dense_2/bias

 main/q1/dense_2/bias/Adam_1/readIdentitymain/q1/dense_2/bias/Adam_1*
_output_shapes
:*'
_class
loc:@main/q1/dense_2/bias*
T0
ľ
;main/q2/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_output_shapes
:*'
_class
loc:@main/q2/dense/kernel*
dtype0

1main/q2/dense/kernel/Adam/Initializer/zeros/ConstConst*'
_class
loc:@main/q2/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0

+main/q2/dense/kernel/Adam/Initializer/zerosFill;main/q2/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1main/q2/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	*
T0*'
_class
loc:@main/q2/dense/kernel*

index_type0
¸
main/q2/dense/kernel/Adam
VariableV2*
_output_shapes
:	*'
_class
loc:@main/q2/dense/kernel*
dtype0*
shared_name *
	container *
shape:	
î
 main/q2/dense/kernel/Adam/AssignAssignmain/q2/dense/kernel/Adam+main/q2/dense/kernel/Adam/Initializer/zeros*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

main/q2/dense/kernel/Adam/readIdentitymain/q2/dense/kernel/Adam*
_output_shapes
:	*'
_class
loc:@main/q2/dense/kernel*
T0
ˇ
=main/q2/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *'
_class
loc:@main/q2/dense/kernel*
_output_shapes
:*
dtype0
Ą
3main/q2/dense/kernel/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@main/q2/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-main/q2/dense/kernel/Adam_1/Initializer/zerosFill=main/q2/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3main/q2/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*
T0*'
_class
loc:@main/q2/dense/kernel*

index_type0
ş
main/q2/dense/kernel/Adam_1
VariableV2*
shape:	*
shared_name *
dtype0*
	container *
_output_shapes
:	*'
_class
loc:@main/q2/dense/kernel
ô
"main/q2/dense/kernel/Adam_1/AssignAssignmain/q2/dense/kernel/Adam_1-main/q2/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	*
validate_shape(*
T0*'
_class
loc:@main/q2/dense/kernel*
use_locking(

 main/q2/dense/kernel/Adam_1/readIdentitymain/q2/dense/kernel/Adam_1*
T0*
_output_shapes
:	*'
_class
loc:@main/q2/dense/kernel

)main/q2/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*%
_class
loc:@main/q2/dense/bias*
valueB*    *
dtype0
Ź
main/q2/dense/bias/Adam
VariableV2*
_output_shapes	
:*%
_class
loc:@main/q2/dense/bias*
shape:*
	container *
shared_name *
dtype0
â
main/q2/dense/bias/Adam/AssignAssignmain/q2/dense/bias/Adam)main/q2/dense/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*%
_class
loc:@main/q2/dense/bias

main/q2/dense/bias/Adam/readIdentitymain/q2/dense/bias/Adam*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes	
:
Ą
+main/q2/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@main/q2/dense/bias*
dtype0*
_output_shapes	
:
Ž
main/q2/dense/bias/Adam_1
VariableV2*
dtype0*
shape:*
shared_name *
	container *%
_class
loc:@main/q2/dense/bias*
_output_shapes	
:
č
 main/q2/dense/bias/Adam_1/AssignAssignmain/q2/dense/bias/Adam_1+main/q2/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*%
_class
loc:@main/q2/dense/bias

main/q2/dense/bias/Adam_1/readIdentitymain/q2/dense/bias/Adam_1*%
_class
loc:@main/q2/dense/bias*
T0*
_output_shapes	
:
š
=main/q2/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0*)
_class
loc:@main/q2/dense_1/kernel
Ł
3main/q2/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes
: 

-main/q2/dense_1/kernel/Adam/Initializer/zerosFill=main/q2/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/q2/dense_1/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@main/q2/dense_1/kernel*
T0*

index_type0* 
_output_shapes
:

ž
main/q2/dense_1/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0*
shared_name * 
_output_shapes
:
*)
_class
loc:@main/q2/dense_1/kernel
÷
"main/q2/dense_1/kernel/Adam/AssignAssignmain/q2/dense_1/kernel/Adam-main/q2/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
T0*)
_class
loc:@main/q2/dense_1/kernel*
use_locking(

 main/q2/dense_1/kernel/Adam/readIdentitymain/q2/dense_1/kernel/Adam*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
*
T0
ť
?main/q2/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
valueB"      *
_output_shapes
:
Ľ
5main/q2/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *)
_class
loc:@main/q2/dense_1/kernel*
dtype0

/main/q2/dense_1/kernel/Adam_1/Initializer/zerosFill?main/q2/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/q2/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*

index_type0*)
_class
loc:@main/q2/dense_1/kernel
Ŕ
main/q2/dense_1/kernel/Adam_1
VariableV2*
shared_name *
shape:
*
	container *)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:
*
dtype0
ý
$main/q2/dense_1/kernel/Adam_1/AssignAssignmain/q2/dense_1/kernel/Adam_1/main/q2/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*)
_class
loc:@main/q2/dense_1/kernel*
T0*
validate_shape(* 
_output_shapes
:

Ł
"main/q2/dense_1/kernel/Adam_1/readIdentitymain/q2/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:

Ł
+main/q2/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q2/dense_1/bias*
valueB*    *
_output_shapes	
:*
dtype0
°
main/q2/dense_1/bias/Adam
VariableV2*
dtype0*'
_class
loc:@main/q2/dense_1/bias*
	container *
shared_name *
shape:*
_output_shapes	
:
ę
 main/q2/dense_1/bias/Adam/AssignAssignmain/q2/dense_1/bias/Adam+main/q2/dense_1/bias/Adam/Initializer/zeros*
use_locking(*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:*
validate_shape(*
T0

main/q2/dense_1/bias/Adam/readIdentitymain/q2/dense_1/bias/Adam*'
_class
loc:@main/q2/dense_1/bias*
T0*
_output_shapes	
:
Ľ
-main/q2/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q2/dense_1/bias*
dtype0*
valueB*    *
_output_shapes	
:
˛
main/q2/dense_1/bias/Adam_1
VariableV2*
	container *
dtype0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:*
shape:*
shared_name 
đ
"main/q2/dense_1/bias/Adam_1/AssignAssignmain/q2/dense_1/bias/Adam_1-main/q2/dense_1/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:*
validate_shape(

 main/q2/dense_1/bias/Adam_1/readIdentitymain/q2/dense_1/bias/Adam_1*
T0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:
Ż
-main/q2/dense_2/kernel/Adam/Initializer/zerosConst*
valueB	*    *
_output_shapes
:	*
dtype0*)
_class
loc:@main/q2/dense_2/kernel
ź
main/q2/dense_2/kernel/Adam
VariableV2*
dtype0*
shape:	*
	container *)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	*
shared_name 
ö
"main/q2/dense_2/kernel/Adam/AssignAssignmain/q2/dense_2/kernel/Adam-main/q2/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	*)
_class
loc:@main/q2/dense_2/kernel*
use_locking(*
validate_shape(*
T0

 main/q2/dense_2/kernel/Adam/readIdentitymain/q2/dense_2/kernel/Adam*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	*
T0
ą
/main/q2/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes
:	*
dtype0*
valueB	*    *)
_class
loc:@main/q2/dense_2/kernel
ž
main/q2/dense_2/kernel/Adam_1
VariableV2*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	*
shared_name *
	container *
dtype0*
shape:	
ü
$main/q2/dense_2/kernel/Adam_1/AssignAssignmain/q2/dense_2/kernel/Adam_1/main/q2/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:	*)
_class
loc:@main/q2/dense_2/kernel*
T0
˘
"main/q2/dense_2/kernel/Adam_1/readIdentitymain/q2/dense_2/kernel/Adam_1*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	*
T0
Ą
+main/q2/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*'
_class
loc:@main/q2/dense_2/bias
Ž
main/q2/dense_2/bias/Adam
VariableV2*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:*
	container *
shared_name *
dtype0*
shape:
é
 main/q2/dense_2/bias/Adam/AssignAssignmain/q2/dense_2/bias/Adam+main/q2/dense_2/bias/Adam/Initializer/zeros*'
_class
loc:@main/q2/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:

main/q2/dense_2/bias/Adam/readIdentitymain/q2/dense_2/bias/Adam*
T0*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:
Ł
-main/q2/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*'
_class
loc:@main/q2/dense_2/bias*
dtype0*
valueB*    
°
main/q2/dense_2/bias/Adam_1
VariableV2*
shape:*
shared_name *
_output_shapes
:*
dtype0*'
_class
loc:@main/q2/dense_2/bias*
	container 
ď
"main/q2/dense_2/bias/Adam_1/AssignAssignmain/q2/dense_2/bias/Adam_1-main/q2/dense_2/bias/Adam_1/Initializer/zeros*'
_class
loc:@main/q2/dense_2/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(

 main/q2/dense_2/bias/Adam_1/readIdentitymain/q2/dense_2/bias/Adam_1*
T0*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:
Y
Adam_1/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Q
Adam_1/beta2Const*
valueB
 *wž?*
_output_shapes
: *
dtype0
S
Adam_1/epsilonConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
­
,Adam_1/update_main/q1/dense/kernel/ApplyAdam	ApplyAdammain/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/q1/dense/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *'
_class
loc:@main/q1/dense/kernel*
_output_shapes
:	
 
*Adam_1/update_main/q1/dense/bias/ApplyAdam	ApplyAdammain/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q1/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*%
_class
loc:@main/q1/dense/bias
ş
.Adam_1/update_main/q1/dense_1/kernel/ApplyAdam	ApplyAdammain/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *)
_class
loc:@main/q1/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
*
T0
Ź
,Adam_1/update_main/q1/dense_1/bias/ApplyAdam	ApplyAdammain/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias*
use_locking( *
T0
š
.Adam_1/update_main/q1/dense_2/kernel/ApplyAdam	ApplyAdammain/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q1/dense_2/MatMul_grad/tuple/control_dependency_1*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
:	*
use_nesterov( *
T0*
use_locking( 
Ť
,Adam_1/update_main/q1/dense_2/bias/ApplyAdam	ApplyAdammain/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
_output_shapes
:*'
_class
loc:@main/q1/dense_2/bias*
use_nesterov( 
­
,Adam_1/update_main/q2/dense/kernel/ApplyAdam	ApplyAdammain/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/q2/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes
:	*
T0*
use_nesterov( *'
_class
loc:@main/q2/dense/kernel
 
*Adam_1/update_main/q2/dense/bias/ApplyAdam	ApplyAdammain/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/q2/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*%
_class
loc:@main/q2/dense/bias*
T0*
use_locking( *
use_nesterov( 
ş
.Adam_1/update_main/q2/dense_1/kernel/ApplyAdam	ApplyAdammain/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q2/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *)
_class
loc:@main/q2/dense_1/kernel* 
_output_shapes
:

Ź
,Adam_1/update_main/q2/dense_1/bias/ApplyAdam	ApplyAdammain/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/q2/dense_1/bias*
use_nesterov( *
_output_shapes	
:
š
.Adam_1/update_main/q2/dense_2/kernel/ApplyAdam	ApplyAdammain/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/q2/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
use_locking( *
use_nesterov( *)
_class
loc:@main/q2/dense_2/kernel*
T0
Ť
,Adam_1/update_main/q2/dense_2/bias/ApplyAdam	ApplyAdammain/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*
use_nesterov( *'
_class
loc:@main/q2/dense_2/bias
ł

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1+^Adam_1/update_main/q1/dense/bias/ApplyAdam-^Adam_1/update_main/q1/dense/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_1/bias/ApplyAdam/^Adam_1/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_2/bias/ApplyAdam/^Adam_1/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_1/update_main/q2/dense/bias/ApplyAdam-^Adam_1/update_main/q2/dense/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_1/bias/ApplyAdam/^Adam_1/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_2/bias/ApplyAdam/^Adam_1/update_main/q2/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
Ł
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *%
_class
loc:@main/q1/dense/bias*
T0*
_output_shapes
: *
validate_shape(
ľ
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2+^Adam_1/update_main/q1/dense/bias/ApplyAdam-^Adam_1/update_main/q1/dense/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_1/bias/ApplyAdam/^Adam_1/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_2/bias/ApplyAdam/^Adam_1/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_1/update_main/q2/dense/bias/ApplyAdam-^Adam_1/update_main/q2/dense/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_1/bias/ApplyAdam/^Adam_1/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_2/bias/ApplyAdam/^Adam_1/update_main/q2/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
§
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
č
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1+^Adam_1/update_main/q1/dense/bias/ApplyAdam-^Adam_1/update_main/q1/dense/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_1/bias/ApplyAdam/^Adam_1/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q1/dense_2/bias/ApplyAdam/^Adam_1/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_1/update_main/q2/dense/bias/ApplyAdam-^Adam_1/update_main/q2/dense/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_1/bias/ApplyAdam/^Adam_1/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_1/update_main/q2/dense_2/bias/ApplyAdam/^Adam_1/update_main/q2/dense_2/kernel/ApplyAdam
L
mul_2/xConst*
dtype0*
valueB
 *R¸~?*
_output_shapes
: 
\
mul_2Mulmul_2/xtarget/pi/dense/kernel/read*
T0*
_output_shapes
:	
L
mul_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×Ł;
Z
mul_3Mulmul_3/xmain/pi/dense/kernel/read*
T0*
_output_shapes
:	
F
add_2AddV2mul_2mul_3*
T0*
_output_shapes
:	
­
AssignAssigntarget/pi/dense/kerneladd_2*
validate_shape(*
use_locking(*
_output_shapes
:	*)
_class
loc:@target/pi/dense/kernel*
T0
L
mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *R¸~?
V
mul_4Mulmul_4/xtarget/pi/dense/bias/read*
_output_shapes	
:*
T0
L
mul_5/xConst*
valueB
 *
×Ł;*
dtype0*
_output_shapes
: 
T
mul_5Mulmul_5/xmain/pi/dense/bias/read*
_output_shapes	
:*
T0
B
add_3AddV2mul_4mul_5*
T0*
_output_shapes	
:
§
Assign_1Assigntarget/pi/dense/biasadd_3*'
_class
loc:@target/pi/dense/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
L
mul_6/xConst*
_output_shapes
: *
valueB
 *R¸~?*
dtype0
_
mul_6Mulmul_6/xtarget/pi/dense_1/kernel/read* 
_output_shapes
:
*
T0
L
mul_7/xConst*
_output_shapes
: *
valueB
 *
×Ł;*
dtype0
]
mul_7Mulmul_7/xmain/pi/dense_1/kernel/read* 
_output_shapes
:
*
T0
G
add_4AddV2mul_6mul_7*
T0* 
_output_shapes
:

´
Assign_2Assigntarget/pi/dense_1/kerneladd_4*
validate_shape(*+
_class!
loc:@target/pi/dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:

L
mul_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *R¸~?
X
mul_8Mulmul_8/xtarget/pi/dense_1/bias/read*
_output_shapes	
:*
T0
L
mul_9/xConst*
_output_shapes
: *
valueB
 *
×Ł;*
dtype0
V
mul_9Mulmul_9/xmain/pi/dense_1/bias/read*
T0*
_output_shapes	
:
B
add_5AddV2mul_8mul_9*
_output_shapes	
:*
T0
Ť
Assign_3Assigntarget/pi/dense_1/biasadd_5*)
_class
loc:@target/pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:
M
mul_10/xConst*
valueB
 *R¸~?*
_output_shapes
: *
dtype0
`
mul_10Mulmul_10/xtarget/pi/dense_2/kernel/read*
T0*
_output_shapes
:	
M
mul_11/xConst*
_output_shapes
: *
valueB
 *
×Ł;*
dtype0
^
mul_11Mulmul_11/xmain/pi/dense_2/kernel/read*
T0*
_output_shapes
:	
H
add_6AddV2mul_10mul_11*
T0*
_output_shapes
:	
ł
Assign_4Assigntarget/pi/dense_2/kerneladd_6*
validate_shape(*
use_locking(*
_output_shapes
:	*+
_class!
loc:@target/pi/dense_2/kernel*
T0
M
mul_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *R¸~?
Y
mul_12Mulmul_12/xtarget/pi/dense_2/bias/read*
_output_shapes
:*
T0
M
mul_13/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×Ł;
W
mul_13Mulmul_13/xmain/pi/dense_2/bias/read*
_output_shapes
:*
T0
C
add_7AddV2mul_12mul_13*
T0*
_output_shapes
:
Ş
Assign_5Assigntarget/pi/dense_2/biasadd_7*
T0*
_output_shapes
:*
use_locking(*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(
M
mul_14/xConst*
valueB
 *R¸~?*
_output_shapes
: *
dtype0
^
mul_14Mulmul_14/xtarget/q1/dense/kernel/read*
_output_shapes
:	*
T0
M
mul_15/xConst*
_output_shapes
: *
valueB
 *
×Ł;*
dtype0
\
mul_15Mulmul_15/xmain/q1/dense/kernel/read*
T0*
_output_shapes
:	
H
add_8AddV2mul_14mul_15*
_output_shapes
:	*
T0
Ż
Assign_6Assigntarget/q1/dense/kerneladd_8*
_output_shapes
:	*
T0*
use_locking(*)
_class
loc:@target/q1/dense/kernel*
validate_shape(
M
mul_16/xConst*
valueB
 *R¸~?*
_output_shapes
: *
dtype0
X
mul_16Mulmul_16/xtarget/q1/dense/bias/read*
_output_shapes	
:*
T0
M
mul_17/xConst*
valueB
 *
×Ł;*
_output_shapes
: *
dtype0
V
mul_17Mulmul_17/xmain/q1/dense/bias/read*
_output_shapes	
:*
T0
D
add_9AddV2mul_16mul_17*
T0*
_output_shapes	
:
§
Assign_7Assigntarget/q1/dense/biasadd_9*
_output_shapes	
:*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
T0*
use_locking(
M
mul_18/xConst*
dtype0*
_output_shapes
: *
valueB
 *R¸~?
a
mul_18Mulmul_18/xtarget/q1/dense_1/kernel/read* 
_output_shapes
:
*
T0
M
mul_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×Ł;
_
mul_19Mulmul_19/xmain/q1/dense_1/kernel/read* 
_output_shapes
:
*
T0
J
add_10AddV2mul_18mul_19* 
_output_shapes
:
*
T0
ľ
Assign_8Assigntarget/q1/dense_1/kerneladd_10*
T0*
validate_shape(* 
_output_shapes
:
*
use_locking(*+
_class!
loc:@target/q1/dense_1/kernel
M
mul_20/xConst*
_output_shapes
: *
valueB
 *R¸~?*
dtype0
Z
mul_20Mulmul_20/xtarget/q1/dense_1/bias/read*
T0*
_output_shapes	
:
M
mul_21/xConst*
_output_shapes
: *
valueB
 *
×Ł;*
dtype0
X
mul_21Mulmul_21/xmain/q1/dense_1/bias/read*
T0*
_output_shapes	
:
E
add_11AddV2mul_20mul_21*
T0*
_output_shapes	
:
Ź
Assign_9Assigntarget/q1/dense_1/biasadd_11*)
_class
loc:@target/q1/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(
M
mul_22/xConst*
valueB
 *R¸~?*
_output_shapes
: *
dtype0
`
mul_22Mulmul_22/xtarget/q1/dense_2/kernel/read*
_output_shapes
:	*
T0
M
mul_23/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×Ł;
^
mul_23Mulmul_23/xmain/q1/dense_2/kernel/read*
_output_shapes
:	*
T0
I
add_12AddV2mul_22mul_23*
_output_shapes
:	*
T0
ľ
	Assign_10Assigntarget/q1/dense_2/kerneladd_12*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *R¸~?
Y
mul_24Mulmul_24/xtarget/q1/dense_2/bias/read*
_output_shapes
:*
T0
M
mul_25/xConst*
valueB
 *
×Ł;*
_output_shapes
: *
dtype0
W
mul_25Mulmul_25/xmain/q1/dense_2/bias/read*
_output_shapes
:*
T0
D
add_13AddV2mul_24mul_25*
_output_shapes
:*
T0
Ź
	Assign_11Assigntarget/q1/dense_2/biasadd_13*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(
M
mul_26/xConst*
dtype0*
valueB
 *R¸~?*
_output_shapes
: 
^
mul_26Mulmul_26/xtarget/q2/dense/kernel/read*
T0*
_output_shapes
:	
M
mul_27/xConst*
valueB
 *
×Ł;*
dtype0*
_output_shapes
: 
\
mul_27Mulmul_27/xmain/q2/dense/kernel/read*
T0*
_output_shapes
:	
I
add_14AddV2mul_26mul_27*
T0*
_output_shapes
:	
ą
	Assign_12Assigntarget/q2/dense/kerneladd_14*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*)
_class
loc:@target/q2/dense/kernel
M
mul_28/xConst*
dtype0*
valueB
 *R¸~?*
_output_shapes
: 
X
mul_28Mulmul_28/xtarget/q2/dense/bias/read*
_output_shapes	
:*
T0
M
mul_29/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×Ł;
V
mul_29Mulmul_29/xmain/q2/dense/bias/read*
_output_shapes	
:*
T0
E
add_15AddV2mul_28mul_29*
T0*
_output_shapes	
:
Š
	Assign_13Assigntarget/q2/dense/biasadd_15*
validate_shape(*
T0*'
_class
loc:@target/q2/dense/bias*
_output_shapes	
:*
use_locking(
M
mul_30/xConst*
dtype0*
_output_shapes
: *
valueB
 *R¸~?
a
mul_30Mulmul_30/xtarget/q2/dense_1/kernel/read* 
_output_shapes
:
*
T0
M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×Ł;
_
mul_31Mulmul_31/xmain/q2/dense_1/kernel/read*
T0* 
_output_shapes
:

J
add_16AddV2mul_30mul_31* 
_output_shapes
:
*
T0
ś
	Assign_14Assigntarget/q2/dense_1/kerneladd_16*+
_class!
loc:@target/q2/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

M
mul_32/xConst*
dtype0*
valueB
 *R¸~?*
_output_shapes
: 
Z
mul_32Mulmul_32/xtarget/q2/dense_1/bias/read*
T0*
_output_shapes	
:
M
mul_33/xConst*
_output_shapes
: *
valueB
 *
×Ł;*
dtype0
X
mul_33Mulmul_33/xmain/q2/dense_1/bias/read*
_output_shapes	
:*
T0
E
add_17AddV2mul_32mul_33*
_output_shapes	
:*
T0
­
	Assign_15Assigntarget/q2/dense_1/biasadd_17*)
_class
loc:@target/q2/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
M
mul_34/xConst*
_output_shapes
: *
valueB
 *R¸~?*
dtype0
`
mul_34Mulmul_34/xtarget/q2/dense_2/kernel/read*
T0*
_output_shapes
:	
M
mul_35/xConst*
valueB
 *
×Ł;*
_output_shapes
: *
dtype0
^
mul_35Mulmul_35/xmain/q2/dense_2/kernel/read*
T0*
_output_shapes
:	
I
add_18AddV2mul_34mul_35*
_output_shapes
:	*
T0
ľ
	Assign_16Assigntarget/q2/dense_2/kerneladd_18*+
_class!
loc:@target/q2/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	
M
mul_36/xConst*
_output_shapes
: *
dtype0*
valueB
 *R¸~?
Y
mul_36Mulmul_36/xtarget/q2/dense_2/bias/read*
T0*
_output_shapes
:
M
mul_37/xConst*
dtype0*
_output_shapes
: *
valueB
 *
×Ł;
W
mul_37Mulmul_37/xmain/q2/dense_2/bias/read*
T0*
_output_shapes
:
D
add_19AddV2mul_36mul_37*
_output_shapes
:*
T0
Ź
	Assign_17Assigntarget/q2/dense_2/biasadd_19*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*)
_class
loc:@target/q2/dense_2/bias
Ţ

group_depsNoOp^Assign	^Assign_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
Ä
	Assign_18Assigntarget/pi/dense/kernelmain/pi/dense/kernel/read*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(*)
_class
loc:@target/pi/dense/kernel
ş
	Assign_19Assigntarget/pi/dense/biasmain/pi/dense/bias/read*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias
Ë
	Assign_20Assigntarget/pi/dense_1/kernelmain/pi/dense_1/kernel/read*
use_locking(*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
*
T0*
validate_shape(
Ŕ
	Assign_21Assigntarget/pi/dense_1/biasmain/pi/dense_1/bias/read*
_output_shapes	
:*
use_locking(*
validate_shape(*
T0*)
_class
loc:@target/pi/dense_1/bias
Ę
	Assign_22Assigntarget/pi/dense_2/kernelmain/pi/dense_2/kernel/read*
_output_shapes
:	*
T0*
validate_shape(*
use_locking(*+
_class!
loc:@target/pi/dense_2/kernel
ż
	Assign_23Assigntarget/pi/dense_2/biasmain/pi/dense_2/bias/read*
_output_shapes
:*)
_class
loc:@target/pi/dense_2/bias*
use_locking(*
validate_shape(*
T0
Ä
	Assign_24Assigntarget/q1/dense/kernelmain/q1/dense/kernel/read*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	*)
_class
loc:@target/q1/dense/kernel
ş
	Assign_25Assigntarget/q1/dense/biasmain/q1/dense/bias/read*
T0*
use_locking(*
validate_shape(*'
_class
loc:@target/q1/dense/bias*
_output_shapes	
:
Ë
	Assign_26Assigntarget/q1/dense_1/kernelmain/q1/dense_1/kernel/read* 
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel
Ŕ
	Assign_27Assigntarget/q1/dense_1/biasmain/q1/dense_1/bias/read*)
_class
loc:@target/q1/dense_1/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
Ę
	Assign_28Assigntarget/q1/dense_2/kernelmain/q1/dense_2/kernel/read*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
ż
	Assign_29Assigntarget/q1/dense_2/biasmain/q1/dense_2/bias/read*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*)
_class
loc:@target/q1/dense_2/bias
Ä
	Assign_30Assigntarget/q2/dense/kernelmain/q2/dense/kernel/read*
_output_shapes
:	*
T0*)
_class
loc:@target/q2/dense/kernel*
use_locking(*
validate_shape(
ş
	Assign_31Assigntarget/q2/dense/biasmain/q2/dense/bias/read*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*'
_class
loc:@target/q2/dense/bias
Ë
	Assign_32Assigntarget/q2/dense_1/kernelmain/q2/dense_1/kernel/read*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*+
_class!
loc:@target/q2/dense_1/kernel
Ŕ
	Assign_33Assigntarget/q2/dense_1/biasmain/q2/dense_1/bias/read*
_output_shapes	
:*)
_class
loc:@target/q2/dense_1/bias*
T0*
use_locking(*
validate_shape(
Ę
	Assign_34Assigntarget/q2/dense_2/kernelmain/q2/dense_2/kernel/read*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*+
_class!
loc:@target/q2/dense_2/kernel
ż
	Assign_35Assigntarget/q2/dense_2/biasmain/q2/dense_2/bias/read*
validate_shape(*
T0*)
_class
loc:@target/q2/dense_2/bias*
use_locking(*
_output_shapes
:
ě
group_deps_1NoOp
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
č
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^main/pi/dense/bias/Adam/Assign!^main/pi/dense/bias/Adam_1/Assign^main/pi/dense/bias/Assign!^main/pi/dense/kernel/Adam/Assign#^main/pi/dense/kernel/Adam_1/Assign^main/pi/dense/kernel/Assign!^main/pi/dense_1/bias/Adam/Assign#^main/pi/dense_1/bias/Adam_1/Assign^main/pi/dense_1/bias/Assign#^main/pi/dense_1/kernel/Adam/Assign%^main/pi/dense_1/kernel/Adam_1/Assign^main/pi/dense_1/kernel/Assign!^main/pi/dense_2/bias/Adam/Assign#^main/pi/dense_2/bias/Adam_1/Assign^main/pi/dense_2/bias/Assign#^main/pi/dense_2/kernel/Adam/Assign%^main/pi/dense_2/kernel/Adam_1/Assign^main/pi/dense_2/kernel/Assign^main/q1/dense/bias/Adam/Assign!^main/q1/dense/bias/Adam_1/Assign^main/q1/dense/bias/Assign!^main/q1/dense/kernel/Adam/Assign#^main/q1/dense/kernel/Adam_1/Assign^main/q1/dense/kernel/Assign!^main/q1/dense_1/bias/Adam/Assign#^main/q1/dense_1/bias/Adam_1/Assign^main/q1/dense_1/bias/Assign#^main/q1/dense_1/kernel/Adam/Assign%^main/q1/dense_1/kernel/Adam_1/Assign^main/q1/dense_1/kernel/Assign!^main/q1/dense_2/bias/Adam/Assign#^main/q1/dense_2/bias/Adam_1/Assign^main/q1/dense_2/bias/Assign#^main/q1/dense_2/kernel/Adam/Assign%^main/q1/dense_2/kernel/Adam_1/Assign^main/q1/dense_2/kernel/Assign^main/q2/dense/bias/Adam/Assign!^main/q2/dense/bias/Adam_1/Assign^main/q2/dense/bias/Assign!^main/q2/dense/kernel/Adam/Assign#^main/q2/dense/kernel/Adam_1/Assign^main/q2/dense/kernel/Assign!^main/q2/dense_1/bias/Adam/Assign#^main/q2/dense_1/bias/Adam_1/Assign^main/q2/dense_1/bias/Assign#^main/q2/dense_1/kernel/Adam/Assign%^main/q2/dense_1/kernel/Adam_1/Assign^main/q2/dense_1/kernel/Assign!^main/q2/dense_2/bias/Adam/Assign#^main/q2/dense_2/bias/Adam_1/Assign^main/q2/dense_2/bias/Assign#^main/q2/dense_2/kernel/Adam/Assign%^main/q2/dense_2/kernel/Adam_1/Assign^main/q2/dense_2/kernel/Assign^target/pi/dense/bias/Assign^target/pi/dense/kernel/Assign^target/pi/dense_1/bias/Assign ^target/pi/dense_1/kernel/Assign^target/pi/dense_2/bias/Assign ^target/pi/dense_2/kernel/Assign^target/q1/dense/bias/Assign^target/q1/dense/kernel/Assign^target/q1/dense_1/bias/Assign ^target/q1/dense_1/kernel/Assign^target/q1/dense_2/bias/Assign ^target/q1/dense_2/kernel/Assign^target/q2/dense/bias/Assign^target/q2/dense/kernel/Assign^target/q2/dense_1/bias/Assign ^target/q2/dense_1/kernel/Assign^target/q2/dense_2/bias/Assign ^target/q2/dense_2/kernel/Assign
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_4262425cdc004966ad311a0380801f3e/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Ţ
save/SaveV2/tensor_namesConst*
valueBLBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:L
ţ
save/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*­
valueŁB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ť
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kernel*Z
dtypesP
N2L

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*
_output_shapes
:*

axis *
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
á
save/RestoreV2/tensor_namesConst*
_output_shapes
:L*
dtype0*
valueBLBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel

save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*­
valueŁB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Z
dtypesP
N2L*Ć
_output_shapesł
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ł
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
T0
Š
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
_output_shapes
: *
validate_shape(*%
_class
loc:@main/q1/dense/bias*
T0*
use_locking(
§
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
_output_shapes
: *
T0*
validate_shape(*%
_class
loc:@main/pi/dense/bias*
use_locking(
Š
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
use_locking(*%
_class
loc:@main/q1/dense/bias*
T0*
validate_shape(*
_output_shapes
: 
ł
save/Assign_4Assignmain/pi/dense/biassave/RestoreV2:4*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(*%
_class
loc:@main/pi/dense/bias
¸
save/Assign_5Assignmain/pi/dense/bias/Adamsave/RestoreV2:5*
_output_shapes	
:*
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(*
validate_shape(
ş
save/Assign_6Assignmain/pi/dense/bias/Adam_1save/RestoreV2:6*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*%
_class
loc:@main/pi/dense/bias
ť
save/Assign_7Assignmain/pi/dense/kernelsave/RestoreV2:7*
use_locking(*
_output_shapes
:	*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0
Ŕ
save/Assign_8Assignmain/pi/dense/kernel/Adamsave/RestoreV2:8*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
Â
save/Assign_9Assignmain/pi/dense/kernel/Adam_1save/RestoreV2:9*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	*
use_locking(
š
save/Assign_10Assignmain/pi/dense_1/biassave/RestoreV2:10*
_output_shapes	
:*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
use_locking(*
T0
ž
save/Assign_11Assignmain/pi/dense_1/bias/Adamsave/RestoreV2:11*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes	
:*
T0
Ŕ
save/Assign_12Assignmain/pi/dense_1/bias/Adam_1save/RestoreV2:12*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_1/bias
Â
save/Assign_13Assignmain/pi/dense_1/kernelsave/RestoreV2:13*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

Ç
save/Assign_14Assignmain/pi/dense_1/kernel/Adamsave/RestoreV2:14*)
_class
loc:@main/pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

É
save/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save/RestoreV2:15*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(
¸
save/Assign_16Assignmain/pi/dense_2/biassave/RestoreV2:16*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
˝
save/Assign_17Assignmain/pi/dense_2/bias/Adamsave/RestoreV2:17*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias
ż
save/Assign_18Assignmain/pi/dense_2/bias/Adam_1save/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
Á
save/Assign_19Assignmain/pi/dense_2/kernelsave/RestoreV2:19*)
_class
loc:@main/pi/dense_2/kernel*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(
Ć
save/Assign_20Assignmain/pi/dense_2/kernel/Adamsave/RestoreV2:20*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0
Č
save/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save/RestoreV2:21*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_2/kernel
ľ
save/Assign_22Assignmain/q1/dense/biassave/RestoreV2:22*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*%
_class
loc:@main/q1/dense/bias
ş
save/Assign_23Assignmain/q1/dense/bias/Adamsave/RestoreV2:23*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
ź
save/Assign_24Assignmain/q1/dense/bias/Adam_1save/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0*%
_class
loc:@main/q1/dense/bias
˝
save/Assign_25Assignmain/q1/dense/kernelsave/RestoreV2:25*
use_locking(*
_output_shapes
:	*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
T0
Â
save/Assign_26Assignmain/q1/dense/kernel/Adamsave/RestoreV2:26*'
_class
loc:@main/q1/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Ä
save/Assign_27Assignmain/q1/dense/kernel/Adam_1save/RestoreV2:27*
_output_shapes
:	*
T0*
validate_shape(*'
_class
loc:@main/q1/dense/kernel*
use_locking(
š
save/Assign_28Assignmain/q1/dense_1/biassave/RestoreV2:28*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*'
_class
loc:@main/q1/dense_1/bias
ž
save/Assign_29Assignmain/q1/dense_1/bias/Adamsave/RestoreV2:29*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*'
_class
loc:@main/q1/dense_1/bias
Ŕ
save/Assign_30Assignmain/q1/dense_1/bias/Adam_1save/RestoreV2:30*
_output_shapes	
:*
validate_shape(*'
_class
loc:@main/q1/dense_1/bias*
use_locking(*
T0
Â
save/Assign_31Assignmain/q1/dense_1/kernelsave/RestoreV2:31*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*)
_class
loc:@main/q1/dense_1/kernel
Ç
save/Assign_32Assignmain/q1/dense_1/kernel/Adamsave/RestoreV2:32*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(*)
_class
loc:@main/q1/dense_1/kernel
É
save/Assign_33Assignmain/q1/dense_1/kernel/Adam_1save/RestoreV2:33* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*)
_class
loc:@main/q1/dense_1/kernel
¸
save/Assign_34Assignmain/q1/dense_2/biassave/RestoreV2:34*'
_class
loc:@main/q1/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
˝
save/Assign_35Assignmain/q1/dense_2/bias/Adamsave/RestoreV2:35*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*'
_class
loc:@main/q1/dense_2/bias
ż
save/Assign_36Assignmain/q1/dense_2/bias/Adam_1save/RestoreV2:36*'
_class
loc:@main/q1/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
Á
save/Assign_37Assignmain/q1/dense_2/kernelsave/RestoreV2:37*)
_class
loc:@main/q1/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	
Ć
save/Assign_38Assignmain/q1/dense_2/kernel/Adamsave/RestoreV2:38*)
_class
loc:@main/q1/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
Č
save/Assign_39Assignmain/q1/dense_2/kernel/Adam_1save/RestoreV2:39*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	
ľ
save/Assign_40Assignmain/q2/dense/biassave/RestoreV2:40*%
_class
loc:@main/q2/dense/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
ş
save/Assign_41Assignmain/q2/dense/bias/Adamsave/RestoreV2:41*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:
ź
save/Assign_42Assignmain/q2/dense/bias/Adam_1save/RestoreV2:42*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes	
:*
use_locking(*
validate_shape(
˝
save/Assign_43Assignmain/q2/dense/kernelsave/RestoreV2:43*'
_class
loc:@main/q2/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
Â
save/Assign_44Assignmain/q2/dense/kernel/Adamsave/RestoreV2:44*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes
:	
Ä
save/Assign_45Assignmain/q2/dense/kernel/Adam_1save/RestoreV2:45*
_output_shapes
:	*
use_locking(*
validate_shape(*'
_class
loc:@main/q2/dense/kernel*
T0
š
save/Assign_46Assignmain/q2/dense_1/biassave/RestoreV2:46*
_output_shapes	
:*
use_locking(*'
_class
loc:@main/q2/dense_1/bias*
T0*
validate_shape(
ž
save/Assign_47Assignmain/q2/dense_1/bias/Adamsave/RestoreV2:47*
use_locking(*
_output_shapes	
:*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(
Ŕ
save/Assign_48Assignmain/q2/dense_1/bias/Adam_1save/RestoreV2:48*
validate_shape(*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
Â
save/Assign_49Assignmain/q2/dense_1/kernelsave/RestoreV2:49*
use_locking(*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
T0* 
_output_shapes
:

Ç
save/Assign_50Assignmain/q2/dense_1/kernel/Adamsave/RestoreV2:50*
validate_shape(*)
_class
loc:@main/q2/dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
É
save/Assign_51Assignmain/q2/dense_1/kernel/Adam_1save/RestoreV2:51* 
_output_shapes
:
*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
use_locking(*
T0
¸
save/Assign_52Assignmain/q2/dense_2/biassave/RestoreV2:52*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
˝
save/Assign_53Assignmain/q2/dense_2/bias/Adamsave/RestoreV2:53*
use_locking(*
T0*
validate_shape(*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:
ż
save/Assign_54Assignmain/q2/dense_2/bias/Adam_1save/RestoreV2:54*
use_locking(*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
Á
save/Assign_55Assignmain/q2/dense_2/kernelsave/RestoreV2:55*)
_class
loc:@main/q2/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	*
use_locking(
Ć
save/Assign_56Assignmain/q2/dense_2/kernel/Adamsave/RestoreV2:56*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(
Č
save/Assign_57Assignmain/q2/dense_2/kernel/Adam_1save/RestoreV2:57*
use_locking(*)
_class
loc:@main/q2/dense_2/kernel*
T0*
validate_shape(*
_output_shapes
:	
š
save/Assign_58Assigntarget/pi/dense/biassave/RestoreV2:58*
T0*
validate_shape(*'
_class
loc:@target/pi/dense/bias*
use_locking(*
_output_shapes	
:
Á
save/Assign_59Assigntarget/pi/dense/kernelsave/RestoreV2:59*)
_class
loc:@target/pi/dense/kernel*
T0*
use_locking(*
_output_shapes
:	*
validate_shape(
˝
save/Assign_60Assigntarget/pi/dense_1/biassave/RestoreV2:60*
use_locking(*
validate_shape(*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes	
:*
T0
Ć
save/Assign_61Assigntarget/pi/dense_1/kernelsave/RestoreV2:61*
T0* 
_output_shapes
:
*
use_locking(*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(
ź
save/Assign_62Assigntarget/pi/dense_2/biassave/RestoreV2:62*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*)
_class
loc:@target/pi/dense_2/bias
Ĺ
save/Assign_63Assigntarget/pi/dense_2/kernelsave/RestoreV2:63*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
š
save/Assign_64Assigntarget/q1/dense/biassave/RestoreV2:64*'
_class
loc:@target/q1/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
Á
save/Assign_65Assigntarget/q1/dense/kernelsave/RestoreV2:65*
use_locking(*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	
˝
save/Assign_66Assigntarget/q1/dense_1/biassave/RestoreV2:66*
_output_shapes	
:*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
use_locking(
Ć
save/Assign_67Assigntarget/q1/dense_1/kernelsave/RestoreV2:67*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*+
_class!
loc:@target/q1/dense_1/kernel
ź
save/Assign_68Assigntarget/q1/dense_2/biassave/RestoreV2:68*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*)
_class
loc:@target/q1/dense_2/bias
Ĺ
save/Assign_69Assigntarget/q1/dense_2/kernelsave/RestoreV2:69*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*+
_class!
loc:@target/q1/dense_2/kernel
š
save/Assign_70Assigntarget/q2/dense/biassave/RestoreV2:70*
use_locking(*'
_class
loc:@target/q2/dense/bias*
_output_shapes	
:*
validate_shape(*
T0
Á
save/Assign_71Assigntarget/q2/dense/kernelsave/RestoreV2:71*
T0*
_output_shapes
:	*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
use_locking(
˝
save/Assign_72Assigntarget/q2/dense_1/biassave/RestoreV2:72*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*)
_class
loc:@target/q2/dense_1/bias
Ć
save/Assign_73Assigntarget/q2/dense_1/kernelsave/RestoreV2:73*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

ź
save/Assign_74Assigntarget/q2/dense_2/biassave/RestoreV2:74*
validate_shape(*)
_class
loc:@target/q2/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
Ĺ
save/Assign_75Assigntarget/q2/dense_2/kernelsave/RestoreV2:75*+
_class!
loc:@target/q2/dense_2/kernel*
use_locking(*
T0*
_output_shapes
:	*
validate_shape(


save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_60c09b8eae714010b96d083c52f862f4/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_1/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
ŕ
save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:L*
valueBLBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel

save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*­
valueŁB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ă
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kernel*Z
dtypesP
N2L

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
Ł
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*
_output_shapes
:*

axis 

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
ă
save_1/RestoreV2/tensor_namesConst*
valueBLBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
_output_shapes
:L*
dtype0

!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*­
valueŁB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*Z
dtypesP
N2L*Ć
_output_shapesł
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
§
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
use_locking(*
_output_shapes
: *
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0
­
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ť
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
_output_shapes
: *
validate_shape(*%
_class
loc:@main/pi/dense/bias*
T0*
use_locking(
­
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*%
_class
loc:@main/q1/dense/bias
ˇ
save_1/Assign_4Assignmain/pi/dense/biassave_1/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
use_locking(*
_output_shapes	
:*
validate_shape(*
T0
ź
save_1/Assign_5Assignmain/pi/dense/bias/Adamsave_1/RestoreV2:5*%
_class
loc:@main/pi/dense/bias*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(
ž
save_1/Assign_6Assignmain/pi/dense/bias/Adam_1save_1/RestoreV2:6*
_output_shapes	
:*
validate_shape(*
T0*%
_class
loc:@main/pi/dense/bias*
use_locking(
ż
save_1/Assign_7Assignmain/pi/dense/kernelsave_1/RestoreV2:7*
T0*
use_locking(*
_output_shapes
:	*'
_class
loc:@main/pi/dense/kernel*
validate_shape(
Ä
save_1/Assign_8Assignmain/pi/dense/kernel/Adamsave_1/RestoreV2:8*
validate_shape(*
_output_shapes
:	*'
_class
loc:@main/pi/dense/kernel*
T0*
use_locking(
Ć
save_1/Assign_9Assignmain/pi/dense/kernel/Adam_1save_1/RestoreV2:9*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
˝
save_1/Assign_10Assignmain/pi/dense_1/biassave_1/RestoreV2:10*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:
Â
save_1/Assign_11Assignmain/pi/dense_1/bias/Adamsave_1/RestoreV2:11*
_output_shapes	
:*
use_locking(*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
T0
Ä
save_1/Assign_12Assignmain/pi/dense_1/bias/Adam_1save_1/RestoreV2:12*
_output_shapes	
:*'
_class
loc:@main/pi/dense_1/bias*
T0*
validate_shape(*
use_locking(
Ć
save_1/Assign_13Assignmain/pi/dense_1/kernelsave_1/RestoreV2:13*
use_locking(*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel*
T0* 
_output_shapes
:

Ë
save_1/Assign_14Assignmain/pi/dense_1/kernel/Adamsave_1/RestoreV2:14* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*)
_class
loc:@main/pi/dense_1/kernel
Í
save_1/Assign_15Assignmain/pi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(*)
_class
loc:@main/pi/dense_1/kernel
ź
save_1/Assign_16Assignmain/pi/dense_2/biassave_1/RestoreV2:16*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
Á
save_1/Assign_17Assignmain/pi/dense_2/bias/Adamsave_1/RestoreV2:17*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ă
save_1/Assign_18Assignmain/pi/dense_2/bias/Adam_1save_1/RestoreV2:18*
_output_shapes
:*
validate_shape(*'
_class
loc:@main/pi/dense_2/bias*
use_locking(*
T0
Ĺ
save_1/Assign_19Assignmain/pi/dense_2/kernelsave_1/RestoreV2:19*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel
Ę
save_1/Assign_20Assignmain/pi/dense_2/kernel/Adamsave_1/RestoreV2:20*
_output_shapes
:	*
use_locking(*)
_class
loc:@main/pi/dense_2/kernel*
T0*
validate_shape(
Ě
save_1/Assign_21Assignmain/pi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
T0*)
_class
loc:@main/pi/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	
š
save_1/Assign_22Assignmain/q1/dense/biassave_1/RestoreV2:22*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*%
_class
loc:@main/q1/dense/bias
ž
save_1/Assign_23Assignmain/q1/dense/bias/Adamsave_1/RestoreV2:23*
use_locking(*
_output_shapes	
:*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
T0
Ŕ
save_1/Assign_24Assignmain/q1/dense/bias/Adam_1save_1/RestoreV2:24*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
Á
save_1/Assign_25Assignmain/q1/dense/kernelsave_1/RestoreV2:25*'
_class
loc:@main/q1/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	
Ć
save_1/Assign_26Assignmain/q1/dense/kernel/Adamsave_1/RestoreV2:26*
validate_shape(*'
_class
loc:@main/q1/dense/kernel*
use_locking(*
T0*
_output_shapes
:	
Č
save_1/Assign_27Assignmain/q1/dense/kernel/Adam_1save_1/RestoreV2:27*
validate_shape(*
use_locking(*
_output_shapes
:	*'
_class
loc:@main/q1/dense/kernel*
T0
˝
save_1/Assign_28Assignmain/q1/dense_1/biassave_1/RestoreV2:28*
_output_shapes	
:*
validate_shape(*'
_class
loc:@main/q1/dense_1/bias*
use_locking(*
T0
Â
save_1/Assign_29Assignmain/q1/dense_1/bias/Adamsave_1/RestoreV2:29*
T0*
_output_shapes	
:*'
_class
loc:@main/q1/dense_1/bias*
use_locking(*
validate_shape(
Ä
save_1/Assign_30Assignmain/q1/dense_1/bias/Adam_1save_1/RestoreV2:30*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(*'
_class
loc:@main/q1/dense_1/bias
Ć
save_1/Assign_31Assignmain/q1/dense_1/kernelsave_1/RestoreV2:31*)
_class
loc:@main/q1/dense_1/kernel* 
_output_shapes
:
*
T0*
use_locking(*
validate_shape(
Ë
save_1/Assign_32Assignmain/q1/dense_1/kernel/Adamsave_1/RestoreV2:32*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:
*)
_class
loc:@main/q1/dense_1/kernel
Í
save_1/Assign_33Assignmain/q1/dense_1/kernel/Adam_1save_1/RestoreV2:33*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
*)
_class
loc:@main/q1/dense_1/kernel
ź
save_1/Assign_34Assignmain/q1/dense_2/biassave_1/RestoreV2:34*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
Á
save_1/Assign_35Assignmain/q1/dense_2/bias/Adamsave_1/RestoreV2:35*
validate_shape(*
_output_shapes
:*
use_locking(*'
_class
loc:@main/q1/dense_2/bias*
T0
Ă
save_1/Assign_36Assignmain/q1/dense_2/bias/Adam_1save_1/RestoreV2:36*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*'
_class
loc:@main/q1/dense_2/bias
Ĺ
save_1/Assign_37Assignmain/q1/dense_2/kernelsave_1/RestoreV2:37*
_output_shapes
:	*
use_locking(*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
T0
Ę
save_1/Assign_38Assignmain/q1/dense_2/kernel/Adamsave_1/RestoreV2:38*
T0*
_output_shapes
:	*
validate_shape(*
use_locking(*)
_class
loc:@main/q1/dense_2/kernel
Ě
save_1/Assign_39Assignmain/q1/dense_2/kernel/Adam_1save_1/RestoreV2:39*)
_class
loc:@main/q1/dense_2/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
š
save_1/Assign_40Assignmain/q2/dense/biassave_1/RestoreV2:40*
T0*
_output_shapes	
:*
validate_shape(*%
_class
loc:@main/q2/dense/bias*
use_locking(
ž
save_1/Assign_41Assignmain/q2/dense/bias/Adamsave_1/RestoreV2:41*
_output_shapes	
:*
T0*
validate_shape(*
use_locking(*%
_class
loc:@main/q2/dense/bias
Ŕ
save_1/Assign_42Assignmain/q2/dense/bias/Adam_1save_1/RestoreV2:42*
_output_shapes	
:*
validate_shape(*
use_locking(*%
_class
loc:@main/q2/dense/bias*
T0
Á
save_1/Assign_43Assignmain/q2/dense/kernelsave_1/RestoreV2:43*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	*'
_class
loc:@main/q2/dense/kernel
Ć
save_1/Assign_44Assignmain/q2/dense/kernel/Adamsave_1/RestoreV2:44*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*'
_class
loc:@main/q2/dense/kernel
Č
save_1/Assign_45Assignmain/q2/dense/kernel/Adam_1save_1/RestoreV2:45*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*'
_class
loc:@main/q2/dense/kernel
˝
save_1/Assign_46Assignmain/q2/dense_1/biassave_1/RestoreV2:46*
use_locking(*
validate_shape(*
T0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes	
:
Â
save_1/Assign_47Assignmain/q2/dense_1/bias/Adamsave_1/RestoreV2:47*
_output_shapes	
:*
validate_shape(*
T0*'
_class
loc:@main/q2/dense_1/bias*
use_locking(
Ä
save_1/Assign_48Assignmain/q2/dense_1/bias/Adam_1save_1/RestoreV2:48*
validate_shape(*'
_class
loc:@main/q2/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
Ć
save_1/Assign_49Assignmain/q2/dense_1/kernelsave_1/RestoreV2:49*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ë
save_1/Assign_50Assignmain/q2/dense_1/kernel/Adamsave_1/RestoreV2:50*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:

Í
save_1/Assign_51Assignmain/q2/dense_1/kernel/Adam_1save_1/RestoreV2:51*)
_class
loc:@main/q2/dense_1/kernel*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(
ź
save_1/Assign_52Assignmain/q2/dense_2/biassave_1/RestoreV2:52*
use_locking(*'
_class
loc:@main/q2/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:
Á
save_1/Assign_53Assignmain/q2/dense_2/bias/Adamsave_1/RestoreV2:53*
_output_shapes
:*
validate_shape(*
use_locking(*'
_class
loc:@main/q2/dense_2/bias*
T0
Ă
save_1/Assign_54Assignmain/q2/dense_2/bias/Adam_1save_1/RestoreV2:54*
_output_shapes
:*'
_class
loc:@main/q2/dense_2/bias*
T0*
use_locking(*
validate_shape(
Ĺ
save_1/Assign_55Assignmain/q2/dense_2/kernelsave_1/RestoreV2:55*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(
Ę
save_1/Assign_56Assignmain/q2/dense_2/kernel/Adamsave_1/RestoreV2:56*
T0*
validate_shape(*)
_class
loc:@main/q2/dense_2/kernel*
use_locking(*
_output_shapes
:	
Ě
save_1/Assign_57Assignmain/q2/dense_2/kernel/Adam_1save_1/RestoreV2:57*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes
:	*
use_locking(*
validate_shape(*
T0
˝
save_1/Assign_58Assigntarget/pi/dense/biassave_1/RestoreV2:58*
T0*
_output_shapes	
:*
use_locking(*'
_class
loc:@target/pi/dense/bias*
validate_shape(
Ĺ
save_1/Assign_59Assigntarget/pi/dense/kernelsave_1/RestoreV2:59*
T0*
_output_shapes
:	*
use_locking(*)
_class
loc:@target/pi/dense/kernel*
validate_shape(
Á
save_1/Assign_60Assigntarget/pi/dense_1/biassave_1/RestoreV2:60*
_output_shapes	
:*
T0*
use_locking(*
validate_shape(*)
_class
loc:@target/pi/dense_1/bias
Ę
save_1/Assign_61Assigntarget/pi/dense_1/kernelsave_1/RestoreV2:61*
T0*
validate_shape(*+
_class!
loc:@target/pi/dense_1/kernel* 
_output_shapes
:
*
use_locking(
Ŕ
save_1/Assign_62Assigntarget/pi/dense_2/biassave_1/RestoreV2:62*)
_class
loc:@target/pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
É
save_1/Assign_63Assigntarget/pi/dense_2/kernelsave_1/RestoreV2:63*
_output_shapes
:	*
validate_shape(*+
_class!
loc:@target/pi/dense_2/kernel*
T0*
use_locking(
˝
save_1/Assign_64Assigntarget/q1/dense/biassave_1/RestoreV2:64*
_output_shapes	
:*
use_locking(*
validate_shape(*'
_class
loc:@target/q1/dense/bias*
T0
Ĺ
save_1/Assign_65Assigntarget/q1/dense/kernelsave_1/RestoreV2:65*
use_locking(*
T0*
validate_shape(*)
_class
loc:@target/q1/dense/kernel*
_output_shapes
:	
Á
save_1/Assign_66Assigntarget/q1/dense_1/biassave_1/RestoreV2:66*)
_class
loc:@target/q1/dense_1/bias*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(
Ę
save_1/Assign_67Assigntarget/q1/dense_1/kernelsave_1/RestoreV2:67*
T0* 
_output_shapes
:
*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(*
use_locking(
Ŕ
save_1/Assign_68Assigntarget/q1/dense_2/biassave_1/RestoreV2:68*)
_class
loc:@target/q1/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
É
save_1/Assign_69Assigntarget/q1/dense_2/kernelsave_1/RestoreV2:69*
_output_shapes
:	*
validate_shape(*+
_class!
loc:@target/q1/dense_2/kernel*
T0*
use_locking(
˝
save_1/Assign_70Assigntarget/q2/dense/biassave_1/RestoreV2:70*
T0*
use_locking(*
validate_shape(*'
_class
loc:@target/q2/dense/bias*
_output_shapes	
:
Ĺ
save_1/Assign_71Assigntarget/q2/dense/kernelsave_1/RestoreV2:71*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*)
_class
loc:@target/q2/dense/kernel
Á
save_1/Assign_72Assigntarget/q2/dense_1/biassave_1/RestoreV2:72*)
_class
loc:@target/q2/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(
Ę
save_1/Assign_73Assigntarget/q2/dense_1/kernelsave_1/RestoreV2:73*
use_locking(* 
_output_shapes
:
*+
_class!
loc:@target/q2/dense_1/kernel*
T0*
validate_shape(
Ŕ
save_1/Assign_74Assigntarget/q2/dense_2/biassave_1/RestoreV2:74*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
É
save_1/Assign_75Assigntarget/q2/dense_2/kernelsave_1/RestoreV2:75*+
_class!
loc:@target/q2/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0
´
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"÷T
	variableséTćT

main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:023main/pi/dense/kernel/Initializer/truncated_normal:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08

main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:025main/pi/dense_1/kernel/Initializer/truncated_normal:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08

main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08

main/q1/dense/kernel:0main/q1/dense/kernel/Assignmain/q1/dense/kernel/read:023main/q1/dense/kernel/Initializer/truncated_normal:08
v
main/q1/dense/bias:0main/q1/dense/bias/Assignmain/q1/dense/bias/read:02&main/q1/dense/bias/Initializer/zeros:08

main/q1/dense_1/kernel:0main/q1/dense_1/kernel/Assignmain/q1/dense_1/kernel/read:025main/q1/dense_1/kernel/Initializer/truncated_normal:08
~
main/q1/dense_1/bias:0main/q1/dense_1/bias/Assignmain/q1/dense_1/bias/read:02(main/q1/dense_1/bias/Initializer/zeros:08

main/q1/dense_2/kernel:0main/q1/dense_2/kernel/Assignmain/q1/dense_2/kernel/read:023main/q1/dense_2/kernel/Initializer/random_uniform:08
~
main/q1/dense_2/bias:0main/q1/dense_2/bias/Assignmain/q1/dense_2/bias/read:02(main/q1/dense_2/bias/Initializer/zeros:08

main/q2/dense/kernel:0main/q2/dense/kernel/Assignmain/q2/dense/kernel/read:023main/q2/dense/kernel/Initializer/truncated_normal:08
v
main/q2/dense/bias:0main/q2/dense/bias/Assignmain/q2/dense/bias/read:02&main/q2/dense/bias/Initializer/zeros:08

main/q2/dense_1/kernel:0main/q2/dense_1/kernel/Assignmain/q2/dense_1/kernel/read:025main/q2/dense_1/kernel/Initializer/truncated_normal:08
~
main/q2/dense_1/bias:0main/q2/dense_1/bias/Assignmain/q2/dense_1/bias/read:02(main/q2/dense_1/bias/Initializer/zeros:08

main/q2/dense_2/kernel:0main/q2/dense_2/kernel/Assignmain/q2/dense_2/kernel/read:023main/q2/dense_2/kernel/Initializer/random_uniform:08
~
main/q2/dense_2/bias:0main/q2/dense_2/bias/Assignmain/q2/dense_2/bias/read:02(main/q2/dense_2/bias/Initializer/zeros:08

target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:025target/pi/dense/kernel/Initializer/truncated_normal:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08

target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:027target/pi/dense_1/kernel/Initializer/truncated_normal:08

target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08

target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08

target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08

target/q1/dense/kernel:0target/q1/dense/kernel/Assigntarget/q1/dense/kernel/read:025target/q1/dense/kernel/Initializer/truncated_normal:08
~
target/q1/dense/bias:0target/q1/dense/bias/Assigntarget/q1/dense/bias/read:02(target/q1/dense/bias/Initializer/zeros:08

target/q1/dense_1/kernel:0target/q1/dense_1/kernel/Assigntarget/q1/dense_1/kernel/read:027target/q1/dense_1/kernel/Initializer/truncated_normal:08

target/q1/dense_1/bias:0target/q1/dense_1/bias/Assigntarget/q1/dense_1/bias/read:02*target/q1/dense_1/bias/Initializer/zeros:08

target/q1/dense_2/kernel:0target/q1/dense_2/kernel/Assigntarget/q1/dense_2/kernel/read:025target/q1/dense_2/kernel/Initializer/random_uniform:08

target/q1/dense_2/bias:0target/q1/dense_2/bias/Assigntarget/q1/dense_2/bias/read:02*target/q1/dense_2/bias/Initializer/zeros:08

target/q2/dense/kernel:0target/q2/dense/kernel/Assigntarget/q2/dense/kernel/read:025target/q2/dense/kernel/Initializer/truncated_normal:08
~
target/q2/dense/bias:0target/q2/dense/bias/Assigntarget/q2/dense/bias/read:02(target/q2/dense/bias/Initializer/zeros:08

target/q2/dense_1/kernel:0target/q2/dense_1/kernel/Assigntarget/q2/dense_1/kernel/read:027target/q2/dense_1/kernel/Initializer/truncated_normal:08

target/q2/dense_1/bias:0target/q2/dense_1/bias/Assigntarget/q2/dense_1/bias/read:02*target/q2/dense_1/bias/Initializer/zeros:08

target/q2/dense_2/kernel:0target/q2/dense_2/kernel/Assigntarget/q2/dense_2/kernel/read:025target/q2/dense_2/kernel/Initializer/random_uniform:08

target/q2/dense_2/bias:0target/q2/dense_2/bias/Assigntarget/q2/dense_2/bias/read:02*target/q2/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

main/pi/dense/kernel/Adam:0 main/pi/dense/kernel/Adam/Assign main/pi/dense/kernel/Adam/read:02-main/pi/dense/kernel/Adam/Initializer/zeros:0

main/pi/dense/kernel/Adam_1:0"main/pi/dense/kernel/Adam_1/Assign"main/pi/dense/kernel/Adam_1/read:02/main/pi/dense/kernel/Adam_1/Initializer/zeros:0

main/pi/dense/bias/Adam:0main/pi/dense/bias/Adam/Assignmain/pi/dense/bias/Adam/read:02+main/pi/dense/bias/Adam/Initializer/zeros:0

main/pi/dense/bias/Adam_1:0 main/pi/dense/bias/Adam_1/Assign main/pi/dense/bias/Adam_1/read:02-main/pi/dense/bias/Adam_1/Initializer/zeros:0

main/pi/dense_1/kernel/Adam:0"main/pi/dense_1/kernel/Adam/Assign"main/pi/dense_1/kernel/Adam/read:02/main/pi/dense_1/kernel/Adam/Initializer/zeros:0
 
main/pi/dense_1/kernel/Adam_1:0$main/pi/dense_1/kernel/Adam_1/Assign$main/pi/dense_1/kernel/Adam_1/read:021main/pi/dense_1/kernel/Adam_1/Initializer/zeros:0

main/pi/dense_1/bias/Adam:0 main/pi/dense_1/bias/Adam/Assign main/pi/dense_1/bias/Adam/read:02-main/pi/dense_1/bias/Adam/Initializer/zeros:0

main/pi/dense_1/bias/Adam_1:0"main/pi/dense_1/bias/Adam_1/Assign"main/pi/dense_1/bias/Adam_1/read:02/main/pi/dense_1/bias/Adam_1/Initializer/zeros:0

main/pi/dense_2/kernel/Adam:0"main/pi/dense_2/kernel/Adam/Assign"main/pi/dense_2/kernel/Adam/read:02/main/pi/dense_2/kernel/Adam/Initializer/zeros:0
 
main/pi/dense_2/kernel/Adam_1:0$main/pi/dense_2/kernel/Adam_1/Assign$main/pi/dense_2/kernel/Adam_1/read:021main/pi/dense_2/kernel/Adam_1/Initializer/zeros:0

main/pi/dense_2/bias/Adam:0 main/pi/dense_2/bias/Adam/Assign main/pi/dense_2/bias/Adam/read:02-main/pi/dense_2/bias/Adam/Initializer/zeros:0

main/pi/dense_2/bias/Adam_1:0"main/pi/dense_2/bias/Adam_1/Assign"main/pi/dense_2/bias/Adam_1/read:02/main/pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0

main/q1/dense/kernel/Adam:0 main/q1/dense/kernel/Adam/Assign main/q1/dense/kernel/Adam/read:02-main/q1/dense/kernel/Adam/Initializer/zeros:0

main/q1/dense/kernel/Adam_1:0"main/q1/dense/kernel/Adam_1/Assign"main/q1/dense/kernel/Adam_1/read:02/main/q1/dense/kernel/Adam_1/Initializer/zeros:0

main/q1/dense/bias/Adam:0main/q1/dense/bias/Adam/Assignmain/q1/dense/bias/Adam/read:02+main/q1/dense/bias/Adam/Initializer/zeros:0

main/q1/dense/bias/Adam_1:0 main/q1/dense/bias/Adam_1/Assign main/q1/dense/bias/Adam_1/read:02-main/q1/dense/bias/Adam_1/Initializer/zeros:0

main/q1/dense_1/kernel/Adam:0"main/q1/dense_1/kernel/Adam/Assign"main/q1/dense_1/kernel/Adam/read:02/main/q1/dense_1/kernel/Adam/Initializer/zeros:0
 
main/q1/dense_1/kernel/Adam_1:0$main/q1/dense_1/kernel/Adam_1/Assign$main/q1/dense_1/kernel/Adam_1/read:021main/q1/dense_1/kernel/Adam_1/Initializer/zeros:0

main/q1/dense_1/bias/Adam:0 main/q1/dense_1/bias/Adam/Assign main/q1/dense_1/bias/Adam/read:02-main/q1/dense_1/bias/Adam/Initializer/zeros:0

main/q1/dense_1/bias/Adam_1:0"main/q1/dense_1/bias/Adam_1/Assign"main/q1/dense_1/bias/Adam_1/read:02/main/q1/dense_1/bias/Adam_1/Initializer/zeros:0

main/q1/dense_2/kernel/Adam:0"main/q1/dense_2/kernel/Adam/Assign"main/q1/dense_2/kernel/Adam/read:02/main/q1/dense_2/kernel/Adam/Initializer/zeros:0
 
main/q1/dense_2/kernel/Adam_1:0$main/q1/dense_2/kernel/Adam_1/Assign$main/q1/dense_2/kernel/Adam_1/read:021main/q1/dense_2/kernel/Adam_1/Initializer/zeros:0

main/q1/dense_2/bias/Adam:0 main/q1/dense_2/bias/Adam/Assign main/q1/dense_2/bias/Adam/read:02-main/q1/dense_2/bias/Adam/Initializer/zeros:0

main/q1/dense_2/bias/Adam_1:0"main/q1/dense_2/bias/Adam_1/Assign"main/q1/dense_2/bias/Adam_1/read:02/main/q1/dense_2/bias/Adam_1/Initializer/zeros:0

main/q2/dense/kernel/Adam:0 main/q2/dense/kernel/Adam/Assign main/q2/dense/kernel/Adam/read:02-main/q2/dense/kernel/Adam/Initializer/zeros:0

main/q2/dense/kernel/Adam_1:0"main/q2/dense/kernel/Adam_1/Assign"main/q2/dense/kernel/Adam_1/read:02/main/q2/dense/kernel/Adam_1/Initializer/zeros:0

main/q2/dense/bias/Adam:0main/q2/dense/bias/Adam/Assignmain/q2/dense/bias/Adam/read:02+main/q2/dense/bias/Adam/Initializer/zeros:0

main/q2/dense/bias/Adam_1:0 main/q2/dense/bias/Adam_1/Assign main/q2/dense/bias/Adam_1/read:02-main/q2/dense/bias/Adam_1/Initializer/zeros:0

main/q2/dense_1/kernel/Adam:0"main/q2/dense_1/kernel/Adam/Assign"main/q2/dense_1/kernel/Adam/read:02/main/q2/dense_1/kernel/Adam/Initializer/zeros:0
 
main/q2/dense_1/kernel/Adam_1:0$main/q2/dense_1/kernel/Adam_1/Assign$main/q2/dense_1/kernel/Adam_1/read:021main/q2/dense_1/kernel/Adam_1/Initializer/zeros:0

main/q2/dense_1/bias/Adam:0 main/q2/dense_1/bias/Adam/Assign main/q2/dense_1/bias/Adam/read:02-main/q2/dense_1/bias/Adam/Initializer/zeros:0

main/q2/dense_1/bias/Adam_1:0"main/q2/dense_1/bias/Adam_1/Assign"main/q2/dense_1/bias/Adam_1/read:02/main/q2/dense_1/bias/Adam_1/Initializer/zeros:0

main/q2/dense_2/kernel/Adam:0"main/q2/dense_2/kernel/Adam/Assign"main/q2/dense_2/kernel/Adam/read:02/main/q2/dense_2/kernel/Adam/Initializer/zeros:0
 
main/q2/dense_2/kernel/Adam_1:0$main/q2/dense_2/kernel/Adam_1/Assign$main/q2/dense_2/kernel/Adam_1/read:021main/q2/dense_2/kernel/Adam_1/Initializer/zeros:0

main/q2/dense_2/bias/Adam:0 main/q2/dense_2/bias/Adam/Assign main/q2/dense_2/bias/Adam/read:02-main/q2/dense_2/bias/Adam/Initializer/zeros:0

main/q2/dense_2/bias/Adam_1:0"main/q2/dense_2/bias/Adam_1/Assign"main/q2/dense_2/bias/Adam_1/read:02/main/q2/dense_2/bias/Adam_1/Initializer/zeros:0"­'
trainable_variables''

main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:023main/pi/dense/kernel/Initializer/truncated_normal:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08

main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:025main/pi/dense_1/kernel/Initializer/truncated_normal:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08

main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08

main/q1/dense/kernel:0main/q1/dense/kernel/Assignmain/q1/dense/kernel/read:023main/q1/dense/kernel/Initializer/truncated_normal:08
v
main/q1/dense/bias:0main/q1/dense/bias/Assignmain/q1/dense/bias/read:02&main/q1/dense/bias/Initializer/zeros:08

main/q1/dense_1/kernel:0main/q1/dense_1/kernel/Assignmain/q1/dense_1/kernel/read:025main/q1/dense_1/kernel/Initializer/truncated_normal:08
~
main/q1/dense_1/bias:0main/q1/dense_1/bias/Assignmain/q1/dense_1/bias/read:02(main/q1/dense_1/bias/Initializer/zeros:08

main/q1/dense_2/kernel:0main/q1/dense_2/kernel/Assignmain/q1/dense_2/kernel/read:023main/q1/dense_2/kernel/Initializer/random_uniform:08
~
main/q1/dense_2/bias:0main/q1/dense_2/bias/Assignmain/q1/dense_2/bias/read:02(main/q1/dense_2/bias/Initializer/zeros:08

main/q2/dense/kernel:0main/q2/dense/kernel/Assignmain/q2/dense/kernel/read:023main/q2/dense/kernel/Initializer/truncated_normal:08
v
main/q2/dense/bias:0main/q2/dense/bias/Assignmain/q2/dense/bias/read:02&main/q2/dense/bias/Initializer/zeros:08

main/q2/dense_1/kernel:0main/q2/dense_1/kernel/Assignmain/q2/dense_1/kernel/read:025main/q2/dense_1/kernel/Initializer/truncated_normal:08
~
main/q2/dense_1/bias:0main/q2/dense_1/bias/Assignmain/q2/dense_1/bias/read:02(main/q2/dense_1/bias/Initializer/zeros:08

main/q2/dense_2/kernel:0main/q2/dense_2/kernel/Assignmain/q2/dense_2/kernel/read:023main/q2/dense_2/kernel/Initializer/random_uniform:08
~
main/q2/dense_2/bias:0main/q2/dense_2/bias/Assignmain/q2/dense_2/bias/read:02(main/q2/dense_2/bias/Initializer/zeros:08

target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:025target/pi/dense/kernel/Initializer/truncated_normal:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08

target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:027target/pi/dense_1/kernel/Initializer/truncated_normal:08

target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08

target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08

target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08

target/q1/dense/kernel:0target/q1/dense/kernel/Assigntarget/q1/dense/kernel/read:025target/q1/dense/kernel/Initializer/truncated_normal:08
~
target/q1/dense/bias:0target/q1/dense/bias/Assigntarget/q1/dense/bias/read:02(target/q1/dense/bias/Initializer/zeros:08

target/q1/dense_1/kernel:0target/q1/dense_1/kernel/Assigntarget/q1/dense_1/kernel/read:027target/q1/dense_1/kernel/Initializer/truncated_normal:08

target/q1/dense_1/bias:0target/q1/dense_1/bias/Assigntarget/q1/dense_1/bias/read:02*target/q1/dense_1/bias/Initializer/zeros:08

target/q1/dense_2/kernel:0target/q1/dense_2/kernel/Assigntarget/q1/dense_2/kernel/read:025target/q1/dense_2/kernel/Initializer/random_uniform:08

target/q1/dense_2/bias:0target/q1/dense_2/bias/Assigntarget/q1/dense_2/bias/read:02*target/q1/dense_2/bias/Initializer/zeros:08

target/q2/dense/kernel:0target/q2/dense/kernel/Assigntarget/q2/dense/kernel/read:025target/q2/dense/kernel/Initializer/truncated_normal:08
~
target/q2/dense/bias:0target/q2/dense/bias/Assigntarget/q2/dense/bias/read:02(target/q2/dense/bias/Initializer/zeros:08

target/q2/dense_1/kernel:0target/q2/dense_1/kernel/Assigntarget/q2/dense_1/kernel/read:027target/q2/dense_1/kernel/Initializer/truncated_normal:08

target/q2/dense_1/bias:0target/q2/dense_1/bias/Assigntarget/q2/dense_1/bias/read:02*target/q2/dense_1/bias/Initializer/zeros:08

target/q2/dense_2/kernel:0target/q2/dense_2/kernel/Assigntarget/q2/dense_2/kernel/read:025target/q2/dense_2/kernel/Initializer/random_uniform:08

target/q2/dense_2/bias:0target/q2/dense_2/bias/Assigntarget/q2/dense_2/bias/read:02*target/q2/dense_2/bias/Initializer/zeros:08"
train_op

Adam
Adam_1*
serving_defaultř
)
x$
Placeholder:0˙˙˙˙˙˙˙˙˙
+
a&
Placeholder_1:0˙˙˙˙˙˙˙˙˙*
q1$
main/q1/Squeeze:0˙˙˙˙˙˙˙˙˙*
q2$
main/q2/Squeeze:0˙˙˙˙˙˙˙˙˙*
pi$
main/pi/mul:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict