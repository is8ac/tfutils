package tfutils

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// VarCache caches data in a veriable.
func VarCache(s *op.Scope, input tf.Output, name string) (init *tf.Operation, output tf.Output) {
	variable := op.VarHandleOp(s, input.DataType(), input.Shape(), op.VarHandleOpSharedName(name))
	init = op.AssignVariableOp(s, variable, input)
	output = op.ReadVariableOp(s, variable, input.DataType())
	return
}

// Zero returns a function which makes a const of shape of type dType.
func Zero(dType tf.DataType, shape tf.Shape) func(*op.Scope) tf.Output {
	dims, err := shape.ToSlice()
	if err != nil {
		panic(err)
	}
	return func(s *op.Scope) (zero tf.Output) {
		floatZero := op.Const(s.SubScope("float_zero"), float32(0))
		if dType == tf.Quint8 {
			zero, _, _ = op.QuantizeV2(s, floatZero, op.Const(s.SubScope("min"), float32(-1)), op.Const(s.SubScope("min"), float32(+1)), tf.Quint8)
			return
		}
		scalarZero := op.Cast(s.SubScope("zero"), floatZero, dType)
		zero = op.Fill(s, op.Cast(s, op.Const(s.SubScope("shape"), dims), tf.Int32), scalarZero)
		return
	}
}
