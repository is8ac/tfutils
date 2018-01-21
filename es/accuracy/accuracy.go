// Package accuracy provides OPs to calculate accurecy.
package accuracy

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Percent calculates the percent of the actual whoes top value index matches the target.
// Expects actual to be onehot, and target to be indexes.
func Percent(s *op.Scope, actual, target tf.Output) (accuracy tf.Output) {
	actualLabels := op.Cast(s.SubScope("actual"),
		op.ArgMax(s, actual, op.Const(s.SubScope("argmax_dim"), int32(-1)), op.ArgMaxOutputType(tf.Int32)),
		target.DataType(),
	)
	correct := op.Reshape(s, op.Equal(s, actualLabels, target), op.Const(s.SubScope("all"), []int32{-1}))
	accuracy = op.Mean(s, op.Cast(s.SubScope("accuracy"), correct, tf.Float), op.Const(s.SubScope("mean_dim"), int32(0)))
	return
}
